/*
 * kmeans_gpu_single.cu — single-GPU Lloyd's K-means (CUDA).
 *
 * Data: CSV with header row skipped; column 0 = song id (string), columns 1..15 = features.
 *       Feature cells are floats, or boolean tokens (true/false, case-insensitive) mapped to 1/0.
 *       Features live in a flat host array features[i*D + d] for one H2D copy of all points.
 *
 * Algorithm (host/device split):
 *   1) Host loads data, picks K initial centroids (first K rows, or shuffled if seed != 0).
 *   2) Each iteration:
 *        - Zero device buffers for per-cluster coordinate sums and point counts.
 *        - Launch one thread per point: assign cluster by minimum squared Euclidean distance
 *          (no sqrt); atomically accumulate sums and counts in global memory (no shared mem).
 *        - Copy sums/counts to host; host divides to form new centroids (empty clusters keep
 *          previous centroid); copy centroids back to device.
 *        - Stop when max coordinate change < tol or max_iters reached.
 *   3) Copy final cluster labels to host and write id,cluster CSV.
 *
 * Default input file: tracks_features_cleaned.csv
 */

#include <cuda_runtime.h>

#include <cctype>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

// Feature dimension (columns 1..15 of the CSV).
static constexpr int D = 15;
// CUDA block size for 1D grid over points.
static constexpr int BLOCK_SIZE = 256;

// Wrap CUDA API calls: on failure print a message and exit.
#define CUDA_CHECK(call)                                                                 \
  do {                                                                                   \
    cudaError_t err__ = (call);                                                          \
    if (err__ != cudaSuccess) {                                                          \
      std::cerr << "CUDA error " << cudaGetErrorString(err__) << " at " << __FILE__      \
                << ":" << __LINE__ << "\n";                                              \
      std::exit(1);                                                                      \
    }                                                                                    \
  } while (0)

// -----------------------------------------------------------------------------
// CSV parsing (host)
// -----------------------------------------------------------------------------

// Split one CSV record into fields; commas inside double quotes are not delimiters.
static std::vector<std::string> parse_csv_line(const std::string& line) {
  std::vector<std::string> result;
  std::string current;
  bool in_quotes = false;
  for (char c : line) {
    if (c == '"') {
      in_quotes = !in_quotes;
    } else if (c == ',' && !in_quotes) {
      result.push_back(current);
      current.clear();
    } else {
      current += c;
    }
  }
  result.push_back(current);
  return result;
}

// Parse numeric feature tokens and boolean-like tokens (true/false, 1/0).
static float parse_feature_value(std::string token) {
  for (char& c : token) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  if (token == "true") return 1.0f;
  if (token == "false") return 0.0f;
  if (token == "1") return 1.0f;
  if (token == "0") return 0.0f;
  try {
    return std::stof(token);
  } catch (...) {
    return 0.0f;
  }
}

// Read the dataset: skip the first line (header), append each id to ids, append D floats per row
// to features in row-major order. Returns false on I/O or parse error; warns and skips short rows.
static bool load_dataset(const std::string& path, std::vector<std::string>& ids,
                         std::vector<float>& features) {
  std::ifstream in(path);
  if (!in) {
    std::cerr << "Error: could not open " << path << "\n";
    return false;
  }
  std::string line;
  if (!std::getline(in, line)) {
    std::cerr << "Error: empty file " << path << "\n";
    return false;
  }
  size_t row = 0;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    std::vector<std::string> cols = parse_csv_line(line);
    if (cols.size() < static_cast<size_t>(D + 1)) {
      std::cerr << "Warning: skipping row " << row << " (expected " << (D + 1)
                << " columns, got " << cols.size() << ")\n";
      ++row;
      continue;
    }
    ids.push_back(cols[0]);
    for (int d = 0; d < D; ++d) {
      float v = parse_feature_value(cols[1 + d]);
      features.push_back(v);
    }
    ++row;
  }
  return !ids.empty();
}

// -----------------------------------------------------------------------------
// CUDA kernel: assignment + reduction via atomics
// -----------------------------------------------------------------------------

// One thread per point. Writes cluster index for that point, then atomically adds its feature
// vector to centroid_sums[best_c] and increments centroid_counts[best_c]. Sums must be zeroed
// before each launch. Uses squared distance only (sqrt omitted).
__global__ void kmeans_assign_accumulate(const float* __restrict__ points,
                                         const float* __restrict__ centroids, int n_points,
                                         int k, int* __restrict__ assignments,
                                         float* __restrict__ centroid_sums,
                                         int* __restrict__ centroid_counts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_points) {
    return;
  }
  int best_c = 0;
  float best_dist = FLT_MAX;
  for (int c = 0; c < k; ++c) {
    float dist = 0.f;
    for (int d = 0; d < D; ++d) {
      float diff = points[idx * D + d] - centroids[c * D + d];
      dist += diff * diff;
    }
    if (dist < best_dist) {
      best_dist = dist;
      best_c = c;
    }
  }
  assignments[idx] = best_c;
  for (int d = 0; d < D; ++d) {
    atomicAdd(&centroid_sums[best_c * D + d], points[idx * D + d]);
  }
  atomicAdd(&centroid_counts[best_c], 1);
}

// Print argv/positional-parameter summary to stderr.
static void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " [csv_path=" << "tracks_features_cleaned.csv"
            << "] [K=8] [max_iters=100] [tol=1e-4] [seed=0] [out_csv=assignments.csv]\n";
}

// -----------------------------------------------------------------------------
// main: CLI, Lloyd loop on host, single H2D copy of all points
// -----------------------------------------------------------------------------

int main(int argc, char** argv) {
  // --- CLI defaults and optional overrides ---
  std::string csv_path = "tracks_features_cleaned.csv";
  int k = 8;
  int max_iters = 100;
  float tol = 1e-4f;
  unsigned seed = 0;
  std::string out_csv = "assignments.csv";

  if (argc > 1) {
    if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
      usage(argv[0]);
      return 0;
    }
    csv_path = argv[1];
  }
  if (argc > 2) {
    k = std::atoi(argv[2]);
  }
  if (argc > 3) {
    max_iters = std::atoi(argv[3]);
  }
  if (argc > 4) {
    tol = std::strtof(argv[4], nullptr);
  }
  if (argc > 5) {
    seed = static_cast<unsigned>(std::strtoul(argv[5], nullptr, 10));
  }
  if (argc > 6) {
    out_csv = argv[6];
  }

  if (k < 1) {
    std::cerr << "K must be >= 1\n";
    return 1;
  }

  // --- Load points into host vectors ---
  std::vector<std::string> ids;
  std::vector<float> features;
  if (!load_dataset(csv_path, ids, features)) {
    return 1;
  }
  const int n = static_cast<int>(ids.size());
  if (n < k) {
    std::cerr << "Need at least K data points (n=" << n << ", K=" << k << ")\n";
    return 1;
  }

  // --- Initial centroids: rows perm[c] for c in 0..K-1 (shuffle if seed != 0) ---
  std::vector<float> fmin(D, FLT_MAX);
  std::vector<float> fmax(D, -FLT_MAX);
  for (int i = 0; i < n; ++i) {
    for (int d = 0; d < D; ++d) {
      float val = features[i * D + d];
      if (val < fmin[d]) fmin[d] = val;
      if (val > fmax[d]) fmax[d] = val;
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int d = 0; d < D; ++d) {
      float range = fmax[d] - fmin[d];
      features[i * D + d] = (range > 0) ? (features[i * D + d] - fmin[d]) / range : 0.0f;
    }
  }

  srand(42);
  std::vector<int> indices;
  while ((int)indices.size() < k) {
    int idx = rand() % n;
    bool dup = false;
    for (int x : indices) if (x == idx) { dup = true; break; }
    if (!dup) indices.push_back(idx);
  }

  std::vector<float> h_centroids(static_cast<size_t>(k) * D);
  for (int c = 0; c < k; ++c) {
    int row = indices[c];
    for (int d = 0; d < D; ++d) {
      h_centroids[c * D + d] = features[row * D + d];
    }
  }

  // --- Device allocation: full dataset once; per-iter centroid stats ---
  float* d_points = nullptr;
  float* d_centroids = nullptr;
  float* d_centroid_sums = nullptr;
  int* d_centroid_counts = nullptr;
  int* d_assignments = nullptr;

  CUDA_CHECK(cudaMalloc(&d_points, static_cast<size_t>(n) * D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_centroids, static_cast<size_t>(k) * D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_centroid_sums, static_cast<size_t>(k) * D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_centroid_counts, static_cast<size_t>(k) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_assignments, static_cast<size_t>(n) * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_points, features.data(), static_cast<size_t>(n) * D * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids.data(),
                        static_cast<size_t>(k) * D * sizeof(float), cudaMemcpyHostToDevice));

  // Host buffers for reduction results and final labels
  std::vector<float> h_sums(static_cast<size_t>(k) * D);
  std::vector<int> h_counts(static_cast<size_t>(k));
  std::vector<int> h_assignments(static_cast<size_t>(n));

  const int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  auto t0 = std::chrono::steady_clock::now();

  // --- Lloyd iterations: device assign+atomic reduce, host centroid update + convergence ---
  bool converged = false;
  for (int it = 0; it < max_iters; ++it) {
    CUDA_CHECK(cudaMemset(d_centroid_sums, 0, static_cast<size_t>(k) * D * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_centroid_counts, 0, static_cast<size_t>(k) * sizeof(int)));

    kmeans_assign_accumulate<<<grid, BLOCK_SIZE>>>(d_points, d_centroids, n, k, d_assignments,
                                                    d_centroid_sums, d_centroid_counts);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_sums.data(), d_centroid_sums,
                          static_cast<size_t>(k) * D * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_counts.data(), d_centroid_counts,
                          static_cast<size_t>(k) * sizeof(int), cudaMemcpyDeviceToHost));

    float max_move = 0.f;
    for (int c = 0; c < k; ++c) {
      if (h_counts[static_cast<size_t>(c)] > 0) {
        for (int d = 0; d < D; ++d) {
          float old_v = h_centroids[static_cast<size_t>(c) * D + d];
          float new_v =
              h_sums[static_cast<size_t>(c) * D + d] /
              static_cast<float>(h_counts[static_cast<size_t>(c)]);
          float ad = std::fabs(new_v - old_v);
          if (ad > max_move) {
            max_move = ad;
          }
          h_centroids[static_cast<size_t>(c) * D + d] = new_v;
        }
      }
    }

    CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids.data(),
                          static_cast<size_t>(k) * D * sizeof(float), cudaMemcpyHostToDevice));

    if (max_move < tol) {
      converged = true;
      break;
    }
  }

  // --- Final labels and CSV output ---
  CUDA_CHECK(cudaMemcpy(h_assignments.data(), d_assignments,
                        static_cast<size_t>(n) * sizeof(int), cudaMemcpyDeviceToHost));

  auto t1 = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = t1 - t0;
  std::cout << "K-means finished"
            << (converged ? " (converged)" : " (max iterations)") << " in " << elapsed.count()
            << " s\n";

  std::ofstream out(out_csv);
  if (!out) {
    std::cerr << "Error: could not write " << out_csv << "\n";
    return 1;
  }
  out << "id,explicit,danceability,energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo,duration_ms,time_signature,year,cluster\n";
  for (int i = 0; i < n; ++i) {
    out << ids[i] << ",";
    for (int d = 0; d < D; ++d) {
      out << features[i * D + d] << ",";
    }
    out << h_assignments[i] << "\n";
  }

  cudaFree(d_points);
  cudaFree(d_centroids);
  cudaFree(d_centroid_sums);
  cudaFree(d_centroid_counts);
  cudaFree(d_assignments);

  return 0;
}
