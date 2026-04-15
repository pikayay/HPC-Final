/*
 * kmeans_gpu_dist.cu — distributed-memory Lloyd's K-means (MPI + CUDA).
 *
 * Roles:
 *   - Rank 0 reads the full CSV (same layout as the single-GPU program: ids + 15 numeric/boolean
 *     features; true/false tokens become 1.0/0.0), keeps all ids for the final output, and
 *     broadcasts the global row count n.
 *   - Row-major feature chunks are distributed with MPI_Scatterv (handles n % P != 0).
 *   - Song IDs are scattered per chunk via MPI_Scatterv on string lengths and raw bytes so every
 *     rank materializes local_ids (mirrors the assignment spec); rank 0 still writes output using
 *     its full ids vector and gathered labels.
 *
 * Per iteration (all ranks):
 *   - Each rank runs the same CUDA kernel as the single-GPU code on its local points (or skips
 *     GPU work if local_n == 0).
 *   - Local centroid sums and counts are MPI_Allreduce'd (sum); every rank recomputes identical
 *     centroids and pushes them to its GPU.
 *   - Convergence uses the same max coordinate delta test on all ranks (deterministic).
 *
 * After the loop, MPI_Gatherv collects cluster labels to rank 0 for CSV export.
 */

#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

static constexpr int D = 15;
static constexpr int BLOCK_SIZE = 256;

// CUDA call wrapper: abort the whole MPI job on failure so ranks do not hang.
#define CUDA_CHECK(call)                                                                 \
  do {                                                                                   \
    cudaError_t err__ = (call);                                                          \
    if (err__ != cudaSuccess) {                                                          \
      std::cerr << "[rank ?] CUDA error " << cudaGetErrorString(err__) << " at "        \
                << __FILE__ << ":" << __LINE__ << "\n";                                  \
      MPI_Abort(MPI_COMM_WORLD, 1);                                                      \
    }                                                                                    \
  } while (0)

// -----------------------------------------------------------------------------
// CSV parsing (host, rank 0 only in practice)
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
  if (token == "false" || token == "flase") return 0.0f;
  if (token == "1") return 1.0f;
  if (token == "0") return 0.0f;
  try {
    return std::stof(token);
  } catch (...) {
    return 0.0f;
  }
}

// Read dataset after header: fills ids and flat features (row-major). Same contract as single-GPU.
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
// CUDA kernel (same logic as kmeans_gpu_single.cu)
// -----------------------------------------------------------------------------

// One thread per local point: squared-distance assignment; atomic sum/count into global buffers.
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

// -----------------------------------------------------------------------------
// MPI helpers
// -----------------------------------------------------------------------------

// Print usage from rank 0 only (stderr).
static void usage(const char* argv0, int rank) {
  if (rank == 0) {
    std::cerr << "Usage: mpirun -np P " << argv0
              << " [csv_path=" << "tracks_features_cleaned.csv"
              << "] [K=8] [max_iters=100] [tol=1e-4] [seed=0] [out_csv=assignments_mpi.csv]\n";
  }
}

// Partition n rows across P ranks: first (n % P) ranks get ceil(n/P) rows, the rest get floor(n/P).
// counts[p] = local row count; displs[p] = starting global row index for rank p.
static void compute_counts_displs(int n, int size, std::vector<int>& counts,
                                  std::vector<int>& displs) {
  counts.resize(static_cast<size_t>(size));
  displs.resize(static_cast<size_t>(size));
  int base = n / size;
  int rem = n % size;
  for (int p = 0; p < size; ++p) {
    counts[static_cast<size_t>(p)] = base + (p < rem ? 1 : 0);
  }
  displs[0] = 0;
  for (int p = 1; p < size; ++p) {
    displs[static_cast<size_t>(p)] =
        displs[static_cast<size_t>(p - 1)] + counts[static_cast<size_t>(p - 1)];
  }
}

// -----------------------------------------------------------------------------
// main: init MPI/CUDA, scatter data, parallel Lloyd loop, gather + CSV on rank 0
// -----------------------------------------------------------------------------

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Pick a GPU on shared-memory islands (multiple ranks on one node): round-robin by local rank.
  int dev_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&dev_count));
  if (dev_count > 0) {
    int local_rank = 0;
    MPI_Comm shm_comm;
    if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                            &shm_comm) == MPI_SUCCESS) {
      MPI_Comm_rank(shm_comm, &local_rank);
      MPI_Comm_free(&shm_comm);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % dev_count));
  }

  // --- CLI (all ranks parse the same argv for consistent parameters) ---
  std::string csv_path = "tracks_features_cleaned.csv";
  int k = 8;
  int max_iters = 100;
  float tol = 1e-4f;
  unsigned seed = 0;
  std::string out_csv = "assignments_mpi.csv";

  if (argc > 1) {
    if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
      usage(argv[0], rank);
      MPI_Finalize();
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
    if (rank == 0) {
      std::cerr << "K must be >= 1\n";
    }
    MPI_Finalize();
    return 1;
  }

  // --- Rank 0 loads full dataset; n is then broadcast ---
  std::vector<std::string> ids;
  std::vector<float> features;
  int n = 0;

  if (rank == 0) {
    if (!load_dataset(csv_path, ids, features)) {
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    n = static_cast<int>(ids.size());
    if (n < k) {
      std::cerr << "Need at least K data points (n=" << n << ", K=" << k << ")\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // --- Normalize before scattering ---
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
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // --- Partition rows: each rank knows its chunk size and global row offset ---
  std::vector<int> row_counts;
  std::vector<int> row_displs;
  compute_counts_displs(n, size, row_counts, row_displs);
  const int local_n = row_counts[static_cast<size_t>(rank)];

  std::vector<int> sendcounts_feat(static_cast<size_t>(size));
  std::vector<int> displs_feat(static_cast<size_t>(size));
  for (int p = 0; p < size; ++p) {
    sendcounts_feat[static_cast<size_t>(p)] = row_counts[static_cast<size_t>(p)] * D;
    displs_feat[static_cast<size_t>(p)] = row_displs[static_cast<size_t>(p)] * D;
  }

  std::vector<float> local_features(static_cast<size_t>(local_n) * D);
  float* feat_send = (rank == 0) ? features.data() : nullptr;
  // Scatter contiguous float blocks (local_n * D values per rank).
  MPI_Scatterv(feat_send, sendcounts_feat.data(), displs_feat.data(), MPI_FLOAT,
               local_features.data(), local_n * D, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // --- Scatter string IDs: lengths per row, then concatenated UTF-8 bytes (variable length) ---
  std::vector<int> all_lengths;
  std::vector<char> all_id_bytes;
  std::vector<int> sendcounts_bytes(static_cast<size_t>(size));
  std::vector<int> displs_bytes(static_cast<size_t>(size));

  if (rank == 0) {
    all_lengths.resize(static_cast<size_t>(n));
    size_t total_bytes = 0;
    for (int i = 0; i < n; ++i) {
      int len = static_cast<int>(ids[static_cast<size_t>(i)].size());
      all_lengths[static_cast<size_t>(i)] = len;
      total_bytes += static_cast<size_t>(len);
    }
    all_id_bytes.resize(total_bytes);
    size_t off = 0;
    for (int i = 0; i < n; ++i) {
      const std::string& s = ids[static_cast<size_t>(i)];
      int len = static_cast<int>(s.size());
      if (len > 0) {
        std::memcpy(all_id_bytes.data() + off, s.data(), static_cast<size_t>(len));
      }
      off += static_cast<size_t>(len);
    }
    for (int p = 0; p < size; ++p) {
      int cnt = 0;
      int start = row_displs[static_cast<size_t>(p)];
      int rc = row_counts[static_cast<size_t>(p)];
      for (int j = 0; j < rc; ++j) {
        cnt += all_lengths[static_cast<size_t>(start + j)];
      }
      sendcounts_bytes[static_cast<size_t>(p)] = cnt;
    }
    displs_bytes[0] = 0;
    for (int p = 1; p < size; ++p) {
      displs_bytes[static_cast<size_t>(p)] =
          displs_bytes[static_cast<size_t>(p - 1)] +
          sendcounts_bytes[static_cast<size_t>(p - 1)];
    }
  }
  // Non-root ranks need byte Scatterv metadata (MPI only uses it on root, but we need valid arrays).
  MPI_Bcast(sendcounts_bytes.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs_bytes.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> local_lengths(static_cast<size_t>(local_n));
  MPI_Scatterv(rank == 0 ? all_lengths.data() : nullptr, row_counts.data(),
               row_displs.data(), MPI_INT, local_lengths.data(), local_n, MPI_INT, 0,
               MPI_COMM_WORLD);

  int my_byte_count = 0;
  for (int i = 0; i < local_n; ++i) {
    my_byte_count += local_lengths[static_cast<size_t>(i)];
  }
  std::vector<char> local_id_bytes(static_cast<size_t>(my_byte_count));
  MPI_Scatterv(rank == 0 ? all_id_bytes.data() : nullptr, sendcounts_bytes.data(),
               displs_bytes.data(), MPI_CHAR, local_id_bytes.data(), my_byte_count, MPI_CHAR, 0,
               MPI_COMM_WORLD);

  std::vector<std::string> local_ids(static_cast<size_t>(local_n));
  {
    int bo = 0;
    for (int i = 0; i < local_n; ++i) {
      int len = local_lengths[static_cast<size_t>(i)];
      if (len > 0) {
        local_ids[static_cast<size_t>(i)].assign(local_id_bytes.data() + bo,
                                                 local_id_bytes.data() + bo + len);
      }
      bo += len;
    }
  }

  // --- Initial centroids on rank 0, then broadcast so every rank runs identical Lloyd math ---
  std::vector<float> h_centroids(static_cast<size_t>(k) * D);
  if (rank == 0) {
    srand(42);
    std::vector<int> indices;
    while ((int)indices.size() < k) {
      int idx = rand() % n;
      bool dup = false;
      for (int x : indices) if (x == idx) { dup = true; break; }
      if (!dup) indices.push_back(idx);
    }

    for (int c = 0; c < k; ++c) {
      int row = indices[c];
      for (int d = 0; d < D; ++d) {
        h_centroids[c * D + d] = features[row * D + d];
      }
    }
  }
  MPI_Bcast(h_centroids.data(), k * D, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // --- Per-rank device memory (skipped if this rank owns zero rows) ---
  float* d_points = nullptr;
  float* d_centroids = nullptr;
  float* d_centroid_sums = nullptr;
  int* d_centroid_counts = nullptr;
  int* d_assignments = nullptr;

  const bool have_points = local_n > 0;
  if (have_points) {
    CUDA_CHECK(cudaMalloc(&d_points, static_cast<size_t>(local_n) * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroids, static_cast<size_t>(k) * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroid_sums, static_cast<size_t>(k) * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroid_counts, static_cast<size_t>(k) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_assignments, static_cast<size_t>(local_n) * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_points, local_features.data(),
                          static_cast<size_t>(local_n) * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids.data(),
                          static_cast<size_t>(k) * D * sizeof(float), cudaMemcpyHostToDevice));
  }

  std::vector<float> local_sums(static_cast<size_t>(k) * D);
  std::vector<int> local_counts(static_cast<size_t>(k));
  std::vector<float> global_sums(static_cast<size_t>(k) * D);
  std::vector<int> global_counts(static_cast<size_t>(k));

  std::vector<int> local_assignments(static_cast<size_t>(std::max(local_n, 1)));
  std::vector<int> all_assignments;
  if (rank == 0) {
    all_assignments.resize(static_cast<size_t>(n));
  }

  const int grid = have_points ? (local_n + BLOCK_SIZE - 1) / BLOCK_SIZE : 0;
  auto t0 = std::chrono::steady_clock::now();

  // --- Lloyd iterations: local GPU reduce, global Allreduce, identical centroid update ---
  bool converged = false;
  for (int it = 0; it < max_iters; ++it) {
    if (have_points) {
      CUDA_CHECK(cudaMemset(d_centroid_sums, 0, static_cast<size_t>(k) * D * sizeof(float)));
      CUDA_CHECK(cudaMemset(d_centroid_counts, 0, static_cast<size_t>(k) * sizeof(int)));

      kmeans_assign_accumulate<<<grid, BLOCK_SIZE>>>(
          d_points, d_centroids, local_n, k, d_assignments, d_centroid_sums, d_centroid_counts);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      CUDA_CHECK(cudaMemcpy(local_sums.data(), d_centroid_sums,
                            static_cast<size_t>(k) * D * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(local_counts.data(), d_centroid_counts,
                            static_cast<size_t>(k) * sizeof(int), cudaMemcpyDeviceToHost));
    } else {
      std::fill(local_sums.begin(), local_sums.end(), 0.f);
      std::fill(local_counts.begin(), local_counts.end(), 0);
    }

    // Sum partial sums/counts across ranks so every process sees global cluster statistics.
    MPI_Allreduce(local_sums.data(), global_sums.data(), k * D, MPI_FLOAT, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(local_counts.data(), global_counts.data(), k, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);

    float max_move = 0.f;
    for (int c = 0; c < k; ++c) {
      if (global_counts[static_cast<size_t>(c)] > 0) {
        for (int d = 0; d < D; ++d) {
          float old_v = h_centroids[static_cast<size_t>(c) * D + d];
          float new_v =
              global_sums[static_cast<size_t>(c) * D + d] /
              static_cast<float>(global_counts[static_cast<size_t>(c)]);
          float ad = std::fabs(new_v - old_v);
          if (ad > max_move) {
            max_move = ad;
          }
          h_centroids[static_cast<size_t>(c) * D + d] = new_v;
        }
      }
    }

    if (have_points) {
      CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids.data(),
                            static_cast<size_t>(k) * D * sizeof(float), cudaMemcpyHostToDevice));
    }

    if (max_move < tol) {
      converged = true;
      break;
    }
  }

  // --- Pull final labels and assemble global assignment vector on rank 0 ---
  if (have_points) {
    CUDA_CHECK(cudaMemcpy(local_assignments.data(), d_assignments,
                          static_cast<size_t>(local_n) * sizeof(int), cudaMemcpyDeviceToHost));
  }

  MPI_Gatherv(local_assignments.data(), local_n, MPI_INT,
              rank == 0 ? all_assignments.data() : nullptr, row_counts.data(),
              row_displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  // --- Rank 0 writes using the all_assignments array ---
  auto t1 = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = t1 - t0;

  if (rank == 0) {
    std::cout << "K-means (MPI+CUDA) finished" << (converged ? " (converged)" : " (max iterations)") << " in " << elapsed.count() << " s\n";

    std::ofstream out(out_csv);
    if (!out) {
      std::cerr << "Error: could not write " << out_csv << "\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    out << "explicit,danceability,energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo,duration_ms,time_signature,year,cluster\n";
    for (int i = 0; i < n; ++i) {
      for (int d = 0; d < D; ++d) {
        out << features[i * D + d] << ",";
      }
      out << all_assignments[i] << "\n";
    }
  }

  if (have_points) {
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_centroid_sums);
    cudaFree(d_centroid_counts);
    cudaFree(d_assignments);
  }

  MPI_Finalize();
  return 0;
}
