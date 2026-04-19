#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <limits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <omp.h>

// ---------------------------------------------------------------------------
// Feature layout (col 0=id stored separately, col 1..15 = features)
// Index: 0=explicit 1=danceability 2=energy 3=key 4=loudness 5=mode
//        6=speechiness 7=acousticness 8=instrumentalness 9=liveness
//       10=valence 11=tempo 12=duration_ms 13=time_signature 14=year
// ---------------------------------------------------------------------------
static const int NUM_FEATURES = 15;
static const char* FEATURE_NAMES[NUM_FEATURES] = {
    "explicit", "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms", "time_signature", "year"
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "  --input   <file>           CSV input (default: tracks_features_cleaned.csv)\n"
              << "  --output  <file>           CSV output (default: kmeans_results.csv)\n"
              << "  --k       <int>            Number of clusters (default: 8)\n"
              << "  --iters   <int>            Max iterations (default: 100)\n"
              << "  --mode    serial|parallel  Implementation (default: parallel)\n"
              << "  --threads <int>            OMP threads (default: system max)\n"
              << "  --rows    <int>            Max rows to load (default: all)\n";
}

static int featureIndex(const std::string& name) {
    for (int i = 0; i < NUM_FEATURES; ++i)
        if (name == FEATURE_NAMES[i]) return i;
    return -1;
}

static float parseSafe(const std::string& s) {
    if (s == "True" || s == "true")  return 1.0f;
    if (s == "False" || s == "false") return 0.0f;
    try { return std::stof(s); }
    catch (...) { return 0.0f; }
}

// ---------------------------------------------------------------------------
// CSV loading
// ---------------------------------------------------------------------------
// Returns data[row][feature] and populates ids. Skips rows with parse errors.
static std::vector<std::array<float, NUM_FEATURES>>
loadCSV(const std::string& filename, int maxRows, std::vector<std::string>& ids) {
    std::ifstream f(filename);
    if (!f.is_open()) {
        std::cerr << "[Error] Cannot open " << filename << "\n";
        std::exit(1);
    }

    std::vector<std::array<float, NUM_FEATURES>> data;
    std::string line;

    // Skip header
    std::getline(f, line);

    int loaded = 0;
    while (std::getline(f, line)) {
        if (maxRows > 0 && loaded >= maxRows) break;

        // Parse CSV: col 0=id, col 1..15=features (explicit first, then the rest)
        std::array<float, NUM_FEATURES> row;
        std::stringstream ss(line);
        std::string tok;
        int col = 0;
        int feat = 0;
        bool ok = true;
        std::string rowId;

        while (std::getline(ss, tok, ',')) {
            if (col == 0) {
                rowId = tok;
            } else if (feat < NUM_FEATURES) {
                // col 1 = explicit, col 2..15 = remaining features
                row[feat++] = parseSafe(tok);
            }
            ++col;
        }
        if (feat < NUM_FEATURES) ok = false;

        if (ok) {
            data.push_back(row);
            ids.push_back(rowId);
            ++loaded;
        }

        if (loaded % 100000 == 0 && loaded > 0) {
            std::cout << "[Loading] " << loaded << " rows loaded...\n";
            std::cout.flush();
        }
    }

    std::cout << "[Loading] Done — " << loaded << " rows loaded.\n";
    return data;
}

// ---------------------------------------------------------------------------
// Normalization (min-max per feature, in-place)
// ---------------------------------------------------------------------------
static void normalize(std::vector<std::array<float, NUM_FEATURES>>& data) {
    if (data.empty()) return;
    float fmin[NUM_FEATURES], fmax[NUM_FEATURES];
    for (int j = 0; j < NUM_FEATURES; ++j) {
        fmin[j] = std::numeric_limits<float>::max();
        fmax[j] = std::numeric_limits<float>::lowest();
    }
    for (auto& row : data)
        for (int j = 0; j < NUM_FEATURES; ++j) {
            if (row[j] < fmin[j]) fmin[j] = row[j];
            if (row[j] > fmax[j]) fmax[j] = row[j];
        }
    for (auto& row : data)
        for (int j = 0; j < NUM_FEATURES; ++j) {
            float range = fmax[j] - fmin[j];
            row[j] = (range > 0) ? (row[j] - fmin[j]) / range : 0.0f;
        }
}

// ---------------------------------------------------------------------------
// Distance (squared Euclidean)
// ---------------------------------------------------------------------------
static inline float distSq(const float* a, const float* b) {
    float d = 0;
    for (int j = 0; j < NUM_FEATURES; ++j) {
        float diff = a[j] - b[j];
        d += diff * diff;
    }
    return d;
}

// ---------------------------------------------------------------------------
// Centroid initialization (random selection from data)
// ---------------------------------------------------------------------------
static std::vector<std::array<float, NUM_FEATURES>>
initCentroids(const std::vector<std::array<float, NUM_FEATURES>>& data, int k) {
    std::vector<std::array<float, NUM_FEATURES>> centroids;
    centroids.reserve(k);
    int n = (int)data.size();
    // Simple: pick k distinct random indices
    std::vector<int> indices;
    indices.reserve(k);
    srand(42);
    while ((int)indices.size() < k) {
        int idx = rand() % n;
        bool dup = false;
        for (int x : indices) if (x == idx) { dup = true; break; }
        if (!dup) indices.push_back(idx);
    }
    for (int idx : indices) centroids.push_back(data[idx]);
    return centroids;
}

// ---------------------------------------------------------------------------
// Serial K-Means
// ---------------------------------------------------------------------------
static std::vector<int> kmeansSerial(
    const std::vector<std::array<float, NUM_FEATURES>>& data,
    int k, int maxIters)
{
    int n = (int)data.size();
    std::vector<int> assign(n, 0);
    auto centroids = initCentroids(data, k);

    std::cout << "[K-Means Serial] Initialized " << k << " centroids.\n";

    for (int iter = 0; iter < maxIters; ++iter) {
        // Assignment
        int changed = 0;
        for (int i = 0; i < n; ++i) {
            float best = std::numeric_limits<float>::max();
            int bestK = 0;
            for (int c = 0; c < k; ++c) {
                float d = distSq(data[i].data(), centroids[c].data());
                if (d < best) { best = d; bestK = c; }
            }
            if (bestK != assign[i]) { assign[i] = bestK; ++changed; }
        }

        std::cout << "[K-Means Serial] Iter " << (iter + 1) << "/" << maxIters
                  << " — reassigned " << changed << " points\n";
        std::cout.flush();

        if (changed == 0) {
            std::cout << "[K-Means Serial] Converged at iteration " << (iter + 1) << "\n";
            break;
        }

        // Update centroids
        std::vector<std::array<float, NUM_FEATURES>> sums(k);
        std::vector<int> counts(k, 0);
        for (auto& s : sums) s.fill(0.0f);

        for (int i = 0; i < n; ++i) {
            int c = assign[i];
            for (int j = 0; j < NUM_FEATURES; ++j)
                sums[c][j] += data[i][j];
            ++counts[c];
        }
        for (int c = 0; c < k; ++c)
            if (counts[c] > 0)
                for (int j = 0; j < NUM_FEATURES; ++j)
                    centroids[c][j] = sums[c][j] / counts[c];
    }
    return assign;
}

// ---------------------------------------------------------------------------
// Parallel K-Means (OpenMP)
// ---------------------------------------------------------------------------
static std::vector<int> kmeansParallel(
    const std::vector<std::array<float, NUM_FEATURES>>& data,
    int k, int maxIters, int threads)
{
    if (threads > 0) omp_set_num_threads(threads);
    int n = (int)data.size();
    std::vector<int> assign(n, 0);
    auto centroids = initCentroids(data, k);

    int nThreads = omp_get_max_threads();
    std::cout << "[K-Means Parallel] Initialized " << k << " centroids"
              << " using " << nThreads << " threads.\n";

    for (int iter = 0; iter < maxIters; ++iter) {
        int changed = 0;

        // --- Assignment (embarrassingly parallel) ---
        #pragma omp parallel for reduction(+:changed) schedule(static)
        for (int i = 0; i < n; ++i) {
            float best = std::numeric_limits<float>::max();
            int bestK = 0;
            for (int c = 0; c < k; ++c) {
                float d = distSq(data[i].data(), centroids[c].data());
                if (d < best) { best = d; bestK = c; }
            }
            if (bestK != assign[i]) { assign[i] = bestK; ++changed; }
        }

        std::cout << "[K-Means Parallel] Iter " << (iter + 1) << "/" << maxIters
                  << " — reassigned " << changed << " points\n";
        std::cout.flush();

        if (changed == 0) {
            std::cout << "[K-Means Parallel] Converged at iteration " << (iter + 1) << "\n";
            break;
        }

        // --- Update centroids (thread-local accumulators + reduction) ---
        // Each thread owns its own sums/counts to avoid false sharing
        int T = omp_get_max_threads();
        std::vector<std::vector<std::array<float, NUM_FEATURES>>> localSums(T,
            std::vector<std::array<float, NUM_FEATURES>>(k));
        std::vector<std::vector<int>> localCounts(T, std::vector<int>(k, 0));

        for (int t = 0; t < T; ++t)
            for (int c = 0; c < k; ++c)
                localSums[t][c].fill(0.0f);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) {
                int c = assign[i];
                for (int j = 0; j < NUM_FEATURES; ++j)
                    localSums[tid][c][j] += data[i][j];
                ++localCounts[tid][c];
            }
        }

        // Merge thread-local results
        for (int c = 0; c < k; ++c) {
            std::array<float, NUM_FEATURES> sum; sum.fill(0.0f);
            int count = 0;
            for (int t = 0; t < T; ++t) {
                for (int j = 0; j < NUM_FEATURES; ++j)
                    sum[j] += localSums[t][c][j];
                count += localCounts[t][c];
            }
            if (count > 0)
                for (int j = 0; j < NUM_FEATURES; ++j)
                    centroids[c][j] = sum[j] / count;
        }
    }
    return assign;
}

// ---------------------------------------------------------------------------
// Write output CSV
// ---------------------------------------------------------------------------
static void writeResults(
    const std::string& filename,
    const std::vector<std::array<float, NUM_FEATURES>>& data,
    const std::vector<int>& assign,
    const std::vector<std::string>& ids)
{
    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "[Error] Cannot write " << filename << "\n";
        return;
    }

    // Header
    f << "id,";
    for (int j = 0; j < NUM_FEATURES; ++j) {
        f << FEATURE_NAMES[j];
        if (j < NUM_FEATURES - 1) f << ',';
    }
    f << ",cluster\n";

    int n = (int)data.size();
    for (int i = 0; i < n; ++i) {
        f << ids[i] << ',';
        for (int j = 0; j < NUM_FEATURES; ++j)
            f << data[i][j] << ',';
        f << assign[i] << '\n';
    }

    std::cout << "[Output] Wrote " << n << " rows to " << filename << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::string inputFile  = "tracks_features_cleaned.csv";
    std::string outputFile = "kmeans_results.csv";
    int k        = 8;
    int maxIters = 100;
    int maxRows  = 0;   // 0 = all
    int threads  = 0;   // 0 = system default
    bool parallel = true;

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--input")   && i+1<argc) inputFile  = argv[++i];
        else if (!strcmp(argv[i], "--output")  && i+1<argc) outputFile = argv[++i];
        else if (!strcmp(argv[i], "--k")       && i+1<argc) k          = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters")   && i+1<argc) maxIters   = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rows")    && i+1<argc) maxRows    = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i+1<argc) threads    = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mode")    && i+1<argc) {
            std::string m = argv[++i];
            if (m == "serial")   parallel = false;
            else if (m == "parallel") parallel = true;
            else { std::cerr << "[Error] --mode must be serial or parallel\n"; return 1; }
        }
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            printUsage(argv[0]); return 0;
        }
        else { std::cerr << "[Warning] Unknown argument: " << argv[i] << "\n"; }
    }

    std::cout << "=== K-Means Song Clustering ===\n"
              << "Input:   " << inputFile  << "\n"
              << "Output:  " << outputFile << "\n"
              << "k:       " << k          << "\n"
              << "iters:   " << maxIters   << "\n"
              << "rows:    " << (maxRows > 0 ? std::to_string(maxRows) : "all") << "\n"
              << "mode:    " << (parallel ? "parallel" : "serial") << "\n\n";

    std::vector<std::string> ids;
    auto data = loadCSV(inputFile, maxRows, ids);
    if (data.empty()) { std::cerr << "[Error] No data loaded.\n"; return 1; }

    std::cout << "[Normalize] Normalizing " << NUM_FEATURES << " features to [0,1]...\n";
    normalize(data);

    double t0 = omp_get_wtime();
    std::vector<int> assign;

    if (parallel) {
        assign = kmeansParallel(data, k, maxIters, threads);
    } else {
        assign = kmeansSerial(data, k, maxIters);
    }

    double elapsed = omp_get_wtime() - t0;
    std::cout << "\n[Timing] " << (parallel ? "Parallel" : "Serial")
              << " elapsed: " << elapsed << "s\n\n";

    writeResults(outputFile, data, assign, ids);

    // Print cluster sizes
    std::vector<int> sizes(k, 0);
    for (int c : assign) ++sizes[c];
    std::cout << "[Clusters]\n";
    for (int c = 0; c < k; ++c)
        std::cout << "  Cluster " << c << ": " << sizes[c] << " songs\n";

    return 0;
}
