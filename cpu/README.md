### **CHPC Build and Execution Instructions (CPU Implementations)**

The codebase can be compiled on the CHPC Notchpeak cluster. The following commands load the necessary compiler and OpenMP-enabled build toolchain, then build the executable using the provided Makefile:

```shell
module load gcc
make clean
make
```

Example usage :

```shell
# Serial mode
./kmeans --mode serial --k 4 --iters 100

# Parallel mode (e.g. 8 threads)
./kmeans --mode parallel --threads 8 --k 4 --iters 100
```

**Full CLI Reference**

```
./kmeans [options]
  --input   <file>           CSV input  (default: tracks_features_cleaned.csv)
  --output  <file>           CSV output (default: kmeans_results.csv)
  --k       <int>            Number of clusters (default: 8)
  --iters   <int>            Max iterations    (default: 100)
  --mode    serial|parallel  Implementation    (default: parallel)
  --threads <int>            OMP thread count  (default: system max)
  --rows    <int>            Max rows to load  (default: all)
```

---

### **Implementation 1: Serial CPU**

The serial implementation serves as the baseline. After loading the dataset from CSV, all 15 features are normalized to [0, 1] using min-max normalization. Initial centroids are chosen by randomly sampling `k` distinct rows from the dataset (seeded at 42 for reproducibility).

Each iteration consists of two sequential phases. In the **assignment phase**, every data point is compared against all `k` centroids using squared Euclidean distance, and the index of the closest centroid is recorded. In the **update phase**, the coordinate sums and counts for each cluster are accumulated in a single pass over the assignments, and new centroid positions are computed as the per-feature mean. Convergence is declared when the maximum centroid displacement across all clusters falls below a tolerance of `1e-4`.

---

### **Implementation 2: Parallel CPU (OpenMP)**

The parallel implementation extends the serial baseline with OpenMP, targeting the two most computationally expensive loops in each iteration.

**Assignment loop** — The outer loop over all `n` data points is decorated with `#pragma omp parallel for schedule(static)`. Each thread independently computes the nearest centroid for its assigned subset of points with no data dependencies between iterations, making this phase embarrassingly parallel.

**Update accumulation** — A naïve shared accumulator would require costly atomic operations or locks for every feature update. Instead, each thread maintains its own local copy of the centroid sums and counts arrays (`localSums[tid]`, `localCounts[tid]`). After the parallel accumulation pass, these per-thread buffers are merged serially on the main thread, and the new centroid positions are finalized. This avoids all synchronization during the hot accumulation loop.

The thread count is set at runtime via `--threads` or defaults to the OpenMP maximum reported by the environment.

---

### **CPU Thread Scaling Study and Performance Analysis**

A scaling study was conducted on the OpenMP parallel implementation across 1 through 8 threads on the Notchpeak cluster. Each configuration was run 3 times and averaged. The dataset was the full `tracks_features_cleaned.csv` with `k=4` and a 100-iteration cap.

**Results**

| Threads | Time (s) | Speedup | Efficiency |
|---------|----------|---------|------------|
| serial  | 0.5740   | 1.000x  | 100.0%     |
| 1       | 0.5954   | 0.964x  |  96.4%     |
| 2       | 0.4188   | 1.370x  |  68.5%     |
| 3       | 0.3284   | 1.748x  |  58.3%     |
| 4       | 0.3132   | 1.833x  |  45.8%     |
| 5       | 0.2833   | 2.026x  |  40.5%     |
| 6       | 0.2848   | 2.015x  |  33.6%     |
| 7       | 0.2939   | 1.953x  |  27.9%     |
| 8       | 0.2326   | 2.468x  |  30.8%     |

**Peak speedup: 2.468x at 8 threads (30.8% efficiency)**

**Analysis**

The scaling behavior reflects several characteristics of the K-means workload on a shared-memory CPU.

The 1-thread parallel run is slightly slower (0.5954 s) than the pure serial baseline (0.5740 s). This is expected — OpenMP introduces a small fixed overhead for thread team creation and barrier synchronization on every iteration, which is not present in the serial path.

From 1 to 5 threads the speedup climbs steadily, reaching a peak of 2.026x at 5 threads. The efficiency drops from ~96% to ~40% over the same range, which is a classic Amdahl's Law signature: the serial fraction of each iteration (the centroid merge reduction and the convergence check) becomes an increasingly dominant fraction of the total runtime as the parallel assignment and accumulation phases are divided across more threads.

The plateau between 5 and 7 threads (speedup ~2.0x, times ranging 0.283–0.294 s) and the minor dip at 7 threads suggest that the job is competing with other workloads on the node, or that NUMA locality effects begin to counteract the benefit of additional threads once the working set is spread across memory controllers.

At 8 threads the best observed time of 0.2326 s yields the highest absolute speedup of 2.468x. However, the parallel efficiency has fallen to 30.8%, meaning nearly 70% of the theoretical throughput is lost to overhead. The modest absolute speedup across the full range is consistent with the algorithm's structure: only the inner distance-calculation and accumulation loops are parallelized, while centroid initialization, normalization, CSV I/O, and the reduction merge remain serial. For a dataset of this scale (~1.2 M rows, 15 features), memory bandwidth also begins to limit returns as thread count increases, since each iteration reads the entire dataset multiple times.

---

### **Team Task Breakdown**

* **Person A:** Responsible for the serial CPU implementation (Implementation 1), the shared-memory OpenMP implementation (Implementation 2).
