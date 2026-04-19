# Section 1: Serial and Parallel Implementations
#### associated files in /cpu

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


# Section 2: Distributed CPU implementation
## Associated Files in /dist-cpu
#### Distributed CPU Implementation - Sam Elliss
### CHPC Build and Exec Instructions (Distributed CPU Implementation)
1) Upload kmeans_slurm.sh, kmeans_cpu_dist.cpp, rapidcsv.h (separate library found at https://github.com/d99kris/rapidcsv), and tracks_features_cleaned.csv (can be obtained from running the csv_parser.cpp file in the pre-processing section on the source dataset from Kaggle) somewhere in your home directory.
2) Edit the bash script to specifications (different account / partition, output / error files, etc).
3) Open up a shell in the desired cluster (Kingspeak was my go-to).
4) Navigate to the directory where you placed the files.
5) Run $ sbatch --nodes=\<number-of-nodes> kmeans-slurm.sh
6) Batch should be submitted, and the designated output files will appear in the working directory when the job is done.
That includes the output file, the error file, and cpu-dist-results.csv.


### Implementation 4: Parallel CPU - Distributed Memory
I'll try to keep the description pretty high-level, since the code is right there and pretty well-commented (in my unbiased opinion).

I start with some pretty standard imports, as well as the rapidcsv library for more intuitive csv handling. Then I set up standard MPI stuff like establishing the threadcount, ranks, and some variables that every thread will need (like the feature count, the maximum iterations, convergence tolerance, data vectors, cluster vectors, etc). I also establish an srand seed (42) and a cluster count for how many genres I want to sort the songs into.

Process 0 loads the source csv into memory, processes it into a 1-dimensional vector (storing the ID column separately), normalizes every feature into a 0-1 range, and randomly picks initial centroids from the dataset. The total number of rows (songs) and the initial clusters are broadcast to every thread. The root process scatters the data to all processes, and then step 1 is complete.

For step 2, each process iterates through their assigned portion of songs, and finds the closest centroid for each. Each process also sums up every point in a cluster to later calculate the average point of that cluster.

For step 3, MPI Allreduce is used to calculate the global cluster sums and counts for the averages. Old centroids are stored to check for convergence. Then the centroids are updated, which is done on each cluster since the compute work is pretty easy and every process needs the updated centroids anyway. Convergence is then checked by measuring the maximum change of all centroids from cycle to cycle. Steps 2 and 3 are repeated if convergence has not been reached.

If convergence is reached, then the root process gathers all of the song clustering information from all threads. Then it writes the output to a csv, which looks the same as the input csv but with an extra column for the cluster a song was assigned to. Then it's done!

As an addendum, there may be some minor differences in results based on cores running the program (and between the various implementations), due to floating point addition imprecision. This is gone into more detail in the Validation section of the report. Here's some example output from comparing the results of the distributed CPU implementation with the distributed GPU implementation:

```
Validation passed with acceptable floating-point variance.
Mismatches: 77 out of 1204025 (0.00640%)
Note: Minor deviations are expected due to parallel reduction non-associativity.
```

### Scaling Study Reports
The distributed memory CPU code was ran on the Kingspeak cluster with 2, 3, and 4 nodes. It should be noted that this is different than the GPU distributed execution, but I couldn't get my code to run with 3 or 4 nodes on Notchpeak for whatever reason. 

- 2 Nodes: 1.4418s
- 3 Nodes: 0.9448s
- 4 Nodes: 0.7310s

This is actually a very solid runtime improvement from node to node, indicative of strong scaling potential with the implementation. Almost a 2x speedup from 2 to 4 nodes. It should be noted that the timing begins with step 2, so after the data has been loaded by process 0, and is stopped after the gather has been completed. That means that the time actually measured here is almost exclusively highly parallelizeable code with the overhead removed.

In comparison with the GPU implementations: This obviously runs much slower, which is what I'd expect for the CPU vs GPU runs of this code. GPU implementations of this type of project generally run much faster than strictly CPU implementations. The CPU implementation is seeing a lot more speedup, though, which could be due to the physical limitations described in that report. If we had a larger dataset that would likely be minimized.


### Visualizations
Visualizations for the results are included in the visualizations folder. You will recall the email chain discussing it, haha. Included are:
- The output from the visualize.py script, which recolors data to better represent the clustering in a flattened 3d space.
- A ParaView representation of the same data, not recolored (which makes it seem messier). 
- A ParaView visualization of three clusters in three dimensions to better show the separation of the algorithm. Clusters are much more distinctly separated here (although the consequences of the "mode" feature mean it is more so two 2D visualizations than a true 3d representation).


### Team Task Breakdown
I was person C, responsible for the distributed CPU implementation (#4), everything that entailed, and output visualizations. I also did some of the initial data processing (although we ended up using a C program developed by Braden for consistency on the starting data). I discovered a small error in the starter data late into the project and patched the C program to address it. I created a fully functional clustering program myself and then when I regrouped with Braden I reused some of his GPU code for clustering, normalization, and convergence to ensure that we had the same output. Once I had access to Dillan's serial program I double checked the output again with the validation program written by Braden. Overall I would say we all did our parts and nobody was dragging the team down (besides the Canvas messaging system, which ate a few messages).



# Section 3 GPU Implementation
#### Associated files in /gpu

### **CHPC Build and Execution Instructions (GPU Implementations)**

The codebase can be compiled on the CHPC Notchpeak cluster. The following commands load the necessary compiler, CUDA toolkit, and MPI modules, and build the executables using the provided Makefile:

```shell
module load gcc cuda openmpi
make clean
make
```

To run the implementations, submit the corresponding SLURM scripts included in the source code directory. For example:

```shell
sbatch run_single.sh
sbatch run_dist_2.sh
sbatch run_dist_3.sh
sbatch run_dist_4.sh
```

### **Implementation 3: Parallel CUDA GPU (Shared Memory Alternative)**

For the single GPU implementation, the host parses the dataset, normalizes the features using min-max normalization, and flattens all features into a single 1D float array in row-major order. This flat memory layout ensures efficient and contiguous memory transfers when copying the entire dataset to the device.

The CUDA kernel is launched with one thread assigned to each data point. To simplify the architecture and avoid the complexity of CUDA shared memory, the kernel utilizes atomic operations. Each thread calculates the minimum squared Euclidean distance to find its closest centroid. It then uses atomicAdd to update the global device buffers that contain the cluster coordinate sums and point counts. Once the kernel finishes, the host retrieves these reduced arrays, calculates the updated centroids, and checks for convergence before launching the next iteration.

### **Implementation 5: Distributed Memory GPU (MPI \+ CUDA)**

The distributed implementation combines MPI and CUDA to process the data across multiple nodes. Rank 0 reads the full dataset, normalizes the features, and acts as the master distributor. To split the workload, Rank 0 uses MPI\_Scatterv to distribute variable-sized chunks of the flattened 1D feature array to all participating ranks. This ensures that rows are divided evenly even when the total row count is not perfectly divisible by the number of nodes.

Each MPI rank allocates memory on its local GPU and runs the exact same assignment kernel used in the single GPU implementation on its local subset of points. After the local kernel execution, each rank copies its partial centroid sums and counts back to its host memory. We then call MPI\_Allreduce with the MPI\_SUM operation. This globally aggregates the cluster statistics across all nodes so that every rank receives the exact same global totals. As a result, every rank independently computes identical new centroids without requiring a secondary broadcast step. Finally, upon convergence, MPI\_Gatherv is used to collect the cluster labels from all nodes back to Rank 0 for CSV generation.

### **GPU Scaling Study and Performance Analysis**

**Block Size Study (Single GPU)**

A block size study was conducted on the single GPU implementation to observe the impact of thread block configuration on runtime.

* Block Size 32: 0.2645 s
* Block Size 64: 0.2688 s
* Block Size 128: 0.2684 s
* Block Size 256: 0.2428 s
* Block Size 512: 0.2511 s
* Block Size 1024: 0.2637 s

The performance remains largely stable across all block sizes, with a slight performance peak at 256 threads per block. Because the arithmetic intensity of calculating Euclidean distances is relatively low and we are utilizing atomic operations in global memory, the bottleneck is primarily memory bandwidth rather than occupancy.

**Node Scaling Study (Distributed GPU)**

The distributed memory GPU implementation was tested across 2, 3, and 4 nodes on the Notchpeak cluster, with 5 runs per configuration to account for variance.

* 2 Nodes (Average of 5 runs): 0.2209 s  
* 3 Nodes (Average of 5 runs): 0.1319 s  
* 4 Nodes (Average of 5 runs): 0.1096 s

The scaling behavior shows a consistent performance improvement as more nodes are added. The transition from 2 nodes to 4 nodes cuts the execution time in half. The efficient scaling at 4 nodes in this 5-run sample indicates that the scheduler may have allocated nodes within the same physical rack, minimizing network communication overhead.

### **Scaling Study Comparison: Distributed CPU vs. Distributed GPU (Implementation 4 vs. 5)**

To evaluate the performance differences between distributed environments, we compared the distributed memory CPU implementation (Implementation 4) against the distributed memory GPU implementation (Implementation 5). Both implementations were tested across 2, 3, and 4 nodes. 

Implementation 4: Distributed CPU Runtimes
* 2 Nodes: 1.4418 s
* 3 Nodes: 0.9448 s
* 4 Nodes: 0.7310 s

Implementation 5: Distributed GPU Runtimes
* 2 Nodes: 0.2209 s
* 3 Nodes: 0.1319 s
* 4 Nodes: 0.1096 s

Analysis

Both implementations exhibit strong scaling behavior, with execution times decreasing significantly as computational resources increase. The transition from 2 nodes to 4 nodes cuts the execution time nearly in half for both architectures. 

However, the distributed GPU implementation is substantially faster across all node counts, running approximately 6.5 times faster than the distributed CPU implementation at both the 2 node and 4 node marks. This massive performance gap is expected. K-means clustering requires calculating the Euclidean distance between every data point and every centroid during each iteration. The highly parallel architecture of the GPU is vastly superior at handling these simultaneous calculations compared to the limited thread count of a CPU. 

While the GPU is faster overall, the CPU implementation shows a slightly more consistent scaling trajectory. Because the GPU implementation calculates the distances so rapidly, the network communication overhead required by the MPI reduction steps begins to occupy a larger percentage of the total runtime, which slightly bottlenecks its scaling efficiency at higher node counts.

### **Validation and Numerical Precision**

To ensure the parallel implementations produce correct results, a Python validation script compares the output CSVs against the serial baseline. 

Usage:
python validate.py serial_results.csv parallel_results.csv

Note on Numerical Precision:
When comparing parallel results to the serial baseline, a microscopic divergence in cluster assignments is expected. The serial CPU implementation adds coordinates sequentially, while the GPU implementations aggregate data non-deterministically using atomic operations or network-based MPI reductions. 

Because floating-point addition is not perfectly associative, these different reduction orders result in minor decimal variations in the final centroids. Over many iterations, data points resting exactly on the mathematical boundary between clusters may be classified differently. The validation script calculates this mismatch percentage and passes the check if it falls within an acceptable high performance computing tolerance.

### **Team Task Breakdown**

* **Person B:** Responsible for the single GPU implementation (Implementation 3), the distributed GPU implementation (Implementation 5), the GPU block size study, the distributed GPU node scaling study, and writing the CHPC build and execution instructions for the GPU codebase.

