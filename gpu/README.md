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

