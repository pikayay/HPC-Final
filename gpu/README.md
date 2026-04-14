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

For the single GPU implementation, the host parses the dataset and flattens all features into a single 1D float array in row-major order. This flat memory layout ensures efficient and contiguous memory transfers when copying the entire dataset to the device.

The CUDA kernel is launched with one thread assigned to each data point. To simplify the architecture and avoid the complexity of CUDA shared memory, the kernel utilizes atomic operations. Each thread calculates the minimum squared Euclidean distance to find its closest centroid. It then uses atomicAdd to update the global device buffers that contain the cluster coordinate sums and point counts. Once the kernel finishes, the host retrieves these reduced arrays, calculates the updated centroids, and checks for convergence before launching the next iteration.

### **Implementation 5: Distributed Memory GPU (MPI \+ CUDA)**

The distributed implementation combines MPI and CUDA to process the data across multiple nodes. Rank 0 reads the full dataset, normalizes the features, and acts as the master distributor. To split the workload, Rank 0 uses MPI\_Scatterv to distribute variable-sized chunks of the flattened 1D feature array to all participating ranks. This ensures that rows are divided evenly even when the total row count is not perfectly divisible by the number of nodes.

Each MPI rank allocates memory on its local GPU and runs the exact same assignment kernel used in the single GPU implementation on its local subset of points. After the local kernel execution, each rank copies its partial centroid sums and counts back to its host memory. We then call MPI\_Allreduce with the MPI\_SUM operation. This globally aggregates the cluster statistics across all nodes so that every rank receives the exact same global totals. As a result, every rank independently computes identical new centroids without requiring a secondary broadcast step. Finally, upon convergence, MPI\_Gatherv is used to collect the cluster labels from all nodes back to Rank 0 for CSV generation.

### **GPU Scaling Study and Performance Analysis**

**Block Size Study (Single GPU)**

A block size study was conducted on the single GPU implementation to observe the impact of thread block configuration on runtime.

* Block Size 32: 0.1276 s  
* Block Size 64: 0.1256 s  
* Block Size 128: 0.1277 s  
* Block Size 256: 0.1247 s  
* Block Size 512: 0.1259 s  
* Block Size 1024: 0.1236 s

The performance remains largely stable across all block sizes, with a slight performance peak at 1024 threads per block. Because the arithmetic intensity of calculating Euclidean distances is relatively low and we are utilizing atomic operations in global memory, the bottleneck is primarily memory bandwidth rather than occupancy.

**Node Scaling Study (Distributed GPU)**

The distributed memory GPU implementation was tested across 2, 3, and 4 nodes on the Notchpeak cluster, with 10 runs per configuration to account for variance.

* 2 Nodes (Average of 10 runs): 0.0982 s  
* 3 Nodes (Average of 10 runs): 0.0656 s  
* 4 Nodes (Average of 10 runs): 0.0847 s

The scaling behavior reveals a hardware and network constraint on the cluster. Moving from 2 to 3 nodes shows a clear performance improvement. However, scaling to 4 nodes results in a higher average runtime. Looking closely at the 4-node data, the performance exhibited a severe bimodal distribution. Exactly half of the 10 runs finished rapidly (around 0.05 seconds), while the other half clustered at the extreme high end (over 0.12 seconds), with zero runs falling in between. This distinct split heavily indicates the execution became unpredictably network bound based on the physical location of the GPUs allocated by the SLURM scheduler. When SLURM allocates 4 nodes within the same physical rack, the MPI communication benefits from low latency switches. If the nodes are scattered across different racks, the MPI traffic must traverse the core network, exposing it to physical distance latency and cluster traffic contention which more than doubles the runtime.

### **Team Task Breakdown**

* **Person B:** Responsible for the single GPU implementation (Implementation 3), the distributed GPU implementation (Implementation 5), the GPU block size study, the distributed GPU node scaling study, and writing the CHPC build and execution instructions for the GPU codebase.

