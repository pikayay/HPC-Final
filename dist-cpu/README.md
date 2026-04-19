## Distributed CPU Implementation - Sam Elliss
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