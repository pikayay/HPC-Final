### CHPC Build and Exec Instructions (Distributed CPU Implementation)
1) Upload kmeans_slurm.sh, kmeans_cpu_dist.cpp, rapidcsv.h, and tracks_features_cleaned.csv somewhere in your home directory.
2) Edit the bash script to specifications (different account / partition, output / error files, etc).
3) Open up a shell in the desired cluster (Kingspeak was my go-to).
4) Navigate to the directory where you placed the files.
5) Run $ sbatch --nodes=\<number-of-nodes> kmeans-slurm.sh
6) Batch should be submitted, and the designated output files will appear in the working directory when the job is done.
That includes the output file, the error file, and cpu-dist-results.csv.


### Implementation 4: Parallel CPU - Distributed Memory
#### (description of the approach used)
I'll try to keep the description pretty high-level, since the code is right there and pretty well-commented (in my unbiased opinion).

We start with some pretty standard imports, as well as the rapidcsv library for more intuitive csv handling. Then we set up standard MPI stuff like establishing the threadcount, ranks, and some variables that every thread will need (like the feature count, the maximum iterations, convergence tolerance, data vectors, cluster vectors, etc). We also establish an srand seed (42) and a cluster count for how many genres we want to sort the songs into.

Process 0 loads the source csv into memory, processes it into a 1-dimensional vector (storing the ID column separately), normalizes every feature into a 0-1 range, and randomly picks initial centroids from the dataset. The total number of rows (songs) and the initial clusters are broadcast to every thread. The root process scatters the data to all processes, and then step 1 is complete.

For step 2, each process iterates through their assigned portion of songs, and finds the closest centroid for each. Each process also sums up every point in a cluster to later calculate the average point of that cluster.

For step 3, MPI Allreduce is used to calculate the global cluster sums and counts for the averages. Old centroids are stored to check for convergence. Then the centroids are updated, which is done on each cluster since the compute work is pretty easy and every process needs the updated centroids anyway. Convergence is then checked by measuring the maximum change of all centroids from cycle to cycle. Steps 2 and 3 are repeated if convergence has not been reached.

If convergence is reached, then the root process gathers all of the song clustering information from all threads. Then it writes the output to a csv, which looks the same as the input csv but with an extra column for the cluster a song was assigned to. Then it's done!