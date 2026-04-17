### CHPC Build and Exec Instructions (Distributed CPU Implementation)
1) Upload kmeans_slurm.sh, kmeans_cpu_dist.cpp, rapidcsv.h, and tracks_features_cleaned.csv somewhere in your home directory.
2) Edit the bash script to specifications (different account / partition, output / error files, etc).
3) Open up a shell in the desired cluster (Kingspeak was my go-to).
4) Navigate to the directory where you placed the files.
5) Run $ sbatch --nodes=\<number-of-nodes> kmeans-slurm.sh
6) Batch should be submitted, and the designated output files will appear in the working directory when the job is done.
That includes the output file, the error file, and cpu-dist-results.csv.


