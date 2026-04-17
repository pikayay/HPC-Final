#!/bin/bash
#
#SBATCH --partition=kingspeak
#SBATCH --account=usucs6030
#
#SBATCH --job-name=kmeans_cpu4
#SBATCH --output=4cpuSE.out
#SBATCH -e 4cpuSE.err
#

#SBATCH --nodes=4
#SBATCH --time=10:00

cd $SLURM_SUBMIT_DIR

module purge
module load gcc openmpi

export PATH
export LD_LIBRARY_PATH

OMPI_PREFIX=$(dirname $(dirname $(which mpirun)))

mpicxx -O2 kmeans_cpu_dist.cpp -o kmeans_cpu_dist

mpirun --prefix $OMPI_PREFIX \
       --mca btl tcp,self \
       ./kmeans_cpu_dist