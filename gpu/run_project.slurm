#!/bin/bash
#SBATCH --job-name=kmeans_gpu
#SBATCH --output=kmeans_%j.out
#SBATCH --error=kmeans_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --account=kingspeak-gpu
#SBATCH --partition=kingspeak-gpu

# Load the required modules
module load cuda
module load openmpi

echo "Running Single GPU Implementation..."
./kmeans_gpu_single tracks_features_cleaned.csv 5

echo "Running Distributed GPU Implementation on $SLURM_NNODES nodes..."
mpirun -np $SLURM_NTASKS ./kmeans_gpu_dist tracks_features_cleaned.csv 5