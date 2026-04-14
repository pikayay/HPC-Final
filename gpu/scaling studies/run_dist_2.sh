#!/bin/bash
#SBATCH --job-name=kmeans_2gpu
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu
#SBATCH -o 2gpu_%j.out
#SBATCH -e 2gpu_%j.err

module load gcc cuda openmpi
cd $SLURM_SUBMIT_DIR

echo "Running Distributed GPU (2 GPUs)..."
mpirun ./kmeans_gpu_dist tracks_features_cleaned.csv 8