#!/bin/bash
#SBATCH --job-name=kmeans_4gpu
#SBATCH --nodes=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu
#SBATCH -o 4gpu_%j.out
#SBATCH -e 4gpu_%j.err

module load gcc cuda openmpi
cd $SLURM_SUBMIT_DIR

echo "Running Distributed GPU (4 GPUs)..."
mpirun ./kmeans_gpu_dist tracks_features_cleaned.csv 8