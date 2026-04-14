#!/bin/bash
#SBATCH --job-name=kmeans_1gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu
#SBATCH -o 1gpu_%j.out
#SBATCH -e 1gpu_%j.err

module load gcc cuda openmpi
cd $SLURM_SUBMIT_DIR

echo "Running Single GPU with Block Size 256..."
./kmeans_gpu_single tracks_features_cleaned.csv 8