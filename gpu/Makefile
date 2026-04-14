# K-means CUDA builds for Linux / SLURM clusters (nvcc + OpenMPI).
#
# Typical modules (adjust for your site):
#   module load cuda/12.2
#   module load openmpi/4.1
#
# If MPI headers are not found, set MPI_HOME to your OpenMPI prefix, e.g.:
#   export MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi
#   make kmeans_gpu_dist
#
# CUDA architecture: set GPU_ARCH to match the cluster GPUs, e.g. sm_70, sm_80, sm_90.
GPU_ARCH ?= sm_52

NVCC ?= nvcc
NVCCFLAGS = -O3 -std=c++17 -lineinfo -arch=$(GPU_ARCH)

all: kmeans_gpu_single kmeans_gpu_dist

# Single-GPU binary (pure CUDA)
kmeans_gpu_single: kmeans_gpu_single.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# MPI + CUDA: use the MPI C++ wrapper as the host compiler so linking finds libmpi.
# Drop -Xcompiler -fopenmp if your mpicxx does not support it.
kmeans_gpu_dist: kmeans_gpu_dist.cu
	$(NVCC) $(NVCCFLAGS) -ccbin mpicxx -o $@ $< -Xcompiler -fopenmp

clean:
	rm -f kmeans_gpu_single kmeans_gpu_dist

.PHONY: all clean
