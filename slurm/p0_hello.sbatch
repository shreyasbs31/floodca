#!/bin/bash
# Tiny P-0 compiler check. One GPU, two MPI ranks, two OpenMP threads.
#SBATCH --job-name=floodca-p0
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_GPU__
#SBATCH --exclude=__FCA_NODE_SECOND__
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs/slurm probes/bin

# shellcheck source=/dev/null
source slurm/env.sh
# shellcheck disable=SC1091
source slurm/gpu_driver_preflight.sh
gpu_driver_preflight

echo "===== host ====="
hostname
date -u
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo "===== uptime / mem ====="
uptime
free -g || true

echo "===== lscpu (compute node) ====="
lscpu

echo "===== nvidia-smi ====="
nvidia-smi
echo "===== nvidia-smi -q (selected) ====="
nvidia-smi -q | awk '
  /^GPU /{p=1}
  p && (/Product Name/ || /Product Brand/ || /Serial Number/ || /CUDA Version/ || /Driver Version/ || /VBIOS/ || /FB Memory Usage/ || /Total/ || /Minor Number/ || /Persistence Mode/ || /Compute Mode/ || /GPU UUID/ || /Multiprocessor Count/ || /CUDA Cores/ || /Max Clocks/ || /PCIe/) {print}
  /^Processes/{p=0}
'

echo "===== compile ====="
bash probes/compile_hello.sh

echo "===== OpenMP run ====="
export OMP_NUM_THREADS=2
export OMP_PROC_BIND=close
export OMP_PLACES=cores
./probes/bin/hello_omp

echo "===== MPI run (2 ranks) ====="
mpirun --report-bindings -np 2 ./probes/bin/hello_mpi

echo "===== CUDA run ====="
./probes/bin/hello_cuda

echo "===== cmake / gcc / nvcc / mpicc ====="
cmake --version | head -n 1
gcc --version | head -n 1
nvcc --version
mpicc --showme:version || true
ompi_info --version | head -n 3

echo "P-0 hello job finished OK"
