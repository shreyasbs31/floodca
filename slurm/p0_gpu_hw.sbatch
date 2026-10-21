#!/bin/bash
# SPEC 9.1 [VERIFY]: GPU-node CPU model and RAM from inside a Slurm job.
#SBATCH --job-name=floodca-gpuhw
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_GPU__
#SBATCH --exclude=__FCA_NODE_SECOND__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p results/slurm
source slurm/env.sh
# shellcheck disable=SC1091
source slurm/gpu_driver_preflight.sh
gpu_driver_preflight
echo "===== job ====="
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION}"
hostname
date -u

echo "===== lscpu ====="
lscpu

echo "===== numactl -H ====="
numactl -H || echo "numactl missing"

echo "===== free -g ====="
free -g

echo "===== nproc / meminfo ====="
nproc
grep -E 'MemTotal|MemAvailable' /proc/meminfo

echo "GPUHW_OK"
