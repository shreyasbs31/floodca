#!/bin/bash
#SBATCH --job-name=floodca-p6e1
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_GPU__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# P-6 exit check (E1: CUDA vs serial, SPEC 7.8 and 12.1): double, --fmad=false on the
# device and -ffp-contract=off on the host, K=4 and 8, blocks 8x8/16x16/32x32,
# CUB and hand-written dt reductions.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export PYTHONPATH="${SLURM_SUBMIT_DIR}/prep:${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export FCA_GIT="$(git rev-parse HEAD 2>/dev/null || true)"
mkdir -p results/slurm
echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

B="${B:-build/p6_cuda}"
export BUILD="${SLURM_SUBMIT_DIR}/${B}"

echo "===== E1 CUDA synthetic 257x263 ====="
E1_GRID=data/processed/e1_257x263 E1_GRID_TAG=257x263 E1_OUT=results/e1_cuda_257x263 \
  E1_THREADS=" " E1_CUDA_BLOCKS="8x8 16x16 32x32" E1_CUDA_REDUCES="cub custom" \
  E1_NEIGHS="4 8" bash tests/c/test_equivalence.sh

echo "===== E1 CUDA grid M ====="
E1_GRID=data/processed/udupi_coast/M E1_GRID_TAG=gridM E1_OUT=results/e1_cuda_gridM \
  E1_THREADS=" " E1_CUDA_BLOCKS="8x8 16x16 32x32" E1_CUDA_REDUCES="cub" \
  E1_NEIGHS="4 8" bash tests/c/test_equivalence.sh
echo "P6_E1_OK"
