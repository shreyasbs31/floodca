#!/bin/bash
#SBATCH --job-name=floodca-p10e12
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_GPU__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# E12 ensemble products (SPEC 6.2, 6.4, 6.5), production float build, not timed:
#  - main ensemble N=500 (90 m, 24 h, K=4) with B6 on one GPU, in nested chunks
#    [0,8) [8,16) [16,32) [32,64) [64,128) [128,250) [250,500) so that the counts of
#    the first N scenarios are exact cumulative sums for the convergence study;
#  - validation ensemble N=64 (scenarios_val_N64.csv) as a B4 loop that keeps every
#    member's hmax for the Sec 10.1 comparison on the 250 m grid.
# Finished chunks are skipped, so the job can be resubmitted after a timeout.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export PYTHONPATH="${SLURM_SUBMIT_DIR}/prep:${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export FCA_GIT="$(git rev-parse HEAD 2>/dev/null || true)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=index,name,driver_version,utilization.gpu,memory.used --format=csv
B="${B:-build/p10_gpu}"
PROC=data/processed/udupi_coast
OUT=results/e12
mkdir -p "${OUT}/main"

for chunk in "0 8" "8 8" "16 16" "32 32" "64 64" "128 122" "250 250"; do
  read -r first n <<< "${chunk}"
  d="${OUT}/main/chunk_$(printf %03d "${first}")_$(printf %03d "${n}")"
  if [[ -f "${d}/count_050.bin" ]]; then
    echo "===== main chunk first=${first} n=${n}: done, skipping ====="
    continue
  fi
  echo "===== main chunk first=${first} n=${n} ====="
  mpirun -np 1 --bind-to none "${B}/floodca_ens_mpi_cuda" --grid "${PROC}/90" \
    --scenario-file "${PROC}/scenarios_N500.csv" --rain-dir "${PROC}/rain/90" \
    --first "${first}" --nscen "${n}" --tsim 86400 --neigh 4 --U 1 --block 16x16 --reduce cub \
    --out "${d}" --ens-csv results/e12_ensemble_main.csv
done

if [[ -f "${OUT}/val/count_050.bin" ]]; then
  echo "===== validation ensemble: done, skipping ====="
else
  echo "===== validation ensemble N=64 (B4 loop) ====="
  python tests/py/ens_reference.py run --exe "${B}/floodca_cuda" --grid "${PROC}/90" \
    --scenarios "${PROC}/scenarios_val_N64.csv" --rain-dir "${PROC}/rain/90" --n 64 \
    --tsim 86400 --neigh 4 --threads 1 --out "${OUT}/val"
fi
echo "P10_E12_RUNS_OK"
