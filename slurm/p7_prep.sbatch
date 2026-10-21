#!/bin/bash
#SBATCH --job-name=floodca-p7prep
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_CPU__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# P-7 inputs (not timed): E7 crop grids C128, C256, C768, C1536 sliced from the 30 m
# products, and their B0 6 h warm starts with the E1-verified OpenMP double build.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export PYTHONPATH="${SLURM_SUBMIT_DIR}/prep:${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_PROC_BIND=close OMP_PLACES=cores
export FCA_GIT="$(git rev-parse HEAD 2>/dev/null || true)"
mkdir -p results/slurm
echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK}"

SIZES="${SIZES:-128,256,768,1536}"
echo "===== E7 crops ${SIZES} ====="
python -m floodca_prep.crops --sizes "${SIZES}"

echo "===== data tests ====="
python -m pytest -v -rs tests/py/test_data.py

echo "===== B0 warm starts ====="
GRIDS=$(echo "${SIZES}" | sed 's/\([0-9]\+\)/C\1/g')
python -m floodca_prep.warmstart --exe "${WS_EXE:-build/p4b_cpu/floodca_omp}" \
  --grids "${GRIDS}" --threads "${SLURM_CPUS_PER_TASK}"
cat data/processed/udupi_coast/warmstart/warmstart_report.json
echo "P7_PREP_OK"
