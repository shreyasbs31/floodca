#!/bin/bash
#SBATCH --job-name=floodca-p3ws
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_CPU__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# P-3 close-out (SPEC 6.1, 6.3): scenarios_N500.csv and per-date rain fields, the
# data tests (event pool size, scenario reproducibility), then the B0 6 h warm
# starts on S, M, L, 30 (XL) and 90 with the OpenMP double build.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export PYTHONPATH="${SLURM_SUBMIT_DIR}/prep:${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_PROC_BIND=close OMP_PLACES=cores
export FCA_GIT="$(git rev-parse HEAD 2>/dev/null || true)"
mkdir -p results/slurm
echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK}"

# SKIP_SCEN=1 keeps the existing scenarios; WS_GRIDS and WS_EXE select grids and binary.
if [[ "${SKIP_SCEN:-0}" != 1 ]]; then
  echo "===== scenarios N=500 + rain fields ====="
  python -m floodca_prep.scenarios --n 500
fi

echo "===== data tests ====="
python -m pytest -v -rs tests/py/test_data.py tests/py/test_scenarios.py

echo "===== B0 warm starts ====="
python -m floodca_prep.warmstart --exe "${WS_EXE:-build/p3_ws/floodca_omp}" \
  --grids "${WS_GRIDS:-S,M,L,30,90}" --threads "${SLURM_CPUS_PER_TASK}"
cat data/processed/udupi_coast/warmstart/warmstart_report.json
echo "P3_WS_OK"
