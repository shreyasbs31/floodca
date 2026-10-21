#!/bin/bash
#SBATCH --job-name=floodca-p3fix90
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_CPU__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# Products that read 90/mask.bin or 90/lc.bin, regenerated after the 3x3 majority
# aggregation fix (scripts/rebuild_90m_majority.py): GFD 2734 validation rasters
# and land flood fraction, the Sec 10.1 exclusion mask, and the F1 draft figure.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export PYTHONPATH="${SLURM_SUBMIT_DIR}/prep:${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg MPLCONFIGDIR="/tmp/${USER}_mplconfig"
mkdir -p results/slurm "${MPLCONFIGDIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname)"

echo "===== GFD 2734 validation (jrc_perm_water band) ====="
python -m floodca_prep.validation_data \
  --gfd data/raw/gee/floodca_gfd_udupi_2734.tif \
  --out data/processed/udupi_coast \
  --event-id 2734

echo "===== Sec 10.1 exclusion mask ====="
python -m analysis.validation --proc data/processed/udupi_coast

echo "===== F1 draft ====="
python -m analysis.fig_f1

echo "===== data tests ====="
python -m pytest -v -rs tests/py/test_data.py tests/py/test_validation.py
echo "P3_FIX90_OK"
