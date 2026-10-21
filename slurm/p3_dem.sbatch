#!/bin/bash
#SBATCH --job-name=floodca-p3dem
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_CPU__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export PYTHONPATH="${SLURM_SUBMIT_DIR}/prep${PYTHONPATH:+:${PYTHONPATH}}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname) python=$(python --version)"
python -m floodca_prep.dem \
  --aoi config/aoi_udupi.yaml \
  --raw data/raw/dem \
  --out data/processed/udupi_coast
python -m pytest tests/py/test_data.py -q
echo "P3_DEM_OK"
