#!/bin/bash
#SBATCH --job-name=floodca-p3imd
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_CPU__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# IMD 1991 to 2023 via imdlib on a CPU compute node (login HTTPS is blocked),
# then the event pool, AOI-mean series, and event_fields.nc (SPEC 4.3).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export PYTHONPATH="${SLURM_SUBMIT_DIR}/prep:${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname)"

echo "===== IMD download 1991-2023 ====="
python scripts/download_imd.py --out data/raw/imd --start 1991 --end 2023
ls -l data/raw/imd/rain/ | head -40

echo "===== rainfall: V0 forcing + event pool ====="
python -m floodca_prep.rainfall \
  --raw data/raw/imd \
  --out data/processed/udupi_coast \
  --aoi config/aoi_udupi.yaml \
  --start 2005-09-19 \
  --end 2005-10-11
cat data/processed/udupi_coast/event_pool_report.json
echo "P3_IMD_OK"
