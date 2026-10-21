#!/bin/bash
#SBATCH --job-name=floodca-v4198
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_CPU__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# Second validation event, declared in docs/DECISIONS.md before this job.
# GFD 4198, Tropical Storm Hudhud. Forcing 2014-10-09 to 2014-10-27 (the GFD
# window 2014-10-12 to 2014-10-27 plus 3 antecedent days). 90 m, K=4, U=1,
# default parameters, from dry. No tuning. Scoring runs only if the GFD
# GeoTIFF has been copied in; the model run does not need it.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export PYTHONPATH="${SLURM_SUBMIT_DIR}/prep:${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_PROC_BIND=close OMP_PLACES=cores
export MPLCONFIGDIR="/tmp/${USER}_mplconfig_${SLURM_JOB_ID}"
EXE=build/p4b_cpu/floodca_omp
test -x "${EXE}"
python -m floodca_prep.rainfall --start 2014-10-09 --end 2014-10-27 --name 4198 --skip-pool
TSIM=$(python -c 'import json; print(json.load(open("data/processed/udupi_coast/forcing_4198_report.json"))["T_sim_s"])')
mkdir -p ${FCA_SCRATCH}/results/val_4198
ln -sfn ${FCA_SCRATCH}/results/val_4198 results/val_4198
if [[ ! -f results/val_4198/hmax.bin ]]; then
  "${EXE}" --grid data/processed/udupi_coast/90 --forcing data/processed/udupi_coast/forcing_4198.csv \
    --tsim "${TSIM}" --neigh 4 --U 1 --threads "${SLURM_CPUS_PER_TASK}" \
    --diag_every 1000 --progress 100000 \
    --out results/val_4198 --exp V4198 --csv results/val_4198_runs.csv
fi
GFD=data/raw/gee/floodca_gfd_udupi_4198.tif
if [[ -f "${GFD}" ]]; then
  python -m floodca_prep.validation_data --gfd "${GFD}" --event-id 4198
  python -m analysis.v0_diagnostics --run results/val_4198 --event 4198
else
  echo "NEED_GFD_4198 model run finished; scoring waits for floodca_gfd_udupi_4198.tif"
fi
echo "P10_VAL_4198_OK"
