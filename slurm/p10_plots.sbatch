#!/bin/bash
#SBATCH --job-name=floodca-p10fig
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_CPU__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# Regenerate paper figures on a compute node from the CSVs that exist when this job
# starts. Missing experiments are skipped by the plot functions; they are not invented.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export PYTHONPATH="${SLURM_SUBMIT_DIR}/prep:${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="/tmp/${USER}_mplconfig_${SLURM_JOB_ID}"
python -m analysis.plots --only F7,F8,F9,F10,F11,F12,F13
python -m analysis.e12
python -m analysis.alignment
python -m analysis.v0_diagnostics
python -m analysis.e8_accuracy || echo "e8_accuracy skipped"
echo "P10_PLOTS_OK"
