#!/bin/bash
#SBATCH --job-name=floodca-paperfig
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_CPU__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# Redraw F7, F9 and F10 from finished jobs only. Not part of the experiment chain.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export PYTHONPATH="${SLURM_SUBMIT_DIR}/prep:${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="/tmp/${USER}_mplconfig_${SLURM_JOB_ID}"
python -m analysis.plots --only F7,F9,F10 --jobs 242994,242998,242999
echo "PAPER_FIGS_OK"
