#!/bin/bash
#SBATCH --job-name=floodca-v0diag
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_CPU__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# V0 diagnostics on the fixed mask. No model run and no parameter change.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export PYTHONPATH="${SLURM_SUBMIT_DIR}/prep:${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
python -m analysis.v0_diagnostics
echo "P10_V0_DIAG_OK"
