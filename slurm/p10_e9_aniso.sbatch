#!/bin/bash
#SBATCH --job-name=floodca-e9an
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_CPU__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# E9 anisotropy on a CPU node with the existing double OpenMP binary.
# Flat plane, closed boundaries, initial 5x5 water column, no rain, no physics change.
# Stops when the wet-cell count stops growing. A 40 percent wet target is unreachable
# at this volume (docs/DECISIONS.md). physics.h is not modified.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export PYTHONPATH="${SLURM_SUBMIT_DIR}/prep:${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_PROC_BIND=close OMP_PLACES=cores
test -x build/p4b_cpu/floodca_omp
python analysis/e9_aniso.py build/p4b_cpu/floodca_omp "${SLURM_CPUS_PER_TASK}"
echo "P10_E9_ANISO_OK"
