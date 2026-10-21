#!/bin/bash
#SBATCH --job-name=floodca-p5e1
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_CPU__
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# P-5 exit check (E1: MPI vs serial, SPEC 7.7 and 12.1): double, FMA off, K=4 and 8,
# rank counts that give non-divisible blocks, including a prime P (1-D split).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export PYTHONPATH="${SLURM_SUBMIT_DIR}/prep:${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export FCA_GIT="$(git rev-parse HEAD 2>/dev/null || true)"
mkdir -p results/slurm
echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname) ntasks=${SLURM_NTASKS}"
mpirun --version | head -1

B="${B:-build/p45_cpu}"
export BUILD="${SLURM_SUBMIT_DIR}/${B}"
export MPIRUN=mpirun MPIRUN_ARGS="--bind-to core"

SETS="${E1_SETS:-synth gridM}"
if [[ " ${SETS} " == *" synth "* ]]; then
  echo "===== E1 MPI synthetic 257x263 (K=4,8; P 1 2 3 4 6 7 12) ====="
  E1_GRID=data/processed/e1_257x263 E1_GRID_TAG=257x263 E1_OUT=results/e1_mpi_257x263 \
    E1_THREADS=" " E1_MPI_PS="1 2 3 4 6 7 12" E1_NEIGHS="4 8" bash tests/c/test_equivalence.sh
fi
if [[ " ${SETS} " == *" gridM "* ]]; then
  echo "===== E1 MPI grid M (K=4,8; P 4 12 48) ====="
  E1_GRID=data/processed/udupi_coast/M E1_GRID_TAG=gridM E1_OUT=results/e1_mpi_gridM \
    E1_THREADS=" " E1_MPI_PS="4 12 48" E1_NEIGHS="4 8" bash tests/c/test_equivalence.sh
fi
echo "P5_E1_OK"
