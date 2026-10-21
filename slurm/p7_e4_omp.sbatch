#!/bin/bash
#SBATCH --job-name=floodca-p7e4
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_GPU__
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# E4 (SPEC 11.3): OpenMP strong scaling on H1 (Gold 6248, 40 cores), float, K=4,
# B0 warm start, 2000 steps, 1 warm-up + 5 repeats per configuration. Serial (B1)
# on the same node is the speedup baseline. Split into chunks with E4_GRIDS/E4_MODES.
cd "${SLURM_SUBMIT_DIR}"
source slurm/p7_lib.sh

GRIDS="${E4_GRIDS:-S M}"
MODES="${E4_MODES:-serial omp}"
THREADS="${E4_THREADS:-1 2 4 8 16 20 32 40}"
BD="build/p7cpu_${SLURM_JOB_ID}"
p7_build "${BD}" OFF OFF
for g in ${GRIDS}; do
  p7_need_grid "${g}"
  for mode in ${MODES}; do
    if [[ ${mode} == serial ]]; then
      p7_run E4 "${g}" "${BD}/floodca_serial"
    else
      for t in ${THREADS}; do
        p7_run E4 "${g}" "${BD}/floodca_omp" --threads "${t}"
      done
    fi
  done
done
echo "P7_E4_CHUNK_OK grids=${GRIDS} modes=${MODES} threads=${THREADS}"
