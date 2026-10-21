#!/bin/bash
#SBATCH --job-name=floodca-p8e3
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_CPU__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-4%3
#SBATCH --output=results/slurm/%x_%A_%a.out
#SBATCH --error=results/slurm/%x_%A_%a.err

# E3 (SPEC 11.3, 10.3): V0 at 90 m (2005-09-19 to 2005-10-11, 23 days, U=1) for
# K in {4, 8} x k_C in {0.7, 1.0, 1.3}. K=4, k_C=1.0 is the V0 run of job 242817
# (same binary and inputs), so this array runs the other five. Accuracy runs, not
# timed: E1-verified OpenMP double build, 48 threads.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export OMP_PROC_BIND=close OMP_PLACES=cores
export FCA_GIT="$(git rev-parse HEAD 2>/dev/null || true)"
PROC=data/processed/udupi_coast
CASES=("8 1.0" "4 0.7" "8 0.7" "4 1.3" "8 1.3")
read -r K KC <<< "${CASES[${SLURM_ARRAY_TASK_ID}]}"
if [[ "${KC}" == 1.0 ]]; then
  GRID="${PROC}/90"
else
  GRID="${PROC}/e3/grid_90_kC${KC}"
fi
TAG="K${K}_kC${KC}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} host=$(hostname) case=${TAG} grid=${GRID}"
"${B:-build/p4b_cpu}/floodca_omp" --grid "${GRID}" \
  --forcing "${PROC}/forcing_V0.csv" \
  --tsim 1987200 --neigh "${K}" --U 1 --threads "${SLURM_CPUS_PER_TASK}" \
  --diag_every 1000 --progress 50000 \
  --out "results/e3/${TAG}" --exp "E3_${TAG}" --csv results/e3_runs.csv
echo "P8_E3_RUN_OK ${TAG}"
