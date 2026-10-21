#!/bin/bash
#SBATCH --job-name=floodca-p10e8a
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

# E8 accuracy part (SPEC 11, RQ3): effect of the time-step update interval U.
#  tasks 0-3: V0 at 90 m (23 days, K=4) with U in {5, 10, 25, 50}; U=1 is the V0 run
#             of job 242817 (results/v0_smoke), same binary and inputs.
#  task 4:    B0 (90 m, 24 h, lambda 1, D 24 h, phi 0.5, from the dry state) with
#             U in {1, 5, 10, 25, 50}.
# Accuracy runs, not timed: E1-verified OpenMP double build, 48 threads.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export OMP_PROC_BIND=close OMP_PLACES=cores
export FCA_GIT="$(git rev-parse HEAD 2>/dev/null || true)"
PROC=data/processed/udupi_coast
EXE="${B:-build/p4b_cpu}/floodca_omp"
B0_DATE="${B0_DATE:-2011-07-17}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} host=$(hostname)"
if (( SLURM_ARRAY_TASK_ID < 4 )); then
  US=(5 10 25 50)
  U=${US[${SLURM_ARRAY_TASK_ID}]}
  "${EXE}" --grid "${PROC}/90" --forcing "${PROC}/forcing_V0.csv" \
    --tsim 1987200 --neigh 4 --U "${U}" --threads "${SLURM_CPUS_PER_TASK}" \
    --diag_every 1000 --progress 50000 \
    --out "results/e8/V0_U${U}" --exp "E8_V0_U${U}" --csv results/e8_runs.csv
  echo "P10_E8_RUN_OK V0_U${U}"
else
  for U in 1 5 10 25 50; do
    "${EXE}" --grid "${PROC}/90" --rain "${PROC}/rain/90/rain_${B0_DATE}.bin" \
      --lambda 1 --D_h 24 --phi 0.5 --hyeto triangle \
      --tsim 86400 --neigh 4 --U "${U}" --threads "${SLURM_CPUS_PER_TASK}" \
      --out "results/e8/B0_U${U}" --exp "E8_B0_U${U}" --csv results/e8_runs.csv
    echo "P10_E8_RUN_OK B0_U${U}"
  done
fi
