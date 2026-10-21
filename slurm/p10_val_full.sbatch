#!/bin/bash
#SBATCH --job-name=floodca-p10vf
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_CPU__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-63%3
#SBATCH --output=results/slurm/%x_%A_%a.out
#SBATCH --error=results/slurm/%x_%A_%a.err

# Validation ensemble, full-event reading (SPEC 6.5, human decision 2026-10-09): member
# k is V0 (90 m, 2005-09-19 to 2005-10-11, K=4, U=1) with every daily field scaled by
# lambda_k from scenarios_val_N64.csv (lambda ~ U(0.9, 1.1)). Accuracy runs, not timed:
# E1-verified OpenMP double build, 48 threads, same binary and inputs as V0.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export OMP_PROC_BIND=close OMP_PLACES=cores
export FCA_GIT="$(git rev-parse HEAD 2>/dev/null || true)"
PROC=data/processed/udupi_coast
K=${SLURM_ARRAY_TASK_ID}
read -r SID LAM < <(awk -F, -v k="$((K + 2))" 'NR == k {print $1, $3}' "${PROC}/scenarios_val_N64.csv")
OUT="results/e12/val_full/scen_$(printf %04d "${SID}")"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} task=${K} host=$(hostname) scenario=${SID} lambda=${LAM}"
if [[ -f "${OUT}/hmax.bin" ]]; then
  echo "done already: ${OUT}"
  exit 0
fi
"${B:-build/p4b_cpu}/floodca_omp" --grid "${PROC}/90" --forcing "${PROC}/forcing_V0.csv" \
  --lambda "${LAM}" --tsim 1987200 --neigh 4 --U 1 --threads "${SLURM_CPUS_PER_TASK}" \
  --diag_every 1000 --progress 100000 \
  --out "${OUT}" --exp E12_valfull --csv results/e12_valfull_runs.csv
echo "P10_VALFULL_RUN_OK ${SID} ${LAM}"
