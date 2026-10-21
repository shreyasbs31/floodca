#!/bin/bash
#SBATCH --job-name=floodca-p10val
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_CPU__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# Remaining members of the N=64 full-event validation ensemble, packed into one job
# (SPEC 6.5). Member k is the 90 m V0 event (2005-09-19 to 2005-10-11, K=4, U=1,
# double) with daily rain scaled by lambda_k. Members whose hmax.bin already exists
# (array 243055 tasks 0-5) are skipped. This is not the single V0 smoke run.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export OMP_PROC_BIND=close OMP_PLACES=cores
export FCA_GIT="$(git rev-parse HEAD 2>/dev/null || true)"
PROC=data/processed/udupi_coast
EXE="${B:-build/p4b_cpu}/floodca_omp"
test -x "${EXE}"
for K in $(seq 0 63); do
  read -r SID LAM < <(awk -F, -v k="$((K + 2))" 'NR == k {print $1, $3}' "${PROC}/scenarios_val_N64.csv")
  OUT="results/e12/val_full/scen_$(printf %04d "${SID}")"
  if [[ -f "${OUT}/hmax.bin" ]]; then
    echo "skip scenario ${SID}"
    continue
  fi
  echo "scenario ${SID} lambda ${LAM} host $(hostname)"
  "${EXE}" --grid "${PROC}/90" --forcing "${PROC}/forcing_V0.csv" --lambda "${LAM}" \
    --tsim 1987200 --neigh 4 --U 1 --threads "${SLURM_CPUS_PER_TASK}" \
    --diag_every 1000 --progress 100000 \
    --out "${OUT}" --exp E12_valfull --csv results/e12_valfull_runs.csv
done
echo "P10_VAL_PACKED_OK"
