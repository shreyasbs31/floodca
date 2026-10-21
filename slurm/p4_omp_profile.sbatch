#!/bin/bash
#SBATCH --job-name=floodca-p4prof
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_CPU__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# Diagnostic only (not a benchmark): OpenMP thread scaling on warm-started grid M,
# 200 steps, per-phase timers, then a perf profile at 48 threads.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export OMP_PROC_BIND=close OMP_PLACES=cores
echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname)"
echo "affinity: $(taskset -cp $$)"
nproc
B="${B:-build/p45_cpu}"
W=data/processed/udupi_coast/warmstart
ARGS=(--grid "${W}/grid_M" --rain data/processed/udupi_coast/rain/M/rain_2011-07-17.bin
      --warmstart "${W}/h0_B0_M.bin" --steps 200 --neigh 4 --U 1)
for t in 1 2 4 8 16 24 48; do
  OMP_DISPLAY_ENV=$([[ $t == 48 ]] && echo true || echo false) \
    "${B}/floodca_omp" "${ARGS[@]}" --threads "$t" --csv results/omp_profile_diag.csv \
    --exp P4diag --out /tmp/fca_prof_$t 2>&1 | grep -E "done|OMP_PROC_BIND|OMP_PLACES|OMP_NUM"
done
"${B}/floodca_serial" "${ARGS[@]}" --csv results/omp_profile_diag.csv --exp P4diag \
  --out /tmp/fca_prof_serial | grep done
echo "===== perf 48 threads ====="
perf record -g -o /tmp/fca_perf_${SLURM_JOB_ID}.data "${B}/floodca_omp" "${ARGS[@]}" \
  --threads 48 --out /tmp/fca_prof_perf > /dev/null 2>&1 || echo "perf record failed"
perf report -i /tmp/fca_perf_${SLURM_JOB_ID}.data --stdio --no-children --sort symbol 2>/dev/null \
  | grep -vE "^$|^#" | head -40 || true
