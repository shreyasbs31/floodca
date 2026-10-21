#!/bin/bash
#SBATCH --job-name=floodca-p4v0
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_CPU__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# P-4 exit check (E1: OpenMP vs serial, K=4 and K=8, threads 1/4/48), a check that
# the window-aware core reproduces the pre-refactor serial outputs exactly, and,
# only if both pass, the V0 smoke run (90 m, 23 days, K=4, defaults, OpenMP).
# Binaries are prebuilt on the login node into ${B} (double, -ffp-contract=off).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export PYTHONPATH="${SLURM_SUBMIT_DIR}/prep:${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg MPLCONFIGDIR="/tmp/${USER}_mplconfig"
export OMP_PROC_BIND=close OMP_PLACES=cores
export FCA_GIT="$(git rev-parse HEAD 2>/dev/null || true)"
mkdir -p results/slurm "${MPLCONFIGDIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK}"
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket"

B="${B:-build/p45_cpu}"
echo "===== unit tests ====="
"${B}/test_physics"
"${B}/test_mass"
"${B}/test_lake_at_rest"
"${B}/test_rain"

export BUILD="${SLURM_SUBMIT_DIR}/${B}"
# SKIP_E1=1 reuses the E1 outputs of an earlier run of this script with the same ${B}.
if [[ "${SKIP_E1:-0}" != 1 ]]; then
  echo "===== E1 synthetic 257x263 (K=4,8; threads 1,4,48) ====="
  E1_GRID=data/processed/e1_257x263 E1_GRID_TAG=257x263 \
    E1_THREADS="1 4 ${SLURM_CPUS_PER_TASK}" E1_NEIGHS="4 8" bash tests/c/test_equivalence.sh

  echo "===== E1 grid M (K=4,8; threads ${SLURM_CPUS_PER_TASK}) ====="
  E1_GRID=data/processed/udupi_coast/M E1_GRID_TAG=gridM \
    E1_THREADS="${SLURM_CPUS_PER_TASK}" E1_NEIGHS="4 8" bash tests/c/test_equivalence.sh
fi

echo "===== refactored serial vs pre-refactor serial (job 242690 outputs) ====="
for t in 257x263 gridM; do
  for K in 4 8; do
    for f in hmax h_final; do
      python tests/py/compare_hmax.py "results/e1_ref_prerefactor/${t}/K${K}/${f}.bin" \
        "results/e1_${t}/K${K}/serial/${f}.bin" 1e-10 \
        --csv results/e1_equivalence.csv --label "${t},${K},serial_vs_prerefactor,${f}"
    done
  done
done
echo "P4_E1_OK"
if [[ "${SKIP_V0:-0}" == 1 ]]; then
  exit 0
fi

echo "===== V0 smoke: 90 m, 23 days, K=4, defaults ====="
"${B}/floodca_omp" --grid data/processed/udupi_coast/90 \
  --forcing data/processed/udupi_coast/forcing_V0.csv \
  --tsim 1987200 --neigh 4 --U 1 --threads "${SLURM_CPUS_PER_TASK}" \
  --diag_every 1000 --progress 20000 \
  --out results/v0_smoke --exp V0smoke --csv results/v0_runs.csv
tail -n 1 results/v0_runs.csv

echo "===== V0 smoke metrics ====="
python -m analysis.v0_smoke --proc data/processed/udupi_coast --run results/v0_smoke
echo "P4_V0_SMOKE_DONE"
