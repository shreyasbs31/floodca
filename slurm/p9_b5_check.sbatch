#!/bin/bash
#SBATCH --job-name=floodca-p9b5
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_CPU__
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# P-9 exit check for B5 (SPEC 22): ensemble counts at 0.10/0.25/0.50 m must equal a
# plain loop over the same N=8 scenarios (90 m, 24 h, K=4) with the single-scenario
# OpenMP binary. Double build, prebuilt on the login node into ${B}.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export PYTHONPATH="${SLURM_SUBMIT_DIR}/prep:${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_PROC_BIND=close OMP_PLACES=cores
export FCA_GIT="$(git rev-parse HEAD 2>/dev/null || true)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname)"
B="${B:-build/p9_cpu}"
PROC=data/processed/udupi_coast
OUT=results/p9_b5
N="${N:-8}"
COMMON=(--grid "${PROC}/90" --scenario-file "${PROC}/scenarios_N500.csv" --rain-dir "${PROC}/rain/90"
        --nscen "${N}" --tsim 86400 --neigh 4 --U 1)

if [[ -f "${OUT}/reference/count_050.bin" && "${REUSE_REF:-0}" == 1 ]]; then
  echo "===== reusing reference loop in ${OUT}/reference ====="
else
echo "===== reference loop (floodca_omp, 48 threads) ====="
OMP_NUM_THREADS=48 python tests/py/ens_reference.py run --exe "${B}/floodca_omp" --grid "${PROC}/90" \
  --scenarios "${PROC}/scenarios_N500.csv" --rain-dir "${PROC}/rain/90" --n "${N}" --tsim 86400 \
  --neigh 4 --threads 48 --out "${OUT}/reference"
fi

run_ens() {
  local tag=$1 np=$2 thr=$3 mc=$4
  echo "===== B5 ${tag}: ${np} ranks x ${thr} threads, master_computes=${mc} ====="
  mpirun -np "${np}" --bind-to none -x OMP_NUM_THREADS="${thr}" -x OMP_PROC_BIND=false \
    "${B}/floodca_ens_mpi_omp" "${COMMON[@]}" --threads "${thr}" --master-computes "${mc}" \
    --out "${OUT}/${tag}" --ens-csv "${OUT}/ensemble_${tag}.csv"
  python tests/py/ens_reference.py compare "${OUT}/reference" "${OUT}/${tag}" \
    --csv results/e1_equivalence.csv --label "90,4,b5_${tag}"
}
run_ens r1x48_mc 1 48 1
run_ens r4x12 4 12 0
run_ens r3x16_mc 3 16 1
echo "P9_B5_CHECK_OK"
