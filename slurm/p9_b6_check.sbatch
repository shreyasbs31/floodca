#!/bin/bash
#SBATCH --job-name=floodca-p9b6
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_GPU__
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# P-9 exit check for B6 (SPEC 22): ensemble counts at 0.10/0.25/0.50 m must equal the
# serial-equivalent loop over the same N=8 scenarios (90 m, 24 h, K=4) written by the
# B5 check (results/p9_b5/reference). Double build with --fmad=false and
# -ffp-contract=off, prebuilt on the login node into ${B}. Correctness only, not timed.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export PYTHONPATH="${SLURM_SUBMIT_DIR}/prep:${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export FCA_GIT="$(git rev-parse HEAD 2>/dev/null || true)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=index,name,driver_version,utilization.gpu,memory.used --format=csv
B="${B:-build/p9_gpu}"
PROC=data/processed/udupi_coast
REF=results/p9_b5/reference
OUT="${OUT:-results/p9_b6}"
VAR="${VAR:-}"
N="${N:-8}"
COMMON=(--grid "${PROC}/90" --scenario-file "${PROC}/scenarios_N500.csv" --rain-dir "${PROC}/rain/90"
        --nscen "${N}" --tsim 86400 --neigh 4 --U 1 --block 16x16 --reduce cub)
test -f "${REF}/count_050.bin"
fail=0

echo "===== B4 loop (floodca_cuda, 1 GPU) ====="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" python tests/py/ens_reference.py run \
  --exe "${B}/floodca_cuda" --grid "${PROC}/90" --scenarios "${PROC}/scenarios_N500.csv" \
  --rain-dir "${PROC}/rain/90" --n "${N}" --tsim 86400 --neigh 4 --threads 1 --out "${OUT}/b4_loop"
python tests/py/ens_reference.py compare "${REF}" "${OUT}/b4_loop" \
  --csv results/e1_equivalence.csv --label "90,4,b4_loop${VAR}" || fail=1

run_ens() {
  local tag=$1 np=$2 mc=$3
  echo "===== B6 ${tag}: ${np} ranks, master_computes=${mc} ====="
  mpirun -np "${np}" --bind-to none "${B}/floodca_ens_mpi_cuda" "${COMMON[@]}" \
    --master-computes "${mc}" --out "${OUT}/${tag}" --ens-csv "${OUT}/ensemble_${tag}.csv"
  python tests/py/ens_reference.py compare "${REF}" "${OUT}/${tag}" \
    --csv results/e1_equivalence.csv --label "90,4,b6_${tag}${VAR}" || fail=1
}
run_ens r1_mc 1 1
run_ens r2_mc 2 1
run_ens r2 2 0
[[ ${fail} -eq 0 ]] && echo "P9_B6_CHECK_OK" || { echo "P9_B6_CHECK_FAIL"; exit 1; }
