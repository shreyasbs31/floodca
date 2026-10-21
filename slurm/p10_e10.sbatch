#!/bin/bash
#SBATCH --job-name=floodca-p10e10
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_GPU__
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:2
#SBATCH --time=30:00:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# E10 ensemble throughput on one exclusive H1 node (SPEC 7.9, 7.10, 11.3), float, 90 m,
# 24 h scenarios, K=4, U=1. One measured run per configuration (the 5-repeat protocol is
# for the timing experiments; a 24 h ensemble is logged once). Serial and OpenMP use N=8,
# labelled measured per-scenario throughput (human decision 2026-10-09; SPEC 6.2 allows a
# reduced CPU N when it is labelled). GPU and hybrid stay at the planned N. Energy (E11) is
# sampled around each configuration: 30 s idle, then RAPL plus nvidia-smi.
# Order puts the 2-GPU work first so a time-limit stop still keeps B4 and B6-a.
# B5 rank x thread pairs fill the 40 cores of this node (SPEC 7.9 lists a 16-core node).
# B6-b (2 nodes) is a separate job. B7 is out of scope.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/p10_lib.sh
mkdir -p ${FCA_SCRATCH}/results/e10/stamps
ln -sfn ${FCA_SCRATCH}/results/e10 results/e10

BD="build/p10e10_${SLURM_JOB_ID}"
p10_build "${BD}" OFF ON ON
PROC=data/processed/udupi_coast
SCEN="${PROC}/scenarios_N500.csv"
RAIN="${PROC}/rain/90"

run_loop() {
  local n=$1 exe=$2 threads=$3
  shift 3
  local i=0 sid date lam dh phi
  while read -r sid date lam dh phi; do
    i=$((i + 1))
    [[ "${i}" -le "${n}" ]] || break
    OMP_NUM_THREADS="${threads}" "${exe}" --grid "${PROC}/90" \
      --rain "${RAIN}/rain_${date}.bin" --lambda "${lam}" --D_h "${dh}" --phi "${phi}" \
      --hyeto triangle --tsim 86400 --neigh 4 --U 1 --threads "${threads}" \
      --no-output --csv "${CSV}" --exp E10 "$@"
  done < <(awk -F, 'NR > 1 {print $1, $2, $3, $4, $5}' "${SCEN}")
}

wrapped() {
  local label=$1 n=$2
  shift 2
  if [[ -f "results/e10/stamps/${label}.ok" ]]; then
    echo "skip ${label}"
    return 0
  fi
  python scripts/energy_sampler.py --label "${label}" --exp E11 --idle 30 --scenarios "${n}" \
    --csv results/e11_energy.csv -- "$@"
  touch "results/e10/stamps/${label}.ok"
  echo "E10_OK ${label}"
}

export -f run_loop
export BD PROC RAIN SCEN CSV
echo "===== B4 single GPU, N=500 ====="
export CUDA_VISIBLE_DEVICES=0
wrapped E10_b4_N500 500 bash -c 'run_loop 500 "'"${BD}"'/floodca_cuda" 1 --block 16x16 --reduce cub'

echo "===== B6-a, 1 node x 2 GPUs, N=500 ====="
unset CUDA_VISIBLE_DEVICES
wrapped E10_b6a_N500 500 mpirun -np 2 --bind-to none "${BD}/floodca_ens_mpi_cuda" \
  --grid "${PROC}/90" --scenario-file "${SCEN}" --rain-dir "${RAIN}" --nscen 500 \
  --tsim 86400 --neigh 4 --U 1 --block 16x16 --reduce cub --master-computes 1 --poll-every 50 \
  --out results/e10/b6a --ens-csv results/e10_b6a.csv --csv "${CSV}" --exp E10

echo "===== OpenMP, N=8, 40 threads (measured per-scenario) ====="
wrapped E10_omp_N8 8 bash -c 'run_loop 8 "'"${BD}"'/floodca_omp" 40'

echo "===== B5, N=64, 40 cores ====="
b5() {
  local tag=$1 np=$2 thr=$3
  local bind=(--bind-to none)
  [[ "${thr}" -eq 1 ]] && bind=(--bind-to core)
  wrapped "E10_b5_${tag}" 64 mpirun -np "${np}" "${bind[@]}" -x OMP_NUM_THREADS="${thr}" \
    -x OMP_PROC_BIND=false "${BD}/floodca_ens_mpi_omp" \
    --grid "${PROC}/90" --scenario-file "${SCEN}" --rain-dir "${RAIN}" --nscen 64 \
    --tsim 86400 --neigh 4 --U 1 --threads "${thr}" --master-computes 1 --poll-every 50 \
    --out "results/e10/b5_${tag}" --ens-csv "results/e10_b5_${tag}.csv" --csv "${CSV}" --exp E10
}
b5 1x40 1 40
b5 2x20 2 20
b5 4x10 4 10
b5 8x5 8 5
b5 20x2 20 2
b5 40x1 40 1

echo "===== serial, N=8 (measured per-scenario) ====="
wrapped E10_serial_N8 8 bash -c 'run_loop 8 "'"${BD}"'/floodca_serial" 1'
echo "P10_E10_OK"
