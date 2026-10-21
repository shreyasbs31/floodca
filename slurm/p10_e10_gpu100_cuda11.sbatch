#!/bin/bash
#SBATCH --job-name=floodca-e10c11
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_GPU__
#SBATCH --nodelist=__FCA_NODE_SECOND__
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:2
#SBATCH --time=00:25:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# PREPARED, NOT SUBMITTED.
# Submit only if job 243817 has not started by 12:00 on 13 October 2026.
# This is a separate CUDA 11.0 build for __FCA_NODE_SECOND__, whose driver is 450.51.05.
# Do not write results/bench.csv. Do not mix these walls into CUDA 12 ratios.
#
# module cuda/11.0 is nvcc release 11.0, V11.0.194 (toolkit version.txt 11.0.207).
# That driver supports CUDA 11.0. The later 11.x modules on this cluster
# (11.2, 11.7, 11.8) need a driver newer than 450.51.05 and are not used.
# The 525-driver preflight is intentionally not sourced: it would refuse __FCA_NODE_SECOND__.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
if type module >/dev/null 2>&1; then
  module unload cuda/12.0 >/dev/null 2>&1 || true
  module load cuda/11.0
fi
export FCA_CUDA_ROOT=/opt/ohpc/pub/cuda/cuda-11.0
export CUDA_HOME="${FCA_CUDA_ROOT}"
export CUDACXX="${FCA_CUDA_ROOT}/bin/nvcc"
export PATH="${FCA_CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${FCA_CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONUNBUFFERED=1
export FCA_GIT="$(git rev-parse HEAD 2>/dev/null || true)"

HOST=$(hostname)
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d '[:space:]')
NVCC_REL=$("${CUDACXX}" --version | awk '/release/ {print; exit}')
echo "E10_CUDA11_SEPARATE host=${HOST} driver=${DRV} nvcc=${NVCC_REL}"
echo "E10_CUDA11_SEPARATE do not mix these walls into CUDA 12 ratios"
MAJOR=${DRV%%.*}
if [[ "${HOST}" != __FCA_NODE_SECOND__ || "${MAJOR}" != 450 ]]; then
  echo "refusing: this script is only for __FCA_NODE_SECOND__ with driver 450" >&2
  exit 3
fi
case "${NVCC_REL}" in
  *release\ 11.0,*) ;;
  *) echo "refusing: nvcc is not CUDA 11.0 (${NVCC_REL})" >&2; exit 3 ;;
esac

BD="build/p10e10gpu11_${SLURM_JOB_ID}"
cmake -S . -B "${BD}" -DCMAKE_BUILD_TYPE=Release -DFCA_DOUBLE=OFF -DFCA_WITH_OMP=ON \
  -DFCA_WITH_MPI=ON -DFCA_WITH_CUDA=ON -DFCA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_COMPILER="${CUDACXX}" > "${BD}.cmake.log" 2>&1
cmake --build "${BD}" -j "$(nproc)" --target floodca_ens_mpi_cuda > "${BD}.build.log" 2>&1

PROC=data/processed/udupi_coast
OUT=results/e10_gpu100_cuda11
mkdir -p "${OUT}"
CSV=results/e10_gpu100_cuda11.csv
echo "config,toolkit,driver,scenarios,ranks,gpus,wall_s,scen_per_hour,slurm_job_id" > "${CSV}"

run_ens() {
  local tag=$1 np=$2
  echo "===== ${tag} np=${np} N=100 CUDA11 ====="
  # Same file and --first 0 as job 243817. Ids 0..7 are the CPU ensemble.
  # --ens-csv records scenario_id and the per-scenario wall. No bench.csv.
  mpirun -np "${np}" --bind-to none "${BD}/floodca_ens_mpi_cuda" \
    --grid "${PROC}/90" --scenario-file "${PROC}/scenarios_N500.csv" --rain-dir "${PROC}/rain/90" \
    --first 0 --nscen 100 --tsim 86400 --neigh 4 --U 1 --block 16x16 --reduce cub \
    --master-computes 1 --poll-every 50 --no-output \
    --out "${OUT}/${tag}" --ens-csv "${OUT}/${tag}.csv" \
    --csv results/e10_gpu100_cuda11_bench.csv --exp E10cuda11 \
    | tee "${OUT}/${tag}.log"
  awk -v tag="${tag}" -v np="${np}" -v job="${SLURM_JOB_ID}" -v drv="${DRV}" '
    /done backend=ens_mpi_cuda/ {
      wall=""; rate=""; scen=""
      for (i = 1; i <= NF; i++) {
        split($i, a, "=")
        if (a[1] == "wall") wall=a[2]
        if (a[1] == "scen_per_hour") rate=a[2]
        if (a[1] == "scenarios") scen=a[2]
      }
      gpus = (np == 1 ? 1 : 2)
      print tag ",11.0," drv "," scen "," np "," gpus "," wall "," rate "," job
    }
  ' "${OUT}/${tag}.log" >> "${CSV}"
}

export CUDA_VISIBLE_DEVICES=0
run_ens B4 1
unset CUDA_VISIBLE_DEVICES
run_ens B6a 2

python3 - << 'PY'
import csv
rows = list(csv.DictReader(open("results/e10_gpu100_cuda11.csv")))
by = {r["config"]: r for r in rows}
b4 = float(by["B4"]["scen_per_hour"])
b6 = float(by["B6a"]["scen_per_hour"])
print(f"E10_CUDA11_RATIO b6_over_b4={b6 / b4:.4f} B4_per_hour={b4:.3f} B6a_per_hour={b6:.3f}")
print("E10_CUDA11_SEPARATE not a CUDA 12 ratio")
PY
echo "P10_E10_GPU100_CUDA11_OK"
