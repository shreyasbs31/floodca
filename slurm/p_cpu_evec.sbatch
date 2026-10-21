#!/bin/bash
#SBATCH --job-name=floodca-evecpu
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_GPU__
#SBATCH --nodelist=__FCA_NODE_SECOND__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --exclusive
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# E-VEC and E-DTDRIFT, CPU only, on __FCA_NODE_SECOND__. No GPU is requested, so the
# driver preflight is not sourced. __FCA_NODE_SECOND__ is a Xeon Gold 6248 node, the
# same node type as __FCA_NODE_MAIN__.
#
# Estimate before submission, from executed walls, not from this job:
#   Two OpenMP Release builds: about 4 min.
#   Fast-math L, 40 threads, 1 warmup + 5 repeats: 6 * 28.019 s = 168 s
#     (job 242997 median, results/p7_e4_summary.csv).
#   Two 90 m scenario runs: 2 * (605.093/8) s = 151 s
#     (job 243816, 8 scenarios, results/e10_cpu.csv).
#   One U=50 drift run. The published U=50 OpenMP wall is 19.1 s and the
#     dt sweep is 9.01 s of the U=1 wall, so one run with an extra sweep
#     is about 1 min.
# Working total about 12 min. --time 00:25:00 is the hard stop (0.42 node-hours).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
source slurm/p7_lib.sh
export OMP_NUM_THREADS=40

HOST=$(hostname)
HOST=${HOST%%.*}
if [[ "${HOST}" != "__FCA_NODE_SECOND__" ]]; then
  echo "refusing: host ${HOST} is not __FCA_NODE_SECOND__" >&2
  exit 2
fi
if ! grep -q "Gold 6248" /proc/cpuinfo; then
  echo "refusing: CPU is not Xeon Gold 6248" >&2
  exit 2
fi
echo "CPU_ONLY host=${HOST} no GPU requested"

OUT=results/evec
mkdir -p "${OUT}"
STD="build/evec_std_${SLURM_JOB_ID}"
FAST="build/evec_fast_${SLURM_JOB_ID}"

cmake -S . -B "${STD}" -DCMAKE_BUILD_TYPE=Release -DFCA_DOUBLE=OFF \
  -DFCA_WITH_OMP=ON -DFCA_WITH_MPI=OFF -DFCA_WITH_CUDA=OFF -DFCA_BUILD_TESTS=OFF
cmake --build "${STD}" -j "${SLURM_CPUS_PER_TASK}" --target floodca_omp

cmake -S . -B "${FAST}" -DCMAKE_BUILD_TYPE=Release -DFCA_DOUBLE=OFF \
  -DFCA_WITH_OMP=ON -DFCA_WITH_MPI=OFF -DFCA_WITH_CUDA=OFF -DFCA_BUILD_TESTS=OFF \
  -DCMAKE_C_FLAGS="-ffast-math -fopenmp-simd"
cmake --build "${FAST}" -j "${SLURM_CPUS_PER_TASK}" --target floodca_omp
echo "FAST_FLAGS $(grep '^C_FLAGS' "${FAST}/CMakeFiles/floodca_core_omp.dir/flags.make")"

CSV="${SLURM_SUBMIT_DIR}/results/evec_runs.csv"
export CSV
rm -f "${CSV}"
p7_run EVEC L "${FAST}/floodca_omp" --threads 40

PROC=data/processed/udupi_coast
read -r sid date lam dh phi < <(awk -F, 'NR==2 {print $1, $2, $3, $4, $5}' "${PROC}/scenarios_N500.csv")
echo "EVEC scenario_id=${sid} date=${date}"
run90() {
  local tag="$1" exe="$2" dest="${OUT}/hmax_${1}"
  mkdir -p "${dest}"
  "${exe}" --grid "${PROC}/90" --rain "${PROC}/rain/90/rain_${date}.bin" \
    --lambda "${lam}" --D_h "${dh}" --phi "${phi}" --hyeto triangle --tsim 86400 \
    --neigh 4 --U 1 --threads 40 --out "${dest}" --exp EVEC90 \
    --csv "${OUT}/hmax_${tag}_run.csv"
}
run90 std "${STD}/floodca_omp"
run90 fast "${FAST}/floodca_omp"
python3 - << 'PY'
import json
from pathlib import Path
import numpy as np
from floodca_prep.io_bin import read_bin
hs, _ = read_bin("results/evec/hmax_std/hmax.bin")
hf, _ = read_bin("results/evec/hmax_fast/hmax.bin")
mask, _ = read_bin("data/processed/udupi_coast/90/mask.bin")
land = mask == 0
a = hs[land].astype(np.float64)
b = hf[land].astype(np.float64)
diff = np.abs(a - b)
thr = {}
for t in (0.10, 0.25, 0.50):
    fa = int((a >= t).sum())
    fb = int((b >= t).sum())
    thr[f"{t:.2f}"] = {
        "standard": fa, "fast_math": fb,
        "difference": fb - fa, "abs_difference": abs(fb - fa),
    }
out = {
    "scenario_id": 0,
    "grid": "90",
    "threads": 40,
    "max_abs_hmax_diff_m": float(diff.max()),
    "land_cells": int(land.sum()),
    "flooded": thr,
}
Path("results/evec_hmax.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
PY

export FCA_DT_DRIFT=1
export FCA_DT_DRIFT_OUT="${SLURM_SUBMIT_DIR}/results/edtdrift.json"
export CSV="${SLURM_SUBMIT_DIR}/results/edtdrift_run.csv"
rm -f "${CSV}"
P7_REPEATS=0 p7_run EDTDRIFT L "${STD}/floodca_omp" --threads 40 --U 50
unset FCA_DT_DRIFT FCA_DT_DRIFT_OUT
python3 - << 'PY'
from pathlib import Path
p = Path("results/edtdrift.json")
print(p.read_text() if p.is_file() else "MISSING edtdrift.json")
PY
echo "EVEC_DTDRIFT_OK"
