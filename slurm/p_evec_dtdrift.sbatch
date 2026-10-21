#!/bin/bash
#SBATCH --job-name=floodca-evec
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_GPU__
#SBATCH --exclude=__FCA_NODE_SECOND__
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:2
#SBATCH --time=00:18:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# E-VEC (cap 0.3 node-hours) and E-DTDRIFT (cap 0.25 node-hours) on one
# exclusive Gold 6248 node, so the OpenMP wall is the same host class as
# the published L median. Estimate before submission, from the published
# single-precision L wall of 28.0 s (job set in results/bench.csv):
#   vectorisation compile plus two Release builds: about 4 min
#   E-VEC L, 40 threads, 1 warmup + 5 repeats: 6 * 28 s = 168 s
#   one 90 m day, two builds: 2 * 76 s = 152 s (the 1x40 scenario timer)
#   E-DTDRIFT one U=50 run, plus a fresh dt sweep each reuse step: under 2 min
# Working total about 12 min. --time 00:18:00 is 0.30 node-hours and is the
# hard stop. Neither experiment quotes the other's wall.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
source slurm/gpu_driver_preflight.sh
gpu_driver_preflight
source slurm/p7_lib.sh
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=40

OUT=results/evec
mkdir -p "${OUT}" results/edtdrift
echo "libmvec:"; ldconfig -p 2>/dev/null | grep libmvec || ls /lib64/libmvec* /usr/lib64/libmvec* 2>/dev/null || echo "NO_LIBMVEC"

# Standard build. -fopt-info-vec-missed does not change codegen.
MISS="${SLURM_SUBMIT_DIR}/${OUT}/vec_missed.txt"
STD="build/evec_std_${SLURM_JOB_ID}"
cmake -S . -B "${STD}" -DCMAKE_BUILD_TYPE=Release -DFCA_DOUBLE=OFF \
  -DFCA_WITH_OMP=ON -DFCA_WITH_MPI=OFF -DFCA_WITH_CUDA=OFF -DFCA_BUILD_TESTS=OFF \
  -DCMAKE_C_FLAGS="-fopt-info-vec-missed=${MISS}"
cmake --build "${STD}" -j "$(nproc)" --target floodca_omp
echo "STD_FLAGS $(grep '^C_FLAGS' "${STD}/CMakeFiles/floodca_core_omp.dir/flags.make")"
if [[ -f "${MISS}" ]]; then
  echo "VEC_MISSED_BYTES $(wc -c < "${MISS}")"
else
  echo "VEC_MISSED_MISSING"
fi

# Fast-math build. -fopenmp-simd is a no-op on the flux loop unless a simd
# pragma is already present. physics.h is not edited. libmvec is used only
# if the toolchain links it.
FAST="build/evec_fast_${SLURM_JOB_ID}"
cmake -S . -B "${FAST}" -DCMAKE_BUILD_TYPE=Release -DFCA_DOUBLE=OFF \
  -DFCA_WITH_OMP=ON -DFCA_WITH_MPI=OFF -DFCA_WITH_CUDA=OFF -DFCA_BUILD_TESTS=OFF \
  -DCMAKE_C_FLAGS="-ffast-math -fopenmp-simd"
cmake --build "${FAST}" -j "$(nproc)" --target floodca_omp
echo "FAST_FLAGS $(grep '^C_FLAGS' "${FAST}/CMakeFiles/floodca_core_omp.dir/flags.make")"
echo "FAST_LINK"; ldd "${FAST}/floodca_omp" | grep -E "libm|libmvec" || true

CSV="${SLURM_SUBMIT_DIR}/results/evec_runs.csv"
export CSV
rm -f "${CSV}"
p7_run EVEC L "${FAST}/floodca_omp" --threads 40

# One 90 m scenario, scenario 0, both builds. Walls are not a timing claim.
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
    thr[f"{t:.2f}"] = {"standard": fa, "fast_math": fb, "difference": fb - fa, "abs_difference": abs(fb - fa)}
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

# E-DTDRIFT. One U=50 run. The fresh dt is computed and discarded.
# The wall includes that extra sweep and is not the published U=50 median.
export FCA_DT_DRIFT=1
export FCA_DT_DRIFT_OUT="${SLURM_SUBMIT_DIR}/results/edtdrift.json"
DRIFT_CSV="${SLURM_SUBMIT_DIR}/results/edtdrift_run.csv"
export CSV="${DRIFT_CSV}"
rm -f "${DRIFT_CSV}"
P7_REPEATS=0 p7_run EDTDRIFT L "${STD}/floodca_omp" --threads 40 --U 50
unset FCA_DT_DRIFT FCA_DT_DRIFT_OUT
python3 - << 'PY'
import json
from pathlib import Path
p = Path("results/edtdrift.json")
print(p.read_text() if p.is_file() else "MISSING edtdrift.json")
PY
echo "EVEC_DTDRIFT_OK"
