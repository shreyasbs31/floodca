#!/bin/bash
#SBATCH --job-name=floodca-prec
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_CPU__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# One 90 m scenario, OpenMP, single and double precision.
# Scenario 0 of the 500-member file. Not a timing result: walls stay out of bench.csv.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export PYTHONPATH="${SLURM_SUBMIT_DIR}/prep:${PYTHONPATH:-}"
export OMP_PROC_BIND=close OMP_PLACES=cores
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

PROC=data/processed/udupi_coast
read -r sid date lam dh phi < <(awk -F, 'NR==2 {print $1, $2, $3, $4, $5}' "${PROC}/scenarios_N500.csv")
echo "PREC scenario_id=${sid} date=${date}"

build_one() {
  local tag="$1" dbl="$2"
  local bd="build/p10prec_${tag}_${SLURM_JOB_ID}"
  cmake -S . -B "${bd}" -DCMAKE_BUILD_TYPE=Release -DFCA_DOUBLE="${dbl}" \
    -DFCA_WITH_OMP=ON -DFCA_WITH_MPI=OFF -DFCA_WITH_CUDA=OFF -DFCA_BUILD_TESTS=OFF
  cmake --build "${bd}" -j "${SLURM_CPUS_PER_TASK}" --target floodca_omp
  local out="results/precision_${tag}"
  mkdir -p "${out}"
  "${bd}/floodca_omp" --grid "${PROC}/90" --rain "${PROC}/rain/90/rain_${date}.bin" \
    --lambda "${lam}" --D_h "${dh}" --phi "${phi}" --hyeto triangle --tsim 86400 \
    --neigh 4 --U 1 --threads "${SLURM_CPUS_PER_TASK}" \
    --out "${out}" --exp PREC --csv "results/precision_${tag}_run.csv"
}

build_one f32 OFF
build_one f64 ON

python3 - << 'PY'
import json
from pathlib import Path
import numpy as np
from floodca_prep.io_bin import read_bin

hf, _ = read_bin("results/precision_f32/hmax.bin")
hd, _ = read_bin("results/precision_f64/hmax.bin")
mask, _ = read_bin("data/processed/udupi_coast/90/mask.bin")
land = mask == 0
a = hf[land].astype(np.float64)
b = hd[land].astype(np.float64)
diff = np.abs(a - b)
thr = {}
for t in (0.10, 0.25, 0.50):
    fa = int((a >= t).sum())
    fb = int((b >= t).sum())
    thr[f"{t:.2f}"] = {"float": fa, "double": fb, "difference": fa - fb}
out = {
    "scenario_id": 0,
    "grid": "90",
    "threads": 40,
    "max_abs_hmax_diff_m": float(diff.max()),
    "mean_abs_hmax_diff_m": float(diff.mean()),
    "land_cells": int(land.sum()),
    "flooded": thr,
}
Path("results/precision_check.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
PY
echo "P10_PRECISION_OK"
