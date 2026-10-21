#!/bin/bash
#SBATCH --job-name=floodca-esparse
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_GPU__
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:2
#SBATCH --exclude=__FCA_NODE_SECOND__
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# E-SPARSE. Cap is 1.0 GPU-node hour, enforced by --time. New CSV only.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/p7_lib.sh
source slurm/gpu_driver_preflight.sh
gpu_driver_preflight
export CUDA_VISIBLE_DEVICES=0

CSV="${SLURM_SUBMIT_DIR}/results/esparse_runs.csv"
export CSV
START=$(date +%s)
LIMIT=$((START + 3300))

BD="build/esparse_${SLURM_JOB_ID}"
p7_build "${BD}" OFF ON
python3 bench/esparse_state.py --exe "${BD}/floodca_cuda" --grids S,M,L

have_time() { [[ $(date +%s) -lt ${LIMIT} ]]; }

sparse_run() {
  local grid=$1 exe=$2
  shift 2
  have_time || { echo "SKIP time cap ${grid} $*"; return 0; }
  local args=(--grid "${WS}/grid_${grid}" --rain "${PROC}/rain/${grid}/rain_${B0_DATE}.bin"
              --warmstart "${WS}/h0_sparse_${grid}.bin" --lambda 1 --D_h 24 --phi 0.5
              --hyeto triangle --steps 2000 --neigh 4 --U 1
              --no-output --csv "${CSV}" --exp Esparse)
  p7_precheck
  local r
  for r in $(seq 0 5); do
    have_time || { echo "SKIP mid-repeat ${grid}"; return 0; }
    export FCA_REPEAT="${r}" FCA_WARMUP=$([[ ${r} == 0 ]] && echo 1 || echo 0)
    "${exe}" "${args[@]}" "$@" | grep -E "^done" | sed "s/^/Esparse ${grid} r${r} /"
  done
  unset FCA_REPEAT FCA_WARMUP
}

dump_end() {
  local grid=$1 exe=$2 tag=$3
  shift 3
  local out="results/esparse_end/${tag}_${grid}"
  rm -rf "${out}"
  mkdir -p "${out}"
  "${exe}" --grid "${WS}/grid_${grid}" --rain "${PROC}/rain/${grid}/rain_${B0_DATE}.bin" \
    --warmstart "${WS}/h0_sparse_${grid}.bin" --lambda 1 --D_h 24 --phi 0.5 \
    --hyeto triangle --steps 2000 --neigh 4 --U 1 \
    --out "${out}" --exp EsparseEnd --csv "${CSV}" "$@" \
    > "${out}/run.log"
  echo "END ${tag} ${grid}"
}

# Cheap and required ratios first, then the one-thread rows.
for g in S M L; do
  sparse_run "${g}" "${BD}/floodca_cuda" --block 16x16 --reduce cub
done
for t in 1 10 20 40; do
  sparse_run S "${BD}/floodca_omp" --threads "${t}"
done
for t in 10 40; do
  sparse_run L "${BD}/floodca_omp" --threads "${t}"
done
for t in 40 20 10 1; do
  sparse_run M "${BD}/floodca_omp" --threads "${t}"
done
for g in S M L; do
  have_time || break
  dump_end "${g}" "${BD}/floodca_omp" omp40 --threads 40
  dump_end "${g}" "${BD}/floodca_cuda" cuda --block 16x16 --reduce cub
done

# Warm-start L at 10 threads only if the cap still has room for about 12 minutes.
if [[ $((LIMIT - $(date +%s))) -gt 720 ]]; then
  echo "WARM10 extra comparison"
  CSV="${SLURM_SUBMIT_DIR}/results/esparse_warm10.csv" p7_run E4warm10 L "${BD}/floodca_omp" --threads 10
fi

python3 - <<'PY'
import json
from pathlib import Path
import numpy as np
from floodca_prep.io_bin import read_bin
out = {}
for p in Path("results/esparse_end").glob("*"):
    hbin = p / "h_final.bin"
    if not hbin.is_file():
        continue
    grid = p.name.split("_", 1)[-1]
    mask = Path("data/processed/udupi_coast/warmstart") / f"grid_{grid}" / "mask.bin"
    h, _ = read_bin(hbin)
    m, _ = read_bin(mask)
    land = np.asarray(m) == 0
    wet = np.asarray(h)[land] >= 1e-3
    out[p.name] = {"wet_frac_end": float(wet.mean()), "n_land": int(land.sum()), "n_wet": int(wet.sum())}
Path("results/esparse_end.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
PY
echo "ESPARSE_OK"
