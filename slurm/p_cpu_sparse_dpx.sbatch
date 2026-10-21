#!/bin/bash
#SBATCH --job-name=floodca-spardpxcpu
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_GPU__
#SBATCH --nodelist=__FCA_NODE_SECOND__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# CPU anchors, then E-SPARSE OpenMP and E-DPX OpenMP, on __FCA_NODE_SECOND__.
# No GPU is requested, so the driver preflight is not sourced.
# scontrol show node __FCA_NODE_SECOND__ on 19 October 2026 reported CPUTot=40,
# ThreadsPerCore=1, RealMemory=192131. This job requests all 40 CPUs
# and --mem=0 (the whole node memory). OpenMP uses 40 threads.
# --exclusive is omitted: on this partition it also allocates both GPUs.
#
# Estimate before submission, from results/p7_e4_summary.csv, one warmup
# plus five repeats:
#   Anchor L at 40 threads: 6 * 28.019063 s = 168 s
#   Anchor M at 1 thread: 6 * 246.17414 s = 1,477 s
#   Default FP32 build: about 3 min
#   Sparse state, OpenMP timings, and double-precision sweep: about 92 min
#     (the earlier 95 min estimate, with the FP32 build counted once)
# Within-3-percent path: about 2.1 h.
# If either anchor differs from the __FCA_NODE_MAIN__ median by more than 3 percent,
# extra dense FP32 baselines, using the 8-thread wall where no 10-thread
# row exists:
#   S at 1, 10, 20, 40: 458 s
#   M at 10, 20, 40: 317 s
#   L at 10: 775 s
#   C32 to C256 thread sweeps: about 5 min
# That extra block is about 32 min, so the long path is about 2.6 h.
# --time 04:00:00 is the hard stop.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
echo "==== allocation ===="
scontrol show job "${SLURM_JOB_ID}"
source slurm/env.sh
source slurm/p7_lib.sh
export OMP_NUM_THREADS=40
export OMP_PROC_BIND=close
export OMP_PLACES=cores
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS} OMP_PROC_BIND=${OMP_PROC_BIND} OMP_PLACES=${OMP_PLACES}"

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

run_locked() {
  (
    flock 9
    "$@"
  ) 9>>results/csv_append.lock
}

BD="build/anchor_omp_${SLURM_JOB_ID}"
p7_build "${BD}" OFF OFF

export CSV="${SLURM_SUBMIT_DIR}/results/anchor_second_node.csv"
rm -f "${CSV}"
p7_run ANCHOR L "${BD}/floodca_omp" --threads 40
p7_run ANCHOR M "${BD}/floodca_omp" --threads 1

python3 - << 'PY'
import csv, json
from pathlib import Path

def walls(path, grid, threads):
    vals = []
    with open(path, newline="") as f:
        for rec in csv.DictReader(f):
            if rec.get("warmup") == "1":
                continue
            name = rec.get("grid_name", "").replace("grid_", "")
            if name == grid and rec.get("threads") == str(threads):
                vals.append(float(rec["wall_total_s"]))
    vals.sort()
    if not vals:
        raise SystemExit("no anchor rows for %s threads %s" % (grid, threads))
    return vals[len(vals) // 2]

ref = {}
with open("results/p7_e4_summary.csv", newline="") as f:
    for rec in csv.DictReader(f):
        ref[(rec["grid"], rec["threads"])] = float(rec["T_median_s"])

pairs = {
    "L_40": (walls("results/anchor_second_node.csv", "L", 40), ref[("L", "40")]),
    "M_1": (walls("results/anchor_second_node.csv", "M", 1), ref[("M", "1")]),
}
out = {"host": "__FCA_NODE_SECOND__", "reference": "results/p7_e4_summary.csv", "rows": {}}
within = True
for key, (got, base) in pairs.items():
    rel = abs(got - base) / base
    out["rows"][key] = {"second_node_median_s": got, "main_node_median_s": base, "abs_rel_diff": rel}
    if rel > 0.03:
        within = False
out["within_3pct"] = within
# Fast-math comparison uses this same-node L/40 median only.
fast = walls("results/evec_runs.csv", "L", 40)
anchor = pairs["L_40"][0]
# (anchor / fast - 1) is the gap relative to the fast-math median.
# Two-decimal percent of the unrounded ratio is 0.97.
# (anchor - fast) / anchor prints as 0.96 and is not the reported figure.
ratio = anchor / fast
percent = (ratio - 1.0) * 100.0
rel_fast = ratio - 1.0
same = {
    "fast_math_median_s": fast,
    "same_node_default_median_s": anchor,
    "ratio_anchor_over_fast": ratio,
    "percent": percent,
    "abs_rel_diff": rel_fast,
    "within_5pct": percent < 5.0,
    "rule": "E-VEC 5 percent test against the __FCA_NODE_SECOND__ default build, not the __FCA_NODE_MAIN__ median. Reported percent is (anchor/fast - 1)*100.",
}
print("EVEC_RATIO %.8f printed_ratio %.4f printed_percent %.2f" % (
    ratio, round(ratio, 4), round(percent, 2)))
Path("results/anchor_second_node.json").write_text(json.dumps(out, indent=2) + "\n")
Path("results/evec_samenode.json").write_text(json.dumps(same, indent=2) + "\n")
print(json.dumps({"within_3pct": within, "fast_within_5pct": rel_fast < 0.05}))
PY

WITHIN=$(python3 -c "import json; print('yes' if json.load(open('results/anchor_second_node.json'))['within_3pct'] else 'no')")
echo "ANCHOR_WITHIN_3PCT ${WITHIN}"

if [[ "${WITHIN}" == "no" ]]; then
  echo "ANCHOR_OVER_3 same-node dense FP32 baselines"
  python3 -m floodca_prep.crops --from-grid S --sizes 32,64
  python3 - << 'PY'
from pathlib import Path
from floodca_prep.warmstart import dry_grid_dir
proc = Path("data/processed/udupi_coast")
for name in ("C32", "C64", "C128", "C256"):
    dry_grid_dir(proc, name)
PY
  export CSV="${SLURM_SUBMIT_DIR}/results/second_node_dense_fp32.csv"
  rm -f "${CSV}"
  for t in 1 10 20 40; do
    p7_run DENSE S "${BD}/floodca_omp" --threads "${t}"
  done
  for t in 10 20 40; do
    p7_run DENSE M "${BD}/floodca_omp" --threads "${t}"
  done
  p7_run DENSE L "${BD}/floodca_omp" --threads 10
  for g in C32 C64 C128 C256; do
    p7_need_grid "${g}"
    for t in 1 4 8 10 20 40; do
      p7_run DENSE "${g}" "${BD}/floodca_omp" --threads "${t}"
    done
  done
fi

(
  flock 9
  python3 bench/esparse_state.py --exe "${BD}/floodca_omp" --grids S,M,L
) 9>results/esparse_state.lock

export CSV="${SLURM_SUBMIT_DIR}/results/esparse_runs.csv"
sparse_omp() {
  local grid=$1 threads=$2
  run_locked p7_run Esparse "${grid}" "${BD}/floodca_omp" --threads "${threads}" \
    --warmstart "${WS}/h0_sparse_${grid}.bin"
}
for t in 1 10 20 40; do
  sparse_omp S "${t}"
  sparse_omp M "${t}"
done
for t in 10 40; do
  sparse_omp L "${t}"
done

dump_end() {
  local grid=$1
  local out="results/esparse_end/omp40_${grid}"
  rm -rf "${out}"
  mkdir -p "${out}"
  run_locked "${BD}/floodca_omp" --grid "${WS}/grid_${grid}" \
    --rain "${PROC}/rain/${grid}/rain_${B0_DATE}.bin" \
    --warmstart "${WS}/h0_sparse_${grid}.bin" --lambda 1 --D_h 24 --phi 0.5 \
    --hyeto triangle --steps 2000 --neigh 4 --U 1 --threads 40 \
    --out "${out}" --exp EsparseEnd --csv "${CSV}"
}
for g in S M L; do
  dump_end "${g}"
done
python3 - << 'PY'
import json
from pathlib import Path
import numpy as np
from floodca_prep.io_bin import read_bin
out = {}
for p in Path("results/esparse_end").glob("omp40_*"):
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
Path("results/esparse_end_omp.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
PY

python3 -m floodca_prep.crops --from-grid S --sizes 32,64
python3 - << 'PY'
from pathlib import Path
from floodca_prep.warmstart import dry_grid_dir
proc = Path("data/processed/udupi_coast")
for name in ("C32", "C64", "C128", "C256"):
    dry_grid_dir(proc, name)
PY
DPX="build/edpx_omp_${SLURM_JOB_ID}"
p7_build "${DPX}" ON OFF
export CSV="${SLURM_SUBMIT_DIR}/results/edpx_runs.csv"
for g in C32 C64 C128 C256; do
  p7_need_grid "${g}"
  for t in 1 4 8 10 20 40; do
    run_locked p7_run EDPX "${g}" "${DPX}/floodca_omp" --threads "${t}"
  done
done
p7_need_grid L
run_locked p7_run EDPX L "${DPX}/floodca_omp" --threads 40
echo "SPARSE_DPX_CPU_OK"
