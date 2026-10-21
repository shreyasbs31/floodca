#!/bin/bash
#SBATCH --job-name=floodca-spardpxgpu
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_GPU__
#SBATCH --exclude=__FCA_NODE_SECOND__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# CUDA half of E-SPARSE (S, M, L) and E-DPX (C32 to C256 and L).
# One GPU, not exclusive. The node may be shared. Driver preflight stays.
#
# Estimate before submission:
#   Single-precision CUDA end-to-end on L is 2.079 s (results/p7_e7_summary.csv).
#   Six runs on S, M, and L, bounded by the L wall: 18 * 2.079 s = 37 s.
#   Double-precision CUDA on L is about 1.6 times that, six runs about 20 s.
#   The smaller double-precision grids are shorter.
#   A CUDA sparse-state search, if the CPU job has not written it yet: a few minutes.
#   Two CUDA builds: about 8 min.
# Working total about 20 min on a quiet device. A shared node can stretch
# the runs. --time 00:45:00 is the hard stop (0.75 GPU-hours).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
source slurm/gpu_driver_preflight.sh
gpu_driver_preflight
source slurm/p7_lib.sh

run_locked() {
  (
    flock 9
    "$@"
  ) 9>>results/csv_append.lock
}

nproc() { echo "${SLURM_CPUS_PER_TASK:-8}"; }

BD="build/esparse_cuda_${SLURM_JOB_ID}"
p7_build "${BD}" OFF ON
(
  flock 9
  python3 bench/esparse_state.py --exe "${BD}/floodca_cuda" --grids S,M,L
) 9>results/esparse_state.lock

CSV="${SLURM_SUBMIT_DIR}/results/esparse_runs.csv"
export CSV
for g in S M L; do
  run_locked p7_run Esparse "${g}" "${BD}/floodca_cuda" --block 16x16 --reduce cub \
    --warmstart "${WS}/h0_sparse_${g}.bin"
done
for g in S M L; do
  out="results/esparse_end/cuda_${g}"
  rm -rf "${out}"
  mkdir -p "${out}"
  run_locked "${BD}/floodca_cuda" --grid "${WS}/grid_${g}" \
    --rain "${PROC}/rain/${g}/rain_${B0_DATE}.bin" \
    --warmstart "${WS}/h0_sparse_${g}.bin" --lambda 1 --D_h 24 --phi 0.5 \
    --hyeto triangle --steps 2000 --neigh 4 --U 1 \
    --block 16x16 --reduce cub \
    --out "${out}" --exp EsparseEnd --csv "${CSV}"
done

python3 -m floodca_prep.crops --from-grid S --sizes 32,64
python3 - << 'PY'
from pathlib import Path
from floodca_prep.warmstart import dry_grid_dir
proc = Path("data/processed/udupi_coast")
for name in ("C32", "C64", "C128", "C256"):
    dry_grid_dir(proc, name)
PY
DPX="build/edpx_cuda_${SLURM_JOB_ID}"
p7_build "${DPX}" ON ON
export CSV="${SLURM_SUBMIT_DIR}/results/edpx_runs.csv"
for g in C32 C64 C128 C256; do
  p7_need_grid "${g}"
  run_locked p7_run EDPX "${g}" "${DPX}/floodca_cuda" --block 16x16 --reduce cub
done
p7_need_grid L
run_locked p7_run EDPX L "${DPX}/floodca_cuda" --block 16x16 --reduce cub

python3 - << 'PY'
import csv, json
from pathlib import Path
import numpy as np

def iqr_report(path):
    rows = []
    if not Path(path).is_file():
        return {"file": path, "status": "missing"}
    with open(path, newline="") as f:
        for rec in csv.DictReader(f):
            if rec.get("warmup") == "1":
                continue
            rows.append(rec)
    groups = {}
    for rec in rows:
        key = (rec.get("exp_id"), rec.get("grid_name"), rec.get("backend"), rec.get("threads"), rec.get("precision"))
        groups.setdefault(key, []).append(float(rec["wall_total_s"]))
    out = []
    worst = 0.0
    for key, vals in groups.items():
        a = np.sort(np.asarray(vals))
        if len(a) < 2:
            ratio = None
        else:
            q1 = float(np.percentile(a, 25))
            q3 = float(np.percentile(a, 75))
            med = float(np.median(a))
            ratio = (q3 - q1) / med if med else None
            if ratio is not None and ratio > worst:
                worst = ratio
        out.append({"exp": key[0], "grid": key[1], "backend": key[2], "threads": key[3],
                    "precision": key[4], "n": len(a), "iqr_over_median": ratio})
    return {"file": path, "worst_iqr_over_median": worst, "groups": out}

report = {
    "esparse": iqr_report("results/esparse_runs.csv"),
    "edpx": iqr_report("results/edpx_runs.csv"),
}
Path("results/shared_gpu_iqr.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"worst_esparse": report["esparse"].get("worst_iqr_over_median"),
                  "worst_edpx": report["edpx"].get("worst_iqr_over_median")}))
PY
echo "SPARSE_DPX_GPU_OK"
