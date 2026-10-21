#!/bin/bash
#SBATCH --job-name=floodca-ensys
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_GPU__
#SBATCH --exclude=__FCA_NODE_SECOND__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# Nsight Systems fallback if Nsight Compute cannot read GPU counters.
# One GPU, not exclusive. Driver preflight stays. __FCA_NODE_SECOND__ is excluded.
# Estimate: one CUDA build about 4 min, a 20-step warmup, then about
# 200 profiled steps on L. The single-precision L step is about 1 ms,
# so the profiled window is a few seconds plus Systems overhead.
# Working total about 10 min. --time 00:20:00 is the hard stop.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
source slurm/gpu_driver_preflight.sh
gpu_driver_preflight
source slurm/p7_lib.sh

nproc() { echo "${SLURM_CPUS_PER_TASK:-8}"; }

NSYS="${FCA_CUDA_ROOT}/bin/nsys"
OUT=results/ensys
mkdir -p "${OUT}"
BD="build/ensys_${SLURM_JOB_ID}"
p7_build "${BD}" OFF ON

PROC=data/processed/udupi_coast
WS="${PROC}/warmstart"
B0_DATE=$(python -c "import json;print(json.load(open('${PROC}/scenarios_N500_report.json'))['b0']['base_date'])")
ARGS=(--grid "${WS}/grid_L" --rain "${PROC}/rain/L/rain_${B0_DATE}.bin"
      --warmstart "${WS}/h0_B0_L.bin" --lambda 1 --D_h 24 --phi 0.5
      --hyeto triangle --neigh 4 --U 1 --no-output
      --block 16x16 --reduce cub)
"${BD}/floodca_cuda" "${ARGS[@]}" --steps 20 --csv "${OUT}/warmup.csv" --exp ENSYS_WARM
"${NSYS}" profile --force-overwrite=true -o "${OUT}/nsys" \
  "${BD}/floodca_cuda" "${ARGS[@]}" --steps 200 --csv "${OUT}/profile_steps.csv" --exp ENSYS
"${NSYS}" stats --report cuda_gpu_kern_sum --format csv --force-export true \
  -o "${OUT}/kern" "${OUT}/nsys.nsys-rep" > "${OUT}/stats.log" 2>&1 || \
  "${NSYS}" stats --report cuda_gpu_kern_sum --format csv \
  -o "${OUT}/kern" "${OUT}/nsys.qdrep" > "${OUT}/stats.log" 2>&1 || true
python3 - << 'PY'
import csv, json
from pathlib import Path
raw = None
for p in Path("results/ensys").glob("kern*.csv"):
    raw = p
    break
if raw is None:
    for p in Path("results/ensys").glob("*.csv"):
        if "warmup" in p.name or "profile_steps" in p.name:
            continue
        raw = p
        break
groups = {"flux": 0.0, "depth_update": 0.0, "dt_reduction": 0.0, "other": 0.0}
kernels = []
if raw is not None and raw.is_file():
    with raw.open(newline="", errors="replace") as f:
        rows = list(csv.DictReader(f))
    name_key = time_key = None
    if rows:
        for k in rows[0]:
            lk = k.lower()
            if name_key is None and "name" in lk:
                name_key = k
            if time_key is None and ("time" in lk and ("total" in lk or lk.endswith("(ns)") or "sum" in lk)):
                time_key = k
        if time_key is None:
            for k in rows[0]:
                if "time" in k.lower():
                    time_key = k
                    break
    for rec in rows:
        name = rec.get(name_key, "") if name_key else ""
        try:
            t = float(str(rec.get(time_key, "0")).replace(",", ""))
        except ValueError:
            continue
        lname = name.lower()
        if "k_flux" in lname:
            bucket = "flux"
        elif "k_update" in lname or "k_outflow" in lname:
            bucket = "depth_update"
        elif "reduce" in lname:
            bucket = "dt_reduction"
        else:
            bucket = "other"
        groups[bucket] += t
        kernels.append({"name": name, "time": t, "bucket": bucket})
    total = sum(groups.values()) or 1.0
    shares = {k: v / total for k, v in groups.items()}
    summary = {"status": "ok" if kernels else "no_kernel_table", "source": str(raw) if raw else "",
               "time_unit": "as reported by nsys", "groups": groups, "shares": shares, "kernels": kernels}
else:
    summary = {"status": "no_kernel_table", "source": ""}
Path("results/ensys_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: summary.get(k) for k in ("status", "shares")}, indent=2))
PY
echo "ENSYS_DONE"
