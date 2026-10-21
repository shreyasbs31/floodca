#!/bin/bash
#SBATCH --job-name=floodca-encu
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

# E-NCU. One GPU, not exclusive. Profiling counters do not need the whole
# node. Driver preflight stays. __FCA_NODE_SECOND__ is excluded.
#
# Estimate before submission: one Release CUDA build is a few minutes, and
# about 15 kernel replays at roughly 20 s each is about 5 min. Working
# total about 8 min on a quiet device. --time 00:20:00 allows a shared node
# (0.33 GPU-hours). Nsight Compute replays each kernel. The binary runs
# 2 steps on L. The first launch is the warmup. Metrics are taken from the
# second launch of each kernel.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
source slurm/gpu_driver_preflight.sh
gpu_driver_preflight
source slurm/p7_lib.sh

nproc() { echo "${SLURM_CPUS_PER_TASK:-8}"; }

NCU="${FCA_CUDA_ROOT}/bin/ncu"
OUT=results/encu
mkdir -p "${OUT}"
BD="build/encu_${SLURM_JOB_ID}"
p7_build "${BD}" OFF ON

PROC=data/processed/udupi_coast
WS="${PROC}/warmstart"
B0_DATE=$(python -c "import json;print(json.load(open('${PROC}/scenarios_N500_report.json'))['b0']['base_date'])")
METRICS="dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.avg,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active"

set +e
"${NCU}" --target-processes all \
  --kernel-name-base demangled \
  --kernel-name regex:"k_flux_x|k_flux_y|k_outflow|k_update|k_vbound|DeviceReduce" \
  --launch-count 24 \
  --metrics "${METRICS}" \
  --csv --page raw \
  -o "${OUT}/ncu" \
  "${BD}/floodca_cuda" \
    --grid "${WS}/grid_L" --rain "${PROC}/rain/L/rain_${B0_DATE}.bin" \
    --warmstart "${WS}/h0_B0_L.bin" --lambda 1 --D_h 24 --phi 0.5 \
    --hyeto triangle --steps 2 --neigh 4 --U 1 --no-output \
    --block 16x16 --reduce cub \
    --csv "${OUT}/warmup_steps.csv" --exp ENCU \
  > "${OUT}/ncu_stdout.txt" 2> "${OUT}/ncu_stderr.txt"
rc=$?
set -e
echo "ncu_exit ${rc}"
if [[ -f "${OUT}/ncu.csv" ]]; then
  cp -f "${OUT}/ncu.csv" "${OUT}/ncu_raw.csv"
elif [[ -s "${OUT}/ncu_stdout.txt" ]]; then
  cp -f "${OUT}/ncu_stdout.txt" "${OUT}/ncu_raw.csv"
fi
python3 - << 'PY'
import json
from pathlib import Path
raw = Path("results/encu/ncu_raw.csv")
err = Path("results/encu/ncu_stderr.txt").read_text(errors="replace") if Path("results/encu/ncu_stderr.txt").is_file() else ""
summary = {"status": "no_csv", "stderr_tail": err[-2000:]}
if raw.is_file() and raw.stat().st_size > 0:
    text = raw.read_text(errors="replace")
    summary["raw_bytes"] = raw.stat().st_size
    summary["raw_head"] = "\n".join(text.splitlines()[:8])
    summary["status"] = "csv_captured"
Path("results/encu_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2)[:2000])
PY
echo "ENCU_DONE"
