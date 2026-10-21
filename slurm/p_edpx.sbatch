#!/bin/bash
#SBATCH --job-name=floodca-edpx
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_GPU__
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:2
#SBATCH --exclude=__FCA_NODE_SECOND__
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# E-DPX. Double precision crossover. Cap is 0.5 GPU-node hour, enforced by --time.
# New CSV only. Same warm start and protocol as the single-precision crossover.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/p7_lib.sh
source slurm/gpu_driver_preflight.sh
gpu_driver_preflight
export CUDA_VISIBLE_DEVICES=0

CSV="${SLURM_SUBMIT_DIR}/results/edpx_runs.csv"
export CSV

python3 -m floodca_prep.crops --from-grid S --sizes 32,64
python3 - <<'PY'
from pathlib import Path
from floodca_prep.warmstart import dry_grid_dir
proc = Path("data/processed/udupi_coast")
for name in ("C32", "C64", "C128", "C256"):
    dry_grid_dir(proc, name)
PY

BD="build/edpx_${SLURM_JOB_ID}"
p7_build "${BD}" ON ON
for g in C32 C64 C128 C256; do
  p7_need_grid "${g}"
  for t in 1 4 8 10 20 40; do
    p7_run EDPX "${g}" "${BD}/floodca_omp" --threads "${t}"
  done
  p7_run EDPX "${g}" "${BD}/floodca_cuda" --block 16x16 --reduce cub
done
p7_need_grid L
p7_run EDPX L "${BD}/floodca_omp" --threads 40
p7_run EDPX L "${BD}/floodca_cuda" --block 16x16 --reduce cub
echo "EDPX_OK"
