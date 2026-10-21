#!/bin/bash
#SBATCH --job-name=floodca-e7rest
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_GPU__
#SBATCH --exclude=__FCA_NODE_SECOND__
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# Replacement for job 243333, which died on __FCA_NODE_SECOND__ (driver 450.51.05) after the
# 32 by 32 CPU rows. This job does not repeat those rows.
#   C32: CUDA 16x16 device minimum only. Per-step time is wall/2000 and the
#        kernel wall from the same rows.
#   C64: serial, OpenMP at 1, 2, 4, 8, 10, 20, 40 threads, and the same CUDA run.
#   E4gap: serial binary with -fopenmp and without FCA_OMP, timed on grid S.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/p7_lib.sh
# shellcheck disable=SC1091
source slurm/gpu_driver_preflight.sh
gpu_driver_preflight
export CUDA_VISIBLE_DEVICES=0

python -m floodca_prep.crops --from-grid S --sizes 32,64
python - <<'PY'
from pathlib import Path
from floodca_prep.warmstart import dry_grid_dir
proc = Path("data/processed/udupi_coast")
for name in ("C32", "C64"):
    dry_grid_dir(proc, name)
    print("grid dir", name)
PY

BD="build/p7rest_${SLURM_JOB_ID}"
p7_build "${BD}" OFF ON
p7_need_grid C32
p7_run E7 C32 "${BD}/floodca_cuda" --block 16x16 --reduce cub
p7_need_grid C64
p7_run E7 C64 "${BD}/floodca_serial"
for t in 1 2 4 8 10 20 40; do
  p7_run E7 C64 "${BD}/floodca_omp" --threads "${t}"
done
p7_run E7 C64 "${BD}/floodca_cuda" --block 16x16 --reduce cub

BDFLAG="build/p7gapflag_${SLURM_JOB_ID}"
cmake -S . -B "${BDFLAG}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DFCA_DOUBLE=OFF -DFCA_WITH_OMP=OFF -DFCA_WITH_MPI=OFF -DFCA_WITH_CUDA=OFF -DFCA_BUILD_TESTS=OFF \
  -DCMAKE_C_FLAGS="-O3 -march=native -fopenmp" > "${BDFLAG}.cmake.log" 2>&1 \
  || { cat "${BDFLAG}.cmake.log"; exit 1; }
cmake --build "${BDFLAG}" -j "$(nproc)" > "${BDFLAG}.build.log" 2>&1 || { tail -40 "${BDFLAG}.build.log"; exit 1; }
p7_run E4gap S "${BDFLAG}/floodca_serial"
echo "P7_E7_REST_OK"
