#!/bin/bash
#SBATCH --job-name=floodca-e7tiny
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_GPU__
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:2
#SBATCH --nice=10000
#SBATCH --time=01:30:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# Extra E7 sizes: centre crops of S at 32 and 64. Same B0 protocol as E7.
# Also times a diagnostic serial binary compiled with -fopenmp but without
# FCA_OMP, on grid S, so the 1-thread gap can be split into flag vs pragma.
# nice=10000 so this does not jump ahead of the existing H1 chain.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/p7_lib.sh
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

BD="build/p7tiny_${SLURM_JOB_ID}"
p7_build "${BD}" OFF ON
for g in C32 C64; do
  p7_need_grid "${g}"
  p7_run E7 "${g}" "${BD}/floodca_serial"
  for t in 1 2 4 8 10 20 40; do
    p7_run E7 "${g}" "${BD}/floodca_omp" --threads "${t}"
  done
  p7_run E7 "${g}" "${BD}/floodca_cuda" --block 16x16 --reduce cub
done

# Serial source, OpenMP compiler flag, no FCA_OMP pragmas. Not an E4 row.
BDFLAG="build/p7gapflag_${SLURM_JOB_ID}"
cmake -S . -B "${BDFLAG}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DFCA_DOUBLE=OFF -DFCA_WITH_OMP=OFF -DFCA_WITH_MPI=OFF -DFCA_WITH_CUDA=OFF -DFCA_BUILD_TESTS=OFF \
  -DCMAKE_C_FLAGS="-O3 -march=native -fopenmp" > "${BDFLAG}.cmake.log" 2>&1 \
  || { cat "${BDFLAG}.cmake.log"; exit 1; }
cmake --build "${BDFLAG}" -j "$(nproc)" > "${BDFLAG}.build.log" 2>&1 || { tail -40 "${BDFLAG}.build.log"; exit 1; }
p7_run E4gap S "${BDFLAG}/floodca_serial"
echo "P7_E7_TINY_OK"
