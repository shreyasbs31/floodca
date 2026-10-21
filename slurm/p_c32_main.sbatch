#!/bin/bash
#SBATCH --job-name=floodca-c32main
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_GPU__
#SBATCH --nodelist=__FCA_NODE_MAIN__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# C32 OpenMP and serial on the main node (__FCA_NODE_MAIN__), single precision.
# Thread counts 1, 2, 4, 8, 10, 16, 20, 40, plus the serial binary.
# One discarded warmup and five repeats, median taken later from bench.csv.
# No GPU is used, so the driver preflight is not sourced.
# Ten minutes is the hard stop so the scheduler can backfill the job.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
echo "==== allocation ===="
scontrol show job "${SLURM_JOB_ID}"
source slurm/p7_lib.sh
export OMP_PROC_BIND=close
export OMP_PLACES=cores

python -m floodca_prep.crops --from-grid S --sizes 32
python - <<'PY'
from pathlib import Path
from floodca_prep.warmstart import dry_grid_dir
proc = Path("data/processed/udupi_coast")
dry_grid_dir(proc, "C32")
print("grid dir C32")
PY

BD="build/c32main_${SLURM_JOB_ID}"
p7_build "${BD}" OFF OFF
p7_need_grid C32
p7_run E7 C32 "${BD}/floodca_serial"
for t in 1 2 4 8 10 16 20 40; do
  p7_run E7 C32 "${BD}/floodca_omp" --threads "${t}"
done
echo "C32_MAIN_OK"
