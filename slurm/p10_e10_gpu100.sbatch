#!/bin/bash
#SBATCH --job-name=floodca-e10gpu
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_GPU__
#SBATCH --exclude=__FCA_NODE_SECOND__
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:2
#SBATCH --time=00:25:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# E10 throughput, GPU half. Same 100 scenarios on B4 (1 GPU) and B6-a (2 GPUs).
# Wall is the ensemble timer: barrier before the first dispatch through the
# final count reduction. Exclusive node. No energy sampler.
# Device-time estimate from job 243032 (500 scenarios, 2951 s, one V100,
# not exclusive): 100 scenarios are about 590 s on one GPU (0.16 GPU-hours).
# B6-a adds a second pass on two GPUs. The two passes together stay under
# about 0.5 GPU-hours of device time if the second GPU takes a share of the work.
# --nscen 100 with the default --first 0 reads scenario ids 0..99 from
# scenarios_N500.csv. The CPU ensemble (jobs 243816 and 243819) ran ids 0..7,
# the first eight rows of that file, so those eight are inside this hundred.
# --ens-csv writes scenario_id and wall_s for each scenario. That wall is the
# scenario timer, not the ensemble wall.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
# shellcheck disable=SC1091
source slurm/gpu_driver_preflight.sh
gpu_driver_preflight
export PYTHONUNBUFFERED=1
export FCA_GIT="$(git rev-parse HEAD 2>/dev/null || true)"

BD="build/p10e10gpu_${SLURM_JOB_ID}"
cmake -S . -B "${BD}" -DCMAKE_BUILD_TYPE=Release -DFCA_DOUBLE=OFF -DFCA_WITH_OMP=ON \
  -DFCA_WITH_MPI=ON -DFCA_WITH_CUDA=ON -DFCA_BUILD_TESTS=OFF > "${BD}.cmake.log" 2>&1
cmake --build "${BD}" -j "$(nproc)" --target floodca_ens_mpi_cuda > "${BD}.build.log" 2>&1

PROC=data/processed/udupi_coast
OUT=results/e10_gpu100
mkdir -p "${OUT}"
CSV=results/e10_gpu100.csv
echo "config,scenarios,ranks,gpus,wall_s,scen_per_hour,slurm_job_id" > "${CSV}"

run_ens() {
  local tag=$1 np=$2
  echo "===== ${tag} np=${np} N=100 ====="
  mpirun -np "${np}" --bind-to none "${BD}/floodca_ens_mpi_cuda" \
    --grid "${PROC}/90" --scenario-file "${PROC}/scenarios_N500.csv" --rain-dir "${PROC}/rain/90" \
    --nscen 100 --tsim 86400 --neigh 4 --U 1 --block 16x16 --reduce cub \
    --master-computes 1 --poll-every 50 --no-output \
    --out "${OUT}/${tag}" --ens-csv "${OUT}/${tag}.csv" --csv results/bench.csv --exp E10 \
    | tee "${OUT}/${tag}.log"
  awk -v tag="${tag}" -v np="${np}" -v job="${SLURM_JOB_ID}" '
    /done backend=ens_mpi_cuda/ {
      wall=""; rate=""; scen=""
      for (i = 1; i <= NF; i++) {
        split($i, a, "=")
        if (a[1] == "wall") wall=a[2]
        if (a[1] == "scen_per_hour") rate=a[2]
        if (a[1] == "scenarios") scen=a[2]
      }
      gpus = (np == 1 ? 1 : 2)
      print tag "," scen "," np "," gpus "," wall "," rate "," job
    }
  ' "${OUT}/${tag}.log" >> "${CSV}"
}

export CUDA_VISIBLE_DEVICES=0
run_ens B4 1
unset CUDA_VISIBLE_DEVICES
run_ens B6a 2

python3 - << 'PY'
import csv
rows = list(csv.DictReader(open("results/e10_gpu100.csv")))
by = {r["config"]: r for r in rows}
b4 = float(by["B4"]["scen_per_hour"])
b6 = float(by["B6a"]["scen_per_hour"])
print(f"E10_GPU_RATIO b6_over_b4={b6 / b4:.4f} B4_per_hour={b4:.3f} B6a_per_hour={b6:.3f}")
PY
echo "P10_E10_GPU100_OK"
