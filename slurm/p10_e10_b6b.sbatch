#!/bin/bash
#SBATCH --job-name=floodca-p10b6b
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_GPU__
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# E10 B6-b (SPEC 7.10): 2 H1 nodes, 1 V100 each, N=500, 24 h, float, K=4.
# srun places one MPI rank per node. Energy is sampled on the first node.
# This job needs two free H1 nodes at once, so it can stay pending after E10.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export PYTHONPATH="${SLURM_SUBMIT_DIR}/prep:${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export FCA_GIT="$(git rev-parse HEAD 2>/dev/null || true)"
mkdir -p ${FCA_SCRATCH}/results/e10/stamps results/slurm
ln -sfn ${FCA_SCRATCH}/results/e10 results/e10
if [[ -f results/e10/stamps/E10_b6b_N500.ok ]]; then
  echo "skip E10_b6b_N500"
  exit 0
fi
BD="build/p10b6b_${SLURM_JOB_ID}"
cmake -S . -B "${BD}" -DCMAKE_BUILD_TYPE=Release -DFCA_DOUBLE=OFF -DFCA_WITH_OMP=ON \
  -DFCA_WITH_MPI=ON -DFCA_WITH_CUDA=ON -DFCA_BUILD_TESTS=OFF > "${BD}.cmake.log" 2>&1
cmake --build "${BD}" -j "$(nproc)" --target floodca_ens_mpi_cuda > "${BD}.build.log" 2>&1
PROC=data/processed/udupi_coast
python scripts/energy_sampler.py --label E10_b6b_N500 --exp E11 --idle 30 --scenarios 500 \
  --csv results/e11_energy.csv -- \
  srun --ntasks=2 --ntasks-per-node=1 "${BD}/floodca_ens_mpi_cuda" \
  --grid "${PROC}/90" --scenario-file "${PROC}/scenarios_N500.csv" --rain-dir "${PROC}/rain/90" \
  --nscen 500 --tsim 86400 --neigh 4 --U 1 --block 16x16 --reduce cub \
  --master-computes 1 --poll-every 50 \
  --out results/e10/b6b --ens-csv results/e10_b6b.csv --csv results/bench.csv --exp E10
touch results/e10/stamps/E10_b6b_N500.ok
echo "P10_B6B_OK"
