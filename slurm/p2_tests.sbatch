#!/bin/bash
#SBATCH --job-name=floodca-p2
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_GPU__
#SBATCH --exclude=__FCA_NODE_SECOND__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
# shellcheck disable=SC1091
source slurm/gpu_driver_preflight.sh
gpu_driver_preflight
mkdir -p build results/slurm
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${CC}" \
  -DCMAKE_CXX_COMPILER="${CXX}" \
  -DCMAKE_CUDA_HOST_COMPILER="${CC}" \
  -DCMAKE_C_FLAGS="-ffp-contract=off -fno-fast-math" \
  -DFCA_DOUBLE=ON \
  -DFCA_WITH_OMP=ON \
  -DFCA_WITH_MPI=ON \
  -DFCA_WITH_CUDA=ON \
  -DFCA_BUILD_TESTS=ON
rm -f build/test_physics build/CMakeFiles/test_physics.dir/tests/c/test_physics.c.o
cmake --build build -j "${SLURM_CPUS_PER_TASK}" --target test_physics test_mass test_lake_at_rest floodca_serial
echo "===== test_physics ====="
./build/test_physics
echo "===== test_mass ====="
./build/test_mass
echo "===== test_lake_at_rest ====="
./build/test_lake_at_rest
echo "P2_TESTS_OK"
