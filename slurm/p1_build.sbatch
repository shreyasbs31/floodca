#!/bin/bash
#SBATCH --job-name=floodca-p1
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_GPU__
#SBATCH --exclude=__FCA_NODE_SECOND__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
# shellcheck disable=SC1091
source slurm/gpu_driver_preflight.sh
gpu_driver_preflight
mkdir -p build results/slurm
echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname) CC=$CC cmake=$(cmake --version | head -n1)"
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${CC}" \
  -DCMAKE_CXX_COMPILER="${CXX}" \
  -DCMAKE_CUDA_HOST_COMPILER="${CC}" \
  -DFCA_DOUBLE=ON \
  -DFCA_WITH_OMP=ON \
  -DFCA_WITH_MPI=ON \
  -DFCA_WITH_CUDA=ON \
  -DFCA_BUILD_TESTS=ON
cmake --build build -j "${SLURM_CPUS_PER_TASK}"
echo "===== binaries ====="
ls -l build/floodca_* build/test_* 2>/dev/null || ls -l build/
echo "===== --help ====="
./build/floodca_serial --help
./build/floodca_omp --help
./build/floodca_mpi --help
./build/floodca_cuda --help
./build/floodca_ens_mpi_omp --help
./build/floodca_ens_mpi_cuda --help
echo "===== pytest io ====="
export PYTHONPATH="${SLURM_SUBMIT_DIR}/prep${PYTHONPATH:+:${PYTHONPATH}}"
python -m pytest tests/py/test_io.py -q
echo "P1_BUILD_OK"
