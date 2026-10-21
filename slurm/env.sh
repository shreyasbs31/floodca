#!/usr/bin/env bash
# Toolchain for a Slurm run. Versions are recorded in docs/ENVIRONMENT.md.
# Copy slurm/config.example.sh to slurm/config.sh before sourcing this file.
#
# Usage, from the repository root, on a compute node:
#   source slurm/env.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "source this file: source slurm/env.sh" >&2
    return 1 2>/dev/null || exit 1
fi

_FCA_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${_FCA_SLURM_DIR}/config.sh" ]]; then
    # shellcheck disable=SC1091
    source "${_FCA_SLURM_DIR}/config.sh"
fi

: "${FCA_SCRATCH:?Set FCA_SCRATCH in slurm/config.sh}"
: "${FCA_GCC_ROOT:?Set FCA_GCC_ROOT in slurm/config.sh}"
: "${FCA_OMPI_ROOT:?Set FCA_OMPI_ROOT in slurm/config.sh}"
: "${FCA_CUDA_ROOT:?Set FCA_CUDA_ROOT in slurm/config.sh}"
: "${FCA_CMAKE_ROOT:?Set FCA_CMAKE_ROOT in slurm/config.sh}"

export CC="${FCA_GCC_ROOT}/bin/gcc"
export CXX="${FCA_GCC_ROOT}/bin/g++"
export OMPI_CC="${CC}"
export OMPI_CXX="${CXX}"
export CUDA_HOME="${FCA_CUDA_ROOT}"
export CUDACXX="${FCA_CUDA_ROOT}/bin/nvcc"
# OpenMPI 4.1.1 still probes the deprecated openib BTL. The paper's jobs
# set the PML to UCX and disabled that BTL.
export OMPI_MCA_btl="${OMPI_MCA_btl:-^openib}"
export OMPI_MCA_pml="${OMPI_MCA_pml:-ucx}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

export PATH="${FCA_CMAKE_ROOT}/bin:${FCA_GCC_ROOT}/bin:${FCA_OMPI_ROOT}/bin:${FCA_CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${FCA_GCC_ROOT}/lib64:${FCA_OMPI_ROOT}/lib:${FCA_CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${FCA_SCRATCH}/conda/pkgs}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-${FCA_SCRATCH}/conda/envs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${FCA_SCRATCH}/conda/pip-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${FCA_SCRATCH}/cache}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${FCA_SCRATCH}/conda}"
export FCA_ENV="${FCA_ENV:-${FCA_SCRATCH}/conda/envs/floodca}"

if type module >/dev/null 2>&1; then
    module load cuda/12.0 >/dev/null 2>&1 || true
fi
