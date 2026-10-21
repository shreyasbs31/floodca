#!/bin/bash
# CUDA 12.0 on this cluster needs an NVIDIA driver at major version 525 or newer.
# __FCA_NODE_SECOND__ was observed at 450.51.05 (jobs 243074 and 243333) and refused cudaMalloc.
# Source this after the environment modules are loaded, then call gpu_driver_preflight.
gpu_driver_preflight() {
  local drv major host
  host=$(hostname)
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')
  if [[ -z "${drv}" ]]; then
    echo "gpu_driver_preflight: node ${host} driver unreadable" >&2
    exit 3
  fi
  major=${drv%%.*}
  if [[ "${major}" -lt 525 ]]; then
    echo "gpu_driver_preflight: node ${host} driver ${drv} is below 525" >&2
    exit 3
  fi
  echo "gpu_driver_preflight: node ${host} driver ${drv}"
}
