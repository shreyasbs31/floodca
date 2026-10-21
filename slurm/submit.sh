#!/usr/bin/env bash
# Fill a job template from slurm/config.sh and hand it to sbatch.
# Usage, from the repository root:
#   bash slurm/submit.sh slurm/p7_e4_omp.sbatch
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="${ROOT}/slurm/config.sh"
if [[ ! -f "${CFG}" ]]; then
    echo "Copy slurm/config.example.sh to slurm/config.sh and edit it." >&2
    exit 1
fi
# shellcheck disable=SC1090
source "${CFG}"
if [[ $# -lt 1 ]]; then
    echo "usage: bash slurm/submit.sh slurm/<job>.sbatch [extra sbatch args]" >&2
    exit 1
fi
script="$1"
shift
tmp="$(mktemp)"
sed \
    -e "s/__FCA_ACCOUNT__/${FCA_ACCOUNT}/g" \
    -e "s/__FCA_PARTITION_GPU__/${FCA_PARTITION_GPU}/g" \
    -e "s/__FCA_PARTITION_CPU__/${FCA_PARTITION_CPU}/g" \
    -e "s/__FCA_NODE_MAIN__/${FCA_NODE_MAIN:-}/g" \
    -e "s/__FCA_NODE_SECOND__/${FCA_NODE_SECOND:-}/g" \
    -e "s/__FCA_EXCLUDE_SECOND__/${FCA_EXCLUDE_SECOND:-}/g" \
    "${script}" \
    | awk '
        /^#SBATCH --nodelist=$/ { next }
        /^#SBATCH --exclude=$/ { next }
        { print }
    ' > "${tmp}"
sbatch "${tmp}" "$@"
rm -f "${tmp}"
