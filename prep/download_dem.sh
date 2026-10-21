#!/usr/bin/env bash
# Download Copernicus GLO-30 COG tiles for the Udupi AOI (SPEC Sec 4.2).
# Run on the login node only (internet). Processing is a separate Slurm job.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${FCA_DEM_DIR:-${ROOT}/data/raw/dem}"
BASE="https://copernicus-dem-30m.s3.amazonaws.com"
mkdir -p "${DEST}"

# Named tiles from SPEC 3.1, plus N12/N14 neighbours for the 0.02 deg reprojection buffer.
TILES=(
  N12_00_E074_00
  N12_00_E075_00
  N13_00_E074_00
  N13_00_E075_00
  N14_00_E074_00
  N14_00_E075_00
)

download_one() {
    local tag="$1"
    local key="Copernicus_DSM_COG_10_${tag}_DEM/Copernicus_DSM_COG_10_${tag}_DEM.tif"
    local out="${DEST}/Copernicus_DSM_COG_10_${tag}_DEM.tif"
    if [[ -s "${out}" ]]; then
        echo "exists ${out} ($(wc -c < "${out}") bytes)"
        return 0
    fi
    echo "GET ${BASE}/${key}"
    if ! curl -fL --retry 3 --retry-delay 2 -o "${out}.part" "${BASE}/${key}"; then
        rm -f "${out}.part"
        echo "WARN: missing or failed ${tag}" >&2
        return 1
    fi
    mv "${out}.part" "${out}"
    echo "saved ${out} ($(wc -c < "${out}") bytes)"
}

ok=0
fail=0
for t in "${TILES[@]}"; do
    if download_one "${t}"; then
        ok=$((ok + 1))
    else
        fail=$((fail + 1))
    fi
done
echo "download_dem: ok=${ok} fail=${fail} dest=${DEST}"
# SPEC 3.1 requires N13 E074 and N13 E075.
test -s "${DEST}/Copernicus_DSM_COG_10_N13_00_E074_00_DEM.tif"
test -s "${DEST}/Copernicus_DSM_COG_10_N13_00_E075_00_DEM.tif"
echo "required N13 tiles present"
