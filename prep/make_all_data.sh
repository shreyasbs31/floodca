#!/usr/bin/env bash
# End-to-end data pipeline. DEM can run on the cluster; IMD waits on Mac files.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
source scripts/env_floodca.sh
export PYTHONPATH="${ROOT}/python:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "DEM tiles: run scripts/download_dem.sh on the login node, then sbatch bench/p3_dem.sbatch"
if [[ ! -s data/raw/dem/Copernicus_DSM_COG_10_N13_00_E074_00_DEM.tif ]]; then
    echo "missing DEM tiles" >&2
    exit 1
fi
if [[ ! -f data/processed/udupi_coast/dem_report.json ]]; then
    python -m floodca_prep.dem --aoi config/aoi_udupi.yaml --raw data/raw/dem --out data/processed/udupi_coast
else
    echo "DEM already processed (dem_report.json present)"
fi

if [[ ! -d data/raw/gee ]]; then
    echo "GEE missing: run scripts/export_gee.py on the Mac and rsync to data/raw/gee/"
    exit 0
fi
python -m floodca_prep.landcover
python -m floodca_prep.validation_data

if [[ ! -d data/raw/imd ]]; then
    echo "IMD missing: run scripts/download_imd_mac.py --start 2005 --end 2005 on the Mac"
    echo "and rsync to data/raw/imd/. V0 forcing and the event pool wait on those files."
    exit 0
fi
python -m floodca_prep.rainfall
python -m floodca_prep.scenarios
python -m floodca_prep.warmstart
