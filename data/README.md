# Data

The rasters are not in this repository. Nothing here is large enough to need Git LFS, and `analysis/make_all.py` does not read them. The scripts that fetch and grid the inputs are under `prep/`. The access date is not recorded in this repository.

| Product | Role | How it is fetched |
| --- | --- | --- |
| Copernicus WorldDEM-30 | Elevation. Copyright DLR e.V. 2010-2014 and Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the EU and ESA | `prep/floodca_prep/dem.py` |
| IMD daily rainfall, 0.25 degree, via IMDLIB | Rain forcing. A day is the total ending 08:30 IST (03:00 UTC) | `prep/download_imd_mac.py`, `prep/floodca_prep/rainfall.py` |
| ESA WorldCover v200 (2021) | Land cover, converted with `config/landcover_params.csv` | `prep/export_gee.py`, `prep/floodca_prep/landcover.py` |
| Global Flood Database (Tellman et al., 250 m, 2000-2018), including the JRC permanent-water band in that product | Observed flood extents for events 2734 and 4198 | `prep/export_gee.py` |

Each product is used under the licence published with that product. This repository does not restate those licences. WorldCover is reached through the Earth Engine collection `ESA/WorldCover/v200`. Set `GEE_PROJECT` to your own Earth Engine project. No project id is stored here.

The 90 m validation grid is 1230 by 1206 cells, origin (445770, 1547820) m, EPSG 32643, with row 0 at the north. Those numbers are the grid header used by the saved diagnostics. The grids themselves (S 512 by 512, M 1024 by 1024, L 2048 by 2048, and the smaller crossover grids) are described in `docs/EXPERIMENTS.md`.
