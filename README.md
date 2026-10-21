# FloodCA

FloodCA times one weighted cellular automata flood kernel, written once in `src/kernels/physics.h`, under OpenMP, intra-node MPI, CUDA and two master-worker ensembles. The grids use open data for the Udupi coast of India. This repository keeps the source, the job templates and the saved results that rebuild the tables and figures 2 to 11 of the paper. Figure 1 needs rasters that are not included.

## How to cite

The title and authors are in `CITATION.cff`.

Shreyas B S, Varun Singh, Vishrut Anand Ojha, and Sandhya Dubey. OpenMP, MPI and CUDA on One Cellular Automata Flood Kernel: Scaling, Time-Step Cost and Ensemble Throughput. 2026.

## Repository map

| Path | Contents |
| --- | --- |
| `src/` | The shared kernel and the six backends |
| `prep/` | Download and preprocess scripts |
| `config/` | Default parameters and the land-cover lookup |
| `slurm/` | Job templates. Account, partition and paths come from `slurm/config.sh` |
| `results/` | Saved outputs used by the paper |
| `analysis/` | Scripts that rebuild tables and figures from `results/` |
| `figures/` | Figure PDFs written by `analysis/make_all.py` |
| `docs/` | Hardware, experiment notes and the redaction log |
| `data/README.md` | Where the input rasters come from. The rasters themselves are not in the repo |

## Reproduce

From the repository root, with no network and no cluster:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python analysis/make_all.py
python analysis/check_numbers.py
```

`make_all.py` writes every table to `analysis/out/tables/` as a CSV and a LaTeX snippet, and writes figures 2 to 11 as vector PDFs under `figures/`. Figure 1 is listed and is not drawn. `check_numbers.py` reprints each paper value, rounds the regenerated number the way that value is printed, and prints PASS or FAIL. It does not change `results/` or `analysis/expected_values.csv`.

## Build the code

The paper's binaries were built with GCC 11.2.0, flags `-O3 -march=native`, CUDA 12.0 for compute capability 7.0, and OpenMPI 4.1.1. `CMakeLists.txt` is unchanged. The C tests are not in this repository, so turn them off.

Serial:

```bash
cmake -S . -B build/serial -DCMAKE_BUILD_TYPE=Release \
  -DFCA_WITH_OMP=OFF -DFCA_WITH_MPI=OFF -DFCA_WITH_CUDA=OFF -DFCA_BUILD_TESTS=OFF
cmake --build build/serial
```

OpenMP:

```bash
cmake -S . -B build/omp -DCMAKE_BUILD_TYPE=Release \
  -DFCA_WITH_OMP=ON -DFCA_WITH_MPI=OFF -DFCA_WITH_CUDA=OFF -DFCA_BUILD_TESTS=OFF
cmake --build build/omp
```

MPI uses the same flags with `-DFCA_WITH_MPI=ON` and an MPI compiler on `PATH`. CUDA uses `-DFCA_WITH_CUDA=ON`. The CUDA architecture in `CMakeLists.txt` is 70.

On this Mac the `gcc` command is Apple Clang and it has no OpenMP. The serial and OpenMP binaries were not built here, and CUDA was not attempted. See `docs/ENVIRONMENT.md`.

## Rerun experiments on a Slurm cluster

```bash
cp slurm/config.example.sh slurm/config.sh
# edit the account, partitions, node names and toolchain prefixes
bash slurm/submit.sh slurm/p7_e4_omp.sbatch
```

`slurm/submit.sh` substitutes `__FCA_ACCOUNT__`, the partition names and the optional node pins, then calls `sbatch`. Resource requests, `module load cuda/12.0` and the compiler flags inside each template are the ones used for the paper. An empty node pin is dropped, so a site that does not have those nodes can still submit. `docs/EXPERIMENTS.md` says which template produced which table.

## Data sources and licences

The rasters are not in this repository. `prep/` is what fetches them, and `data/README.md` names each product. The access date is not recorded here. Use the licence published with each product.

## Paper item, results file, script

| Paper item | Results file | Script |
| --- | --- | --- |
| Table I, datasets | none (bibliographic) | `prep/` |
| Table II, parameters | `config/default.yaml` | `analysis/rebuild.py` |
| Table III, land cover | `config/landcover_params.csv` | `analysis/rebuild.py` |
| Table IV, hardware | none (probe logs are not included) | `docs/ENVIRONMENT.md` |
| Table V, scores | `results/validation_compare.csv`, `results/v0_diagnostics.json`, `results/val_4198_diagnostics.json` | `analysis/rebuild.py` |
| Tables VI to VIII, OpenMP | `results/bench.csv` | `analysis/rebuild.py`, `analysis/plots.py` |
| Table IX, MPI | `results/bench.csv` | `analysis/rebuild.py`, `analysis/plots.py` |
| Table X, crossover | `results/bench.csv` | `analysis/rebuild.py`, `analysis/plots.py` |
| Table XI, flux | `results/bench.csv` | `analysis/rebuild.py` |
| Table XII, reuse | `results/bench.csv` | `analysis/rebuild.py`, `analysis/plots.py` |
| Table XIII, CPU ensemble | `results/e10_cpu.csv`, `results/e10_gpu100.csv` | `analysis/rebuild.py`, `analysis/plots.py` |
| Table XIV, matched scenarios | `results/e10_cpu/b5_1x40.csv`, `results/e10_gpu100/B4.csv` | `analysis/rebuild.py` |
| Table XV, sparse start | `results/esparse_runs.csv`, `results/anchor_second_node.json` | `analysis/rebuild.py` |
| Table XVI, double precision | `results/edpx_runs.csv`, `results/bench.csv`, `results/anchor_second_node.json` | `analysis/rebuild.py` |
| Table XVII, energy | `results/e11_energy.csv` | `analysis/rebuild.py` |
| Figure 1, map | blocked: DEM, mask, flood raster and count grids are absent | `analysis/fig_study.py` (not run) |
| Figure 2, pipeline | TikZ in `analysis/diagrams/fig2_body.tex` | `analysis/make_all.py` |
| Figure 3, halo | `analysis/diagrams/fig3_body.tex` | `analysis/make_all.py` |
| Figure 4, master-worker | `analysis/diagrams/fig4_body.tex` | `analysis/make_all.py` |
| Figure 5, OpenMP | `results/bench.csv` | `analysis/plots.py` |
| Figure 6, MPI | `results/bench.csv` | `analysis/plots.py` |
| Figure 7, CUDA | `results/bench.csv` | `analysis/plots.py` |
| Figure 8, crossover | `results/bench.csv` | `analysis/plots.py` |
| Figure 9, reuse | `results/bench.csv` | `analysis/plots.py` |
| Figure 10, neighbourhood | `results/bench.csv` | `analysis/plots.py` |
| Figure 11, ensemble | `results/e10_cpu.csv`, `results/e10_gpu100.csv` | `analysis/plots.py` |

`results/PROVENANCE.csv` adds the row filter and the Slurm job ids.

## Hardware

The main node has two Xeon Gold 6248 processors at 2.50 GHz (40 cores), 187 GiB of memory and two Tesla V100-SXM2 16 GB accelerators. The second node has the same CPU. The CPU nodes have two Xeon Platinum 8268 processors (48 cores). The software is GCC 11.2.0, CUDA 12.0 and OpenMPI 4.1.1, on PARAM Utkarsh at C-DAC, under Slurm.
