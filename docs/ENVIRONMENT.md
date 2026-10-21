# Environment

The timings in this repository were measured on PARAM Utkarsh at C-DAC, under Slurm.

## Hardware

| Item | Value |
| --- | --- |
| Main node CPU | 2 x Xeon Gold 6248, 2.50 GHz, 40 cores |
| Main node memory | 187 GiB |
| Main node accelerators | 2 x Tesla V100-SXM2, 16 GB |
| Second node | same CPU as the main node |
| CPU nodes | 2 x Xeon Platinum 8268, 48 cores |

The main node is the host for the OpenMP, MPI, CUDA and crossover timings. The second node is the host for the sparse-state runs, the double-precision OpenMP runs, the matched-scenario CPU times and the same-node anchors. The CPU nodes ran the double-precision equivalence checks, the flooded-extent comparison and the flat-plane isotropy run.

## Software

| Item | Value |
| --- | --- |
| C compiler | GCC 11.2.0 |
| Release flags | `-O3 -march=native` |
| CUDA | 12.0, compute capability 7.0 |
| MPI | OpenMPI 4.1.1 |
| MPI settings used | `OMPI_MCA_btl=^openib`, `OMPI_MCA_pml=ucx` |
| OpenMP binding | `OMP_PROC_BIND=close`, `OMP_PLACES=cores` |
| CUDA module | `module load cuda/12.0` |

The GCC install on that cluster was a Spack prefix. `slurm/config.example.sh` takes the prefix as `FCA_GCC_ROOT` and does not record the hash. OpenMPI and CUDA prefixes used on the cluster are the defaults in that file.

## This Mac

`/usr/bin/gcc` is Apple Clang 21.0.0. It does not define `_OPENMP`, and `gcc -fopenmp` is rejected. A separate GCC with OpenMP is not installed, so the serial and OpenMP binaries were not built here. CUDA was not attempted. CMake is available. `pdflatex` is available and is what `analysis/make_all.py` uses for figures 2 to 4.
