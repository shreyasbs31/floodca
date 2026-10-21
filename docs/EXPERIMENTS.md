# Experiments

Every timing ratio in the tables uses single precision, a 4-neighbour stencil, reuse interval 1, 2000 steps, one discarded warmup and the median of five repeats, unless the table says otherwise. The warm start is the depth after 6 hours of scenario B0 (17 July 2011, scale 1, duration 24 h, peak fraction 0.5). Equivalence runs are double precision with fused multiply-add off.

Job templates live in `slurm/`. Copy `slurm/config.example.sh` to `slurm/config.sh` and submit with `bash slurm/submit.sh slurm/<job>.sbatch`. The account, the partition and any node pin are read from that file. The resource lines in the template (nodes, tasks, CPUs, GPUs, memory, exclusive, time) are the ones used.

| Paper item | Template | What was kept |
| --- | --- | --- |
| Tables VI to VIII, OpenMP on S, M and L | `slurm/p7_e4_omp.sbatch` | `results/bench.csv`, experiment E4, warmup 0 |
| Table IX, MPI strong scaling | `slurm/p10_e5e8e9.sbatch` | `results/bench.csv`, experiment E5, grid L, halo mode sendrecv2 |
| Table X, crossover | `slurm/p7_e7_cross.sbatch`, `slurm/p_c32_main.sbatch` | E4 and E7 in `results/bench.csv`. When a small grid was repeated on two hosts, the main-node rows are the ones used |
| Table XI, flux | same OpenMP and crossover jobs | flux phase column of those rows |
| Table XII, reuse | `slurm/p10_e5e8e9.sbatch` | experiment E8 in `results/bench.csv` |
| Figure 10, eight neighbours | `slurm/p10_e9_aniso.sbatch` and the E9 rows of the timing job | experiment E9 |
| Table XIII and Figure 11 | `slurm/p10_e10_cpu.sbatch`, `slurm/p10_e10_gpu100.sbatch` | `results/e10_cpu.csv`, `results/e10_gpu100.csv` |
| Table XIV | those ensemble jobs | per-scenario files under `results/e10_cpu/` and `results/e10_gpu100/` |
| Table XV | `slurm/p_cpu_sparse_dpx.sbatch` | `results/esparse_runs.csv` on the second node, dense comparison from E4 or `results/anchor_second_node.json` |
| Table XVI | `slurm/p_cpu_sparse_dpx.sbatch` | `results/edpx_runs.csv` |
| Table XVII | the energy wrapper around the L timing configuration | `results/e11_energy.csv`, exit code 0, job 243067 |
| Table V | `slurm/p10_v0_diag.sbatch`, `slurm/p10_val_4198.sbatch` | `results/validation_compare.csv` and the two diagnostics JSON files |
| Isotropy | `slurm/p10_e9_aniso.sbatch` | `results/e9_anisotropy.csv` |

Slurm logs and the depth grids (`*.bin`) are not included. They are larger than the tables need, and the logs contain account and path strings. `docs/REDACTIONS.md` lists what was removed from the files that are included.

Figure 1 is the study-area map. `analysis/fig_study.py` would draw it from `data/processed/udupi_coast/90/` and from the ensemble count grids. Neither is in this repository. `analysis/make_all.py` does not call that script and does not invent a PDF.

The plotting script writes summary CSVs to `analysis/out/summaries/` rather than back into `results/`, so a rebuild does not rewrite the saved numeric files.
