# Redactions

Numeric fields were not changed. The edits below replace a username, an account, a hostname, an email address, a cluster domain or an absolute path. The original string is not written here.

## Results

| File | Kind of string |
| --- | --- |
| `results/bench.csv` | hostname |
| `results/e8_runs.csv` | hostname |
| `results/e11_energy.csv` | absolute path, hostname |
| `results/e11_table.csv` | absolute path, hostname |
| `results/edpx_runs.csv` | hostname |
| `results/esparse_runs.csv` | hostname |
| `results/evec_runs.csv` | hostname |
| `results/evec_samenode.json` | hostname |
| `results/precision_f32_run.csv` | hostname |
| `results/precision_f64_run.csv` | hostname |
| `results/edtdrift_run.csv` | hostname |
| `results/anchor_second_node.csv` | hostname |
| `results/anchor_second_node.json` | hostname, including the source filename and two field names |

`results/anchor_second_node.csv` and `results/anchor_second_node.json` were copied from filenames that contained a hostname. Two field names that contained hostnames were renamed. The two median seconds and the relative difference were left as stored.

Hostnames in result rows were replaced with `main-node`, `second-node` or `cpu-node`. That label is not used as a group key by the table scripts, which group on experiment, backend, grid, precision, threads and ranks.

## Scripts

| Files | Kind of string |
| --- | --- |
| `slurm/*.sbatch` | account on every job; hostname where a node was pinned or excluded; absolute path in the paper-build and a few data jobs |
| `slurm/p7_lib.sh`, `slurm/gpu_driver_preflight.sh`, `slurm/p_cpu_sparse_dpx.sbatch` | hostname in comments and in a generated JSON field |
| `prep/env_floodca.sh` | absolute path (the file is now a pointer to `slurm/env.sh`) |
| `prep/export_gee.py` | absolute path, email |
| `prep/download_imd_mac.py` | absolute path, email |
| `prep/floodca_prep/rainfall.py` | absolute path, email |
| `analysis/e9_aniso.py` | absolute path, replaced with `results/e9_aniso` |

Account, partition and node pins in the job templates are the tokens `__FCA_ACCOUNT__`, `__FCA_PARTITION_GPU__`, `__FCA_PARTITION_CPU__`, `__FCA_NODE_MAIN__`, `__FCA_NODE_SECOND__` and `__FCA_EXCLUDE_SECOND__`. `slurm/submit.sh` fills them from `slurm/config.sh`.

`src/` and `CMakeLists.txt` had none of these strings and were copied unchanged.
