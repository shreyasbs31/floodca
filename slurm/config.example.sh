# Copy this file to slurm/config.sh and fill in the site values.
# slurm/submit.sh substitutes them into the job templates.
# Do not commit slurm/config.sh if it contains an account name.

# Slurm account and the partitions used for the paper's jobs.
FCA_ACCOUNT=your-account
FCA_PARTITION_GPU=gpu
FCA_PARTITION_CPU=cpu

# Optional node pins. Leave a value empty to drop that line.
# The paper pinned a main timing node and a second CPU node, and some
# jobs excluded the second node. Put this site's node names here.
FCA_NODE_MAIN=
FCA_NODE_SECOND=
FCA_EXCLUDE_SECOND=

# Scratch tree for data, builds and conda caches.
FCA_SCRATCH=/path/to/floodca-scratch

# GCC 11.2.0, OpenMPI 4.1.1 and CUDA 12.0. The paper used these versions.
# The GCC prefix on the cluster was a Spack install; set the local prefix.
FCA_GCC_ROOT=/path/to/gcc-11.2.0
FCA_OMPI_ROOT=/opt/ohpc/pub/compiler/openmpi/4.1.1
FCA_CUDA_ROOT=/opt/ohpc/pub/cuda/cuda-12.0
FCA_CMAKE_ROOT=/path/to/cmake

# Optional login used by the rsync examples in prep/.
FCA_LOGIN=user@login-host
FCA_DATA=/path/to/floodca-scratch/data
