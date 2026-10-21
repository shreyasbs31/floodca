#!/usr/bin/env bash
# Moved to slurm/env.sh. Job templates source that file.
echo "source slurm/env.sh from the repository root" >&2
return 1 2>/dev/null || exit 1
