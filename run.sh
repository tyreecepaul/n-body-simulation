#!/bin/bash

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <number_of_runs> [sbatch options...]"
    echo "  e.g. $0 3 --cpus-per-task=8"
    exit 1
fi

N="$1"
shift  # remaining arguments are passed through to sbatch

if ! [[ "$N" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: number of runs must be a positive integer."
    exit 1
fi

for ((i=1; i<=N; i++)); do
    echo "Submitting run $i/$N..."
    sbatch "$@" main_slurm.sh
done

echo "Submitted $N jobs."
