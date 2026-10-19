#!/bin/bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <number_of_runs>"
    exit 1
fi

N="$1"

if ! [[ "$N" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: number of runs must be a positive integer."
    exit 1
fi

for ((i=1; i<=N; i++)); do
    echo "Submitting run $i/$N..."
    sbatch main_slurm_cuda.sh
done

echo "Submitted $N jobs."
