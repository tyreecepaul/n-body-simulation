#!/bin/bash
#
# Submit <number_of_runs> copies of main_slurm.sh.
# By default the jobs only time the computation (no .bin file is written);
# "visual" also writes the trajectory .bin for run.py. Any further
# arguments are passed through to sbatch.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <number_of_runs> [visual] [sbatch options...]"
    echo "  e.g. $0 3                 # timing runs, no output file"
    echo "       $0 1 visual          # also write the .bin for run.py"
    echo "       $0 3 --cpus-per-task=8"
    exit 1
fi

N="$1"
shift

if ! [[ "$N" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: number of runs must be a positive integer."
    exit 1
fi

VISUAL=""
if [ "${1:-}" = visual ]; then
    VISUAL=visual
    shift
fi
# remaining arguments are passed through to sbatch

for ((i=1; i<=N; i++)); do
    echo "Submitting run $i/$N${VISUAL:+ (visual)}..."
    sbatch "$@" main_slurm.sh ${VISUAL}
done

echo "Submitted $N jobs."
