#!/bin/bash
#
# Submit <number_of_runs> benchmark jobs from the project root.
#   cpu  (default): cluster/main_slurm.sh, the OpenMP sim
#   cuda          : cluster/main_slurm_cuda.sh, the GPU sim
# By default the jobs only time the computation (no .bin file is written);
# "visual" also writes the trajectory .bin for run.py. Any further
# arguments are passed through to sbatch.

set -euo pipefail

usage() {
    echo "Usage: $0 <number_of_runs> [cpu|cuda] [visual] [sbatch options...]"
    echo "  e.g. $0 5                          # 5 OpenMP timing runs"
    echo "       $0 5 cpu --cpus-per-task=1    # serial-equivalent runs"
    echo "       $0 3 cuda                     # 3 GPU timing runs"
    echo "       $0 1 cuda visual              # also write the .bin for run.py"
    exit 1
}

[ "$#" -ge 1 ] || usage
N="$1"
shift
if ! [[ "$N" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: number of runs must be a positive integer."
    exit 1
fi

SCRIPT=cluster/main_slurm.sh
case "${1:-}" in
    cpu) shift ;;
    cuda) SCRIPT=cluster/main_slurm_cuda.sh; shift ;;
esac

VISUAL=""
if [ "${1:-}" = visual ]; then
    VISUAL=visual
    shift
fi
# remaining arguments are passed through to sbatch

# Jobs run from the submit directory, so submit from the project root.
cd "$(dirname "$0")/.."
for ((i=1; i<=N; i++)); do
    echo "Submitting run $i/$N (${SCRIPT##*/}${VISUAL:+, visual})..."
    sbatch "$@" "$SCRIPT" ${VISUAL}
done

echo "Submitted $N jobs."
