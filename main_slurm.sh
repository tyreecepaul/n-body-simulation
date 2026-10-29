#!/bin/bash
#SBATCH --job-name=nbody_sim
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=cosc3500
#SBATCH --partition=a100        
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1         	# single-threaded; 
##SBATCH --time=0:00:00           	# D-HH:MM
##SBATCH --mem-per-cpu=1G          	# 100k bodies x ~40 bytes each is small;         

set -euo pipefail

# Run from the project root (the directory containing src/, Makefile, etc.)
cd "${SLURM_SUBMIT_DIR:-.}"
 
# ---- simulation parameters ----
N_BODIES=10000
TOTAL_STEPS=1000
OUTPUT_EVERY=2
# Default: computation only, no .bin file (OUTPUT_EVERY is then unused and
# logged as "none"). Pass "visual" (./run.sh N visual, or sbatch <this
# script> visual) to also write the trajectory .bin every OUTPUT_EVERY
# steps, for run.py.
VISUAL=0
if [ "${1:-}" = visual ]; then
    VISUAL=1
elif [ -n "${1:-}" ]; then
    echo "Unknown argument '$1' (expected nothing or 'visual')" >&2
    exit 1
fi
OUTPUT_DIR="sim_output"
OUTPUT_FILE="${OUTPUT_DIR}/nbody_${SLURM_JOB_ID:-local}.bin"
 
mkdir -p "${OUTPUT_DIR}"
 
# ---- build ----
make
 
# ---- run ----
# Recorded in the .err log next to `time`'s output; times.sh reads them.
LOGGED_EVERY=$([ "${VISUAL}" = 1 ] && echo "${OUTPUT_EVERY}" || echo none)
echo "config threads=1 n_bodies=${N_BODIES} steps=${TOTAL_STEPS} output_every=${LOGGED_EVERY}" >&2
echo "node $(hostname) cpu=\"$(lscpu | sed -n 's/^Model name:[[:space:]]*//p')\"" >&2
if [ "${VISUAL}" = 1 ]; then
    time ./sim "${N_BODIES}" "${TOTAL_STEPS}" "${OUTPUT_EVERY}" "${OUTPUT_FILE}"
else
    time ./sim "${N_BODIES}" "${TOTAL_STEPS}" "${OUTPUT_EVERY}"
fi
if [ "${VISUAL}" = 1 ]; then
    echo "Simulation output written to: ${OUTPUT_FILE}"
fi
