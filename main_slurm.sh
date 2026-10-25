#!/bin/bash
#SBATCH --job-name=nbody_sim
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=cosc3500
#SBATCH --partition=a100        
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        	# OpenMP threads; override with sbatch --cpus-per-task=N
##SBATCH --time=0:00:00           	# D-HH:MM
##SBATCH --exclusive               	# whole node to this job: no other jobs' load in the timing (may queue longer)
##SBATCH --mem-per-cpu=1G          	# 100k bodies x ~40 bytes each is small;         

set -euo pipefail

# Run from the project root (the directory containing src/, Makefile, etc.)
cd "${SLURM_SUBMIT_DIR:-.}"
 
# ---- simulation parameters ----
N_BODIES=10000
TOTAL_STEPS=1000
OUTPUT_EVERY=2
OUTPUT_DIR="sim_output"
OUTPUT_FILE="${OUTPUT_DIR}/nbody_${SLURM_JOB_ID:-local}.bin"
 
mkdir -p "${OUTPUT_DIR}"
 
# ---- OpenMP ----
# One thread per allocated CPU, pinned so threads don't migrate between
# cores mid-run. Output is byte-identical at any thread count.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
# Print the effective OpenMP settings (threads, binding, places) to the
# .err log at start-up, to confirm what each run actually used.
export OMP_DISPLAY_ENV=TRUE

# ---- build ----
# Always compile here, on the compute node, into job-private paths:
#  - a binary built elsewhere (e.g. a newer glibc/libstdc++) won't run here,
#    and -march=native must match this node's CPU;
#  - jobs submitted together (run.sh) share this directory, so a shared
#    build/ and ./sim would be rebuilt underneath each other.
# (Not `make clean`: it also deletes the nbody_sim_* job logs.)
JOB_TAG="${SLURM_JOB_ID:-local}"
BUILD_DIR="build_${JOB_TAG}"
BIN="sim_${JOB_TAG}"
trap 'rm -rf "${BUILD_DIR}" "${BIN}"' EXIT
make -s BUILD_DIR="${BUILD_DIR}" TARGET="${BIN}"

# ---- run ----
# Recorded in the .err log next to `time`'s output; times.sh groups by it.
# Compare runs by `real` (or the BH: total_time line), never `user`: user
# is CPU time summed over all threads, so it grows with thread count.
echo "config threads=${OMP_NUM_THREADS} n_bodies=${N_BODIES} steps=${TOTAL_STEPS} output_every=${OUTPUT_EVERY}" >&2
echo "node $(hostname) cpu=\"$(lscpu | sed -n 's/^Model name:[[:space:]]*//p')\" allocated_cpus=$(nproc)" >&2
time "./${BIN}" "${N_BODIES}" "${TOTAL_STEPS}" "${OUTPUT_EVERY}" "${OUTPUT_FILE}"
 
echo "Simulation output written to: ${OUTPUT_FILE}"
