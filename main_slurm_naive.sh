#!/bin/bash
#SBATCH --job-name=nbody_naive
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
OUTPUT_DIR="sim_output"
LOG_FILE="${OUTPUT_DIR}/nbody_naive_${SLURM_JOB_ID:-local}.log"

mkdir -p "${OUTPUT_DIR}"

# ---- build ----
# Naive implementation lives in its own subdirectory so it can be built
# and benchmarked independently of the Barnes-Hut version.
cd src/naive
make

# ---- run ----
# The naive runner doesn't write a trajectory file (see MainNaive.cpp) —
# it just times N steps and prints total/average step time to stderr, so
# that's captured to a log file instead of an output binary.
time ./naive_sim "${N_BODIES}" "${TOTAL_STEPS}" 2>&1 | tee "../../${LOG_FILE}"

echo "Naive simulation log written to: ${LOG_FILE}"
