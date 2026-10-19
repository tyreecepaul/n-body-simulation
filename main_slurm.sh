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
OUTPUT_DIR="sim_output"
OUTPUT_FILE="${OUTPUT_DIR}/nbody_${SLURM_JOB_ID:-local}.bin"
 
mkdir -p "${OUTPUT_DIR}"
 
# ---- build ----
make
 
# ---- run ----
time ./sim "${N_BODIES}" "${TOTAL_STEPS}" "${OUTPUT_EVERY}" "${OUTPUT_FILE}"
 
echo "Simulation output written to: ${OUTPUT_FILE}"
