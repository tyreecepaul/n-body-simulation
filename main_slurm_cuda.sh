#!/bin/bash
#SBATCH --job-name=nbody_cuda
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=cosc3500
#SBATCH --partition=cosc3500       	# course partition: includes a100-a/b, often idle
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4         	# for the parallel build (make -j); the run uses one
#SBATCH --gres=gpu:1               	# one GPU; some clusters want gpu:a100:1 (see: sinfo -p a100 -o "%G")
#SBATCH --time=00:10:00            	# short limit lets the scheduler backfill this job into gaps

set -euo pipefail

# Run from the project root (the directory containing src/, Makefile, etc.)
cd "${SLURM_SUBMIT_DIR:-.}"

# ---- toolchain ----
# nvcc must be on PATH. sbatch passes your environment through, so a
# `module load cuda/12.2` before submitting works; otherwise load it here.
# (Not plain `cuda`: the default module, 11.4, has no nvcc on this cluster.)
CUDA_MODULE="cuda/12.2"
if ! command -v nvcc >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    set +u  # module scripts often reference unset variables
    module load "${CUDA_MODULE}"
    set -u
fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc not found: run 'module load ${CUDA_MODULE}' (see 'module avail cuda') before sbatch" >&2
    exit 1
fi
nvcc --version | tail -1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# ---- build ----
# Compiled on this node into job-private paths, so jobs submitted together
# don't rebuild underneath each other.
JOB_TAG="${SLURM_JOB_ID:-local}"
BUILD_DIR="build_cuda_${JOB_TAG}"
BIN_DIR="bin_cuda_${JOB_TAG}"
trap 'rm -rf "${BUILD_DIR}" "${BIN_DIR}"' EXIT
make -s -j"${SLURM_CPUS_PER_TASK:-1}" cuda CUDA_BUILD_DIR="${BUILD_DIR}" CUDA_BIN_DIR="${BIN_DIR}"

# ---- validate ----
# GPU kernels against the CPU reference; non-zero exit if any check fails.
"./${BIN_DIR}/cuda_check"

# ---- run ----
# Same workload as main_slurm.sh (CPU sim), for a direct comparison.
N_BODIES=10000
TOTAL_STEPS=1000
OUTPUT_EVERY=2
OUTPUT_DIR="sim_output"
OUTPUT_FILE="${OUTPUT_DIR}/nbody_cuda_${JOB_TAG}.bin"
mkdir -p "${OUTPUT_DIR}"

# Recorded in the .err log next to `time`'s output; times.sh reads them.
echo "config device=gpu n_bodies=${N_BODIES} steps=${TOTAL_STEPS} output_every=${OUTPUT_EVERY}" >&2
echo "node $(hostname) gpu=\"$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)\"" >&2
time "./${BIN_DIR}/sim_cuda" "${N_BODIES}" "${TOTAL_STEPS}" "${OUTPUT_EVERY}" "${OUTPUT_FILE}"
echo "Simulation output written to: ${OUTPUT_FILE}"
