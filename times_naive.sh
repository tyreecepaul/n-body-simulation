#!/bin/bash
#
# times_naive.sh — parse nbody_naive_<jobid>.out / .err log pairs produced
# by main_slurm_naive.sh, and print a summary table of N, steps, total
# time, and average step time across all completed naive runs.
#
# Log layout produced by main_slurm_naive.sh:
#   nbody_naive_<jobid>.out — stdout: naive_sim's own timing line
#     "Naive O(N^2): n=<N> steps=<S> total_time=<T> ms avg_step=<A> ms"
#   nbody_naive_<jobid>.err — stderr: bash's `time` output for the whole
#     pipeline (real/user/sys)
#
# Usage:
#   ./times_naive.sh                 # scan current directory
#   ./times_naive.sh <log_dir>       # scan a specific directory
#   ./times_naive.sh --csv           # print CSV instead of a table
#   ./times_naive.sh <log_dir> --csv

set -euo pipefail

LOG_DIR="."
CSV=0

for arg in "$@"; do
    if [[ "$arg" == "--csv" ]]; then
        CSV=1
    else
        LOG_DIR="$arg"
    fi
done

shopt -s nullglob
out_files=("${LOG_DIR}"/nbody_naive_*.out)
shopt -u nullglob

if [ ${#out_files[@]} -eq 0 ]; then
    echo "No nbody_naive_*.out files found in ${LOG_DIR}" >&2
    exit 1
fi

rows=()

for out_file in "${out_files[@]}"; do
    job_id=$(basename "$out_file" .out | sed -E 's/^nbody_naive_//')
    err_file="${LOG_DIR}/nbody_naive_${job_id}.err"

    # Pull the naive_sim summary line out of the .out file.
    summary_line=$(grep -m1 '^Naive O(N\^2):' "$out_file" || true)
    if [ -z "$summary_line" ]; then
        echo "Warning: no timing line found in $out_file, skipping" >&2
        continue
    fi

    n=$(echo "$summary_line"     | grep -oP 'n=\K[0-9]+')
    steps=$(echo "$summary_line" | grep -oP 'steps=\K[0-9]+')
    total_ms=$(echo "$summary_line" | grep -oP 'total_time=\K[0-9.]+')
    avg_ms=$(echo "$summary_line"   | grep -oP 'avg_step=\K[0-9.]+')

    # Pull wall-clock "real" time from the .err file, if present.
    real_time="n/a"
    if [ -f "$err_file" ]; then
        real_time=$(grep -m1 '^real' "$err_file" | awk '{print $2}' || true)
        [ -z "$real_time" ] && real_time="n/a"
    fi

    rows+=("${job_id}	${n}	${steps}	${total_ms}	${avg_ms}	${real_time}")
done

if [ ${#rows[@]} -eq 0 ]; then
    echo "No parsable runs found." >&2
    exit 1
fi

# Sort numerically by N (field 2) so scaling trends are easy to read.
IFS=$'\n' sorted=($(printf '%s\n' "${rows[@]}" | sort -t$'\t' -k2,2n))
unset IFS

if [ "$CSV" -eq 1 ]; then
    echo "job_id,n_bodies,total_steps,total_time_ms,avg_step_ms,wall_real"
    for row in "${sorted[@]}"; do
        echo "$row" | awk -F'\t' '{printf "%s,%s,%s,%s,%s,%s\n",$1,$2,$3,$4,$5,$6}'
    done
else
    printf "%-12s %-10s %-10s %-14s %-12s %-10s\n" \
        "JOB_ID" "N_BODIES" "STEPS" "TOTAL_MS" "AVG_STEP_MS" "WALL_REAL"
    printf '%s\n' "----------------------------------------------------------------------"
    for row in "${sorted[@]}"; do
        echo "$row" | awk -F'\t' '{printf "%-12s %-10s %-10s %-14s %-12s %-10s\n",$1,$2,$3,$4,$5,$6}'
    done
fi
