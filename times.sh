#!/bin/bash
#
# Summarise timings from the nbody_sim_<jobid>.err logs written by
# main_slurm.sh: one row per job, then the average per configuration
# (bodies/steps/output_every) and thread count, with speedup relative to
# the 1-thread runs of the same configuration.
#
# Usage: ./times.sh [log_dir]    (default: current directory)
#
# Logs from before main_slurm.sh recorded its config show "?" and are
# grouped together.

set -euo pipefail

LOG_DIR="${1:-.}"

shopt -s nullglob
files=("${LOG_DIR}"/nbody_sim_*.err)
shopt -u nullglob

if [ ${#files[@]} -eq 0 ]; then
    echo "No nbody_sim_*.err files found in ${LOG_DIR}" >&2
    exit 1
fi

for file in "${files[@]}"; do
    job_id=$(basename "$file" .err | sed -E 's/^nbody_sim_//')
    real=$(grep -m1 '^real' "$file" | awk '{print $2}' || true)
    user=$(grep -m1 '^user' "$file" | awk '{print $2}' || true)
    sys=$(grep -m1 '^sys' "$file" | awk '{print $2}' || true)

    # Skip jobs that didn't finish (no `time` output).
    if [ -z "$real" ] || [ -z "$user" ] || [ -z "$sys" ]; then
        continue
    fi

    config=$(grep -m1 '^config ' "$file" || true)
    threads=$(echo "$config" | grep -oP 'threads=\K[0-9]+' || echo "?")
    n=$(echo "$config" | grep -oP 'n_bodies=\K[0-9]+' || echo "?")
    steps=$(echo "$config" | grep -oP 'steps=\K[0-9]+' || echo "?")
    every=$(echo "$config" | grep -oP 'output_every=\K[0-9]+' || echo "?")

    printf '%s\t%s\t%s/%s/%s\t%s\t%s\t%s\n' \
        "$job_id" "$threads" "$n" "$steps" "$every" "$real" "$user" "$sys"
done | awk -F'\t' '
    # Convert bash `time` output (e.g. 1m3.537s) to seconds.
    function secs(t,    parts) {
        split(t, parts, /[ms]/)
        return parts[1] * 60 + parts[2]
    }
    {
        job[NR] = $1; thr[NR] = $2; cfg[NR] = $3
        real[NR] = secs($4); user[NR] = secs($5); sys[NR] = secs($6)
        key = $3 SUBSEP $2
        if (!(key in count)) { order[++groups] = key }
        count[key]++; sum[key] += real[NR]
    }
    END {
        if (NR == 0) { print "No completed jobs with timing information found."; exit 1 }

        printf "%-10s %-8s %-22s %-10s %-10s %-10s\n", "Job ID", "Threads", "Bodies/Steps/OutEvery", "Real (s)", "User (s)", "Sys (s)"
        print "--------------------------------------------------------------------------"
        for (i = 1; i <= NR; i++)
            printf "%-10s %-8s %-22s %-10.3f %-10.3f %-10.3f\n", job[i], thr[i], cfg[i], real[i], user[i], sys[i]

        # Sort groups by config, then numerically by thread count.
        for (i = 2; i <= groups; i++) {
            k = order[i]
            for (j = i - 1; j >= 1; j--) {
                split(order[j], a, SUBSEP); split(k, b, SUBSEP)
                if (a[1] < b[1] || (a[1] == b[1] && (a[2] == "?" ? 1e9 : a[2]) <= (b[2] == "?" ? 1e9 : b[2]))) break
                order[j + 1] = order[j]
            }
            order[j + 1] = k
        }

        print ""
        printf "%-22s %-8s %-6s %-14s %-8s\n", "Bodies/Steps/OutEvery", "Threads", "Jobs", "Avg real (s)", "Speedup"
        print "--------------------------------------------------------------"
        for (i = 1; i <= groups; i++) {
            split(order[i], p, SUBSEP)
            avg = sum[order[i]] / count[order[i]]
            base = p[1] SUBSEP "1"
            speedup = (base in count) ? sprintf("%.2fx", (sum[base] / count[base]) / avg) : "-"
            printf "%-22s %-8s %-6d %-14.3f %-8s\n", p[1], p[2], count[order[i]], avg, speedup
        }
        print ""
        print "Completed jobs: " NR
        print "Speedup is vs the 1-thread average of the same configuration (\"-\" if none)."
    }'
