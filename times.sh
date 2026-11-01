#!/bin/bash
#
# Summarise timings from the nbody_sim_<jobid>.err logs written by
# main_slurm.sh: one row per job, then the average per configuration
# (bodies/steps/output_every) and thread count, with speedup relative to
# the 1-thread runs of the same configuration.
#
# Speed is wall time: `real` from bash's `time`, and the program's own step
# loop time ("Loop", from its "BH: ... total_time=" line) where available.
# `user` is CPU time summed over all threads, so it is shown only as CPU
# cost: it grows with thread count and is not a speed measure.
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
    node=$(grep -m1 '^node ' "$file" | awk '{print $2}' || true)
    loop_ms=$(grep -m1 -oP '^BH: .*total_time=\K[0-9.e+-]+' "$file" || true)

    printf '%s\t%s\t%s/%s/%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$job_id" "$threads" "$n" "$steps" "$every" "$real" "$user" "$sys" \
        "${node:-?}" "${loop_ms:--}"
done | awk -F'\t' '
    # "?" (unknown) sorts after every number.
    function num(v) { return (v == "?" || v == "") ? 1e18 : v + 0 }
    # Does group key x sort after group key y?
    function groupAfter(x, y,    a, b, ca, cb, i) {
        split(x, a, SUBSEP); split(y, b, SUBSEP)
        split(a[1], ca, "/"); split(b[1], cb, "/")
        for (i = 1; i <= 3; i++)
            if (num(ca[i]) != num(cb[i])) return num(ca[i]) > num(cb[i])
        return num(a[2]) > num(b[2])
    }
    # Convert bash `time` output (e.g. 1m3.537s) to seconds.
    function secs(t,    parts) {
        split(t, parts, /[ms]/)
        return parts[1] * 60 + parts[2]
    }
    {
        job[NR] = $1; thr[NR] = $2; cfg[NR] = $3
        real[NR] = secs($4); user[NR] = secs($5); sys[NR] = secs($6)
        node[NR] = $7; loop[NR] = ($8 == "-") ? "-" : sprintf("%.3f", $8 / 1000)
        key = $3 SUBSEP $2
        if (!(key in count)) { order[++groups] = key }
        count[key]++; sum[key] += real[NR]
        # Distinct nodes per group: mixing node types makes timings noisy.
        if (!((key, $7) in seenNode)) { seenNode[key, $7] = 1; nodes[key] = nodes[key] (nodes[key] == "" ? "" : ",") $7 }
    }
    END {
        if (NR == 0) { print "No completed jobs with timing information found."; exit 1 }

        printf "%-10s %-8s %-10s %-22s %-10s %-10s %-12s %-8s\n", "Job ID", "Threads", "Node", "Bodies/Steps/OutEvery", "Real (s)", "Loop (s)", "CPU user (s)", "Sys (s)"
        print "------------------------------------------------------------------------------------------------"
        for (i = 1; i <= NR; i++)
            printf "%-10s %-8s %-10s %-22s %-10.3f %-10s %-12.3f %-8.3f\n", job[i], thr[i], node[i], cfg[i], real[i], loop[i], user[i], sys[i]

        # Sort groups numerically by bodies, steps, output_every, then threads.
        for (i = 2; i <= groups; i++) {
            k = order[i]
            for (j = i - 1; j >= 1; j--) {
                if (!groupAfter(order[j], k)) break
                order[j + 1] = order[j]
            }
            order[j + 1] = k
        }

        print ""
        printf "%-22s %-8s %-6s %-14s %-9s %-11s %s\n", "Bodies/Steps/OutEvery", "Threads", "Jobs", "Avg real (s)", "Speedup", "Efficiency", "Nodes"
        print "------------------------------------------------------------------------------------------"
        for (i = 1; i <= groups; i++) {
            split(order[i], p, SUBSEP)
            avg = sum[order[i]] / count[order[i]]
            base = p[1] SUBSEP "1"
            speedup = "-"; eff = "-"
            if (base in count) {
                s = (sum[base] / count[base]) / avg
                speedup = sprintf("%.2fx", s)
                if (p[2] != "?" && p[2] > 0) eff = sprintf("%.0f%%", 100 * s / p[2])
            }
            printf "%-22s %-8s %-6d %-14.3f %-9s %-11s %s\n", p[1], p[2], count[order[i]], avg, speedup, eff, nodes[order[i]]
        }
        print ""
        print "Completed jobs: " NR
        print "Speedup = 1-thread avg real / this avg real, same configuration (\"-\" if no 1-thread runs)."
        print "Efficiency = speedup / threads. CPU user time is summed over threads: not a speed measure."
        print "If a group lists several nodes (or differs from its 1-thread group), node differences add noise."
    }'
