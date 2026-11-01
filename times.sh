#!/bin/bash
#
# Summarise timings from Slurm job logs: CPU jobs (main_slurm.sh,
# nbody_sim_<jobid>.err) and GPU jobs (main_slurm_cuda.sh,
# nbody_cuda_<jobid>.err). One row per job, then one row per configuration
# (bodies/steps/output_every) and kind (CPU-<threads> or GPU), with speedup
# against the serial (CPU-1) runs of the same configuration.
#
# Usage:
#   ./times.sh                 # logs in the current directory
#   ./times.sh DIR [DIR ...]   # combine directories, e.g. the CUDA and CPU
#                              # checkouts: ./times.sh . ../n-body-simulation
#   ./times.sh nbody_cuda      # only logs with this prefix (current directory)
#
# Speed is wall time: `real` from bash's `time`, and the program's own step
# loop time ("Loop", from its "BH:" / "CUDA BH:" ... total_time= line),
# which excludes start-up (for GPU jobs: CUDA context creation and upload).
# `user` is host CPU time (summed over threads; for GPU jobs mostly waiting
# on the GPU), so it is shown only as CPU cost, never as a speed measure.
#
# Older logs without the "config" / "node" lines still work: the
# configuration then comes from the program's "Done. n=... steps=...
# output_every=..." line, and a GPU name from nvidia-smi's line in the
# matching .out file. Anything unknown shows as "?".

set -euo pipefail

prefixes=(nbody_sim nbody_cuda)
dirs=()
if [ "$#" -eq 1 ] && [ ! -d "$1" ]; then
    prefixes=("$1")
    dirs=(.)
else
    dirs=("$@")
fi
[ "${#dirs[@]}" -eq 0 ] && dirs=(.)

files=()
shopt -s nullglob
for d in "${dirs[@]}"; do
    for p in "${prefixes[@]}"; do
        files+=("$d"/"$p"_*.err)
    done
done
shopt -u nullglob

if [ "${#files[@]}" -eq 0 ]; then
    echo "No ${prefixes[*]} *.err logs found in: ${dirs[*]}" >&2
    exit 1
fi

# Value of key=... in a "key=value key=value" line ("" if absent).
field() {
    echo "$1" | grep -oP "(^|\s)$2=\K[^ ]+" || true
}

for file in "${files[@]}"; do
    real=$(grep -m1 '^real' "$file" | awk '{print $2}' || true)
    user=$(grep -m1 '^user' "$file" | awk '{print $2}' || true)
    sys=$(grep -m1 '^sys' "$file" | awk '{print $2}' || true)

    # Skip jobs that didn't finish (no `time` output).
    if [ -z "$real" ] || [ -z "$user" ] || [ -z "$sys" ]; then
        continue
    fi

    job_id=$(basename "$file" .err)
    job_id=${job_id##*_}
    config=$(grep -m1 '^config ' "$file" || true)
    done_line=$(grep -m1 '^Done\. ' "$file" || true)
    timing=$(grep -m1 -E '^(CUDA )?BH: ' "$file" || true)
    node_line=$(grep -m1 '^node ' "$file" || true)

    n=$(field "$config" n_bodies)
    [ -z "$n" ] && n=$(field "$done_line" n)
    steps=$(field "$config" steps)
    [ -z "$steps" ] && steps=$(field "$done_line" steps)
    every=$(field "$config" output_every)
    [ -z "$every" ] && every=$(field "$done_line" output_every)

    if [[ "$timing" == CUDA* || "$config" == *device=gpu* ]]; then
        kind=GPU
    else
        threads=$(field "$config" threads)
        [ -z "$threads" ] && threads=$(field "$timing" threads)
        kind="CPU-${threads:-?}"
    fi

    node=$(echo "$node_line" | awk '{print $2}')
    device=$(echo "$node_line" | grep -oP '(gpu|cpu)="\K[^"]*' || true)
    if [ -z "$device" ] && [ "$kind" = GPU ] && [ -f "${file%.err}.out" ]; then
        device=$(grep -m1 -oP '^NVIDIA [^,]+' "${file%.err}.out" || true)
    fi
    loop_ms=$(field "$timing" total_time)

    printf '%s\t%s\t%s\t%s\t%s/%s/%s\t%s\t%s\t%s\t%s\n' \
        "$job_id" "$kind" "${node:-?}" "${device:-?}" \
        "${n:-?}" "${steps:-?}" "${every:-?}" \
        "$real" "$user" "$sys" "${loop_ms:--}"
done | awk -F'\t' '
    # Convert bash `time` output (e.g. 1m3.537s) to seconds.
    function secs(t,    parts) {
        split(t, parts, /[ms]/)
        return parts[1] * 60 + parts[2]
    }
    # "?" (unknown) sorts after every number.
    function num(v) { return (v == "?" || v == "") ? 1e18 : v + 0 }
    # Sort rank of a kind: CPU by thread count, then GPU, then unknown.
    function kindRank(k,    t) {
        if (k == "GPU") return 1e9
        t = substr(k, 5)
        return (t == "?" || t == "") ? 1e12 : t + 0
    }
    # Does group key x sort after group key y?
    function groupAfter(x, y,    a, b, ca, cb, i) {
        split(x, a, SUBSEP); split(y, b, SUBSEP)
        split(a[1], ca, "/"); split(b[1], cb, "/")
        for (i = 1; i <= 3; i++)
            if (num(ca[i]) != num(cb[i])) return num(ca[i]) > num(cb[i])
        return kindRank(a[2]) > kindRank(b[2])
    }
    function shorten(s, width) {
        return length(s) > width ? substr(s, 1, width - 1) "~" : s
    }
    {
        job[NR] = $1; kind[NR] = $2; node[NR] = $3; dev[NR] = $4; cfg[NR] = $5
        real[NR] = secs($6); user[NR] = secs($7)
        loop[NR] = ($9 == "-") ? "-" : sprintf("%.3f", $9 / 1000)
        key = $5 SUBSEP $2
        if (!(key in count)) { order[++groups] = key }
        count[key]++; sum[key] += real[NR]
        if ($9 != "-") { loopCount[key]++; loopSum[key] += $9 / 1000 }
        # Distinct nodes per group: mixing hardware makes timings noisy.
        if (!((key, $3) in seen)) { seen[key, $3] = 1; nodes[key] = nodes[key] (nodes[key] == "" ? "" : ",") $3 }
    }
    END {
        if (NR == 0) { print "No completed jobs with timing information found."; exit 1 }

        printf "%-9s %-7s %-9s %-24s %-22s %-9s %-9s %-9s\n", "Job ID", "Kind", "Node", "Device", "Bodies/Steps/OutEvery", "Real (s)", "Loop (s)", "CPU user"
        print "-----------------------------------------------------------------------------------------------------------"
        for (i = 1; i <= NR; i++)
            printf "%-9s %-7s %-9s %-24s %-22s %-9.3f %-9s %-9.3f\n", job[i], kind[i], node[i], shorten(dev[i], 24), cfg[i], real[i], loop[i], user[i]

        for (i = 2; i <= groups; i++) {
            k = order[i]
            for (j = i - 1; j >= 1; j--) {
                if (!groupAfter(order[j], k)) break
                order[j + 1] = order[j]
            }
            order[j + 1] = k
        }

        print ""
        printf "%-22s %-7s %-5s %-13s %-13s %-17s %-11s %s\n", "Bodies/Steps/OutEvery", "Kind", "Jobs", "Avg real (s)", "Avg loop (s)", "Speedup vs CPU-1", "Efficiency", "Nodes"
        print "-------------------------------------------------------------------------------------------------------------"
        for (i = 1; i <= groups; i++) {
            split(order[i], p, SUBSEP)
            avg = sum[order[i]] / count[order[i]]
            avgLoop = (order[i] in loopCount) ? sprintf("%.3f", loopSum[order[i]] / loopCount[order[i]]) : "-"
            base = p[1] SUBSEP "CPU-1"
            speedup = "-"; eff = "-"
            if (base in count) {
                s = (sum[base] / count[base]) / avg
                speedup = sprintf("%.2fx", s)
                t = substr(p[2], 5)
                if (p[2] != "GPU" && t != "?" && t + 0 > 0) eff = sprintf("%.0f%%", 100 * s / t)
            }
            printf "%-22s %-7s %-5d %-13.3f %-13s %-17s %-11s %s\n", p[1], p[2], count[order[i]], avg, avgLoop, speedup, eff, nodes[order[i]]
        }
        print ""
        print "Completed jobs: " NR
        print "Speedup = CPU-1 (serial) avg real / this avg real, same configuration (\"-\" if no serial runs)."
        print "Efficiency (CPU only) = speedup / threads. GPU real includes CUDA start-up; Loop excludes it."
        print "CPU user is host CPU time summed over threads: not a speed measure."
    }'
