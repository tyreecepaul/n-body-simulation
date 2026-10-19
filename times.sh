#!/bin/bash

echo "Job ID        Real (s)       User (s)       Sys (s)"
echo "------------------------------------------------------"

total_real=0
total_user=0
total_sys=0
count=0

for file in nbody_sim_*.err; do
    [ -f "$file" ] || continue

    real=$(grep '^real' "$file" | awk '{print $2}')
    user=$(grep '^user' "$file" | awk '{print $2}')
    sys=$(grep '^sys' "$file" | awk '{print $2}')

    # Skip files that don't contain timing information
    if [ -z "$real" ] || [ -z "$user" ] || [ -z "$sys" ]; then
        continue
    fi

    job_id=$(echo "$file" | sed -E 's/nbody_sim_([0-9]+)\.err/\1/')

    # Convert 0m14.512s -> 14.512 seconds
    real_sec=$(echo "$real" | sed -E 's/([0-9]+)m([0-9.]+)s/\1*60+\2/' | bc -l)
    user_sec=$(echo "$user" | sed -E 's/([0-9]+)m([0-9.]+)s/\1*60+\2/' | bc -l)
    sys_sec=$(echo "$sys" | sed -E 's/([0-9]+)m([0-9.]+)s/\1*60+\2/' | bc -l)

    printf "%-13s %-14.3f %-14.3f %-14.3f\n" \
        "$job_id" "$real_sec" "$user_sec" "$sys_sec"

    total_real=$(echo "$total_real + $real_sec" | bc -l)
    total_user=$(echo "$total_user + $user_sec" | bc -l)
    total_sys=$(echo "$total_sys + $sys_sec" | bc -l)

    ((count++))
done

if [ "$count" -gt 0 ]; then
    avg_real=$(echo "$total_real / $count" | bc -l)
    avg_user=$(echo "$total_user / $count" | bc -l)
    avg_sys=$(echo "$total_sys / $count" | bc -l)

    echo "------------------------------------------------------"
    printf "%-13s %-14.3f %-14.3f %-14.3f\n" \
        "Average" "$avg_real" "$avg_user" "$avg_sys"
    echo
    echo "Completed jobs: $count"
else
    echo "No completed jobs with timing information found."
fi
