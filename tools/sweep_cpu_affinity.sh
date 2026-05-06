#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# sweep_cpu_affinity.sh — sweep AICPU affinity configurations and benchmark each.
#
# For each ALLOWED_CPUS combo:
#   1. Patch platform_aicpu_affinity.cpp
#   2. Rebuild (pip install --no-build-isolation .)
#   3. Run benchmark_rounds.sh N_REPEAT times
#   4. Collect Trimmed Avg (or Avg) for Elapsed / Sched / Orch
#
# Usage:
#   ./tools/sweep_cpu_affinity.sh [-p a5] [-d 0] [-n 100] [--repeats 3]
#
# Edit the COMBOS array below to control which CPU sets to test.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ============================================================================
# CPU affinity combinations to sweep.
# Each entry: "cpu1,cpu2,...  label"
# The script patches ALLOWED_CPUS[] and ALLOWED_CPU_COUNT automatically.
# ============================================================================
COMBOS=(
    "4,5,11,12         4c-default"
    "4,5               2c-4-5"
    "11,12             2c-11-12"
    "4,11              2c-cross-4-11"
    "4,5,6,11,12,13    6c-extended"
)

# ============================================================================
# Argument parsing (pass-through to benchmark_rounds.sh)
# ============================================================================
PLATFORM=a5
DEVICE_ID=0
ROUNDS=100
N_REPEAT=3
BENCH_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--platform)  PLATFORM="$2";  shift 2 ;;
        -d|--device)    DEVICE_ID="$2"; shift 2 ;;
        -n|--rounds)    ROUNDS="$2";    shift 2 ;;
        --repeats)      N_REPEAT="$2";  shift 2 ;;
        -h|--help)
            cat <<'USAGE'
sweep_cpu_affinity.sh — sweep AICPU affinity configs and benchmark

Usage:
  ./tools/sweep_cpu_affinity.sh [options]

Options:
  -p, --platform   Platform (default: a5)
  -d, --device     Device ID (default: 0)
  -n, --rounds     Rounds per benchmark run (default: 100)
  --repeats        How many times to run each combo (default: 3)
  -h, --help       Show this help

Edit the COMBOS array in this script to define CPU sets.
USAGE
            exit 0 ;;
        *) BENCH_EXTRA_ARGS+=("$1"); shift ;;
    esac
done

AFFINITY_FILE="$PROJECT_ROOT/src/a5/platform/onboard/aicpu/platform_aicpu_affinity.cpp"

if [[ ! -f "$AFFINITY_FILE" ]]; then
    echo "ERROR: $AFFINITY_FILE not found"
    exit 1
fi

# Save original for restoration
cp "$AFFINITY_FILE" "$AFFINITY_FILE.bak"
trap 'mv "$AFFINITY_FILE.bak" "$AFFINITY_FILE"; echo "Restored original affinity file."' EXIT

# ============================================================================
# patch_affinity <csv_cpus>
#   Rewrite ALLOWED_CPUS[] and ALLOWED_CPU_COUNT in the source file.
# ============================================================================
patch_affinity() {
    local csv="$1"
    local count
    count=$(echo "$csv" | tr ',' '\n' | wc -l | tr -d ' ')

    sed -i \
        -e "s/^static constexpr int32_t ALLOWED_CPUS\[\].*/static constexpr int32_t ALLOWED_CPUS[] = {${csv}};/" \
        -e "s/^static constexpr int32_t ALLOWED_CPU_COUNT.*/static constexpr int32_t ALLOWED_CPU_COUNT = ${count};/" \
        "$AFFINITY_FILE"

    echo "  Patched: ALLOWED_CPUS={${csv}}, count=${count}"
}

# ============================================================================
# extract_avg <benchmark_output>
#   Extract Trimmed Avg (preferred) or Avg line. Returns "elapsed sched orch".
# ============================================================================
extract_avg() {
    local output="$1"

    # Prefer Trimmed Avg
    local trimmed_line
    trimmed_line=$(echo "$output" | grep "Trimmed Avg:" | head -1 || true)
    local elapsed="-" sched="-" orch="-"

    if [[ -n "$trimmed_line" ]]; then
        elapsed=$(echo "$trimmed_line" | awk '{print $3}')
        # Sched/Orch trimmed lines follow
        sched=$(echo "$output" | grep "Sched Trimmed Avg:" | head -1 | awk '{print $4}' || true)
        orch=$(echo "$output" | grep "Orch Trimmed Avg:" | head -1 | awk '{print $4}' || true)
    else
        # Fallback to Avg
        local avg_line
        avg_line=$(echo "$output" | grep "^  Avg:" | head -1 || true)
        if [[ -n "$avg_line" ]]; then
            elapsed=$(echo "$avg_line" | awk '{print $2}')
            sched=$(echo "$avg_line" | grep -o 'Sched Avg: [0-9.]*' | awk '{print $3}' || true)
            orch=$(echo "$avg_line" | grep -o 'Orch Avg: [0-9.]*' | awk '{print $3}' || true)
        fi
    fi

    [[ -z "$sched" ]] && sched="-"
    [[ -z "$orch" ]] && orch="-"
    echo "$elapsed $sched $orch"
}

# ============================================================================
# Main sweep
# ============================================================================
echo ""
echo "========================================================"
echo "  CPU Affinity Sweep"
echo "  Platform: $PLATFORM  Device: $DEVICE_ID  Rounds: $ROUNDS  Repeats: $N_REPEAT"
echo "========================================================"

# Results storage: RESULT_<combo_idx>_<repeat> = "elapsed sched orch"
declare -A RESULTS
COMBO_LABELS=()

for combo_idx in "${!COMBOS[@]}"; do
    line="${COMBOS[$combo_idx]}"
    csv=$(echo "$line" | awk '{print $1}')
    label=$(echo "$line" | awk '{print $2}')
    [[ -z "$label" ]] && label="combo-${combo_idx}"
    COMBO_LABELS+=("$label [$csv]")

    echo ""
    echo "========================================================"
    echo "  [$((combo_idx+1))/${#COMBOS[@]}] $label  CPUS={$csv}"
    echo "========================================================"

    # 1. Patch source
    patch_affinity "$csv"

    # 2. Rebuild
    echo "  Rebuilding..."
    cd "$PROJECT_ROOT"
    source .venv/bin/activate 2>/dev/null || true
    pip install --no-build-isolation . > /dev/null 2>&1
    echo "  Build done."

    # 3. Run benchmark N_REPEAT times
    for r in $(seq 1 "$N_REPEAT"); do
        echo "  --- Run $r/$N_REPEAT ---"
        bench_output=$("$SCRIPT_DIR/benchmark_rounds.sh" \
            -p "$PLATFORM" -d "$DEVICE_ID" -n "$ROUNDS" \
            "${BENCH_EXTRA_ARGS[@]}" 2>&1) || true
        echo "$bench_output" | grep -E "(Avg:|FAILED)" || true

        avgs=$(extract_avg "$bench_output")
        RESULTS["${combo_idx}_${r}"]="$avgs"
    done
done

# ============================================================================
# Summary table
# ============================================================================
echo ""
echo ""
echo "========================================================"
echo "  SWEEP RESULTS SUMMARY"
echo "========================================================"
echo ""

# Header
printf "  %-30s" "Config"
for r in $(seq 1 "$N_REPEAT"); do
    printf "  %14s" "Run${r} (us)"
done
printf "  %14s\n" "Mean (us)"

printf "  %-30s" "------------------------------"
for r in $(seq 1 "$N_REPEAT"); do
    printf "  %14s" "--------------"
done
printf "  %14s\n" "--------------"

for combo_idx in "${!COMBO_LABELS[@]}"; do
    label="${COMBO_LABELS[$combo_idx]}"
    printf "  %-30s" "$label"

    sum=0
    count=0
    for r in $(seq 1 "$N_REPEAT"); do
        avgs="${RESULTS["${combo_idx}_${r}"]:-"- - -"}"
        elapsed=$(echo "$avgs" | awk '{print $1}')
        printf "  %14s" "$elapsed"
        if [[ "$elapsed" != "-" ]]; then
            sum=$(echo "$sum + $elapsed" | bc)
            ((count++)) || true
        fi
    done

    if [[ $count -gt 0 ]]; then
        mean=$(echo "scale=1; $sum / $count" | bc)
        printf "  %14s" "$mean"
    else
        printf "  %14s" "-"
    fi
    printf "\n"
done

# Sched/Orch breakdown if available
first_avgs="${RESULTS["0_1"]:-"- - -"}"
first_sched=$(echo "$first_avgs" | awk '{print $2}')
if [[ "$first_sched" != "-" ]]; then
    echo ""
    echo "  --- Sched Phase ---"
    printf "  %-30s" "Config"
    for r in $(seq 1 "$N_REPEAT"); do
        printf "  %14s" "Run${r} (us)"
    done
    printf "  %14s\n" "Mean (us)"
    printf "  %-30s" "------------------------------"
    for r in $(seq 1 "$N_REPEAT"); do printf "  %14s" "--------------"; done
    printf "  %14s\n" "--------------"

    for combo_idx in "${!COMBO_LABELS[@]}"; do
        label="${COMBO_LABELS[$combo_idx]}"
        printf "  %-30s" "$label"
        sum=0; count=0
        for r in $(seq 1 "$N_REPEAT"); do
            avgs="${RESULTS["${combo_idx}_${r}"]:-"- - -"}"
            val=$(echo "$avgs" | awk '{print $2}')
            printf "  %14s" "$val"
            if [[ "$val" != "-" ]]; then sum=$(echo "$sum + $val" | bc); ((count++)) || true; fi
        done
        if [[ $count -gt 0 ]]; then printf "  %14s" "$(echo "scale=1; $sum / $count" | bc)"; else printf "  %14s" "-"; fi
        printf "\n"
    done

    echo ""
    echo "  --- Orch Phase ---"
    printf "  %-30s" "Config"
    for r in $(seq 1 "$N_REPEAT"); do
        printf "  %14s" "Run${r} (us)"
    done
    printf "  %14s\n" "Mean (us)"
    printf "  %-30s" "------------------------------"
    for r in $(seq 1 "$N_REPEAT"); do printf "  %14s" "--------------"; done
    printf "  %14s\n" "--------------"

    for combo_idx in "${!COMBO_LABELS[@]}"; do
        label="${COMBO_LABELS[$combo_idx]}"
        printf "  %-30s" "$label"
        sum=0; count=0
        for r in $(seq 1 "$N_REPEAT"); do
            avgs="${RESULTS["${combo_idx}_${r}"]:-"- - -"}"
            val=$(echo "$avgs" | awk '{print $3}')
            printf "  %14s" "$val"
            if [[ "$val" != "-" ]]; then sum=$(echo "$sum + $val" | bc); ((count++)) || true; fi
        done
        if [[ $count -gt 0 ]]; then printf "  %14s" "$(echo "scale=1; $sum / $count" | bc)"; else printf "  %14s" "-"; fi
        printf "\n"
    done
fi

echo ""
echo "Sweep complete."
