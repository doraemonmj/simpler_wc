#!/usr/bin/env bash
# Sweep ALLOWED_CPUS configurations and benchmark each one.
#
# For each configuration:
#   1. Patch ALLOWED_CPUS in platform_aicpu_affinity.cpp
#   2. Patch aicpu_thread_num in test_paged_attention_unroll.py to match count
#   3. Rebuild (pip install --no-build-isolation .)
#   4. Run benchmark_rounds.sh 3 times (-n 2)
#   5. Save raw output to results directory
#
# Usage:
#   ./tools/sweep_allowed_cpus.sh [-n <bench_rounds>] [-r <repeats>]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CPP_FILE="$PROJECT_ROOT/src/a5/platform/onboard/aicpu/platform_aicpu_affinity.cpp"
PY_FILE="$PROJECT_ROOT/tests/st/a5/tensormap_and_ringbuffer/paged_attention_unroll/test_paged_attention_unroll.py"

# ---------------------------------------------------------------------------
# Configurations — order matters, do not reorder
# ---------------------------------------------------------------------------
CONFIGS=(
    "4,5,11,12"
    "4,5,6,7"
    "4,5,9,10"
)

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
BENCH_ROUNDS=2      # -n flag passed to benchmark_rounds.sh
REPEATS=3           # how many times to run the benchmark per config

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--bench-rounds) BENCH_ROUNDS="$2"; shift 2 ;;
        -r|--repeats)      REPEATS="$2";      shift 2 ;;
        -h|--help)
            echo "Usage: $0 [-n <bench_rounds>] [-r <repeats>]"
            echo "  -n  Rounds per benchmark run (default: 2)"
            echo "  -r  Repeat count per config  (default: 3)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Results directory
# ---------------------------------------------------------------------------
RESULTS_DIR="$PROJECT_ROOT/benchmark_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Back up original files
# ---------------------------------------------------------------------------
cp "$CPP_FILE" "$RESULTS_DIR/platform_aicpu_affinity.cpp.orig"
cp "$PY_FILE"  "$RESULTS_DIR/test_paged_attention_unroll.py.orig"

restore_files() {
    echo ""
    echo "Restoring original files..."
    cp "$RESULTS_DIR/platform_aicpu_affinity.cpp.orig" "$CPP_FILE"
    cp "$RESULTS_DIR/test_paged_attention_unroll.py.orig" "$PY_FILE"
    echo "Done."
}
trap restore_files EXIT

# ---------------------------------------------------------------------------
# Portable in-place sed (macOS vs GNU)
# ---------------------------------------------------------------------------
sedi() {
    if sed --version 2>/dev/null | grep -q GNU; then
        sed -i "$@"
    else
        sed -i '' "$@"
    fi
}

# ---------------------------------------------------------------------------
# Summary CSV header
# ---------------------------------------------------------------------------
SUMMARY_CSV="$RESULTS_DIR/summary.csv"
echo "config,cpu_count,repeat,case,elapsed_us,sched_us,orch_us" > "$SUMMARY_CSV"

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
TOTAL_CONFIGS=${#CONFIGS[@]}

for cfg_idx in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$cfg_idx]}"
    # Count CPUs by counting comma-separated elements
    cpu_count=$(echo "$config" | awk -F',' '{print NF}')
    # Build a filesystem-safe label
    config_label=$(echo "$config" | tr -d ' ' | tr ',' '_')

    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "  Config $((cfg_idx+1))/$TOTAL_CONFIGS: ALLOWED_CPUS = {$config}"
    echo "  CPU count = $cpu_count  |  aicpu_thread_num = $cpu_count"
    echo "╚══════════════════════════════════════════════════════════════╝"

    # --- Patch ALLOWED_CPUS ---
    sedi "s/static constexpr int32_t ALLOWED_CPUS\[\] = {[^}]*};/static constexpr int32_t ALLOWED_CPUS[] = {$config};/" "$CPP_FILE"

    # --- Patch aicpu_thread_num (all occurrences) ---
    sedi "s/\"aicpu_thread_num\": [0-9]*/\"aicpu_thread_num\": $cpu_count/g" "$PY_FILE"

    # Verify patches
    echo "  [patched] $(grep 'ALLOWED_CPUS\[\]' "$CPP_FILE" | xargs)"
    echo "  [patched] aicpu_thread_num → $cpu_count"

    # --- Rebuild ---
    echo "  Building..."
    build_log="$RESULTS_DIR/${config_label}_build.log"
    if ! pip install --no-build-isolation . > "$build_log" 2>&1; then
        echo "  BUILD FAILED — see $build_log"
        continue
    fi
    echo "  Build OK"

    # --- Swimlane profiling (once per config, error-tolerant) ---
    echo "  Running swimlane profiling..."
    pre_swimlane_dirs=$(ls -d "$PROJECT_ROOT"/outputs/*/ 2>/dev/null | sort || true)
    if TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
        pytest tests/st/a5/tensormap_and_ringbuffer/paged_attention_unroll \
        --platform a5 --device 0 --case Case1 --enable-l2-swimlane \
        > "$RESULTS_DIR/${config_label}_swimlane.log" 2>&1; then
        echo "  Swimlane OK"
    else
        echo "  Swimlane FAILED (non-fatal) — see ${config_label}_swimlane.log"
    fi
    # Rename newly created outputs/ directories to include config label
    post_swimlane_dirs=$(ls -d "$PROJECT_ROOT"/outputs/*/ 2>/dev/null | sort || true)
    while IFS= read -r new_dir; do
        [[ -z "$new_dir" ]] && continue
        base=$(basename "$new_dir")
        mv "$new_dir" "$PROJECT_ROOT/outputs/${config_label}_${base}"
        echo "  Swimlane output: outputs/${config_label}_${base}"
    done < <(comm -13 <(echo "$pre_swimlane_dirs") <(echo "$post_swimlane_dirs"))

    # --- Run benchmarks ---
    for r in $(seq 1 "$REPEATS"); do
        echo ""
        echo "  ── Run $r/$REPEATS (config {$config}) ──"
        out_file="$RESULTS_DIR/${config_label}_run${r}.txt"

        TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
            "$SCRIPT_DIR/benchmark_rounds.sh" -p a5 -d 0 -n "$BENCH_ROUNDS" \
            2>&1 | tee "$out_file"

        # Parse averages from the Performance Summary table and append to CSV
        # Match only summary rows that contain "(CaseN)", not the bare heading
        while IFS= read -r line; do
            if echo "$line" | grep -qE 'paged_attention_unroll \(Case[0-9]+\)'; then
                case_name=$(echo "$line" | grep -oE '\(Case[0-9]+\)' | tr -d '()')
                elapsed=$(echo "$line" | awk '{print $(NF-2)}')
                sched=$(echo "$line" | awk '{print $(NF-1)}')
                orch=$(echo "$line" | awk '{print $NF}')
                echo "${config},${cpu_count},${r},${case_name},${elapsed},${sched},${orch}" >> "$SUMMARY_CSV"
            fi
        done < "$out_file"
    done
done

# ---------------------------------------------------------------------------
# Print final summary table from CSV
# ---------------------------------------------------------------------------
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "  SWEEP COMPLETE — All Results"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
printf "  %-18s  %-4s  %-3s  %-8s  %12s  %12s  %12s\n" \
       "ALLOWED_CPUS" "#CPU" "Run" "Case" "Elapsed(us)" "Sched(us)" "Orch(us)"
printf "  %-18s  %-4s  %-3s  %-8s  %12s  %12s  %12s\n" \
       "------------------" "----" "---" "--------" "------------" "------------" "------------"

# Skip CSV header, print rows
tail -n +2 "$SUMMARY_CSV" | while IFS=',' read -r cfg cnt rep cas el sc or; do
    printf "  %-18s  %-4s  %-3s  %-8s  %12s  %12s  %12s\n" \
           "{$cfg}" "$cnt" "$rep" "$cas" "$el" "$sc" "$or"
done

echo ""
echo "Raw results:  $RESULTS_DIR/"
echo "Summary CSV:  $SUMMARY_CSV"
echo ""
