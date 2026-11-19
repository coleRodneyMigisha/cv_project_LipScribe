#!/usr/bin/env bash
#
# Run the three Week 5 faith ablations and write a summary report to
#   ./reports/faith_ablation_results.md
#
# Usage (from repository root, with project venv activated):
#   ./scripts/run_faith_ablations.sh
#   ./scripts/run_faith_ablations.sh --steps 50
#   ./scripts/run_faith_ablations.sh --steps 50 --allow-cpu
#

set -euo pipefail

STEPS=50
ALLOW_CPU=0

usage() {
    cat <<EOF
Usage: $0 [--steps N] [--allow-cpu]

  --steps N      Number of optimizer updates per ablation (default: 50)
  --allow-cpu    Allow running without CUDA
  -h, --help     Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --allow-cpu)
            ALLOW_CPU=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

stop_with_guidance() {
    echo "ERROR: $1" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

REPORT_DIR="$REPO_ROOT/reports"
REPORT_FILE="$REPORT_DIR/faith_ablation_results.md"
mkdir -p "$REPORT_DIR"

if [[ "$STEPS" -lt 1 ]]; then
    stop_with_guidance "Steps must be a positive integer."
fi

# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------
if ! command -v python &>/dev/null; then
    stop_with_guidance "Python was not found. Activate the project virtual environment first."
fi

if ! PROBE_OUTPUT=$(python -c "
import sys
import torch
print(f'PY={sys.version_info.major}.{sys.version_info.minor}')
print(f'TORCH={torch.__version__}')
print(f'CUDA={torch.cuda.is_available()}')
" 2>&1); then
    stop_with_guidance "PyTorch could not be imported.
$PROBE_OUTPUT"
fi

echo "$PROBE_OUTPUT"

if ! echo "$PROBE_OUTPUT" | grep -q 'PY=3\.11'; then
    stop_with_guidance "This project specifies Python 3.11. Activate the correct virtual environment."
fi

if [[ "$ALLOW_CPU" -eq 0 ]] && ! echo "$PROBE_OUTPUT" | grep -q 'CUDA=True'; then
    stop_with_guidance "CUDA is not available. Use --allow-cpu if you intentionally want a CPU run."
fi

# ---------------------------------------------------------------------------
# Required files
# ---------------------------------------------------------------------------
REQUIRED_FILES=(
    "data/manifest.csv"
    "data/splits.csv"
    "data/stats/feature_norm.npz"
    "checkpoints/b1_audio_only_best.pt"
    "checkpoints/b2_visual_only_v2_best.pt"
    "configs/x1_no_modality_dropout.yaml"
    "configs/x2_concat_fusion.yaml"
    "configs/x3_2d_frontend.yaml"
)

MISSING=()
for f in "${REQUIRED_FILES[@]}"; do
    [[ -f "$f" ]] || MISSING+=("$f")
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    DETAILS=("Required files are missing:")
    for m in "${MISSING[@]}"; do
        DETAILS+=("  - $m")
    done
    stop_with_guidance "$(printf '%s\n' "${DETAILS[@]}")"
fi

# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------
STAMP=$(date +%Y%m%d_%H%M%S)
RUNS_ROOT="$REPO_ROOT/runs/faith_ablations/$STAMP"
mkdir -p "$RUNS_ROOT"

CONFIGS=(
    "configs/x1_no_modality_dropout.yaml"
    "configs/x2_concat_fusion.yaml"
    "configs/x3_2d_frontend.yaml"
)

# Human-readable names for the report
declare -A ABLATION_NAMES=(
    ["x1_no_modality_dropout"]="X1 – No modality dropout"
    ["x2_concat_fusion"]="X2 – Concat fusion"
    ["x3_2d_frontend"]="X3 – 2-D frontend"
)

declare -A ABLATION_DESC=(
    ["x1_no_modality_dropout"]="Removes modality dropout; both audio and visual streams always present."
    ["x2_concat_fusion"]="Simple concatenation fusion instead of the default attention/fusion block."
    ["x3_2d_frontend"]="Uses a pure 2-D convolutional visual frontend."
)

# ---------------------------------------------------------------------------
# Helpers to extract useful numbers from a log
# ---------------------------------------------------------------------------
extract_last() {
    # extract_last <file> <regex>
    # Returns the last matching line (or empty)
    grep -E "$2" "$1" 2>/dev/null | tail -n 1 || true
}

extract_value() {
    # extract_value <line> <key>
    # Tries to pull a numeric value after the key
    local line="$1"
    local key="$2"
    echo "$line" | grep -oE "${key}[=: ]+[0-9.eE+-]+" | grep -oE "[0-9.eE+-]+$" || true
}

# ---------------------------------------------------------------------------
# Run the ablations
# ---------------------------------------------------------------------------
echo
echo "============================================================"
echo " Faith Ablation Smoke Run  (${STEPS} optimizer steps each)"
echo " Logs will be written under: $RUNS_ROOT"
echo "============================================================"

RESULTS=()          # array of "name|status|log|extra"
OVERALL_OK=1

for config in "${CONFIGS[@]}"; do
    name=$(basename "$config" .yaml)
    log="$RUNS_ROOT/${name}.log"
    pretty="${ABLATION_NAMES[$name]:-$name}"

    echo
    echo "=== $pretty  ($STEPS steps) ==="
    echo "Config : $config"
    echo "Log    : $log"

    set +e
    python -u -m scripts.train \
        --config "$config" \
        --max-steps "$STEPS" \
        --no-results \
        --runs-root "$RUNS_ROOT" \
        2>&1 | tee "$log"
    code=${PIPESTATUS[0]}
    set -e

    status="FAIL"
    extra=""

    if [[ $code -ne 0 ]]; then
        status="FAIL (exit $code)"
        OVERALL_OK=0
        echo "  → FAILED with exit code $code"
    elif ! grep -q "Reached requested optimizer step limit: $STEPS" "$log"; then
        status="FAIL (did not reach step limit)"
        OVERALL_OK=0
        echo "  → Exited without confirming $STEPS optimizer updates"
    else
        status="PASS"
        echo "  → PASS"
    fi

    # Try to pull a few useful metrics if present in the log
    last_loss_line=$(extract_last "$log" "(loss|train_loss|Loss)")
    last_step_line=$(extract_last "$log" "(step|optimizer step|Reached requested)")

    RESULTS+=("${name}|${status}|${log}|${last_loss_line}|${last_step_line}")
done

# ---------------------------------------------------------------------------
# Write the markdown report
# ---------------------------------------------------------------------------
{
    cat <<EOF
# Faith Ablation Results

**Generated:** $(date -u +"%Y-%m-%d %H:%M:%S UTC")  
**Repository root:** \`$REPO_ROOT\`  
**Run directory:** \`$RUNS_ROOT\`  
**Optimizer steps per ablation:** $STEPS  

---

## Summary

| Ablation | Status | Description |
|----------|--------|-------------|
EOF

    for entry in "${RESULTS[@]}"; do
        IFS='|' read -r name status log last_loss last_step <<< "$entry"
        pretty="${ABLATION_NAMES[$name]:-$name}"
        desc="${ABLATION_DESC[$name]:-}"
        echo "| **$pretty** | $status | $desc |"
    done

    cat <<EOF

---

## Detailed results

EOF

    for entry in "${RESULTS[@]}"; do
        IFS='|' read -r name status log last_loss last_step <<< "$entry"
        pretty="${ABLATION_NAMES[$name]:-$name}"
        desc="${ABLATION_DESC[$name]:-}"

        cat <<EOF
### $pretty

- **Config:** \`configs/${name}.yaml\`
- **Status:** $status
- **Log file:** \`$log\`
- **Description:** $desc

EOF

        if [[ -n "$last_step" ]]; then
            echo "- **Last step message:** \`$last_step\`"
        fi
        if [[ -n "$last_loss" ]]; then
            echo "- **Last loss-related line:** \`$last_loss\`"
        fi

        # Include a short tail of the log for quick inspection
        echo
        echo "<details>"
        echo "<summary>Last 30 lines of log</summary>"
        echo
        echo '```'
        tail -n 30 "$log" 2>/dev/null || echo "(log empty or unreadable)"
        echo '```'
        echo "</details>"
        echo
        echo "---"
        echo
    done

    cat <<EOF
## Environment

\`\`\`
$PROBE_OUTPUT
\`\`\`

## Notes

- This is a **smoke / fidelity** run (fixed number of optimizer steps).
- Full training curves and final metrics live under the individual run directories inside \`$RUNS_ROOT\`.
- Re-run with a larger \`--steps\` value (or remove the step limit) for a proper experimental comparison.
EOF

} > "$REPORT_FILE"

echo
echo "============================================================"
if [[ $OVERALL_OK -eq 1 ]]; then
    echo " All ablations completed successfully."
else
    echo " One or more ablations failed – see the report for details."
fi
echo " Report written to: $REPORT_FILE"
echo "============================================================"

# Exit non-zero if any ablation failed
[[ $OVERALL_OK -eq 1 ]]
