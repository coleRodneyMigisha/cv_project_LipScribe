#!/usr/bin/env bash
#
# Run the three Week 5 P ablations for exactly 50 optimizer updates each.
#
# Run from the repository root after activating the project's Python environment:
#   ./scripts/run_ablations_smoke.sh
#
# Use --allow-cpu only when deliberately running without an NVIDIA CUDA GPU.
#

set -euo pipefail

STEPS=50
ALLOW_CPU=0

usage() {
    cat <<EOF
Usage: $0 [--steps N] [--allow-cpu]

  --steps N      Number of optimizer updates (default: 50)
  --allow-cpu    Allow running without CUDA (for testing only)
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

# Resolve repo root (script is expected to live in scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ "$STEPS" -lt 1 ]]; then
    stop_with_guidance "Steps must be a positive integer."
fi

if ! command -v python &>/dev/null; then
    stop_with_guidance "Python was not found. Install Python 3.11, activate the project virtual environment, then rerun this script."
fi

# Probe Python / PyTorch / CUDA
if ! PROBE_OUTPUT=$(python -c "
import sys
import torch
print(f'PY={sys.version_info.major}.{sys.version_info.minor}')
print(f'TORCH={torch.__version__}')
print(f'CUDA={torch.cuda.is_available()}')
" 2>&1); then
    stop_with_guidance "PyTorch could not be imported. Activate the configured environment and install the dependencies first.
$PROBE_OUTPUT"
fi

echo "$PROBE_OUTPUT"

if ! echo "$PROBE_OUTPUT" | grep -q 'PY=3\.11'; then
    stop_with_guidance "This project specifies Python 3.11. Activate the Python 3.11 virtual environment and rerun."
fi

if [[ "$ALLOW_CPU" -eq 0 ]] && ! echo "$PROBE_OUTPUT" | grep -q 'CUDA=True'; then
    stop_with_guidance "CUDA is not available to PyTorch. Install the CUDA-enabled PyTorch build for this Alienware GPU, or explicitly use --allow-cpu."
fi

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
    if [[ ! -f "$f" ]]; then
        MISSING+=("$f")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    DETAILS=("Required files are missing:")
    for m in "${MISSING[@]}"; do
        DETAILS+=("  - $m")
    done

    if [[ " ${MISSING[*]} " == *" data/manifest.csv "* ]]; then
        DETAILS+=(
            ""
            "Create the manifest on the Alienware after verifying GRID audio/video and extracting mouth crops:"
            "  python -m scripts.verify_dataset --data-root <GRID_ROOT>"
            "  python -m scripts.extract_mouths --data-root <GRID_ROOT>"
            "  python -m scripts.prepare_audio --raw-dir data/raw/grid"
            "  python -m scripts.build_manifest"
            "The manifest builder expects reports/integrity_report.csv plus data/processed/mouths,"
            "data/processed/mouth_valid, data/processed/audio16k, and data/splits.csv."
        )
    fi

    if [[ " ${MISSING[*]} " == *" data/stats/feature_norm.npz "* ]]; then
        DETAILS+=(
            ""
            "After data/manifest.csv is built, compute the training normalization statistics:"
            "  python -m scripts.compute_norm_stats --manifest data/manifest.csv --splits data/splits.csv"
        )
    fi

    if [[ " ${MISSING[*]} " == *" checkpoints/b1_audio_only_best.pt "* ]] || \
       [[ " ${MISSING[*]} " == *" checkpoints/b2_visual_only_v2_best.pt "* ]]; then
        DETAILS+=(
            ""
            "Copy the trained B1 and B2 v2 checkpoint files into checkpoints/ on the Alienware."
        )
    fi

    stop_with_guidance "$(printf '%s\n' "${DETAILS[@]}")"
fi

STAMP=$(date +%Y%m%d_%H%M%S)
RUNS_ROOT="$REPO_ROOT/runs/ablations_smoke/$STAMP"
mkdir -p "$RUNS_ROOT"

CONFIGS=(
    "configs/x1_no_modality_dropout.yaml"
    "configs/x2_concat_fusion.yaml"
    "configs/x3_2d_frontend.yaml"
)

FAILED=()

for config in "${CONFIGS[@]}"; do
    name=$(basename "$config" .yaml)
    log="$RUNS_ROOT/$name.log"

    echo
    echo "=== $name : $STEPS optimizer updates ==="
    echo "Log: $log"

    # Run training, tee output to log, capture exit status
    set +e
    python -u -m scripts.train --config "$config" --max-steps "$STEPS" --no-results --runs-root "$RUNS_ROOT" 2>&1 | tee "$log"
    code=${PIPESTATUS[0]}
    set -e

    if [[ $code -ne 0 ]]; then
        FAILED+=("$name")
        echo "WARNING: $name failed with exit code $code. See $log" >&2
        break
    fi

    if ! grep -q "Reached requested optimizer step limit: $STEPS" "$log"; then
        FAILED+=("$name")
        echo "WARNING: $name exited without confirming exactly $STEPS optimizer updates. See $log" >&2
        break
    fi

    echo "PASS: $name completed $STEPS optimizer updates."
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "ERROR: Ablation smoke run did not complete: ${FAILED[*]}. Logs: $RUNS_ROOT" >&2
    exit 1
fi

echo
echo "All three ablations completed $STEPS optimizer updates. Logs: $RUNS_ROOT"
