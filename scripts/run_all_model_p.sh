#!/usr/bin/env bash
# Finish evaluating all 4 already-trained Model P checkpoints (or train any missing ones),
# update experiments/results.csv and reports/, and regenerate comparison plots.

set -uo pipefail

PY="${PYTHON:-.venv/bin/python}"
BUDGET="${P_BUDGET:-30}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

mkdir -p reports/w4_run
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="reports/w4_run/run_model_p_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

START=$(date +%s)
say() { printf '\n[%s | +%dm] %s\n' "$(date +%H:%M:%S)" $(( ($(date +%s) - START) / 60 )) "$*"; }

latest_run() {
    ls -d runs/"$1"_2[0-9]*-[0-9]* 2>/dev/null | sort | tail -n 1
}

run_and_eval() {
    local cfg="$1"
    local exp
    exp="$(basename "$cfg" .yaml)"
    local run_dir
    run_dir="$(latest_run "$exp")"

    if [[ -n "$run_dir" && -f "$run_dir/best.pt" && "${RETRAIN:-0}" != "1" ]]; then
        say "=== Found completed training checkpoint for $exp: $run_dir/best.pt (skipping training; set RETRAIN=1 to force) ==="
    else
        say "=== Training $exp ($cfg) for ${BUDGET} min on full dataset ==="
        "$PY" -m scripts.train --config "$cfg" --time-budget-min "$BUDGET"
        run_dir="$(latest_run "$exp")"
    fi

    if [[ -n "$run_dir" && -f "$run_dir/best.pt" ]]; then
        say "=== Evaluating $exp ($run_dir/best.pt): greedy, grammar, drop-audio, drop-visual ==="
        "$PY" -m scripts.evaluate --checkpoint "$run_dir/best.pt"
        "$PY" -m scripts.evaluate --checkpoint "$run_dir/best.pt" --decode grammar
        "$PY" -m scripts.evaluate --checkpoint "$run_dir/best.pt" --drop-modality audio
        "$PY" -m scripts.evaluate --checkpoint "$run_dir/best.pt" --drop-modality visual
    else
        say "ERROR: No best.pt found for $exp in '$run_dir'"
    fi
}

for cfg in \
    configs/p_init_b1_b2_v2.yaml \
    configs/p_init_b1_b2.yaml \
    configs/p_init_b2.yaml \
    configs/p_scratch.yaml
do
    run_and_eval "$cfg"
done

# Copy the best overall Model P checkpoint to checkpoints/p_proposed_best.pt
if [[ -f checkpoints/p_init_b1_b2_v2_best.pt ]]; then
    cp checkpoints/p_init_b1_b2_v2_best.pt checkpoints/p_proposed_best.pt
fi

say "=== Regenerating comparison plots ==="
"$PY" -m scripts.plot_all_model_comparisons

say "=== All 4 full Model P evaluations and plots finished! ==="
