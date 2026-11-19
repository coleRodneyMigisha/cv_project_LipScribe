#!/usr/bin/env bash
# W4: run every B2/B3 improvement experiment on the Alienware, push the results to
# `cole` after each stage, and finally try to merge `cole` into `main`.
#
#   tmux new -s w4            # so it survives a closed terminal / SSH drop
#   bash scripts/run_w4.sh    # then detach with Ctrl-b d
#
# Time budgets in minutes (defaults fit a ~2 h window; raise them if you have longer):
#   B2_BUDGET=50 B3_BUDGET=30 bash scripts/run_w4.sh
# Other knobs: PYTHON=python  SKIP_TESTS=1  SKIP_MERGE=1
#
# Stages (results are committed and pushed to `cole` after each one, so whatever has
# finished is on GitHub even if a later stage fails or runs over time):
#   0  pull, GPU check, test suite
#   1  B1 test evaluation (never done before; the week-3 report used B1 validation WER)
#   2  B2 v2: smoke test (falls back to the conv3d config on failure/OOM), train, evaluate
#   3  B3 v2: train from the B1 + B2 v2 encoders, evaluate
#   4  grammar decoding of the v1 models, modality ablations of B3 v2
#   5  merge cole -> main and push (falls back to opening a PR with `gh`)

set -uo pipefail

PY="${PYTHON:-python}"
B2_BUDGET="${B2_BUDGET:-50}"
B3_BUDGET="${B3_BUDGET:-30}"
BRANCH=cole

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="reports/w4_run"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/run_w4_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

START=$(date +%s)
FAILED=()
SUMMARY=()

say()  { printf '\n[%s | +%dm] %s\n' "$(date +%H:%M:%S)" $(( ($(date +%s) - START) / 60 )) "$*"; }
fail() { say "FAILED: $*"; FAILED+=("$*"); }

latest_run() {  # newest runs/<experiment>_YYYYMMDD-HHMMSS directory
    ls -d runs/"$1"_2[0-9]*-[0-9]* 2>/dev/null | sort | tail -n 1
}

evaluate() {    # evaluate <run_dir> [extra evaluate.py args...]
    local run="$1"; shift
    if [[ -z "$run" || ! -f "$run/best.pt" ]]; then
        fail "evaluate: no best.pt in '${run}'"; return 1
    fi
    say "Evaluate $run $*"
    if "$PY" -m scripts.evaluate --checkpoint "$run/best.pt" "$@"; then
        SUMMARY+=("eval ok: $run $*")
    else
        fail "evaluate $run $*"
    fi
}

commit_push() { # commit_push <message>
    # The live log is NOT committed here: it keeps changing, and a tracked-but-modified
    # file blocks the checkout in the merge stage. It is committed once at the end.
    git add experiments/results.csv configs/ 2>/dev/null
    git add reports/ ':!reports/w4_run' 2>/dev/null
    for f in runs/*/history.csv runs/*/learning_curves.png runs/*/config.yaml runs/*/commit.txt; do
        [[ -f "$f" ]] && git add "$f"
    done
    if git diff --cached --quiet; then
        say "Nothing new to commit"; return 0
    fi
    git commit -q -m "$1" -m "Automated by scripts/run_w4.sh on $(hostname)." || { fail "git commit"; return 1; }
    if ! git push -q origin "$BRANCH"; then
        git pull -q --rebase origin "$BRANCH" && git push -q origin "$BRANCH" || { fail "git push"; return 1; }
    fi
    say "Pushed: $1"
}

# ----------------------------------------------------------------------------- 0
say "Stage 0: pull, GPU check, tests  (B2 budget ${B2_BUDGET} min, B3 budget ${B3_BUDGET} min)"
git fetch -q origin || { say "git fetch failed (network?). Aborting."; exit 1; }
git checkout -q "$BRANCH" || { say "Cannot check out $BRANCH (uncommitted changes?). Aborting."; exit 1; }
if ! git pull -q --ff-only origin "$BRANCH"; then
    say "git pull failed. Commit or stash local changes on the Alienware first. Aborting."
    git status --short
    exit 1
fi
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || true
if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    say "CUDA is not available to $PY. Activate the right environment. Aborting."
    exit 1
fi
if [[ "${SKIP_TESTS:-0}" != 1 ]]; then
    "$PY" -m pytest -q -x || { say "Tests failed. Aborting before any training."; exit 1; }
fi

# ----------------------------------------------------------------------------- 1
say "Stage 1: B1 test evaluation"
B1_RUN="runs/b1_audio_only_20261103-193743"
[[ -f "$B1_RUN/best.pt" ]] || B1_RUN="$(latest_run b1_audio_only)"
mkdir -p checkpoints
if [[ ! -f checkpoints/b1_audio_only_best.pt && -f "$B1_RUN/best.pt" ]]; then
    cp "$B1_RUN/best.pt" checkpoints/b1_audio_only_best.pt
fi
evaluate "$B1_RUN"
evaluate "$B1_RUN" --decode grammar
commit_push "results(w4): B1 test evaluation (greedy + grammar)"

# ----------------------------------------------------------------------------- 2
say "Stage 2: B2 v2"
B2_CFG=configs/b2_visual_only_v2.yaml
B3_CFG=configs/b3_concat_v2.yaml
SMOKE_ROOT="$(mktemp -d)"
if ! "$PY" -m scripts.train --config "$B2_CFG" --limit 64 --max-epochs 1 \
        --no-results --runs-root "$SMOKE_ROOT"; then
    fail "B2 v2 ResNet smoke test; falling back to the conv3d encoder"
    B2_CFG=configs/b2_visual_only_v2_conv3d.yaml
    B3_CFG=configs/b3_concat_v2_conv3d.yaml
    "$PY" -m scripts.train --config "$B2_CFG" --limit 64 --max-epochs 1 \
        --no-results --runs-root "$SMOKE_ROOT" || fail "B2 v2 conv3d smoke test"
fi
rm -rf "$SMOKE_ROOT"
B2_EXP="$(basename "$B2_CFG" .yaml)"
say "Training $B2_CFG for at most ${B2_BUDGET} min"
"$PY" -m scripts.train --config "$B2_CFG" --time-budget-min "$B2_BUDGET" || fail "B2 v2 training exited non-zero"
B2_RUN="$(latest_run "$B2_EXP")"
evaluate "$B2_RUN"
evaluate "$B2_RUN" --decode grammar
commit_push "results(w4): B2 v2 training + test evaluation ($B2_EXP)"

# ----------------------------------------------------------------------------- 3
say "Stage 3: B3 v2"
B3_EXP="$(basename "$B3_CFG" .yaml)"
if [[ -f checkpoints/b1_audio_only_best.pt && -f "checkpoints/${B2_EXP}_best.pt" ]]; then
    say "Training $B3_CFG for at most ${B3_BUDGET} min"
    "$PY" -m scripts.train --config "$B3_CFG" --time-budget-min "$B3_BUDGET" || fail "B3 v2 training exited non-zero"
    B3_RUN="$(latest_run "$B3_EXP")"
    evaluate "$B3_RUN"
    evaluate "$B3_RUN" --decode grammar
    commit_push "results(w4): B3 v2 training + test evaluation ($B3_EXP)"
else
    fail "B3 v2 skipped: need checkpoints/b1_audio_only_best.pt and checkpoints/${B2_EXP}_best.pt"
    B3_RUN=""
fi

# ----------------------------------------------------------------------------- 4
say "Stage 4: grammar decoding of the v1 models, B3 v2 ablations"
evaluate "runs/b2_visual_only_20261104-162358" --decode grammar
evaluate "runs/b3_concat_20261104-181916" --decode grammar
if [[ -n "$B3_RUN" ]]; then
    evaluate "$B3_RUN" --drop-modality audio
    evaluate "$B3_RUN" --drop-modality visual
fi
commit_push "results(w4): v1 grammar decoding and B3 v2 modality ablations"

# ----------------------------------------------------------------------------- 5
merge_main() {
    say "Stage 5: merge $BRANCH into main"
    git fetch -q origin || { fail "git fetch"; return 1; }
    # Merge on a detached origin/main so no local branch is moved or reset.
    git checkout -q --detach origin/main || { fail "checkout origin/main"; git checkout -q "$BRANCH"; return 1; }
    if ! git merge --no-ff -q "$BRANCH" -m "Merge branch '$BRANCH': W4 B2/B3 improvements and results"; then
        git merge --abort
        git checkout -q "$BRANCH"
        fail "merge conflict between $BRANCH and main (left unmerged; resolve on the laptop)"
        return 1
    fi
    if git push -q origin HEAD:main; then
        git checkout -q "$BRANCH"
        SUMMARY+=("merged $BRANCH into main and pushed")
        return 0
    fi
    # main is probably protected: drop the detached merge and open a PR instead.
    git checkout -q "$BRANCH"
    if command -v gh >/dev/null 2>&1; then
        gh pr create --base main --head "$BRANCH" \
            --title "W4: B2/B3 improvements and results" \
            --body "Automated by scripts/run_w4.sh. See reports/w4_run/ for the run log." \
            || true
        if gh pr merge "$BRANCH" --merge; then
            SUMMARY+=("merged $BRANCH into main via PR")
            return 0
        fi
        fail "direct push to main rejected; PR opened but not merged (merge it on GitHub)"
    else
        fail "direct push to main rejected and gh is not installed; open a PR $BRANCH -> main on GitHub"
    fi
    return 1
}

if [[ "${SKIP_MERGE:-0}" != 1 ]]; then
    if (( ${#FAILED[@]} == 0 )); then
        merge_main
    else
        say "Skipping merge into main because a stage failed; results are on $BRANCH."
    fi
fi

# ----------------------------------------------------------------------------- done
say "Finished in $(( ($(date +%s) - START) / 60 )) min. Log: $LOG"
for s in "${SUMMARY[@]}"; do echo "  ok   $s"; done
for f in "${FAILED[@]}"; do echo "  FAIL $f"; done
# Push the final log lines too (on cole; main already has everything else).
git add -f "$LOG" && git commit -q -m "logs(w4): run_w4 log" && git push -q origin "$BRANCH" || true
(( ${#FAILED[@]} == 0 ))
