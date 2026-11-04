# W4 Retrain Guide: B2 v2 and B3 v2 (Alienware)

## Quick start: one command

```bash
cd ~/Desktop/cv_project_LipScribe   # repo root on the Alienware
git fetch origin && git checkout cole && git pull origin cole
tmux new -s w4
bash scripts/run_w4.sh              # detach: Ctrl-b d   re-attach: tmux attach -t w4
```

`scripts/run_w4.sh` runs every step below in order, with time-budgeted training:
B2 v2 for 50 min and B3 v2 for 30 min, about 2 h in total. The cosine schedule is
compressed to fit, so the learning rate still anneals. It commits and pushes results to
`cole` after each stage, then merges `cole` into `main`. If `main` is protected it
opens a PR with `gh` instead. The full log is in `reports/w4_run/`.

More time gives better models. For example, `B2_BUDGET=150 B3_BUDGET=90 bash scripts/run_w4.sh`.

The manual steps below are what the script does, if you need to run or redo one by hand.

Run these steps on the Alienware (RTX 4080 SUPER, full GRID data) from the repo root,
in the usual Python environment. Run them **in order**, because B3 v2 is initialised from the B1
and B2 v2 checkpoints. Long runs should go inside `tmux` so they survive a closed terminal.

## What changed and why

| Change | Where | Why |
|---|---|---|
| GRID grammar-constrained decoding (`--decode grammar`) | `lipscribe/decoding/grammar.py` | Output is always a valid 6-word GRID sentence. Removes non-words and wrong word counts. No retraining needed. |
| ResNet-18 visual front end | `lipscribe/models/visual_encoder.py`, `model.visual_encoder: resnet18_bilstm` | Standard lipreading front end; much better on unseen speakers than 3 conv blocks |
| Per-clip input normalisation + spatial dropout | `model.input_norm: clip`, `model.spatial_dropout` | Removes brightness/contrast differences between speakers; regularises the CNN |
| Scale jitter, cutout, time masking | `lipscribe/data/augmentation.py`, `data.visual_aug` | B2 v1's biggest weakness is unseen speakers (test WER 13 % to 34 % per speaker) |
| lr 1e-3, warm-up 3, patience 20 | `configs/b2_visual_only_v2.yaml` | v1 early-stopped at epoch 28/60 while the learning rate was still ~60 % of peak |
| Validate B2 on clean only (`data.val_snrs`) | `scripts/train.py` | Visual-only scores are identical at every SNR, so this is 5x less validation time |
| B3 encoders initialised from B1 + B2 v2 (`model.init_from`) | `scripts/train.py` | B3 v1's visual branch was trained from scratch for 22 epochs and was undertrained |
| B3 modality dropout (audio 30 %, video 10 %) | `lipscribe/models/fusion_concat.py` | Forces the fused model to use the lips, which is what helps at 0 and -5 dB |
| `evaluate.py` writes extra decodes and ablations to their own folders and columns | `scripts/evaluate.py` | Greedy numbers are never overwritten. `reports/<run>_grammar/`, `reports/<run>_no_audio/` |

The old configs (`b2_visual_only.yaml`, `b3_concat.yaml`) are unchanged, so the milestone-3 runs
are still reproducible.

---

## Step 0: Pull and check

```bash
git checkout cole
git pull origin cole
python -m pytest -q          # all tests should pass
nvidia-smi                   # GPU free?
```

Make sure the B1 checkpoint is where B3 v2 expects it:

```bash
ls -la checkpoints/
# If b1_audio_only_best.pt is missing, copy it from the B1 run:
cp runs/b1_audio_only_20261103-193743/best.pt checkpoints/b1_audio_only_best.pt
```

## Step 1: Free gains on the existing checkpoints (no training, ~15 min)

B1 was never evaluated on the test set (its `results.csv` test columns are empty). The week-3
report used B1's *validation* WER instead. Evaluate it, then re-score all three v1 models with
grammar decoding:

```bash
python -m scripts.evaluate --checkpoint runs/b1_audio_only_20261103-193743/best.pt
python -m scripts.evaluate --checkpoint runs/b1_audio_only_20261103-193743/best.pt --decode grammar
python -m scripts.evaluate --checkpoint runs/b2_visual_only_20261104-162358/best.pt --decode grammar
python -m scripts.evaluate --checkpoint runs/b3_concat_20261104-181916/best.pt     --decode grammar
```

## Step 2: Smoke-test B2 v2 (2 min)

```bash
python -m scripts.train --config configs/b2_visual_only_v2.yaml --limit 64 --max-epochs 1
git checkout experiments/results.csv        # drop the smoke-run row
```

It should finish without errors or CUDA out-of-memory. Delete the `runs/b2_visual_only_v2_*`
folder it made.

## Step 3: Train B2 v2 (a few hours)

```bash
tmux new -s b2v2
python -m scripts.train --config configs/b2_visual_only_v2.yaml 2>&1 | tee b2v2_train.log
# detach: Ctrl-b d      re-attach: tmux attach -t b2v2
```

Monitor with `tensorboard --logdir runs` or `tail -f b2v2_train.log`.

- **Sanity check:** val WER should drop below 1.0 within the first 2 to 3 epochs, and should
  beat v1's best (**0.357**) somewhere in the first ~15 to 20 epochs.
- **If an epoch takes more than ~10 min, or you hit CUDA OOM:** stop it (Ctrl-C) and run the
  fallback instead:
  `python -m scripts.train --config configs/b2_visual_only_v2_conv3d.yaml`.
  Then edit `configs/b3_concat_v2.yaml` as its header comment explains (two lines).
- **If the loss goes to NaN or val WER is stuck at 1.0 after 5 epochs:** stop it, set
  `lr: 5.0e-4` in `configs/b2_visual_only_v2.yaml`, and restart.
- If the machine reboots mid-run: `python -m scripts.train --resume runs/b2_visual_only_v2_<stamp>/latest.pt`

When it finishes, `checkpoints/b2_visual_only_v2_best.pt` is written automatically. Evaluate:

```bash
python -m scripts.evaluate --checkpoint runs/b2_visual_only_v2_<stamp>/best.pt
python -m scripts.evaluate --checkpoint runs/b2_visual_only_v2_<stamp>/best.pt --decode grammar
```

## Step 4: Train B3 v2 (a few hours)

```bash
ls -la checkpoints/b1_audio_only_best.pt checkpoints/b2_visual_only_v2_best.pt   # both must exist
tmux new -s b3v2
python -m scripts.train --config configs/b3_concat_v2.yaml 2>&1 | tee b3v2_train.log
```

The log must start with two `Initialised ... from checkpoints/...` lines. If it instead fails with
"does not match this model", the `visual_encoder` setting in `b3_concat_v2.yaml` does not match
the B2 run you trained (see the config's header comment).

Since both encoders are pretrained, epoch-0 val WER should already be low (roughly B1 level).

Evaluate, including the modality ablations for the report:

```bash
python -m scripts.evaluate --checkpoint runs/b3_concat_v2_<stamp>/best.pt
python -m scripts.evaluate --checkpoint runs/b3_concat_v2_<stamp>/best.pt --decode grammar
python -m scripts.evaluate --checkpoint runs/b3_concat_v2_<stamp>/best.pt --drop-modality audio
python -m scripts.evaluate --checkpoint runs/b3_concat_v2_<stamp>/best.pt --drop-modality visual
```

## Step 5: Commit and push the results

Checkpoints (`*.pt`) and TensorBoard logs are git-ignored. Everything below is small:

```bash
git add experiments/results.csv reports/ \
        runs/*/history.csv runs/*/learning_curves.png runs/*/config.yaml
git commit -m "results(w4): B1 test eval, grammar decoding, B2 v2 and B3 v2 runs"
git push origin cole
```

That is everything needed to finalise the report: `results.csv`, every `reports/<run>*/`
folder (metrics, hypotheses, curves) and the training histories.
