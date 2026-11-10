# W3-08 (Experiment X5): Audio Feature Configurations — Alienware Run Guide

Train B1 with A2 (MFCC39) and A3 (log-Mel + MFCC) and decide whether MFCCs add
anything over A1 (80 log-Mel, the current `configs/b1_audio_only.yaml`
baseline). Run only if GPU time allows this week; otherwise move to Week 4.

The A1 baseline already has a result in `experiments/results.csv`
(`b1_audio_only`, mean val WER 0.0772) — no need to retrain it unless it's
been invalidated by other changes.

---

## Checklist Summary

- [ ] Pull latest changes from branch `charis`
- [ ] Compute feature-normalisation stats for A2 and A3
- [ ] Train `configs/b1_mfcc39.yaml` (A2)
- [ ] Train `configs/b1_logmel_mfcc.yaml` (A3)
- [ ] Evaluate both checkpoints on the same protocol as B1
- [ ] Run the decision script and record the decision
- [ ] Commit/push `experiments/results.csv`, `reports/`, and the decision doc

---

## Step 1: Pull Latest Updates on Alienware

```bash
git checkout charis
git pull origin charis
```

---

## Step 2: Compute Feature-Normalisation Stats

A1's stats (`data/stats/feature_norm.npz`) hold 80-dim log-Mel moments and
can't be reused for A2 (39-dim) or A3 (119-dim) — each feature config needs
its own mean/std, computed from the training partition only:

```bash
python scripts/compute_norm_stats.py --audio-features mfcc39 \
    --noise-splits Data/Noise_splits/splits.csv
python scripts/compute_norm_stats.py --audio-features logmel_mfcc \
    --noise-splits Data/Noise_splits/splits.csv
```

This writes `data/stats/feature_norm_mfcc39.npz` and
`data/stats/feature_norm_logmel_mfcc.npz` — the paths already referenced by
`configs/b1_mfcc39.yaml` and `configs/b1_logmel_mfcc.yaml`. Omit
`--noise-splits` if the MUSAN pools aren't set up on this machine; the
dataset falls back to its default noise handling.

---

## Step 3: Train A2 (MFCC39)

```bash
python -m scripts.train --config configs/b1_mfcc39.yaml
```

Same protocol as B1: AdamW, same LR/batch/epoch schedule, early stopping on
mean validation WER (patience 8), SNR curriculum `[clean, 10, 5, 0, -5]`.
Artifacts land in `runs/b1_mfcc39_<timestamp>/`; a summary row is appended to
`experiments/results.csv` automatically when the run finishes or early-stops.

Monitor with TensorBoard if useful:

```bash
tensorboard --logdir runs/
```

---

## Step 4: Train A3 (log-Mel + MFCC)

```bash
python -m scripts.train --config configs/b1_logmel_mfcc.yaml
```

Same as Step 3, writing to `runs/b1_logmel_mfcc_<timestamp>/`.

---

## Step 5: Evaluate Both on the Test Protocol

```bash
python -m scripts.evaluate --checkpoint runs/b1_mfcc39_<timestamp>/best.pt
python -m scripts.evaluate --checkpoint runs/b1_logmel_mfcc_<timestamp>/best.pt
```

Each call fills in `test_wer_by_snr` / `test_cer_by_snr` on that run's row in
`experiments/results.csv` and writes `reports/<run>/evaluation_report.md`
plus the curve plots, exactly as for B1 and B2.

---

## Step 6: Record the Decision

Once both configs have a row in `experiments/results.csv` (training alone is
enough — the decision rule only looks at validation WER, not test), run:

```bash
python -m scripts.decide_audio_features
```

This reads the latest row for `b1_audio_only`, `b1_mfcc39` and
`b1_logmel_mfcc`, applies the card's rule ("keep A1 unless another
configuration improves validation WER"), and writes
`reports/w3_08_audio_features_decision.md` with the comparison table and the
decision sentence. Paste that decision onto card W3-08.

---

## Step 7: Push Results Back

`data/stats/*.npz` is gitignored (generated locally, like checkpoints) —
only the results and reports are committed:

```bash
git add experiments/results.csv reports/
git commit -m "W3-08: Log B1 A2/A3 audio feature experiment results and decision"
git push origin charis
```
