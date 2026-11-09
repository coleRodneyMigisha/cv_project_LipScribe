# B2 Visual-Only Training & Evaluation Guide (Alienware Rig)

This guide walks you through executing, monitoring, and evaluating the **B2 Visual-Only Model** (Card W3-05) on the Alienware GPU machine where the full GRID dataset is located.

---

## Checklist Summary

- [x] Pull latest changes from branch `cole`
- [x] Run GPU Overfit Test on 8 utterances (see `b2_overfit_test_results.md` for full run analysis)
- [x] Launch Full Training run with `configs/b2_visual_only.yaml`
- [x] Monitor validation CER (must fall clearly below untrained model $\approx 1.0$)
- [x] Check if early progress stalls (contingency: raise at review for W4-05 curriculum)
- [x] Evaluate best checkpoint (`python -m scripts.evaluate`)
- [x] Confirm results are logged to `experiments/results.csv` and commit/push

---

## Step 1: Pull Latest Updates on Alienware

Open a terminal in the project repository on Alienware and pull the latest commits:

```bash
git checkout cole
git pull origin cole
```

Verify your environment has GPU access:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), 'GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## Step 2: Run GPU Overfit Test (8 Utterances) — [COMPLETED]

**Result:**
- **Final Loss:** `0.0497` (target: `< 0.05`)
- **Final CER:** `0.0000` (100% character accuracy across all 8 utterances)
- **Steps:** 3099 steps in **138.4 seconds** on NVIDIA GeForce RTX 4080 SUPER
- **Status:** **PASS** (Full breakdown documented in `b2_overfit_test_results.md`)

```bash
# Executed command:
python scripts/check_b2_overfit.py --device cuda -n 8 --steps 4000 --lr 2e-3 --target-loss 0.05 --target-cer 0.0
```

---

## Step 3: Launch Full B2 Training

Start the full training run:

```bash
python -m scripts.train --config configs/b2_visual_only.yaml
```

*Notes:*
- Training artifacts and checkpoints will be saved to `runs/b2_visual_only_<YYYYMMDD-HHMMSS>/`.
- Training-only augmentation (random crop jitter + horizontal flip) is automatically enabled for training and disabled for validation.
- Early stopping is active with patience = 8 epochs on validation WER.

---

## Step 4: Monitor Validation CER Early

In another terminal, launch TensorBoard to observe metrics:

```bash
tensorboard --logdir runs/
```

Open `http://localhost:6006` in your browser.

**What to look for:**
1. **Untrained Baseline:** At Epoch 0/1, an untrained model has validation $\text{CER} \approx 1.0$ (or higher).
2. **Learning Confirmation:** Within the first 3–6 epochs, validation CER should show a clear downward trend (e.g. dropping to $0.70 - 0.85$ or lower).
3. **Completion criterion:** Once validation CER clearly drops below the untrained baseline, tick off:
   - **Action:** `[x] Monitor validation CER; it must fall clearly below that of an untrained model`.

---

## Step 5: Review Trigger (Curriculum Contingency)

- **Condition:** If after 5–8 epochs the validation CER has **not improved** (flatlining near $\ge 1.0$), do not waste prolonged compute.
- **Action:** Raise this at the project review meeting to activate card **W4-05 (Visual pre-training curriculum)**.
- If it is improving steadily, continue training until convergence.

---

## Step 6: Evaluate Test Performance Across SNR

Once training stops (either by early stopping or reaching maximum epochs), locate the best checkpoint: `runs/b2_visual_only_<timestamp>/best.pt`.

Run the evaluation script across all test SNR levels (clean, 10 dB, 5 dB, 0 dB, -5 dB):

```bash
python -m scripts.evaluate --checkpoint runs/b2_visual_only_<timestamp>/best.pt
```

**Expected Behavior & Diagnostic Value:**
- **Constant Performance Across SNR:** Because B2 relies exclusively on lip video, its CER and WER across clean, 10 dB, 5 dB, 0 dB, and -5 dB should remain roughly constant (unlike audio models that degrade sharply as noise increases).
- **Completion criterion:**
  - Evaluator prints table and logs metrics to `experiments/results.csv`.
  - Detailed report written to `reports/b2_visual_only_<timestamp>/`.
  - **Action:** Tick off `[x] Evaluate (performance should be roughly constant across SNR by design)` and `[x] Done when: B2 results are logged`.

---

## Step 7: Push Final Results Back

Commit the generated evaluation results and push to GitHub:

```bash
git add experiments/results.csv reports/
git commit -m "W3-05: Log final B2 visual-only baseline evaluation results"
git push origin cole
```
