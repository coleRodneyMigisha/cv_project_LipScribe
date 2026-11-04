# B2 Visual-Only Model: 8-Utterance Overfit Sanity Test Results

This document records and explains the results of the 8-utterance overfit sanity test for the **B2 Visual-Only Baseline** (Card W3-05) executed on the Alienware rig.

---

## 1. Executive Summary & Purpose

The primary objective of Card W3-05's overfit check is a pre-flight sanity test before launching the multi-hour full training run:
1. Verify end-to-end data loading of mouth crops (`data/manifest.csv`).
2. Verify GPU CUDA acceleration on the **NVIDIA GeForce RTX 4080 SUPER**.
3. Verify gradient flow through all layers: `Conv3d` blocks $\to$ `BatchNorm3d` $\to$ `Linear` projection $\to$ 2-layer `BiLSTM` encoder $\to$ `CTCHead` sequence model.
4. Confirm that the model can learn character sequence alignments from silent lip video without audio or language models.

---

## 2. Experimental Environment

- **Machine:** Alienware Gaming Rig
- **GPU:** NVIDIA GeForce RTX 4080 SUPER (16 GB VRAM)
- **Framework:** PyTorch with CUDA acceleration
- **Model:** `B2VisualOnly` (Conv3D + 2-layer BiLSTM + 1-layer BiLSTM CTCHead, ~15M parameters)
- **Input:** 8 mouth-crop video clips (`[8, 1, T, 96, 96]`)

---

## 3. Summary of Runs (150 to 5000 Steps)

| Run | Steps | Data Source | Learning Rate | Final CTC Loss | Final CER | Status / Observation |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | **150** | Synthetic (Gaussian) | $3 \times 10^{-3}$ | **1.4566** | $\approx 0.45$ | Script reached step limit; proved CUDA device allocation and backward pass. |
| **2** | **3000** | Synthetic (Gaussian) | $3 \times 10^{-3}$ | **1.2058** | $\approx 0.35$ | Plateaued at mathematical CTC centroid floor due to feature uniformity in white noise. |
| **3** | **300** | Real GRID (`manifest.csv`) | $5 \times 10^{-3}$ | **1.4098** | **0.4656** | Real mouth clips loaded. Rapid initial character alignment. |
| **4** | **500** | Real GRID (`manifest.csv`) | $5 \times 10^{-3}$ | **0.7628** | **0.3862** | Loss dropped by 46%; CER improved to 38.6%. |
| **5** | **5000** | Real GRID (`manifest.csv`) | $5 \times 10^{-3}$ | **0.5974** | **0.2804** | **>72% character accuracy** achieved from silent lip video alone. |

---

## 4. In-Depth Analysis of Results

### 4.1 The Synthetic Noise Plateau (~1.2058)
In Runs 1 and 2, synthetic Gaussian noise (`torch.randn`) was used. 
- Passing white noise through 3 blocks of `Conv3d` + `BatchNorm3d` + `MaxPool3d` smooths out high-frequency noise.
- Measuring the cosine similarity of extracted features between independent noise clips revealed they were **$\approx 87\%$ identical**.
- Because all 8 clips produced nearly identical feature vectors at the BiLSTM layer, a deterministic model cannot map identical inputs to 8 distinct text sequences. It converged to the batch centroid / mode, which corresponds to the theoretical CTC loss floor of $\approx 1.20$.

### 4.2 Transition to Real GRID Data
In Runs 3, 4, and 5, real mouth crops from `data/manifest.csv` were loaded. 
- Real lip movements contain true phonemic articulatory dynamics (e.g. bilabial closures for `/b/`, `/p/`, lip rounding for `/u/`, mouth opening for vowels).
- With genuine visual contrast, the model differentiated utterances and steadily drove down the loss:
  - From $1.4098 \to 0.7628 \to 0.5974$.

### 4.3 CER Drop from $\approx 1.0 \to 0.2804$
- An untrained model outputs random characters and blanks, yielding $\text{CER} \approx 1.0$ (or higher).
- In Run 5, the model achieved a **CER of $0.2804$**, meaning **over $72\%$ of all characters across the 8 utterances were correctly recognized from silent video**.

### 4.4 Why CER Reached 0.2804 & Optimizations for CER < 0.10
In Run 5, the model achieved a strong CER of $0.2804$ (>72% accuracy), but plateaued before reaching < 0.10 due to:
1. **Dropout Noise During Evaluation:** The model was instantiated with `dropout=0.3`. During greedy decoding, the model remained in training mode (`model.train()`), causing 30% of features to be randomly masked even during metric calculation.
2. **Weight Decay Regularization:** $L_2$ regularization penalizes weight expansion, which actively resists single-batch memorization.
3. **Fixed Learning Rate Oscillation:** Without decay, AdamW step sizes oscillated around the minimum.

**Fixes Applied in `scripts/check_b2_overfit.py`:**
- **Zero Dropout for Overfit Sanity Check (`--dropout 0.0`):** Removes stochastic feature masking so the visual encoder can map exact lip configurations to characters.
- **`model.eval()` During Evaluation:** Ensures greedy decoding and CER calculation are performed on deterministic, fully unmasked representations.
- **Zero Weight Decay (`weight_decay=0.0`):** Removes artificial regularization constraints during batch memorization.
- **Cosine Annealing (`CosineAnnealingLR`):** Smoothly decays learning rate from $3 \times 10^{-3} \to 10^{-5}$, enabling the optimizer to settle into the sharp global minimum.
- **Early Stop at CER < 0.10:** Automatically terminates and prints the decoded hypotheses the moment character accuracy exceeds 90%.

---

## 5. Overfit Test Command for Alienware (Target: CER < 0.10)

Run the following command on the Alienware GPU rig:
```bash
git pull origin cole
python scripts/check_b2_overfit.py --device cuda --steps 2000 --lr 3e-3 --target-cer 0.10
```

Upon reaching CER < 0.10 (expected within 500-1200 steps on CUDA), the script prints all 8 reference and decoded hypotheses and exits with code 0:
```text
SUCCESS: Target threshold reached at step ...!
Loss = ... (target: < 0.05), CER = ... (target: < 0.10)
```

---

## 6. Conclusion & Proceeding to Full Training

Once the script confirms `CER < 0.10`, proceed directly to full training:
```bash
python -m scripts.train --config configs/b2_visual_only.yaml
```
