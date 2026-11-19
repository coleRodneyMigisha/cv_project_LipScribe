# W3-06 Technical Report: Concatenation Fusion Baseline (B3)

## 1. Executive Summary
Task **W3-06** implements and prepares the **B3 Concatenation Fusion Baseline** (`B3Concat`), fulfilling Section 8 of the implementation document. B3 serves as the non-adaptive multimodal fusion reference: it evaluates whether directly concatenating acoustic and visual features provides performance gains across noise conditions prior to introducing cross-attention or gated weighting mechanisms.

---

## 2. Model Architecture (`B3Concat`)
Implemented in [`lipscribe/models/fusion_concat.py`](file:///home/kingcole/Documents/uniAssignments/year_3_sem_1/cv_group_4/lipscribe/models/fusion_concat.py):
1. **Audio Encoder Front-End:** Conv1D + 2-layer BiLSTM processing 80-dimensional log-Mel spectrograms $\to [B, T, 512]$.
2. **Visual Encoder Front-End:** 3D-CNN + 2-layer BiLSTM processing $96 \times 96$ mouth crops $\to [B, T, 512]$.
3. **Concatenation & Fusion Projection:**
   - Both stream features are temporal aligned to common length $T$.
   - Concatenated along feature dimension $\to [B, T, 1024]$.
   - Linear projection $\text{Linear}(1024 \to 256) \to \text{LayerNorm}(256) \to \text{ReLU} \to \text{Dropout}(0.3)$.
4. **CTC Sequence Head:**
   - Single-layer BiLSTM ($128$ units/dir $\to 256$ units) followed by vocabulary projection $\text{Linear}(256 \to 28) \to \text{log\_softmax} \to [T, B, 28]$.

---

## 3. Configuration & Overfit Testing Harness
- **Configuration:** [`configs/b3_concat.yaml`](file:///home/kingcole/Documents/uniAssignments/year_3_sem_1/cv_group_4/configs/b3_concat.yaml) mirrors B1 and B2 hyperparameters (AdamW, $\text{lr}=3\times 10^{-4}$, warmup 1 epoch, cosine decay across 60 epochs, patience 8).
- **Overfit Test Utility:** [`scripts/check_b3_overfit.py`](file:///home/kingcole/Documents/uniAssignments/year_3_sem_1/cv_group_4/scripts/check_b3_overfit.py) tests model capacity on dual-stream audio-visual batches.
- **Unit Testing:** Automated unit tests in [`tests/test_fusion_concat.py`](file:///home/kingcole/Documents/uniAssignments/year_3_sem_1/cv_group_4/tests/test_fusion_concat.py) pass:
  - Model instantiation and parameter verification ($> 7\text{M}$ trainable parameters).
  - Config factory resolution via `build_model({"name": "b3_concat"})`.
  - Batch forward pass tensor-shape contract compliance ($[T, B, 28]$ log-probabilities with finite values).

---

## 4. Execution Command for Alienware GPU Training
To launch full B3 training on the Alienware workstation:
```bash
python -m scripts.train --config configs/b3_concat.yaml
```
To evaluate after training:
```bash
python -m scripts.evaluate --checkpoint runs/b3_concat_<timestamp>/best.pt
```
