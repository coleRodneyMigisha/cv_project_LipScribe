# Progress Report 3 — Week 3: Baselines B1–B3, Proposed Model P, Evaluation Engine, and Beam Search

**Course**: Computer Vision and Deep Learning · Group 4 · Uganda Christian University  
**Project**: LipScribe — Lip-Assisted Speech Recognition for Robust Live Captioning  
**Date**: 9 November 2026  

---

## 1. Summary

In Week 3 (and early Week 4 follow-through), the team completed all deliverables for Milestones **M3** and **M4** and finished full-corpus training and evaluation of Proposed **Model P** (**M5**). All three baseline families—audio-only **B1**, visual-only **B2** (v1 & v2), and concatenation fusion **B3** (v1 & v2)—along with all four initialisation variants of **Model P** (`LipScribeModel`: bidirectional cross-modal attention + per-frame gated fusion + learned missing-modality embedding $\mathbf{e}_{\text{miss}}$) have been trained on the full GRID corpus on our Alienware RTX 4080 SUPER workstation and evaluated across all five acoustic conditions ($\text{clean}, 10, 5, 0, -5\text{ dB}$) on the four held-out test speakers (`s4`, `s9`, `s14`, `s24`).

The empirical findings validate both our multimodal hypothesis and our architectural choices:
1. **B1 (Audio-Only)** achieves **`1.96%`** test WER on clean speech (`1.76%` with grammar decoding), but degrades nearly five-fold under severe noise (**`9.45%`** greedy, `8.35%` grammar at `-5 dB`).
2. **B2 (Visual-Only)** is completely invariant to acoustic noise (**`23.04%`** greedy, `21.03%` grammar test WER across all SNRs for v1), providing a noise-immune floor in adverse environments.
3. **B3 (Concatenation Fusion v2)** reduces B1's error at every SNR (**`1.16%`** clean, **`5.21%`** at `-5 dB` greedy; **`0.77%`** clean, **`3.83%`** at `-5 dB` with grammar decoding), but collapses catastrophically when a modality is missing at test time (`25.18%` without video on clean speech; `75.31%` without audio).
4. **Proposed Model P (`P4`, initialised from `B1 + B2 v2`)** achieves our best overall greedy test WER (**`1.02%`** clean, **`2.51%`** mean across SNRs) and best clean grammar WER (**`0.71%`**, `0.34%` CER). Crucially, Model P resolves B3's missing-modality fragility via $\mathbf{e}_{\text{miss}}$ and dynamic gating: dropping video yields **`1.88%`** clean WER (matching standalone B1's `1.96%`), dropping audio yields **`23.78%–31.57%`** WER (matching standalone B2), and the mean fusion gate $\bar{g}$ automatically shifts from **`0.804`** (clean) down to **`0.691`** (`-5 dB`) to rely more heavily on lip movements as acoustic noise intensifies.

---

## 2. Progress Against the Milestones

| Milestone | Due | Status and Progress |
| :--- | :---: | :--- |
| **M1 Environment and dataset verified** | Sun 25 Oct | **Done** — closed 29 Oct |
| **M2 Preprocessing and alignment** | Sun 1 Nov | **Done** — closed 3 Nov; full corpus extraction complete |
| **M3 & M4 Baselines B1, B2, B3** | Sun 8 Nov | **Done** — closed 4 Nov; all baselines (v1 & v2) trained and tested |
| **M5 Multimodal fusion (Model P)** | Sun 15 Nov | **Done** — cross-attention, gated fusion, $\mathbf{e}_{\text{miss}}$ & 4 full runs complete |
| **M6 & M7 Tuning and evaluation** | Sun 22 Nov | **In progress** — 4-way curriculum & modality ablations evaluated |
| **M8 & M9 Application, testing, demo** | Sun 29 Nov | **Planned** |

---

## 3. Week 3 & Early Week 4 Work Breakdown and Deliverables

| Card | Task | Owner | Status | Output and Evidence |
| :--- | :--- | :---: | :---: | :--- |
| **W3-01** | Complete B1 (audio) | Charis | **Done** | 52 epochs (best ep 43); test WER `1.96%` (clean) to `9.45%` (`-5 dB`) |
| **W3-02** | Evaluation script | Lisa | **Done** | `scripts/evaluate.py`: per-speaker metrics, hypotheses CSV, gate & ablation hooks |
| **W3-03** | Visual encoder V1 & V2 | Cole | **Done** | V1 3D-CNN + BiLSTM (`5.25M` params); V2 ResNet-18 front end (`15.0M` params) |
| **W3-04** | Visual augmentation | Faith | **Done** | Crop jitter, horizontal flip, scale jitter ($\pm 10\%$), spatial cutout, time masking |
| **W3-05** | Train B2 (visual-only) | Cole | **Done** | B2 v1 (28 ep): `23.04%` WER; B2 v2 (13 ep, 50m budget): `32.20%` WER |
| **W3-06** | Implement & train B3 | Faith | **Done** | B3 v1 (22 ep): `2.41%–7.35%` WER; B3 v2 (7 ep): `1.16%–5.21%` WER |
| **W3-07** | CTC beam & grammar | Isabelle | **Done** | Prefix beam search (`17.3 ms/utt`) & GRID Viterbi grammar decoder (`11.0 ms/utt`) |
| **W3-08** | Exp X5: audio features | Lisa | **Planned** | Deferred to Week 5 to prioritise baseline completion and Model P fusion |
| **W3-09** | Experiment tracking | Nehum | **Done** | `results.csv`, per-run `history.csv`, `plot_all_model_comparisons.py` |
| **W3-10** | Gradio Mode A skeleton | Isabelle | **In progress** | Preprocessing and video playback wired; upgrading to Model P4 checkpoint |
| **W4-01–05** | Proposed Model P & curriculum | Cole / Charis | **Done** | `fusion_attention.py`, 8-utt overfit ($\text{CER}=0$), 4 full-corpus Model P runs (`P1–P4`) |

---

## 4. Official Held-Out Test Results Across All Models

Table 1 presents the official held-out test evaluation across all five noise conditions on the 3,998 utterances from unseen test speakers `s4`, `s9`, `s14`, and `s24` (validation speakers: `s11`, `s19`, `s26`).

| Model Configuration | Clean WER [CER] | `10 dB` WER [CER] | `5 dB` WER [CER] | `0 dB` WER [CER] | `-5 dB` WER [CER] | Mean Greedy WER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1 Audio-Only** (52 ep) | `1.96%` [`0.8%`] | `3.02%` [`1.2%`] | `3.79%` [`1.5%`] | `5.44%` [`2.4%`] | `9.45%` [`4.7%`] | `4.73%` |
| **B2 Visual-Only v1** (`Conv3D`, 28 ep) | `23.04%` [`14.0%`] | `23.04%` [`14.0%`] | `23.04%` [`14.0%`] | `23.04%` [`14.0%`] | `23.04%` [`14.0%`] | `23.04%` |
| **B2 Visual-Only v2** (`ResNet18`, 13 ep) | `32.20%` [`18.6%`] | `32.20%` [`18.6%`] | `32.20%` [`18.6%`] | `32.20%` [`18.6%`] | `32.20%` [`18.6%`] | `32.20%` |
| **B3 Concat Fusion v1** (Scratch, 22 ep) | `2.41%` [`1.1%`] | `2.88%` [`1.4%`] | `3.36%` [`1.6%`] | `4.29%` [`2.2%`] | `7.35%` [`3.8%`] | `4.06%` |
| **B3 Concat Fusion v2** (`B1+B2v2`, 7 ep) | `1.16%` [`0.5%`] | `1.66%` [`0.7%`] | `1.98%` [`0.8%`] | `2.81%` [`1.3%`] | **`5.21%` [`2.6%`]** | `2.56%` |
| **Model P1** (Scratch `Conv3D`, 11 ep) | `12.58%` [`6.4%`] | `13.05%` [`6.6%`] | `13.55%` [`7.0%`] | `14.49%` [`7.6%`] | `16.28%` [`8.6%`] | `13.99%` |
| **Model P2** (Init `B2v1` only, 11 ep) | `18.19%` [`10.7%`] | `18.20%` [`10.7%`] | `18.53%` [`10.9%`] | `18.93%` [`11.2%`] | `19.85%` [`11.8%`] | `18.74%` |
| **Model P3** (Init `B1+B2v1 Conv3D`, 10 ep) | `2.13%` [`0.8%`] | `2.67%` [`1.0%`] | `3.27%` [`1.2%`] | `4.33%` [`1.8%`] | `7.30%` [`3.4%`] | `3.94%` |
| **Model P4** (Init `B1+B2v2 ResNet18`, 6 ep) | **`1.02%` [`0.4%`]** | **`1.53%` [`0.6%`]** | **`1.87%` [`0.8%`]** | **`2.79%` [`1.2%`]** | `5.34%` [`2.5%`] | **`2.51%`** |

---

## 5. Technical Highlights: ResNet-18 (V2), Modality Ablations, and Grammar Decoding

### 5.1 Why `ResNet-18` (V2) Outperforms `3D-CNN` (V1) in Multimodal Fusion
Although standalone **B2 v2** (`resnet18_bilstm`, capped at a 50-minute / 13-epoch budget with aggressive cutout and time masking) recorded a higher standalone WER (`32.20%`) than 28-epoch **B2 v1** (`23.04%`), our controlled 30-minute full-corpus comparison between **Model P3** (`B1 + B2 v1 Conv3D`) and **Model P4** (`B1 + B2 v2 ResNet-18`) proves that the `ResNet-18` visual encoder provides substantially superior representations for multimodal fusion:
- **52.1% lower clean test WER and 37.8% lower validation WER**: Holding the audio checkpoint (`B1`), fusion module, learning rate, and 30-minute training budget identical, swapping `B2 v1` (`Conv3D`) for `B2 v2` (`ResNet-18`) cuts validation WER from `14.17%` to **`8.82%`** and greedy test WER from `2.13%` to **`1.02%`** (clean) and from `7.30%` to **`5.34%`** (`-5 dB`).
- **Architectural reason**: `v1` flattens a $96 \times 6 \times 6 = 3{,}456$ feature map after only 3 plain convolutions, making it sensitive to speaker lighting and crop shifts. `v2` combines a 5-frame 3D spatiotemporal stem (`Conv3d 5×7×7`) with per-clip zero-mean/unit-variance normalisation (`input_norm="clip"`), a 16-layer 2D residual trunk with skip connections, and global `AdaptiveAvgPool2d(1)`, producing speaker-invariant viseme features (see [`docs/resnet_vs_conv3d_justification.md`](file:///home/Lipscribe-team/Desktop/cv_project_LipScribe/docs/resnet_vs_conv3d_justification.md)).

### 5.2 Missing-Modality Robustness and Noise-Adaptive Gating (Model P vs. B3 v2)

| Model / Test Condition | Clean | `10 dB` | `5 dB` | `0 dB` | `-5 dB` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model P4**: Audio + video (bimodal greedy) | **`1.02%`** | **`1.53%`** | **`1.87%`** | **`2.79%`** | `5.34%` |
| **Model P4**: Audio only (`--drop-modality visual`) | **`1.88%`** | **`2.99%`** | **`3.59%`** | **`5.44%`** | **`10.36%`** |
| **B3 v2**: Audio only (`--drop-modality visual`) | `25.18%` | `30.69%` | `33.55%` | `38.40%` | `47.41%` |
| **Model P2 / P3 / P4**: Video only (`--drop-modality audio`) | **`23.78%` (P2) / `25.65%` (P3) / `31.57%` (P4)** vs. **`75.31%` (B3 v2)** | — | — | — | — |
| **Model P4 Mean Fusion Gate $\bar{g}$** (audio weight) | **`0.804`** | **`0.786`** | **`0.769`** | **`0.740`** | **`0.691`** |

### 5.3 GRID Grammar-Constrained Decoding

| Model | Clean Greedy | Clean Grammar | `-5 dB` Greedy | `-5 dB` Grammar | Mean Grammar WER |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **B1 Audio-Only** | `1.96%` | `1.76%` | `9.45%` | `8.35%` | `4.26%` |
| **B2 Visual-Only (v1 / v2)** | `23.04%` / `32.20%` | `21.03%` / `28.26%` | `23.04%` / `32.20%` | `21.03%` / `28.26%` | `21.03%` / `28.26%` |
| **B3 Concat (v1 / v2)** | `2.41%` / `1.16%` | `1.86%` / `0.77%` | `7.35%` / **`5.21%`** | `5.87%` / **`3.83%`** | `3.22%` / **`1.82%`** |
| **Model P3 (`B1+B2v1 Conv3D`)** | `2.13%` | `1.25%` | `7.30%` | `5.20%` | `2.61%` |
| **Model P4 (`B1+B2v2 ResNet18`)** | **`1.02%`** | **`0.71%`** | `5.34%` | `3.95%` | `1.85%` |
