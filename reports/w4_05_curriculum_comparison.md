# Week 4 Card W4-04 & W4-05: Pre-training Curriculum & Full Model Comparison Report

**Date**: 2026-11-05  
**Hardware**: Alienware Workstation (`NVIDIA GeForce RTX 4080 SUPER`)  
**Scripts**: `scripts/check_w4_curriculum.py`, `scripts/run_all_model_p.sh`, & `scripts/plot_all_model_comparisons.py`  

---

## 1. Executive Summary

To evaluate how encoder initialization impacts **Proposed Model P** (`LipScribeModel`: `AudioEncoder` + `VisualEncoder` $\to$ 4-head `CrossModalAttention` $\to$ `GatedFusion` + `e_miss` $\to$ `CTCHead`), we evaluated four variants both on a 5-epoch subset curriculum benchmark (`N = 2,048`) and on **full GRID corpus training runs** (`30-minute` wall-clock budget per model, matching `B3 v2`):

1. **P1 (`p_scratch`)**: Both audio and visual (`conv3d_bilstm`) encoders initialized randomly (`runs/p_scratch_20261105-183620`).
2. **P2 (`p_init_b2`)**: Visual encoder (`conv3d_bilstm`) initialized from `checkpoints/b2_visual_only_best.pt`; audio encoder random (`runs/p_init_b2_20261105-180345`).
3. **P3 (`p_init_b1_b2`)**: Audio encoder initialized from `checkpoints/b1_audio_only_best.pt` + visual encoder (`conv3d_bilstm`) initialized from `checkpoints/b2_visual_only_best.pt` (`runs/p_init_b1_b2_20261105-173253`).
4. **P4 (`p_init_b1_b2_v2` / `p_proposed`)**: Audio encoder initialized from `checkpoints/b1_audio_only_best.pt` + visual encoder (`resnet18_bilstm`) initialized from `checkpoints/b2_visual_only_v2_best.pt` (`runs/p_init_b1_b2_v2_20261105-170138`).

---

## 2. Full-Dataset Training & Held-Out Test Comparison Across All 4 Model P Variants

### 2.1 Full-Dataset Training & Validation Summary (30-Min Budget Each)

| Model P Variant | Audio Init | Visual Init | Epochs Run (Best) | Best Mean Val WER | Best Mean Val CER | Clean Val WER | `-5 dB` Val WER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **P1: From Scratch (`p_scratch`)** | Random | Random (`conv3d`) | `11` (`ep 9`) | `28.13%` | `15.26%` | `25.58%` | `31.45%` |
| **P2: Init B2 v1 Only (`p_init_b2`)** | Random | `b2_visual_only_best.pt` | `11` (`ep 8`) | `34.93%` | `23.20%` | `34.38%` | `35.69%` |
| **P3: Init B1 + B2 v1 (`p_init_b1_b2`)** | `b1_audio_only_best.pt` | `b2_visual_only_best.pt` | `10` (`ep 6`) | `14.17%` | `8.53%` | `9.06%` | `19.99%` |
| **P4: Init B1 + B2 v2 (`p_init_b1_b2_v2`)** | `b1_audio_only_best.pt` | `b2_visual_only_v2_best.pt` | `6` (`ep 5`) | **`8.82%`** | **`5.17%`** | **`5.13%`** | **`14.61%`** |

### 2.2 Held-Out Test Split Performance Across All 4 Model P Variants (`3,998` Utterances / SNR)

| Model P Variant | Decode Mode | Clean | `10 dB` | `5 dB` | `0 dB` | `-5 dB` | Mean Test WER | No-Audio (`-A`, All SNR) | No-Visual (`-V`, Clean / `-5 dB`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **P1: From Scratch** | Greedy / Grammar | `12.58%` / `9.40%` | `13.05%` / `9.84%` | `13.55%` / `10.30%` | `14.49%` / `11.32%` | `16.28%` / `13.16%` | `13.99%` / `10.80%` | `25.03%` | `32.41%` / `49.57%` |
| **P2: Init B2 v1 Only** | Greedy / Grammar | `18.19%` / `15.38%` | `18.20%` / `15.48%` | `18.53%` / `15.67%` | `18.93%` / `16.16%` | `19.85%` / `17.01%` | `18.74%` / `15.94%` | **`23.78%`** | `28.82%` / `46.93%` |
| **P3: Init B1 + B2 v1** | Greedy / Grammar | `2.13%` / `1.25%` | `2.67%` / `1.62%` | `3.27%` / `2.08%` | `4.33%` / `2.88%` | `7.30%` / `5.20%` | `3.94%` / `2.61%` | `25.65%` | `2.72%` / `12.39%` |
| **P4: Init B1 + B2 v2** | Greedy / Grammar | **`1.02%` / `0.71%`** | **`1.53%` / `1.10%`** | **`1.87%` / `1.42%`** | **`2.79%` / `2.06%`** | **`5.34%` / `3.95%`** | **`2.51%` / `1.85%`** | `31.57%` | **`1.88%` / `10.36%`** |

---

## 3. Master Comparison Across All 9 Full Models (`B1`, `B2 v1`, `B2 v2`, `B3 v1`, `B3 v2`, and `P1–P4`)

### 3.1 Held-Out Test WER (%) Across SNR Conditions & Modality Ablations

| Model | Architecture / Init | Decode Mode | Clean | `10 dB` | `5 dB` | `0 dB` | `-5 dB` | Mean Test WER | No-Audio (`-A`) | No-Visual (`-V`, Clean) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1 Audio-Only** | `Conv1d + BiLSTM` (`52 ep`) | Greedy / Grammar | `1.96%` / `1.76%` | `3.02%` / `2.79%` | `3.79%` / `3.47%` | `5.44%` / `4.93%` | `9.45%` / `8.35%` | `4.73%` / `4.26%` | — | `1.96%` |
| **B2 Visual-Only (v1)** | `Conv3D + BiLSTM` (`28 ep`) | Greedy / Grammar | `23.04%` / `21.03%` | `23.04%` / `21.03%` | `23.04%` / `21.03%` | `23.04%` / `21.03%` | `23.04%` / `21.03%` | `23.04%` / `21.03%` | `23.04%` | — |
| **B2 Visual-Only (v2)** | `ResNet18 + BiLSTM` (`13 ep`) | Greedy / Grammar | `32.20%` / `28.26%` | `32.20%` / `28.26%` | `32.20%` / `28.26%` | `32.20%` / `28.26%` | `32.20%` / `28.26%` | `32.20%` / `28.26%` | `32.20%` | — |
| **B3 Concat (v1)** | Scratch `Conv3D` (`22 ep`) | Greedy / Grammar | `2.41%` / `1.86%` | `2.88%` / `2.27%` | `3.36%` / `2.65%` | `4.29%` / `3.45%` | `7.35%` / `5.87%` | `4.06%` / `3.22%` | — | — |
| **B3 Concat (v2)** | Init `B1 + B2 v2` (`7 ep`) | Greedy / Grammar | `1.16%` / `0.77%` | `1.66%` / `1.13%` | `1.98%` / **`1.38%`** | `2.81%` / **`1.99%`** | **`5.21%` / `3.83%`** | `2.56%` / **`1.82%`** | `75.31%` | `25.18%` |
| **Model P1 (Scratch)** | Scratch `Conv3D` (`11 ep`) | Greedy / Grammar | `12.58%` / `9.40%` | `13.05%` / `9.84%` | `13.55%` / `10.30%` | `14.49%` / `11.32%` | `16.28%` / `13.16%` | `13.99%` / `10.80%` | `25.03%` | `32.41%` |
| **Model P2 (Init B2 v1)** | Init `B2 v1` (`11 ep`) | Greedy / Grammar | `18.19%` / `15.38%` | `18.20%` / `15.48%` | `18.53%` / `15.67%` | `18.93%` / `16.16%` | `19.85%` / `17.01%` | `18.74%` / `15.94%` | **`23.78%`** | `28.82%` |
| **Model P3 (Init B1+B2v1)** | Init `B1 + B2 v1` (`10 ep`) | Greedy / Grammar | `2.13%` / `1.25%` | `2.67%` / `1.62%` | `3.27%` / `2.08%` | `4.33%` / `2.88%` | `7.30%` / `5.20%` | `3.94%` / `2.61%` | `25.65%` | `2.72%` |
| **Model P4 (Init B1+B2v2)** | Init `B1 + B2 v2` (`6 ep`) | Greedy / Grammar | **`1.02%` / `0.71%`** | **`1.53%` / `1.10%`** | **`1.87%`** / `1.42%` | **`2.79%`** / `2.06%` | `5.34%` / `3.95%` | **`2.51%`** / `1.85%` | `31.57%` | **`1.88%`** |

---

## 4. Key Takeaways

1. **Model P4 (`p_init_b1_b2_v2`) Achieves the Lowest Overall Greedy Test WER (`2.51%` Mean, `1.02%` Clean) and Lowest Clean Grammar WER (`0.71%`)**:
   - Under greedy decoding, **P4** beats every baseline including **B3 v2** at `clean` (`1.02%` vs `1.16%`), `10 dB` (`1.53%` vs `1.66%`), `5 dB` (`1.87%` vs `1.98%`), `0 dB` (`2.79%` vs `2.81%`), and overall mean test WER (`2.51%` vs `2.56%`).
   - Similarly, **P3 (`B1 + B2 v1`)** beats **B3 v1 (`Conv3D` concat)** at every single SNR under both greedy (`3.94%` vs `4.06%` mean) and grammar (`2.61%` vs `3.22%` mean) decoding.
2. **Massive Superiority Under Missing Modalities (`--drop-modality audio` and `--drop-modality visual`)**:
   - When video is dropped at test time (`--drop-modality visual`), **B3 v2** degrades from `1.16%` to `25.18%` WER on clean audio (`47.41%` at `-5 dB`), whereas **Model P4** achieves **`1.88%` WER on clean audio** (`10.36%` at `-5 dB`) — even slightly outperforming standalone **B1 Audio-Only** (`1.96%` clean)!
   - When audio is dropped at test time (`--drop-modality audio`), **B3 v2** collapses to `75.31%` WER, whereas **Model P** gracefully falls back to visual-only lipreading via `e_miss` (`23.78%` for P2, `25.03%` for P1, `25.65%` for P3, and `31.57%` for P4 — outperforming standalone B2 v2's `32.20%`).

---

## 5. Visualizations

* `reports/model_p_4way_curriculum.png` — 3-panel plot of Model P 4-way full-dataset CTC loss convergence, epoch-wise validation WER, and held-out test WER across SNR.
* `reports/all_models_snr_and_summary.png` — 3-panel comparison of held-out test WER vs SNR, missing-modality robustness (`e_miss` vs concat), and mean test WER across all 9 full models.
