# Technical Justification: Why V2 Used a Hybrid 3D-Conv + ResNet-18 Visual Encoder (`resnet18_bilstm`) Instead of a 3-Block 3D-CNN (`conv3d_bilstm`)

**Project**: LipScribe — Lip-Assisted Speech Recognition for Robust Live Captioning  
**Module**: [`lipscribe/models/visual_encoder.py`](file:///home/Lipscribe-team/Desktop/cv_project_LipScribe/lipscribe/models/visual_encoder.py) (`VisualEncoder` vs. `ResNetVisualEncoder`)  
**Date**: 9 November 2026  

---

## 1. Executive Summary

Our initial visual encoder (**V1**, `conv3d_bilstm`) used three plain 3D convolutional blocks followed by a flattened linear projection (`96 × 6 × 6 = 3,456 -> 512`) and a 2-layer BiLSTM. While V1 trained quickly on the GRID corpus, its evaluation on unseen speakers revealed a **2.5× inter-speaker Word Error Rate (WER) gap** (`13.3%` on test speaker `s24` vs. `33.7%` on `s9`, and `35.74%` on validation speakers vs. `23.04%` on test speakers).

To overcome this speaker-generalisation bottleneck and provide richer frame-level embeddings for multimodal cross-attention, **V2** (`resnet18_bilstm`) adopts the canonical lipreading front-end architecture introduced by Stafylakis & Tzimiropoulos (2017) and Martinez et al. (2020): **per-clip intensity normalisation (`input_norm="clip"`) + a single 5-frame 3D spatiotemporal stem (`Conv3d 5×7×7`) + a deep 2D ResNet-18 residual trunk + global spatial average pooling + a 2-layer BiLSTM**.

In a controlled, head-to-head full-corpus multimodal experiment (**Model P3** vs. **Model P4**, both trained for 30 minutes with the exact same `B1` audio checkpoint, cross-attention fusion module, hyperparameters, and random seed), replacing `conv3d_bilstm` with `resnet18_bilstm`:
- Reduced **validation WER by 37.8% relative** (`14.17%` $\to$ **`8.82%`**).
- Reduced **clean held-out test WER by 52.1% relative** (`2.13%` $\to$ **`1.02%`** greedy; `1.25%` $\to$ **`0.71%`** with grammar decoding).
- Reduced **$-5\text{ dB}$ noisy test WER by 26.8% relative** (`7.30%` $\to$ **`5.34%`** greedy; `5.20%` $\to$ **`3.95%`** grammar).
- Increased **noise-adaptive gate responsiveness by 2.25×** ($\Delta \bar{g} = -0.113$ vs. $-0.050$ from clean to $-5\text{ dB}$).

---

## 2. Architectural Comparison: V1 (`conv3d_bilstm`) vs. V2 (`resnet18_bilstm`)

| Design Dimension | V1: `VisualEncoder` (`conv3d_bilstm`) | V2: `ResNetVisualEncoder` (`resnet18_bilstm`) | Technical Advantage of V2 |
| :--- | :--- | :--- | :--- |
| **Input Normalisation** | Global corpus pixel mean/std (`input_norm="none"`) | Per-clip zero-mean, unit-variance over valid frames (`input_norm="clip"`) | Eliminates per-speaker lighting, skin-tone, and exposure bias before the first convolution |
| **Spatiotemporal Stem** | 3 stacked `Conv3d` blocks ($3\times 5\times 5$, $3\times 5\times 5$, $3\times 3\times 3$) | Single `Conv3d(1, 64, kernel=5×7×7, stride=1×2×2)` + `MaxPool3d(1×3×3)` | Captures $5$-frame ($200\text{ ms}$) co-articulatory viseme motion right at the input while avoiding 3D FLOP explosion in deeper layers |
| **Spatial Feature Extractor** | Only 3 convolutional layers total (`32 -> 64 -> 96` channels); no residual connections | 16-layer 2D ResNet-18 trunk (8 `BasicBlock`s: `64 -> 128 -> 256 -> 512` channels) with identity skip connections | Deep hierarchical feature extraction (lip contours $\to$ teeth/tongue visibility $\to$ viseme posture) without vanishing gradients |
| **Spatial-to-Vector Bottleneck** | Flatten `96 × 6 × 6 = 3,456` $\to$ dense `Linear(3456, 512)` (`1.77M` params in one layer) | `AdaptiveAvgPool2d(1)` over $3\times 3\times 512$ feature maps $\to$ `512`-d vector | Enforces spatial translation invariance to minor mouth-crop centring jitter and removes the overfitting-prone flatten matrix |
| **Regularisation** | Feature-vector dropout (`0.3`) + crop jitter ($\pm 4\text{ px}$) + horizontal flip | `Dropout2d(0.2)` + scale jitter ($\pm 10\%$) + spatial cutout + temporal masking (2 masks $\le 6$ frames) | Forces the encoder to rely on distributed lip dynamics rather than memorising speaker-specific static mouth shapes |
| **Output Interface** | `[B, T, 512]` (via 2-layer BiLSTM) | `[B, T, 512]` (via 2-layer BiLSTM) | Drop-in compatible with `B2`, `B3`, and `Model P` |

---

## 3. Four Core Reasons for Switching from Plain 3D-CNN to Hybrid ResNet-18

### 3.1 Solving the Flatten-Projection Overfitting Bottleneck
In V1 (`conv3d_bilstm`), the third `MaxPool3d` outputs a spatial grid of $6 \times 6$ with $96$ channels per frame. Flattening this into a $3{,}456$-dimensional vector and feeding it into `nn.Linear(3456, 512)` ties specific weights to exact $(x, y)$ spatial coordinates. If an unseen speaker has slightly thicker lips, a wider jaw, or a $2\text{--}3\text{ px}$ MediaPipe landmark shift, the activations land on different spatial indices and degrade recognition.  
In V2 (`resnet18_bilstm`), progressive residual downsampling ($96 \to 48 \to 24 \to 12 \to 6 \to 3$) followed by `nn.AdaptiveAvgPool2d(1)` pools spatially before the BiLSTM, making the $512$-d frame representation invariant to small spatial shifts and morphological differences across speakers.

### 3.2 Depth and Residual Conditioning Without 3D Compute Explosion
Distinguishing visually similar visemes (homophenes such as `/b, p, m/` or `/f, v/` in GRID letters) requires fine-grained spatial discrimination (e.g., subtle teeth-on-lip contact or oral cavity opening). Three plain convolutional layers lack the receptive field hierarchy and non-linear capacity to separate these cues reliably across 34 speakers. By using a **hybrid 3D-stem + 2D-ResNet** design:
- Short-term temporal motion across adjacent video frames ($5$ frames $= 200\text{ ms}$) is fused immediately in the first `Conv3d(5×7×7)` stem.
- All subsequent spatial refinement occurs in 2D residual blocks (`_BasicBlock` with zero-initialised final `BatchNorm2d` weights so residual branches start as exact identity mappings), achieving 17 convolutional layers of depth at a fraction of the memory and compute cost of a full 3D ResNet-18.

### 3.3 Per-Clip Photometric Normalisation (`input_norm="clip"`)
V1 relied solely on corpus-level pixel normalisation (`data/stats/feature_norm.npz`). However, GRID speakers were recorded under varying studio illumination and skin-tone contrast levels. V2 introduces `clip_normalise()`, which standardises each utterance's valid (unpadded) frames to zero mean and unit variance prior to the 3D stem. This guarantees that the convolutional filters respond to **relative lip motion and shadowing** rather than static speaker brightness.

### 3.4 Superior Alignment for Cross-Modal Attention and Gated Fusion
Why did standalone `B2 v2` (`32.20%` test WER) score higher error than standalone `B2 v1` (`23.04%` test WER) in Table 1, yet dominate in both `B3 v2` and `Model P4`?
1. **Training Budget & Augmentation Asymmetry in Standalone B2**: Standalone `B2 v1` trained for **28 epochs** with light augmentation, whereas standalone `B2 v2` was capped at a **50-minute time budget (13 epochs)** while subjected to heavy regularisation (spatial cutout, two 6-frame time masks, scale jitter, and `Dropout2d`).
2. **Controlled Head-to-Head Multimodal Proof (`P3` vs. `P4`)**: When both visual checkpoints were used to initialise **Model P** alongside the exact same `B1` audio checkpoint and trained under an identical 30-minute full-corpus budget:

| Metric | Model P3 (`B1 + B2 v1 [Conv3D]`) | Model P4 (`B1 + B2 v2 [ResNet-18]`) | Relative Improvement of `ResNet-18` |
| :--- | :---: | :---: | :---: |
| **Best Mean Validation WER / CER** | `14.17%` / `8.53%` | **`8.82%` / `5.17%`** | **$-37.8\%$ WER / $-39.4\%$ CER** |
| **Test WER — Clean (Greedy / Grammar)** | `2.13%` / `1.25%` | **`1.02%` / `0.71%`** | **$-52.1\%$ Greedy / $-43.2\%$ Grammar** |
| **Test WER — `10 dB` (Greedy / Grammar)** | `2.67%` / `1.62%` | **`1.53%` / `1.10%`** | **$-42.7\%$ Greedy / $-32.1\%$ Grammar** |
| **Test WER — `5 dB` (Greedy / Grammar)** | `3.27%` / `2.08%` | **`1.87%` / `1.42%`** | **$-42.8\%$ Greedy / $-31.7\%$ Grammar** |
| **Test WER — `0 dB` (Greedy / Grammar)** | `4.33%` / `2.88%` | **`2.79%` / `2.06%`** | **$-35.6\%$ Greedy / $-28.5\%$ Grammar** |
| **Test WER — `-5 dB` (Greedy / Grammar)** | `7.30%` / `5.20%` | **`5.34%` / `3.95%`** | **$-26.8\%$ Greedy / $-24.0\%$ Grammar** |
| **Mean Test WER Across All 5 SNRs** | `3.94%` / `2.61%` | **`2.51%` / `1.85%`** | **$-36.3\%$ Greedy / $-29.1\%$ Grammar** |
| **No-Visual Fallback (`--drop-modality visual`, Clean)** | `2.72%` | **`1.88%`** | **$-30.9\%$ WER** |
| **Fusion Gate Shift $\Delta \bar{g}$ ($\text{Clean} \to -5\text{ dB}$)** | `0.884 -> 0.834` (`-0.050`) | **`0.804 -> 0.691` (`-0.113`)** | **$2.25\times$ stronger visual gating under noise** |

As the last row demonstrates, in **Model P3** (`Conv3D`), the gated fusion layer keeps the audio gate $\bar{g}$ high (`0.884` to `0.834`) because the shallow 3D-CNN features are less discriminative on unseen speakers. In **Model P4** (`ResNet-18`), the fusion gate actively trusts the visual stream—assigning $19.6\%$ weight to visual features in clean conditions and escalating to **$30.9\%$ weight at $-5\text{ dB}$**—which directly drives P4's state-of-the-art accuracy across all noise levels.
