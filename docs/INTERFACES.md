# LipScribe module interfaces

**Card:** W1-11 · Module interface specification
**Owner:** Wakisa Isabelle · **Reviewers:** Cole Rodney Migisha, Charis Opol
**Source:** Implementation document, Table 3 (*Module responsibilities and interfaces*), plus the shape tables in Sections 5–7.

This file fixes the name, inputs, outputs and tensor shapes of every module, so that the data, model and application pairs can build in parallel and plug their parts together later. If you need to change an interface, change this file in the same pull request and tell the owners of the modules on either side.

## Conventions

| Symbol | Meaning | GRID value (3 s clip) |
|---|---|---|
| `B` | batch size | 16 (training default) |
| `T` | video frames = encoder frames (25 fps, 40 ms each) | 75 |
| `T_a` | audio feature frames (100 fps, 10 ms hop), `T_a = 4T` | 300 |
| `N` | audio samples at 16 kHz | 48 000 |
| `L` | transcript length in characters | about 35 at most |
| `V` | vocabulary size: blank (0), space, a–z | 28 |

- Shapes are written `[dim, dim, ...]`. Video tensors use PyTorch's `[B, C, T, H, W]` order.
- Encoder lengths are always `T(b) = ceil(T_a(b) / 4)`. The data loader and the audio encoder use this same formula.
- All shared preprocessing (cropping, features, normalisation, tokens) lives in `lipscribe/preprocessing` and `lipscribe/data`. Training and the demo import the same functions; nothing is re-implemented in the app.
- Vocabulary order (fixed): index 0 = blank, index 1 = space, indices 2–27 = `a`–`z`.

## 1. Preprocessing (`lipscribe/preprocessing`)

| Function | Input | Output | Owner |
|---|---|---|---|
| `read_frames(video_path)` | path to video file | `uint8[T, H, W, 3]` RGB frames at 25 fps, frame indices kept | Cole |
| `crop_mouths(video)` | `uint8[T, H, W, 3]` RGB frames (or a video path) | `crops: uint8[T, 96, 96]` grey, `valid: bool[T]` (False where Face Mesh failed) | Cole |
| `load_audio(path)` | path to audio or video file | `float32[N]` mono waveform at 16 kHz, level-normalised to −25 dBFS | Lisa |
| `mix_noise(clean, noise, snr_db)` | `float32[N]`, `float32[M]`, SNR in dB | `float32[N]` noisy waveform; video and transcript unchanged | Lisa |
| `logmel(wave)` | `float32[N]` (or `[B, N]` batched) | `float32[T_a, 80]` (or `[B, T_a, 80]`); 25 ms Hann window, 10 ms hop, 80 Mel bands, trimmed to `4T` | Lisa |
| `normalise(features, stats)` | `[.., T_a, 80]`, stats from `data/stats/feature_norm.npz` | same shape, standardised with training-set mean and std | Lisa |

Rules
- `crop_mouths` never drops frames (that would break audio sync). Missing detections are interpolated for short gaps; a clip with more than 10 % invalid frames is dropped from the dataset by the data pipeline, not by the crop function.
- Cached crops are stored as `data/processed/mouths/<speaker>/<utt_id>.npy`, `uint8[T, 96, 96]`.

## 2. Data (`lipscribe/data`)

| Function / class | Input | Output | Owner |
|---|---|---|---|
| `tokenizer.encode(text)` | normalised string, e.g. `"place green at b four now"` | `list[int]` in 1–27, never 0 | Lisa |
| `tokenizer.decode(ids)` | `list[int]` | `str` | Lisa |
| `GridDataset(manifest, splits, partition, ...)` | `data/manifest.csv`, `data/splits.csv`, one of `train` / `val` / `test` | one item per utterance (dictionary below) | Faith |
| `collate(items)` | list of item dictionaries | batch dictionary below | Faith |

**Item dictionary** (one utterance, returned by the dataset)

| Key | Type / shape | Notes |
|---|---|---|
| `utt_id` | `str` | e.g. `s1_bbaf2n` |
| `speaker` | `str` | e.g. `s1` |
| `video` | `uint8[T, 96, 96]` | cached mouth crops |
| `video_valid` | `bool[T]` | per-frame Face Mesh flag |
| `audio` | `float32[N]` | 16 kHz; noise already mixed in (random SNR for train, fixed files for val/test) |
| `snr_db` | `float` | `inf` or 30 for clean |
| `text` | `str` | normalised transcript |
| `target` | `int64[L]` | `tokenizer.encode(text)` |

**Batch dictionary** (returned by `collate`, consumed by `train.py`, `evaluate.py` and the model)

| Key | Type / shape | Notes |
|---|---|---|
| `utt_ids` | `list[str]` of length `B` | |
| `video` | `float32[B, 1, T, 96, 96]` | zero-padded to the longest `T` in the batch, scaled to 0–1 |
| `audio` | `float32[B, N_max]` | zero-padded waveforms; log-Mel is computed on the GPU from this |
| `audio_lengths` | `int64[B]` | true samples per item |
| `feat_lengths` | `int64[B]` | true `T_a(b)` |
| `lengths` | `int64[B]` | encoder lengths `T(b) = ceil(T_a(b) / 4)` |
| `mask` | `bool[B, T]` | True on real frames, False on padding |
| `targets` | `int64[sum(L_b)]` | all targets concatenated, as `nn.CTCLoss` expects |
| `target_lengths` | `int64[B]` | `L(b)` |
| `texts` | `list[str]` | reference transcripts for WER/CER |
| `snr_db` | `float32[B]` | |

> The implementation document describes what the collate function must produce but does not name the keys. The names above are a proposal; Faith and Charis should confirm them in review.

## 3. Models (`lipscribe/models`)

| Module | Input | Output | Owner |
|---|---|---|---|
| `VisualEncoder` (V1: 3× Conv3d + Linear 512 + 2-layer BiLSTM) | `video: [B, 1, T, 96, 96]`, `lengths: [B]` | `[B, T, 512]` | Cole |
| `AudioEncoder` (2× strided Conv1d + 2-layer BiLSTM) | `features: [B, T_a, 80]`, `feat_lengths: [B]` | `[B, T, 512]` (time reduced 4×) | Cole |
| `AttentionFusion` (Linear 512→256 + LayerNorm per stream, A→V and V→A attention, 4 heads, gate) | `a: [B, T, 512]`, `v: [B, T, 512]`, `mask: [B, T]` | `fused: [B, T, 256]`, `gate: [B, T]` (per-frame weight; near 1 = audio, near 0 = lips) | Cole |
| Sequence model (1-layer BiLSTM, 128/dir) | `[B, T, 256]` | `[B, T, 256]` | Cole |
| `CTCHead` (Linear 256→28 + log-softmax) | `[B, T, 256]` | `log_probs: [T, B, 28]` (time-first, as `nn.CTCLoss` expects) | Cole |
| `LipScribeModel.forward(batch)` | batch dictionary | `{"log_probs": [T, B, 28], "lengths": [B], "gate": [B, T]}` | Cole |

Rules
- A missing stream (modality dropout in training, or no face at inference) is replaced by the learned vector `e_miss ∈ R^256` repeated over all frames, never by masking every key (that produces NaN).
- Baselines use the same output contract: B1 (audio only), B2 (visual only) and B3 (concatenation) all return `log_probs: [T, B, 28]` and `lengths: [B]`, so decoding and evaluation work on every model unchanged. `gate` is omitted for models without one.
- Loss: `nn.CTCLoss(blank=0, zero_infinity=True)(log_probs, targets, lengths, target_lengths)`.

## 4. Decoding (`lipscribe/decoding`)

| Function | Input | Output | Owner |
|---|---|---|---|
| `collapse(ids)` | `list[int]` frame-level path | `str`: merge repeats, then drop blanks. `collapse([b,b,0,i,0,n,n]) == "bin"`, `collapse([e,0,e]) == "ee"` | Isabelle |
| `greedy_decode(log_probs, lengths)` | `[T, B, 28]`, `[B]` | `list[str]` of length `B` | Isabelle |
| `beam_search(log_probs, lengths, beam_width=10)` | `[T, B, 28]`, `[B]` | `list[str]` of length `B`; no language model | Isabelle |

## 5. Evaluation (`lipscribe/evaluation`)

| Function | Input | Output | Owner |
|---|---|---|---|
| `wer(refs, hyps)` | `list[str]`, `list[str]` | `float`, `(S + D + I) / N` over words | Isabelle |
| `cer(refs, hyps)` | `list[str]`, `list[str]` | `float`, same on characters | Isabelle |
| `scripts/evaluate.py --config ... --checkpoint ...` | config YAML, checkpoint, test partition | results CSV: model, SNR (clean, 10, 5, 0, −5 dB), WER, CER | Lisa |

## 6. Application (`lipscribe/app`)

| Function | Input | Output | Owner |
|---|---|---|---|
| `load_model(checkpoint_path, config_path)` | paths | model in `eval()` mode plus feature-normalisation stats | Isabelle |
| `transcribe(video_path) -> str` | path to a video clip with audio | caption string. Internally: `read_frames` → `crop_mouths` → `load_audio` → `logmel` → `normalise` → model → `beam_search` | Isabelle |
| `scripts/transcribe.py <video>` | command-line wrapper around `transcribe` | prints the caption | Isabelle |
| Gradio Mode A (`lipscribe/app/demo.py`) | uploaded or recorded clip, optional SNR | caption, mouth-crop strip, per-frame gate plot | Isabelle |
| Live Mode B | camera + microphone, 3 s window every 1 s | partial then final caption | Nehum |

Rules
- Inference runs under `model.eval()` and `torch.inference_mode()`, with the same preprocessing functions as training.
- If more than half of a window's frames have no face, the visual stream is replaced by `e_miss` and the caption is audio-only.

## Data flow at a glance

```
video file ──read_frames──► [T,H,W,3] ──crop_mouths──► [T,96,96] + valid[T] ─┐
                                                                             ├─ collate ─► VisualEncoder [B,T,512] ─┐
audio file ──load_audio──► [N] ──(mix_noise)──► [N] ─────────────────────────┘             AudioEncoder  [B,T,512] ─┤
                                         logmel + normalise on GPU: [B,T_a,80] ─────────────────┘                    │
                                                                                                                     ▼
                                                    AttentionFusion [B,T,256] ─► BiLSTM [B,T,256] ─► CTCHead [T,B,28]
                                                                                                                     │
                                                                  greedy_decode / beam_search ─► str ─► WER / CER, caption
```
