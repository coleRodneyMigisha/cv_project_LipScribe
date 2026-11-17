# Models

LipScribe has two families of models. Each family has a frozen final model (decided in the
16 Nov re-plan):

- **GRID models** (Weeks 1–4) are trained only on the GRID corpus and use a 28-symbol character
  CTC. The final model is **P4**.
- **Real-world models** ("Flix", Weeks 4–5) are trained on films and TED talks and use pretrained
  encoders and word-piece CTC. The final model is **ws_v5**.

## GRID models

Configs are in `configs/`, checkpoints in `checkpoints/` on the Alienware
(`~/Desktop/cv_project_LipScribe/checkpoints/`).

| Label | Config / experiment | What it is |
|---|---|---|
| B1 | `b1_audio_only` | Audio only: log-Mel 80 → 2 strided Conv1d → 2-layer BiLSTM → CTC |
| B1-A2, B1-A3 | `b1_mfcc39`, `b1_logmel_mfcc` | X5: B1 with MFCC, and with log-Mel + MFCC features |
| B2 v1 | `b2_visual_only` | Lips only: three Conv3d blocks → BiLSTM → CTC |
| B2 v2 | `b2_visual_only_v2` | Lips only: Conv3d 5×7×7 stem + per-frame ResNet-18 + BiLSTM |
| B3 v1, B3 v2 | `b3_concat`, `b3_concat_v2` | Concatenation fusion of the two streams; v2 starts from B1 and B2 v2 and uses modality dropout |
| P1–P4 | `p_scratch`, `p_init_b2`, `p_init_b1_b2`, `p_init_b1_b2_v2` | Proposed model P (cross-modal attention, per-frame gate, e_miss, modality dropout) with four initialisations: from scratch, from B2 v1, from B1 + B2 v1, and from B1 + B2 v2 |
| **P4** | `p_init_b1_b2_v2` | **Final GRID model** (`checkpoints/p_proposed_best.pt`) |
| X1 | `x1_no_modality_dropout` | Ablation: P4 without modality dropout |
| X2 | `x2_concat_fusion` | Ablation: P4 with concatenation in place of attention and gate |
| X3 | `x3_2d_frontend` | Ablation: P4 with a per-frame 2D front end in place of the 3D one |

## Real-world models (films and TED)

These were trained on the Alienware in `~/Desktop/lipscribe_flix`. The decision logs are
`reports/flix_v4/`, `reports/flix_v5/` and `reports/flix_v6/decision_log.md`.

| Version | Model | What it is |
|---|---|---|
| v1–v2 | P4 retrained on film clips | GRID architecture on films: about 100 % WER (too little data, 51-word design). Week 4 report |
| v4 | `audio` | WavLM Base+ fine-tuned, character CTC |
| v4 | `visual_old`, `visual_c` | Auto-AVSR lip encoder fine-tuned; `visual_c` uses Auto-AVSR-style crops and word pieces |
| v4 | `fusion`, `fusion_wp` | First fusions; `fusion_wp` is lip-anchored and over-trusts the lips |
| v4 | `fusion_wp_audio` = **twin** | Audio-only twin with the same word-piece head (the fair audio-only baseline) |
| v4 | `fusion_gated`, `fusion_gated_ws` = **ws** | Reliability-gated fusion; `ws` (warm start) starts as the twin and learns where the lips help |
| v5 | **twin_v5**, **ws_v5** | The same twin and ws, retrained on clips re-cut with 0.32 s of padding and relabelled from Whisper word timestamps |
| v6 | ws_v6 | ws_v5 retrained on audio that does not belong to the face (silent, talk-over, replace): a negative result |
| v6 | lips_v5 | ws_v5 with its audio switched off: the lips-only reference |

**Final real-world model: ws_v5**
(`runs/flix_v5/flix_v5_fusion_gated_ws_20261116-093641/best.pt`), with **twin_v5** as its
audio-only baseline.
