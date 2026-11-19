# W3-08 (Experiment X5): audio feature configuration decision

Rule (card W3-08): keep A1 unless another configuration improves validation WER.

| Experiment | Features | Mean val WER | Mean val CER | Run |
|---|---|---:|---:|---|
| `b1_audio_only` | A1 (80 log-Mel) | 0.0772 | 0.0436 | `b1_audio_only_20261103-193743` |
| `b1_mfcc39` | A2 (13 MFCC + delta + delta-delta) | 0.0720 | 0.0391 | `b1_mfcc39_20261110-212519` |
| `b1_logmel_mfcc` | A3 (log-Mel + MFCC, 119 channels) | 0.0690 | 0.0378 | `b1_logmel_mfcc_20261110-223919` |

**Decision: Switch to A3 (log-Mel + MFCC, 119 channels) (`b1_logmel_mfcc`): mean val WER 0.0690 improves on A1's 0.0772.**
