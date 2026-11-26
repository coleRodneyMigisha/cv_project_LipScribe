# Faith Ablation Results

**Generated:** 2026-11-19 17:07:37 UTC  
**Repository root:** `/home/Lipscribe-team/Desktop/cv_project_LipScribe`  
**Run directory:** `/home/Lipscribe-team/Desktop/cv_project_LipScribe/runs/faith_ablations/20261119_200658`  
**Optimizer steps per ablation:** 50  

---

## Summary

| Ablation | Status | Description |
|----------|--------|-------------|
| **X1 – No modality dropout** | PASS | Removes modality dropout; both audio and visual streams always present. |
| **X2 – Concat fusion** | PASS | Simple concatenation fusion instead of the default attention/fusion block. |
| **X3 – 2-D frontend** | PASS | Uses a pure 2-D convolutional visual frontend. |

---

## Detailed results

### X1 – No modality dropout

- **Config:** `configs/x1_no_modality_dropout.yaml`
- **Status:** PASS
- **Log file:** `/home/Lipscribe-team/Desktop/cv_project_LipScribe/runs/faith_ablations/20261119_200658/x1_no_modality_dropout.log`
- **Description:** Removes modality dropout; both audio and visual streams always present.

- **Last step message:** `Reached requested optimizer step limit: 50`

<details>
<summary>Last 30 lines of log</summary>

```
Run directory: /home/Lipscribe-team/Desktop/cv_project_LipScribe/runs/faith_ablations/20261119_200658/x1_no_modality_dropout_20261119-200700
Initialised visual_encoder from /home/Lipscribe-team/Desktop/cv_project_LipScribe/checkpoints/b2_visual_only_v2_best.pt (136 tensors)
Initialised audio_encoder from /home/Lipscribe-team/Desktop/cv_project_LipScribe/checkpoints/b1_audio_only_best.pt (30 tensors)
Reached requested optimizer step limit: 50

============================================================
Training finished! Best epoch: -1 (Mean Val WER: inf, CER: inf)
Best model checkpoint saved at: /home/Lipscribe-team/Desktop/cv_project_LipScribe/runs/faith_ablations/20261119_200658/x1_no_modality_dropout_20261119-200700/best.pt
Convenience copy saved at:      /home/Lipscribe-team/Desktop/cv_project_LipScribe/checkpoints/x1_no_modality_dropout_best.pt
============================================================

```
</details>

---

### X2 – Concat fusion

- **Config:** `configs/x2_concat_fusion.yaml`
- **Status:** PASS
- **Log file:** `/home/Lipscribe-team/Desktop/cv_project_LipScribe/runs/faith_ablations/20261119_200658/x2_concat_fusion.log`
- **Description:** Simple concatenation fusion instead of the default attention/fusion block.

- **Last step message:** `Reached requested optimizer step limit: 50`

<details>
<summary>Last 30 lines of log</summary>

```
Run directory: /home/Lipscribe-team/Desktop/cv_project_LipScribe/runs/faith_ablations/20261119_200658/x2_concat_fusion_20261119-200713
Initialised visual_encoder from /home/Lipscribe-team/Desktop/cv_project_LipScribe/checkpoints/b2_visual_only_v2_best.pt (136 tensors)
Initialised audio_encoder from /home/Lipscribe-team/Desktop/cv_project_LipScribe/checkpoints/b1_audio_only_best.pt (30 tensors)
/home/Lipscribe-team/Desktop/cv_project_LipScribe/scripts/train.py:337: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  scheduler.step()
Reached requested optimizer step limit: 50

============================================================
Training finished! Best epoch: -1 (Mean Val WER: inf, CER: inf)
Best model checkpoint saved at: /home/Lipscribe-team/Desktop/cv_project_LipScribe/runs/faith_ablations/20261119_200658/x2_concat_fusion_20261119-200713/best.pt
Convenience copy saved at:      /home/Lipscribe-team/Desktop/cv_project_LipScribe/checkpoints/x2_concat_fusion_best.pt
============================================================

```
</details>

---

### X3 – 2-D frontend

- **Config:** `configs/x3_2d_frontend.yaml`
- **Status:** PASS
- **Log file:** `/home/Lipscribe-team/Desktop/cv_project_LipScribe/runs/faith_ablations/20261119_200658/x3_2d_frontend.log`
- **Description:** Uses a pure 2-D convolutional visual frontend.

- **Last step message:** `Reached requested optimizer step limit: 50`

<details>
<summary>Last 30 lines of log</summary>

```
Run directory: /home/Lipscribe-team/Desktop/cv_project_LipScribe/runs/faith_ablations/20261119_200658/x3_2d_frontend_20261119-200726
visual_encoder.frontend.0.weight: (64, 1, 5, 7, 7) -> (64, 1, 1, 7, 7) by summing over the temporal axis (static clip gives the same response)
Initialised visual_encoder from /home/Lipscribe-team/Desktop/cv_project_LipScribe/checkpoints/b2_visual_only_v2_best.pt (136 tensors)
Initialised audio_encoder from /home/Lipscribe-team/Desktop/cv_project_LipScribe/checkpoints/b1_audio_only_best.pt (30 tensors)
Reached requested optimizer step limit: 50

============================================================
Training finished! Best epoch: -1 (Mean Val WER: inf, CER: inf)
Best model checkpoint saved at: /home/Lipscribe-team/Desktop/cv_project_LipScribe/runs/faith_ablations/20261119_200658/x3_2d_frontend_20261119-200726/best.pt
Convenience copy saved at:      /home/Lipscribe-team/Desktop/cv_project_LipScribe/checkpoints/x3_2d_frontend_best.pt
============================================================

```
</details>

---

## Environment

```
PY=3.11
TORCH=2.14.1+cu130
CUDA=True
```

## Notes

- This is a **smoke / fidelity** run (fixed number of optimizer steps).
- Full training curves and final metrics live under the individual run directories inside `/home/Lipscribe-team/Desktop/cv_project_LipScribe/runs/faith_ablations/20261119_200658`.
- Re-run with a larger `--steps` value (or remove the step limit) for a proper experimental comparison.
