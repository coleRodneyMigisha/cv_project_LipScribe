# Flix v4: decision and setback log

A running record of what we tried, what went wrong and why we changed course, so that it can go into the learning-curve section of the Week 5 report. Newest entries go at the bottom.

## 2026-11-13: v3 diagnosis and change of plan

**State of v3 at 15:24.** v3 was a sweep over 11 ResNet variations of the P4 recipe, trained on film + TED (41,415 clips). Checkpoints were chosen on the mean of film-val and TED-val CER.
- 2 of 11 trials were finished: r00_frozen at 78.9 % CER (best epoch 31/52), and r01_ft0.1 at 77.0 % CER (22/43).
- r02_ft0.3 was at epoch 30, with a best of 79.2 % CER.
- The sweep was expected to finish on 16–17 Nov.

**Finding.** The v2 test results split by modality showed that the audio branch is the bottleneck, not the lips:

| v2 on films (clean audio) | WER % | CER % |
|---|---:|---:|
| Lips + sound | 103.2 | 72.5 |
| Sound only | 103.5 | 72.4 |
| Lips only | 97.1 | 87.6 |

- On films, the fused model does no better than its audio branch.
- 72 % CER from clean audio is far too high for real-world use.
- The GRID-built audio encoder (B1: log-mel input, trained on 51 words) and the ImageNet ResNet18 visual encoder are too small and too narrow to learn open-vocabulary speech from about 28 h of data.
- v3 only varies the visual side, so it cannot fix this.

**Setback noted.** We first misread the identical clean and 0 dB WER as a sign that the model was ignoring the audio. When WER is near 100 %, it hardly changes with audio quality, so that comparison cannot show anything. The split-by-modality test results are the right diagnostic.

**Decision.** We stopped v3 and moved to a three-stage, transfer-learning plan that mirrors what worked for Model P. When we first built Model P, the from-scratch version performed poorly, and initialising it from separately trained encoders fixed that.
1. **Audio-only model.** Fine-tune a pretrained self-supervised speech encoder on film + TED. Target: WER below 10 %.
2. **Visual-only model.** Fine-tune a pretrained lip-reading encoder on film + TED. Target: WER below 30 %.
3. **Fused model.** Initialise from stages 1 and 2 and train the fusion and CTC head the Model P way. Target: WER below 10 % on clean audio, with a clear gain over audio-only in noise.

Stages 1 and 2 train in parallel on the Alienware, and stage 3 starts automatically when both finish.

**Rules agreed.**
- Pretrained models may only be used as starting points for transfer learning. The weights we report must come from our own fine-tuning.
- No Whisper-based audio model. The labels are Whisper transcripts, so such a model would be scored against its own output.

## 2026-11-13 (afternoon): the TED labels were misaligned

**Setback found.** Unmodified, Auto-AVSR scored 89 % WER on film-val and 99 % on TED-val (greedy subword CTC). Zooming our tighter crops out (0.5–0.85) changed nothing, so crop scale was not the cause.

We then ran Whisper large-v3-turbo on the clips' own audio and scored it against our labels:
- **Film-val:** 11.6 % WER, so the film labels are fine.
- **TED-val:** the labels often describe different words. For example, label "its impacting the students trying to go to college", Whisper "intentionally impacting the very people who need it most". The lip reader's guess ("…people who need most") matched the audio, not the label.

The TED subtitle cue times do not line up with the speech in the cut clips. In each clip, audio and video come from the same window, so they are in sync with each other; only the text is wrong.

**Effect on earlier results.** About 38,000 of v3's 40,219 training clips are TED, and half of v3's checkpoint selection (TED-val) used these labels, so v3 was trained and selected on noisy text. Correct labels would very likely have improved v3, but not to the targets: v2, trained mostly on film clips, was still at 72 % CER from sound alone. The small GRID-built encoders remain the main limit.

**Decision.**
- Relabel every training, TED-val and TED-test clip with Whisper from its own audio (`scripts/flix_v4_relabel.py` → `data/flix_v4`).
- Keep the film-val and film-test references as they are.
- TED numbers from now on are measured against Whisper references; film-test remains the human-referenced check.
- There is no need to download the TED videos again or redo the crops, since the crops themselves were fine.

**Memory.** At batch 16, the visual model ran out of GPU memory while v3 still held 5.7 GB. The visual and fusion runs use batch 8 × 2 gradient accumulation (the same effective batch). If they still run out, the pipeline retries at batch 4 × 4.

## Automatic pipeline log (scripts/flix_v4_run.sh)

- 2026-11-13 16:00 Stopped v3 (r00, r01 finished; r02_ft0.3 stopped mid-run). Its runs and logs stay in runs/flix_v3, reports/flix_v3.
- 2026-11-13 16:00 Relabelling with Whisper large-v3-turbo from each clip's own audio (TED cue text was misaligned).
- 2026-11-13 18:01 Relabel done: train_mix: 38677/40219 kept, old-label WER vs Whisper 95.3%; ted_val: 821/855 kept, old-label WER vs Whisper 84.8%; ted_test: 1122/1153 kept, old-label WER vs Whisper 84.3%
- 2026-11-13 18:02 Zero-shot Auto-AVSR (unmodified, greedy subword CTC) on relabelled val: ted_val@zoom1 WER 76.8%/CER 56.0%, film_val@zoom1 WER 89.4%/CER 72.2%, ted_val@zoom0.85 WER 68.0%/CER 48.8%, film_val@zoom0.85 WER 85.3%/CER 67.5%, ted_val@zoom0.7 WER 67.1%/CER 47.5%, film_val@zoom0.7 WER 85.1%/CER 67.3%, ted_val@zoom0.6 WER 68.1%/CER 47.9%, film_val@zoom0.6 WER 84.0%/CER 68.3%. Chosen zoom: 0.7.
- 2026-11-13 18:02 Stage audio started (38678 training rows).
- 2026-11-13 18:03 Stage visual started (38678 training rows).

## 2026-11-13 (evening): the audio model overfits, so we queue a regularised retrain

At epoch 12 the first audio run (WavLM Base+) had a best mean val WER of 34.0 % (film clean 35 %, TED clean 20 %). Val loss had been rising since epoch 3 (1.01 → 1.20) while train loss kept falling (0.70 → 0.40). This is overfitting, and it leaves the model far from the 10 % target.

**Decision.** Run the audio model again (configs/flix_v4/audio_reg.yaml) after the first run stops, starting again from WavLM Base+ with stronger regularisation:
- time masking 0.05 → 0.15
- new channel masking 0.05
- LayerDrop 0.05 → 0.1
- head dropout 0.1 → 0.2
- learning rate halved (head 1.5e-4, WavLM 3e-5)
- weight decay 0.01 → 0.05

The pipeline keeps whichever run has the lower mean val WER for fusion, and reports the other on the test sets as audio_first. scripts/flix_v4_handover.sh adds this stage to the instance that is already running.

## 2026-11-13 (evening): re-crop the mouths the Auto-AVSR way, and keep its word-piece output layer

**Finding.** After 5 epochs, the fine-tuned visual model (old crops at zoom 0.7, with a new 28-character CTC layer) was still behind the unmodified Auto-AVSR on TED-val WER (77 % vs 67 %), although its CER was lower.

We identified two causes:
1. **The crops did not match what Auto-AVSR was trained on.** Its crops are aligned to a mean face and show the mouth from nose to chin. Ours were tight lip boxes whose scale varied.
2. **We discarded the pretrained word-piece CTC layer** and trained a new character layer from random weights.

**Test.** We re-implemented Auto-AVSR's crop with our face landmarks (lipscribe/preprocessing/avsr_crop.py). The steps are: smooth the landmarks over 12 frames, align the 8 stable points to the 20-words mean face at 256×256, and cut 96×96 around the mouth.

On 142 TED-val clips, the unmodified model scored:

| Crops | WER | CER |
|---|---:|---:|
| Old crops, zoom 1 | 67.9 % | 48.4 % |
| Old crops, zoom 0.7 | 58.3 % | 41.1 % |
| New crops | 45.3 % | 31.0 % |

**Decision.**
- Re-crop all clips with scripts/flix_recrop.py:
  - TED: download each talk again on the Alienware, re-cut the same clip windows, crop, then delete the video.
  - Films: crop on the laptop, where the films are, and ship only the crops.
- Clip ids, audio, Whisper labels and splits are unchanged; the new manifests are in data/flix_v4c.
- The new visual stage visual_c keeps Auto-AVSR's word-piece CTC layer (head: avsr_ctc, data.tokenizer: subword).
  - Before training, our own pipeline reproduces the zero-shot 45.3 % WER on these clips.
  - The head learning rate is 1e-4 because the layer is pretrained.
- The old-crop visual run keeps training until the re-crop is ready. It is then stopped and reported as the visual_old baseline.
- Fusion is initialised from visual_c and uses the new crops.
- GRID test (grid_test.csv) still uses the old GRID crops.
- 2026-11-13 21:27 Stage audio done: best epoch 25/30: mean val WER 30.1%, CER 16.8% (wer_film_clean 29.9%, wer_film_0 45.4%, wer_ted_clean 18.6%, wer_ted_0 26.6%). Run: runs/flix_v4/flix_v4_audio_20261113-180217
- 2026-11-13 21:27 Stage audio_reg started (38678 training rows).
- 2026-11-13 23:38 Re-crop done (Auto-AVSR alignment, 37341 TED + 2241 film clips ok): film_val: 69/69 clips with new crops;flix_test: 43/43 clips with new crops;ted_val: 821/821 clips with new crops;train_mix: 38677/38677 clips with new crops;ted_test: 1122/1122 clips with new crops;
- 2026-11-13 23:38 Old-crop visual run stopped as the baseline visual_old: best epoch 15/21: mean val WER 80.0%, CER 49.7% (wer_film_clean 87.4%, wer_ted_clean 72.5%). Run: runs/flix_v4/flix_v4_visual_20261113-180347
- 2026-11-13 23:38 Stage visual_c started (38678 training rows).
- 2026-11-14 00:49 Stage audio_reg done: best epoch 23/30: mean val WER 29.2%, CER 16.3% (wer_film_clean 26.0%, wer_film_0 45.1%, wer_ted_clean 19.0%, wer_ted_0 26.6%). Run: runs/flix_v4/flix_v4_audio_reg_20261113-212724
- 2026-11-14 00:49 Audio model for fusion: audio_reg (val WER 0.2919 vs first run 0.3012).
- 2026-11-14 02:34 Stage visual_c done: best epoch 4/13: mean val WER 61.5%, CER 44.2% (wer_film_clean 75.0%, wer_ted_clean 47.9%). Run: runs/flix_v4/flix_v4_visual_c_20261113-233839
- 2026-11-14 02:34 Stage fusion started (38678 training rows).
- 2026-11-14 04:34 Stage fusion done: best epoch 0/6: mean val WER 31.0%, CER 16.5% (wer_film_clean 30.4%, wer_film_0 45.4%, wer_ted_clean 20.3%, wer_ted_0 27.8%). Run: runs/flix_v4/flix_v4_fusion_20261114-023452
- 2026-11-14 04:37 Test results (greedy CTC) written to reports/flix_v4/test_results.md:

| model | test | SNR | WER % | CER % | n |
|---|---|---|---:|---:|---:|
| audio | film_test | clean | 32.1 | 19.4 | 43 |
| audio | film_test | 0 | 41.4 | 25.6 | 43 |
| audio | film_test | -5 | 53.4 | 35.0 | 43 |
| audio | ted_test | clean | 16.4 | 8.6 | 1122 |
| audio | ted_test | 0 | 24.0 | 12.8 | 1122 |
| audio | ted_test | -5 | 34.3 | 19.4 | 1122 |
| visual | film_test | clean | 60.6 | 50.4 | 43 |
| visual | ted_test | clean | 48.2 | 34.3 | 1122 |
| fusion | film_test | clean | 34.9 | 20.1 | 43 |
| fusion | film_test | 0 | 42.6 | 23.4 | 43 |
| fusion | film_test | -5 | 48.6 | 30.0 | 43 |
| fusion-no_visual | film_test | clean | 33.7 | 20.5 | 43 |
| fusion-no_visual | film_test | 0 | 43.0 | 24.0 | 43 |
| fusion-no_visual | film_test | -5 | 51.4 | 32.3 | 43 |
| fusion-no_audio | film_test | clean | 95.2 | 75.5 | 43 |
| fusion | ted_test | clean | 18.1 | 9.2 | 1122 |
| fusion | ted_test | 0 | 26.6 | 13.8 | 1122 |
| fusion | ted_test | -5 | 36.4 | 19.5 | 1122 |
| fusion-no_visual | ted_test | clean | 17.9 | 9.2 | 1122 |
| fusion-no_visual | ted_test | 0 | 26.5 | 13.9 | 1122 |
| fusion-no_visual | ted_test | -5 | 36.3 | 20.1 | 1122 |
| fusion-no_audio | ted_test | clean | 93.4 | 65.5 | 1122 |
| audio_first | film_test | clean | 36.9 | 21.5 | 43 |
| audio_first | film_test | 0 | 47.0 | 29.7 | 43 |
| audio_first | film_test | -5 | 54.6 | 35.6 | 43 |
| audio_first | ted_test | clean | 15.8 | 8.7 | 1122 |
| audio_first | ted_test | 0 | 24.4 | 13.5 | 1122 |
| audio_first | ted_test | -5 | 35.2 | 20.1 | 1122 |

Old-crop baseline:

| model | test | SNR | WER % | CER % | n |
|---|---|---|---:|---:|---:|
| visual_old | film_test | clean | 73.9 | 53.7 | 43 |
| visual_old | film_test | 0 | 73.9 | 53.7 | 43 |
| visual_old | film_test | -5 | 73.9 | 53.7 | 43 |
| visual_old | ted_test | clean | 72.7 | 43.8 | 1122 |
| visual_old | ted_test | 0 | 72.7 | 43.8 | 1122 |
| visual_old | ted_test | -5 | 72.7 | 43.8 | 1122 |

- 2026-11-14 04:37 Flix v4 pipeline finished.

## Fusion v2 (14 Nov 2026)

- Probe of v4_fusion: the lips were unused. Giving each clip another clip's video left WER unchanged (TED -5 dB 36.1 vs 36.5), and the gate was about 0.9 towards audio.
- Causes:
  - A new character head; the visual encoder had been trained for word pieces.
  - The pretrained audio dominated from epoch 0.
  - The "visual" branch already carried attended audio.
  - Little noise in training.
- v4_fusion_wp:
  - Every head uses Auto-AVSR word pieces at 25 fps.
  - Fusion is lip-anchored: f = v + MLP([a; v]), zero-initialised, so the model starts as visual_c.
  - Auxiliary audio and visual CTC losses.
  - Frozen WavLM.
  - Training SNRs weighted to <= 0 dB, down to -10 dB.
  - Checkpoint selection includes -5 dB.
- Success criterion: beat the audio-only twin (the same model with the lips always hidden) in noise, not just use the lips.
- 2026-11-14 11:03 Stage fusion_wp started (configs/flix_v4/fusion_wp.yaml).
- 2026-11-14 14:26 Stage fusion_wp done: best epoch 11/12: mean val WER 38.0% (film_clean 36.3%, film_0 45.9%, film_-5 54.4%, ted_clean 27.7%, ted_0 30.7%, ted_-5 32.8%). Run: runs/flix_v4/flix_v4_fusion_wp_20261114-110340
- 2026-11-14 14:26 Stage fusion_wp_audio started (configs/flix_v4/fusion_wp_audio.yaml).
- 2026-11-14 15:59 Stage fusion_wp_audio done: best epoch 11/12: mean val WER 33.3% (film_clean 27.3%, film_0 42.3%, film_-5 52.8%, ted_clean 19.6%, ted_0 25.2%, ted_-5 32.6%). Run: runs/flix_v4/flix_v4_fusion_wp_audio_20261114-142621

Fusion v2 (word pieces, lip-anchored) vs its audio-only twin, test sets:

| model | test | SNR | WER % | CER % | n |
|---|---|---|---:|---:|---:|
| fusion_wp | film_test | clean | 34.5 | 26.4 | 43 |
| fusion_wp | film_test | 0 | 39.0 | 30.4 | 43 |
| fusion_wp | film_test | -5 | 45.4 | 35.1 | 43 |
| fusion_wp | film_test | -10 | 47.8 | 39.3 | 43 |
| fusion_wp-no_visual | film_test | clean | 32.9 | 22.4 | 43 |
| fusion_wp-no_visual | film_test | 0 | 39.8 | 28.2 | 43 |
| fusion_wp-no_visual | film_test | -5 | 50.6 | 39.8 | 43 |
| fusion_wp-no_visual | film_test | -10 | 59.4 | 53.2 | 43 |
| fusion_wp-no_audio | film_test | clean | 59.0 | 49.9 | 43 |
| fusion_wp | ted_test | clean | 26.9 | 20.2 | 1122 |
| fusion_wp | ted_test | 0 | 29.9 | 22.3 | 1122 |
| fusion_wp | ted_test | -5 | 33.2 | 24.2 | 1122 |
| fusion_wp | ted_test | -10 | 36.2 | 26.3 | 1122 |
| fusion_wp-no_visual | ted_test | clean | 22.1 | 15.9 | 1122 |
| fusion_wp-no_visual | ted_test | 0 | 27.8 | 20.0 | 1122 |
| fusion_wp-no_visual | ted_test | -5 | 35.7 | 26.6 | 1122 |
| fusion_wp-no_visual | ted_test | -10 | 47.4 | 38.8 | 1122 |
| fusion_wp-no_audio | ted_test | clean | 48.1 | 34.2 | 1122 |
| fusion_wp_audio | film_test | clean | 30.1 | 20.5 | 43 |
| fusion_wp_audio | film_test | 0 | 39.0 | 25.6 | 43 |
| fusion_wp_audio | film_test | -5 | 46.2 | 36.3 | 43 |
| fusion_wp_audio | film_test | -10 | 59.4 | 51.5 | 43 |
| fusion_wp_audio | ted_test | clean | 17.4 | 11.4 | 1122 |
| fusion_wp_audio | ted_test | 0 | 23.4 | 15.7 | 1122 |
| fusion_wp_audio | ted_test | -5 | 31.5 | 22.1 | 1122 |
| fusion_wp_audio | ted_test | -10 | 43.8 | 34.0 | 1122 |

Lip-use probe (WER %, real / shuffled / zeroed video / visual removed):
```
{
 "film_test|clean": {
  "real": 34.5,
  "shuffled": 61.8,
  "zero": 36.1,
  "e_miss": 32.9,
  "n": 43
 },
 "film_test|0": {
  "real": 39.0,
  "shuffled": 68.3,
  "zero": 41.4,
  "e_miss": 39.8,
  "n": 43
 },
 "film_test|-5": {
  "real": 45.4,
  "shuffled": 72.7,
  "zero": 50.6,
  "e_miss": 50.6,
  "n": 43
 },
 "film_test|-10": {
  "real": 47.8,
  "shuffled": 82.3,
  "zero": 62.2,
  "e_miss": 59.4,
  "n": 43
 },
 "ted_test|clean": {
  "real": 26.9,
  "shuffled": 80.2,
  "zero": 26.1,
  "e_miss": 22.1,
  "n": 1122
 },
 "ted_test|0": {
  "real": 29.9,
  "shuffled": 84.1,
  "zero": 31.6,
  "e_miss": 27.8,
  "n": 1122
 },
 "ted_test|-5": {
  "real": 33.2,
  "shuffled": 89.0,
  "zero": 39.7,
  "e_miss": 35.7,
  "n": 1122
 },
 "ted_test|-10": {
  "real": 36.2,
  "shuffled": 93.7,
  "zero": 50.3,
  "e_miss": 47.4,
  "n": 1122
 }
}```
- 2026-11-14 16:03 Fusion v2 finished.
- 2026-11-14 16:25 Combining fusion_wp and fusion_wp_audio frame by frame (weight chosen on validation).

Frame-level combination of fusion_wp (lips + audio) and fusion_wp_audio, chosen on validation: **loglinear@0.1** (weight on the audio-visual model). Test WER %:

| test | SNR | audio-only twin | fusion_wp | combined |
|---|---|---:|---:|---:|
| film_test | clean | 30.12 | 34.54 | **29.72** |
| film_test | 0 | 38.96 | 38.96 | **37.35** |
| film_test | -5 | 46.18 | 45.38 | **45.78** |
| film_test | -10 | 59.44 | 47.79 | **58.23** |
| ted_test | clean | 17.36 | 26.93 | **17.16** |
| ted_test | 0 | 23.35 | 29.87 | **22.79** |
| ted_test | -5 | 31.5 | 33.18 | **30.8** |
| ted_test | -10 | 43.83 | 36.25 | **43.53** |
- 2026-11-14 16:27 Stage fusion_wp_ft started (configs/flix_v4/fusion_wp_ft.yaml).
- 2026-11-14 18:13 Stopped fusion_wp_ft after 5 epochs. It barely moved: best val TED clean 27.4 % vs the twin at 19.6 %, because the lip-anchored residual cannot be unlearnt. best.pt kept in runs/flix_v4/flix_v4_fusion_wp_ft_20261114-162730.

## Fusion v3: symmetric reliability-gated fusion (14 Nov 2026)

- fusion_wp was lip-anchored (f = v + MLP) and over-trusted the lips. A fine-tune could not undo that.
- Combining the outputs at a fixed weight helped at every SNR, but only slightly, and it lost the large −10 dB gain.
- v4_fusion_gated builds in neither stream:
  - Frozen audio stream from the twin and frozen lip stream from fusion_wp.
  - A 768-channel per-frame gate on both features plus each stream's own confidence (CTC entropy and blank probability).
  - A zero-initialised synergy MLP.
  - 3 Transformer layers and a new word-piece head.
  - Balanced modality dropout (0.15 / 0.15), noise levels spread evenly from clean to −10 dB.
- Success criterion: no worse than the twin on clean audio, and better in noise.
- 2026-11-14 18:14 Stage fusion_gated started (configs/flix_v4/fusion_gated.yaml).
- 2026-11-14 19:54 building the LibriSpeech LM failed (logs/flix_v4_lm_big.log)
- 2026-11-14 19:59 LMs built (training transcripts only, and 40M LibriSpeech words + transcripts x5, 3/4-gram singletons pruned): flix_train_4gram.arpa ted_val perplexity 248; flix_train_4gram.arpa film_val perplexity 241; flix_libri_4gram.arpa ted_val perplexity 325; flix_libri_4gram.arpa film_val perplexity 227
- 2026-11-14 20:20 Stage fusion_gated done: best epoch 13/15: mean val WER 36.2% (film_clean 29.9%, film_0 42.5%, film_-5 51.3%, film_-10 56.7%, ted_clean 21.0%, ted_0 24.7%, ted_-5 28.4%, ted_-10 34.7%). Run: runs/flix_v4/flix_v4_fusion_gated_20261114-181447

Fusion v3 (fusion_gated) test results, with fusion_wp and the twin for comparison:

| model | test | SNR | WER % | CER % | n |
|---|---|---|---:|---:|---:|
| fusion_wp | film_test | clean | 34.5 | 26.4 | 43 |
| fusion_wp | film_test | 0 | 39.0 | 30.4 | 43 |
| fusion_wp | film_test | -5 | 45.4 | 35.1 | 43 |
| fusion_wp | film_test | -10 | 47.8 | 39.3 | 43 |
| fusion_wp-no_visual | film_test | clean | 32.9 | 22.4 | 43 |
| fusion_wp-no_visual | film_test | 0 | 39.8 | 28.2 | 43 |
| fusion_wp-no_visual | film_test | -5 | 50.6 | 39.8 | 43 |
| fusion_wp-no_visual | film_test | -10 | 59.4 | 53.2 | 43 |
| fusion_wp-no_audio | film_test | clean | 59.0 | 49.9 | 43 |
| fusion_wp | ted_test | clean | 26.9 | 20.2 | 1122 |
| fusion_wp | ted_test | 0 | 29.9 | 22.3 | 1122 |
| fusion_wp | ted_test | -5 | 33.2 | 24.2 | 1122 |
| fusion_wp | ted_test | -10 | 36.2 | 26.3 | 1122 |
| fusion_wp-no_visual | ted_test | clean | 22.1 | 15.9 | 1122 |
| fusion_wp-no_visual | ted_test | 0 | 27.8 | 20.0 | 1122 |
| fusion_wp-no_visual | ted_test | -5 | 35.7 | 26.6 | 1122 |
| fusion_wp-no_visual | ted_test | -10 | 47.4 | 38.8 | 1122 |
| fusion_wp-no_audio | ted_test | clean | 48.1 | 34.2 | 1122 |
| fusion_wp_audio | film_test | clean | 30.1 | 20.5 | 43 |
| fusion_wp_audio | film_test | 0 | 39.0 | 25.6 | 43 |
| fusion_wp_audio | film_test | -5 | 46.2 | 36.3 | 43 |
| fusion_wp_audio | film_test | -10 | 59.4 | 51.5 | 43 |
| fusion_wp_audio | ted_test | clean | 17.4 | 11.4 | 1122 |
| fusion_wp_audio | ted_test | 0 | 23.4 | 15.7 | 1122 |
| fusion_wp_audio | ted_test | -5 | 31.5 | 22.1 | 1122 |
| fusion_wp_audio | ted_test | -10 | 43.8 | 34.0 | 1122 |
| fusion_gated | film_test | clean | 30.9 | 22.1 | 43 |
| fusion_gated | film_test | 0 | 37.3 | 26.5 | 43 |
| fusion_gated | film_test | -5 | 43.0 | 31.9 | 43 |
| fusion_gated | film_test | -10 | 44.6 | 35.1 | 43 |
| fusion_gated-no_visual | film_test | clean | 32.9 | 21.2 | 43 |
| fusion_gated-no_visual | film_test | 0 | 39.8 | 26.4 | 43 |
| fusion_gated-no_visual | film_test | -5 | 47.0 | 36.2 | 43 |
| fusion_gated-no_visual | film_test | -10 | 59.8 | 50.4 | 43 |
| fusion_gated-no_audio | film_test | clean | 61.0 | 51.5 | 43 |
| fusion_gated | ted_test | clean | 18.5 | 12.6 | 1122 |
| fusion_gated | ted_test | 0 | 22.8 | 16.2 | 1122 |
| fusion_gated | ted_test | -5 | 27.4 | 19.2 | 1122 |
| fusion_gated | ted_test | -10 | 32.9 | 23.6 | 1122 |
| fusion_gated-no_visual | ted_test | clean | 18.4 | 12.2 | 1122 |
| fusion_gated-no_visual | ted_test | 0 | 24.4 | 16.6 | 1122 |
| fusion_gated-no_visual | ted_test | -5 | 32.7 | 22.9 | 1122 |
| fusion_gated-no_visual | ted_test | -10 | 44.6 | 34.4 | 1122 |
| fusion_gated-no_audio | ted_test | clean | 49.2 | 35.3 | 1122 |

Lip-use probe for fusion_gated:
```
{
 "film_test|clean": {
  "real": 30.9,
  "shuffled": 44.2,
  "zero": 32.1,
  "e_miss": 32.9,
  "n": 43,
  "gate_mean": 0.623,
  "gate_min_clip": 0.25
 },
 "film_test|0": {
  "real": 37.3,
  "shuffled": 56.6,
  "zero": 40.2,
  "e_miss": 39.8,
  "n": 43,
  "gate_mean": 0.586,
  "gate_min_clip": 0.276
 },
 "film_test|-5": {
  "real": 43.0,
  "shuffled": 57.8,
  "zero": 48.6,
  "e_miss": 47.0,
  "n": 43,
  "gate_mean": 0.546,
  "gate_min_clip": 0.187
 },
 "film_test|-10": {
  "real": 44.6,
  "shuffled": 75.9,
  "zero": 60.6,
  "e_miss": 59.8,
  "n": 43,
  "gate_mean": 0.504,
  "gate_min_clip": 0.145
 },
 "ted_test|clean": {
  "real": 18.5,
  "shuffled": 35.7,
  "zero": 19.2,
  "e_miss": 18.4,
  "n": 1122,
  "gate_mean": 0.46,
  "gate_min_clip": 0.279
 },
 "ted_test|0": {
  "real": 22.8,
  "shuffled": 47.2,
  "zero": 25.4,
  "e_miss": 24.4,
  "n": 1122,
  "gate_mean": 0.421,
  "gate_min_clip": 0.143
 },
 "ted_test|-5": {
  "real": 27.4,
  "shuffled": 59.0,
  "zero": 34.2,
  "e_miss": 32.7,
  "n": 1122,
  "gate_mean": 0.389,
  "gate_min_clip": 0.097
 },
 "ted_test|-10": {
  "real": 32.9,
  "shuffled": 70.6,
  "zero": 45.5,
  "e_miss": 44.6,
  "n": 1122,
  "gate_mean": 0.346,
  "gate_min_clip": 0.068
 }
}```
- 2026-11-14 20:24 Fusion v3 finished.

Beam search + LM, own-transcript LM:

Greedy vs beam search (width 32) without and with the 4-gram LM `lm/flix_train_4gram.arpa`. alpha/beta chosen per model on validation. Test WER %:

| model | alpha | beta | test | SNR | greedy | beam, no LM | beam + LM |
|---|---:|---:|---|---|---:|---:|---:|
| twin | 0.3 | 1.0 | film_test | -10 | 59.44 | 58.23 | **57.43** |
| twin | 0.3 | 1.0 | film_test | -5 | 46.18 | 46.59 | **46.18** |
| twin | 0.3 | 1.0 | film_test | 0 | 38.96 | 39.36 | **39.36** |
| twin | 0.3 | 1.0 | film_test | clean | 30.12 | 29.72 | **29.72** |
| twin | 0.3 | 1.0 | ted_test | -10 | 43.83 | 43.08 | **42.38** |
| twin | 0.3 | 1.0 | ted_test | -5 | 31.5 | 30.63 | **30.29** |
| twin | 0.3 | 1.0 | ted_test | 0 | 23.35 | 22.63 | **22.3** |
| twin | 0.3 | 1.0 | ted_test | clean | 17.36 | 16.53 | **16.35** |
| av | 0.3 | 1.0 | film_test | -10 | 47.79 | 46.18 | **48.19** |
| av | 0.3 | 1.0 | film_test | -5 | 45.38 | 44.58 | **43.78** |
| av | 0.3 | 1.0 | film_test | 0 | 38.96 | 37.35 | **37.75** |
| av | 0.3 | 1.0 | film_test | clean | 34.54 | 33.33 | **34.54** |
| av | 0.3 | 1.0 | ted_test | -10 | 36.25 | 35.33 | **34.74** |
| av | 0.3 | 1.0 | ted_test | -5 | 33.18 | 32.13 | **31.68** |
| av | 0.3 | 1.0 | ted_test | 0 | 29.87 | 28.74 | **28.35** |
| av | 0.3 | 1.0 | ted_test | clean | 26.93 | 25.87 | **25.48** |
| visual | 0.3 | 1.0 | film_test | clean | 60.64 | 60.64 | **58.63** |
| visual | 0.3 | 1.0 | ted_test | clean | 48.25 | 48.22 | **48.17** |
| gated | 0.3 | 2.0 | film_test | -10 | 44.58 | 44.58 | **45.38** |
| gated | 0.3 | 2.0 | film_test | -5 | 42.97 | 41.37 | **40.96** |
| gated | 0.3 | 2.0 | film_test | 0 | 37.35 | 35.34 | **35.34** |
| gated | 0.3 | 2.0 | film_test | clean | 30.92 | 29.72 | **30.12** |
| gated | 0.3 | 2.0 | ted_test | -10 | 32.93 | 32.11 | **31.93** |
| gated | 0.3 | 2.0 | ted_test | -5 | 27.4 | 26.66 | **26.39** |
| gated | 0.3 | 2.0 | ted_test | 0 | 22.84 | 22.26 | **21.91** |
| gated | 0.3 | 2.0 | ted_test | clean | 18.51 | 17.87 | **17.87** |

Beam search + LM, LibriSpeech + own-transcript LM:

Greedy vs beam search (width 32) without and with the 4-gram LM `lm/flix_libri_4gram.arpa`. alpha/beta chosen per model on validation. Test WER %:

| model | alpha | beta | test | SNR | greedy | beam, no LM | beam + LM |
|---|---:|---:|---|---|---:|---:|---:|
| twin | 0.3 | 1.0 | film_test | -10 | 59.44 | 57.83 | **57.83** |
| twin | 0.3 | 1.0 | film_test | -5 | 46.18 | 46.18 | **46.59** |
| twin | 0.3 | 1.0 | film_test | 0 | 38.96 | 38.96 | **38.55** |
| twin | 0.3 | 1.0 | film_test | clean | 30.12 | 29.72 | **28.51** |
| twin | 0.3 | 1.0 | ted_test | -10 | 43.83 | 42.95 | **42.71** |
| twin | 0.3 | 1.0 | ted_test | -5 | 31.5 | 30.51 | **30.04** |
| twin | 0.3 | 1.0 | ted_test | 0 | 23.35 | 22.41 | **21.65** |
| twin | 0.3 | 1.0 | ted_test | clean | 17.36 | 16.36 | **15.9** |
| av | 0.3 | 2.0 | film_test | -10 | 47.79 | 46.18 | **46.99** |
| av | 0.3 | 2.0 | film_test | -5 | 45.38 | 44.58 | **41.77** |
| av | 0.3 | 2.0 | film_test | 0 | 38.96 | 37.35 | **38.96** |
| av | 0.3 | 2.0 | film_test | clean | 34.54 | 33.33 | **33.33** |
| av | 0.3 | 2.0 | ted_test | -10 | 36.25 | 35.25 | **34.57** |
| av | 0.3 | 2.0 | ted_test | -5 | 33.18 | 32.01 | **31.3** |
| av | 0.3 | 2.0 | ted_test | 0 | 29.87 | 28.58 | **27.69** |
| av | 0.3 | 2.0 | ted_test | clean | 26.93 | 25.78 | **24.68** |
| visual | 0.3 | 1.0 | film_test | clean | 60.64 | 60.64 | **59.44** |
| visual | 0.3 | 1.0 | ted_test | clean | 48.25 | 47.99 | **47.17** |
| gated | 0.5 | 2.0 | film_test | -10 | 44.58 | 44.58 | **43.78** |
| gated | 0.5 | 2.0 | film_test | -5 | 42.97 | 41.37 | **40.56** |
| gated | 0.5 | 2.0 | film_test | 0 | 37.35 | 35.34 | **35.34** |
| gated | 0.5 | 2.0 | film_test | clean | 30.92 | 29.72 | **30.12** |
| gated | 0.5 | 2.0 | ted_test | -10 | 32.93 | 31.93 | **31.37** |
| gated | 0.5 | 2.0 | ted_test | -5 | 27.4 | 26.54 | **25.91** |
| gated | 0.5 | 2.0 | ted_test | 0 | 22.84 | 22.04 | **21.42** |
| gated | 0.5 | 2.0 | ted_test | clean | 18.51 | 17.63 | **17.25** |
- 2026-11-14 20:28 Decoding finished.
