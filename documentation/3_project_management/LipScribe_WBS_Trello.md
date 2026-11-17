# LipScribe — Work Breakdown Structure for Trello

**Project:** LipScribe: Lip-Assisted Speech Recognition for Robust Live Captioning
**Course:** Computer Vision and Deep Learning — Group 4, Uganda Christian University
**Source:** *LipScribe: Technical Implementation and System Development Document*, Section 12 (milestones M1–M9)
**Duration:** 5 weeks, Mon 19 Oct – Sun 22 Nov 2026. The original plan ran six weeks; on Mon 16 Nov the team re-planned to finish a week early, with **P4** as the GRID benchmark model and **ws_v5** as the real-world model (see Week 5).

---

## Part A — Board setup

Use this file to create one Trello board.

1. **Board name:** `LipScribe — CVDL Group 4`
2. **Lists, left to right:**
   `📌 Team & Reference` · `Week 1 — Setup & Data` · `Week 2 — Preprocessing & Pipeline` · `Week 3 — Baselines` · `Week 4 — Multimodal Fusion` · `Week 5 — Final Week` · `In Progress` · `In Review` · `Done`
3. **Labels** (create these first, then attach them to cards as listed):

   | Label | Colour | Used for |
   |---|---|---|
   | PM | purple | Coordination, meetings, submissions |
   | Data | green | Dataset download, splits, integrity, preprocessing |
   | Audio | sky | Audio loading, noise mixing, features |
   | Visual | orange | Face Mesh, mouth crops, visual encoder |
   | Model | blue | Encoders, fusion, CTC head |
   | Training | red | Training runs, tuning, checkpoints |
   | Evaluation | yellow | WER/CER, ablations, analysis |
   | App | pink | Inference script, Gradio, live captioning |
   | Testing | lime | Unit, integration and end-to-end tests, CI |
   | Docs | black | README, report, slides |
   | Milestone | purple (dark) | Milestone checkpoint cards |
   | Optional | grey (no colour) | Extensions; only after core freeze |
   | GPU | red (dark) | Needs GPU time; plan with the GPU owner |

4. **Members:** add the six team members in Part B. Put the card **Owner** on each card as its member. Put the **Reviewer** in the description, and also add them as a member if the board allows.
5. **Each card** is a `###` heading in Part C. For each card:
   - **Title** = the heading text, including its ID (for example `W1-05 · Download and catalogue the GRID corpus`).
   - **Description** = the *Description*, *Depends on*, *Reviewer*, *Estimate* and *Done when* fields.
   - **Checklist** named `Tasks` = the checklist items.
   - **Due date** = the *Due* field (18:00 East Africa Time).
   - **Labels** = the *Labels* field.
6. Put each **Milestone** card at the bottom of its week's list.
7. Put the cards from Part B (roles, working agreement, key references) in the `📌 Team & Reference` list.
8. Cards move `Week N` → `In Progress` → `In Review` → `Done`. A card reaches `Done` only after its Reviewer approves the pull request or output.

---

## Part B — Team, roles and working agreement

> **Note:** The role allocation below is a *proposal* that balances the workload. It is not based on known skills. Agree on it at the kick-off (card W1-01) and swap names before building the board if needed.
>
> **16 Nov re-plan:** Wakisa Isabelle moves to frontend work only. Her `transcribe.py` and demo-video duties pass to Ainomugisha Faith, and the slides to Charis Opol.

| Member | Primary role | Also responsible for |
|---|---|---|
| **Charis Opol** | Project coordinator & training lead | Trello board, weekly reviews, `train.py`, training runs, GPU booking, final submission; from 16 Nov also the slides and the rehearsal |
| **Cole Rodney Migisha** | Model lead | Mouth cropping, visual encoder, audio encoder, attention fusion, modality dropout |
| **Ainomugisha Faith** | Data lead | GRID download, integrity checks, speaker splits, manifest, crop extraction, augmentation, ablation runs; from 16 Nov also `transcribe.py` and the demo video |
| **Lisa Lusinga Namugenyi** | Audio & evaluation lead | MUSAN noise pools, noise mixing, transcripts/tokeniser, `evaluate.py`, statistics, final-report editor |
| **Wakisa Isabelle** | Frontend lead (from 16 Nov; earlier application lead) | The demo page `demo/index.html`: recording, captions, gate plot, noise control. Earlier: decoding, metrics, Gradio Mode A |
| **Ansasire Nehum Naamaye** | QA & live-systems lead | Repository, environment, CI/tests, plots, live Mode B, latency, README |

**Pairs** (from the implementation document; each pair reviews the other's work first):
- Data pair: Ainomugisha Faith + Lisa Lusinga Namugenyi
- Model pair: Cole Rodney Migisha + Charis Opol
- Application pair: Wakisa Isabelle + Ansasire Nehum Naamaye

### REF-01 · Working agreement
- **Labels:** PM
- **Description:** Rules every team member follows.
- **Checklist:**
  - [ ] All code goes through a pull request; the card's Reviewer approves it before merging to `main`
  - [ ] Weekly review meeting (suggested Friday, 1 hour): check the milestone's completion criteria and re-plan
  - [ ] Short daily async update in the group chat: done / doing / blocked
  - [ ] Every experiment has a committed YAML config in `configs/`; no unlogged results
  - [ ] Never commit data, checkpoints or personal recordings to Git
  - [ ] Never regenerate `splits.csv` after week 1
  - [ ] Flag blockers on the card with a comment and mention @Charis Opol

### REF-02 · Key reference files
- **Labels:** PM, Docs
- **Description:** Links to keep on the board: the project proposal PDF, the implementation document (`documentation/2_implementation/implementation_document/`), the GitHub repository, the GPU access instructions, and the shared results sheet/TensorBoard.

### REF-03 · Drop order if the schedule slips
- **Labels:** PM
- **Description:** Drop work in this order: (1) optional extensions, (2) experiments X4–X6, (3) live Mode B (Mode A stays), (4) multiple seeds. **Never dropped:** baselines B1–B3, proposed model P, per-SNR evaluation, ablations X1–X3.
- **Applied on 16 Nov:** (1), (3) and (4) are dropped, and so are the P sweeps; X6 was done as the Flix experiment instead of the SNR head. See card F-DROP.

### REF-04 · Experiment IDs
- **Labels:** Training, Evaluation
- **Description:** B1 audio-only · B2 visual-only · B3 concatenation fusion · P proposed attention fusion + modality dropout · X1 P without modality dropout · X2 P with concatenation in place of attention (keeping dropout) · X3 P with 2D per-frame CNN in place of 3D front-end · X4 P with EfficientNet-B0 trunk (optional) · X5 B1 with MFCC / log-Mel+MFCC features · X6 P with SNR auxiliary head (optional; replaced in Week 4 by **X6 "Flix"**, the out-of-domain test on films and TED). SNR conditions: clean, 10, 5, 0, −5 dB.
- **Real-world models (Flix v4–v6):** twin_v5, the audio-only twin (WavLM) · **ws_v5**, the warm-start gated fusion of WavLM audio and Auto-AVSR lips, the real-world model · ws_v6, ws_v5 retrained on audio that does not belong to the face (negative result). SNR conditions: clean, 0, −5, −10 dB.

---

## Part C — Weekly work breakdown

---

## Week 1 — Setup & Data (Mon 19 Oct – Sun 25 Oct 2026)
**Milestone:** M1 Environment setup and dataset verification (M2 starts mid-week)
**Week goal:** Everyone has a working environment, the raw data is downloaded and verified, and the speaker split is fixed.

### W1-01 · Kick-off meeting, roles and Trello board
- **Owner:** Charis Opol
- **Reviewer:** Cole Rodney Migisha
- **Labels:** PM
- **Due:** Mon 19 Oct 2026
- **Depends on:** —
- **Estimate:** 2 h
- **Description:** Run the kick-off. Confirm the roles in Part B, the working agreement (REF-01) and the weekly review time. Set up the Trello board and invite all members.
- **Checklist:**
  - [x] Walk the team through the implementation document (architecture, schedule, experiment list)
  - [x] Confirm or adjust the role allocation
  - [x] Agree the weekly review slot and the chat channel
  - [x] Create the board, add members, attach REF cards
  - [x] Record decisions in a `docs/meeting_notes.md` file
- **Done when:** All six members have accepted the board invite and agreed to their roles.

### W1-02 · Create GitHub repository skeleton
- **Owner:** Ansasire Nehum Naamaye
- **Reviewer:** Wakisa Isabelle
- **Labels:** Testing, PM
- **Due:** Tue 20 Oct 2026
- **Depends on:** W1-01
- **Estimate:** 3 h
- **Description:** Create the repository with the package layout from the implementation document so that everyone works in the same structure from day one.
- **Checklist:**
  - [x] Create the repo and add all six members as collaborators
  - [x] Package `lipscribe/` with sub-packages `data`, `preprocessing`, `models`, `decoding`, `evaluation`, `app`
  - [x] Folders `configs/`, `scripts/`, `tests/`, `tests/fixtures/`, `experiments/`, `docs/`
  - [x] `.gitignore` excluding `data/`, `runs/`, `*.npy`, `*.pt`, `*.wav`, recordings
  - [x] `pyproject.toml` so `pip install -e .` works
  - [x] Protect `main`: pull requests with one approval
  - [x] Pull request template with a "how I tested this" section
- **Done when:** Every member has cloned the repo and run `pip install -e .`.

### W1-03 · Pinned Python environment
- **Owner:** Ansasire Nehum Naamaye
- **Reviewer:** Cole Rodney Migisha
- **Labels:** Testing
- **Due:** Wed 21 Oct 2026
- **Depends on:** W1-02
- **Estimate:** 4 h
- **Description:** Build one environment that works on every member's machine and on the GPU machine. MediaPipe supports only some Python versions, so check it first.
- **Checklist:**
  - [x] Check that MediaPipe has a wheel for Python 3.11 on every member's operating system
  - [x] Install matching `torch`, `torchaudio`, `torchvision` (CPU build for laptops; CUDA build matching the GPU driver)
  - [x] Install opencv-python, mediapipe, numpy, pandas, soundfile, librosa, sounddevice, scikit-learn, jiwer, matplotlib, tensorboard, pyyaml, pytest, gradio
  - [x] Generate `requirements.txt` and `environment.yml` from the working environment (do not hand-write versions)
  - [x] Write `scripts/check_env.py`: imports every library, prints versions and whether CUDA is available
  - [x] Each member runs `check_env.py` and posts the output on this card
- **Done when:** `check_env.py` passes for all six members and on the GPU machine.

### W1-04 · Secure GPU access and storage
- **Owner:** Charis Opol
- **Reviewer:** Ainomugisha Faith
- **Labels:** PM, GPU
- **Due:** Wed 21 Oct 2026
- **Depends on:** W1-01
- **Estimate:** 3 h
- **Description:** Training needs a CUDA GPU (about 8 GB or more) and about 100 GB of disk. Find out what is actually available. This is the top project risk.
- **Checklist:**
  - [x] Ask the department/lecturer about a university GPU workstation
  - [x] List team members' own GPUs (model, memory)
  - [x] Check a cloud notebook option as a fallback (session limits, storage)
  - [x] Confirm at least 100 GB of free disk where the data will live
  - [x] Write `docs/compute.md`: machine, GPU, memory, how to log in, booking rules
  - [x] Set up a shared GPU booking schedule (who trains when)
- **Done when:** A named GPU machine is confirmed and documented. If none is available, escalate to the lecturer by Wednesday.

### W1-05 · Download and catalogue the GRID corpus
- **Owner:** Ainomugisha Faith
- **Reviewer:** Lisa Lusinga Namugenyi
- **Labels:** Data
- **Due:** Wed 21 Oct 2026
- **Depends on:** W1-04 (storage location)
- **Estimate:** 4 h (mostly download time)
- **Description:** Download video, audio and word-alignment files for all 34 GRID speakers from the official distribution into `data/raw/grid/`. Raw files are never modified.
- **Checklist:**
  - [x] Download video, audio and alignment archives for every speaker
  - [x] Extract to `data/raw/grid/s<N>/{video,audio,align}/`
  - [x] Record source URL, download date and archive checksums in `data/raw/SOURCES.md`
  - [x] Note any speakers with missing or incomplete video
  - [x] Copy 2 utterances (video, audio, align) into `tests/fixtures/` for the tests
- **Done when:** All archives are extracted and `SOURCES.md` is committed.

### W1-06 · Download MUSAN and create noise pools
- **Owner:** Lisa Lusinga Namugenyi
- **Reviewer:** Ainomugisha Faith
- **Labels:** Audio, Data
- **Due:** Thu 22 Oct 2026
- **Depends on:** W1-04
- **Estimate:** 3 h
- **Description:** Download MUSAN. Split the *noise* subset into disjoint train/validation/test pools **by file**, so the test noise is never heard in training.
- **Checklist:**
  - [x] Download MUSAN into `data/raw/musan/`; record it in `SOURCES.md`
  - [x] List all noise-subset files
  - [x] Randomly assign files to train/val/test (suggested 70/15/15) with a fixed seed
  - [x] Save `data/noise_splits.csv` (file path, pool) and commit it
  - [x] Copy one short noise file into `tests/fixtures/`
- **Done when:** `noise_splits.csv` is committed and no file appears in two pools.

### W1-07 · Dataset integrity checker (`verify_dataset.py`)
- **Owner:** Ainomugisha Faith
- **Reviewer:** Ansasire Nehum Naamaye
- **Labels:** Data, Testing
- **Due:** Fri 23 Oct 2026
- **Depends on:** W1-05
- **Estimate:** 6 h
- **Description:** Script that checks every utterance before training and writes an integrity report. Its counts become the dataset statistics in the final report; none are assumed in advance.
- **Checklist:**
  - [x] Check each utterance has video, audio and alignment files
  - [x] Check video decodes and has 75 ± 2 frames
  - [x] Check audio duration matches video duration to within 40 ms
  - [x] Check transcript has exactly six words from the GRID word list
  - [x] Check audio is not silent or clipped throughout (RMS and peak)
  - [x] Leave a hook for the Face Mesh success-rate check (≥ 90 % of frames), added in W2-02
  - [x] Write `reports/integrity_report.csv` plus a summary of pass counts per check and per speaker
- **Done when:** The report runs over the full corpus and its summary is posted on the card.

### W1-08 · Speaker-independent split (`splits.csv`)
- **Owner:** Ainomugisha Faith
- **Reviewer:** Charis Opol
- **Labels:** Data
- **Due:** Sat 24 Oct 2026
- **Depends on:** W1-07
- **Estimate:** 2 h
- **Description:** Split by speaker, never by utterance, to prevent speaker leakage. Generate once with a fixed seed and never regenerate.
- **Checklist:**
  - [x] Exclude speakers with missing or badly corrupted video (from the integrity report)
  - [x] Test: 4 speakers (2 female, 2 male)
  - [x] Validation: 3 speakers
  - [x] Training: all remaining speakers (~27)
  - [x] Check gender balance in each partition
  - [x] Write `data/splits.csv` (speaker → partition) and commit it
  - [x] Add a unit test asserting the three speaker sets are pairwise disjoint
- **Done when:** `splits.csv` is committed and the disjointness test passes.

### W1-09 · Transcript normalisation and tokeniser
- **Owner:** Lisa Lusinga Namugenyi
- **Reviewer:** Wakisa Isabelle
- **Labels:** Data, Model
- **Due:** Fri 23 Oct 2026
- **Depends on:** W1-05
- **Estimate:** 4 h
- **Description:** Build transcripts from the `.align` files and the 28-symbol character vocabulary (blank = 0, space, a–z).
- **Checklist:**
  - [x] Parse `.align` files; drop silence (`sil`) and short-pause (`sp`) tokens
  - [x] Lower-case; letters as single characters ("b"), digits as words ("four")
  - [x] `lipscribe/data/tokenizer.py`: `encode(text) -> list[int]`, `decode(ids) -> str`, blank index 0
  - [x] Unit test: `decode(encode(s)) == s` for all GRID word strings; `encode` never outputs blank
  - [x] Record the longest transcript length (needed for the CTC length check)
- **Done when:** Tokeniser tests pass in CI.

### W1-10 · Mouth-cropping prototype with MediaPipe Face Mesh
- **Owner:** Cole Rodney Migisha
- **Reviewer:** Ainomugisha Faith
- **Labels:** Visual
- **Due:** Sat 24 Oct 2026
- **Depends on:** W1-03, W1-05
- **Estimate:** 8 h
- **Description:** Implement the crop function from Section 5.1 of the implementation document and test it on a few clips.
- **Checklist:**
  - [x] Read frames with OpenCV, convert BGR → RGB, keep frame indices
  - [x] Face Mesh in video mode, `max_num_faces=1`
  - [x] Lip landmark indices from `FACEMESH_LIPS` (not hard-coded)
  - [x] Square box around the lips + 25 % margin
  - [x] Smooth centre and size with a centred 5-frame moving average
  - [x] Crop with edge padding, resize to 96×96 (area interpolation), convert to grey `uint8`
  - [x] Return a per-frame validity flag for failed detections
  - [x] Unit test: output shape (96, 96), dtype `uint8`
  - [x] Save a contact sheet of crops for 5 clips and attach it to the card
- **Done when:** Crops look centred and stable on the contact sheet, and the reviewer agrees.

### W1-11 · Module interface specification
- **Owner:** Wakisa Isabelle
- **Reviewer:** Cole Rodney Migisha
- **Labels:** Docs, App
- **Due:** Thu 22 Oct 2026
- **Depends on:** W1-02
- **Estimate:** 3 h
- **Description:** Write `docs/INTERFACES.md` from Table 3 of the implementation document: function names, inputs, outputs and tensor shapes for every module, so the pairs can work in parallel.
- **Checklist:**
  - [x] Preprocessing: `crop_mouths(video) -> uint8[T,96,96], valid[T]`, `load_audio(path) -> float32[N] @16 kHz`
  - [x] Features: `logmel(wave) -> [T_a, 80]`
  - [x] Encoders: visual `[B,1,T,96,96] -> [B,T,512]`, audio `[B,T_a,80] -> [B,T,512]`
  - [x] Fusion `-> [B,T,256]`; CTC head `-> [T,B,28]` log-probabilities
  - [x] Decoder `-> str`; `transcribe(video_path) -> str`
  - [x] Batch dictionary keys returned by the data loader
- **Done when:** Cole and Charis approve the document.

### W1-12 · Test framework and continuous integration
- **Owner:** Ansasire Nehum Naamaye
- **Reviewer:** Charis Opol
- **Labels:** Testing
- **Due:** Fri 23 Oct 2026
- **Depends on:** W1-02, W1-05
- **Estimate:** 4 h
- **Description:** Set up pytest and GitHub Actions so every pull request runs the tests on the small fixture files, without the full dataset.
- **Checklist:**
  - [x] `tests/conftest.py` with fixtures pointing to `tests/fixtures/`
  - [x] GitHub Actions workflow: install CPU environment, run `pytest`
  - [x] Add a status badge to the README
  - [x] Write `docs/testing.md`: how to run tests locally
- **Done when:** CI runs green on a test pull request.

### W1-M · MILESTONE M1 — Environment & dataset verified
- **Owner:** Charis Opol
- **Reviewer:** All
- **Labels:** Milestone, PM
- **Due:** Sun 25 Oct 2026
- **Depends on:** W1-02 … W1-12
- **Description:** Weekly review. Check the completion criteria and re-plan Week 2.
- **Checklist:**
  - [x] Every member runs the test suite successfully
  - [x] Integrity report lists usable utterances per speaker
  - [x] `splits.csv` committed; speaker sets disjoint
  - [x] GPU access confirmed
  - [x] Mouth-crop prototype approved
  - [x] Unfinished cards moved to Week 2 with a comment explaining why

---

## Week 2 — Preprocessing & Pipeline (Mon 26 Oct – Sun 1 Nov 2026)
**Milestone:** M2 Preprocessing and data alignment (M3 audio-only baseline starts)
**Week goal:** All data is preprocessed and cached, one batch loads correctly, and the audio-only baseline (B1) starts training.

### W2-01 · Failure handling in mouth cropping
- **Owner:** Cole Rodney Migisha
- **Reviewer:** Ainomugisha Faith
- **Labels:** Visual
- **Due:** Tue 27 Oct 2026
- **Depends on:** W1-10
- **Estimate:** 4 h
- **Description:** Make the crop function robust before running it on 33,000 clips.
- **Checklist:**
  - [x] Failed frame → interpolate the box from the nearest successful frames; mark invalid in the mask
  - [x] More than 10 % failed frames → flag the utterance for exclusion
  - [x] Pad or truncate to the audio-derived frame count by repeating the last valid frame; mark padding in the mask
  - [x] Never drop frames (this keeps audio–video alignment)
  - [x] Unit test: a frame with no face gives an interpolated box and an invalid flag
- **Done when:** Tests pass and the function handles a clip with artificially blanked frames.

### W2-02 · Run mouth extraction over the full corpus (`extract_mouths.py`)
- **Owner:** Ainomugisha Faith
- **Reviewer:** Cole Rodney Migisha
- **Labels:** Data, Visual
- **Due:** Thu 29 Oct 2026
- **Depends on:** W2-01
- **Estimate:** 4 h of work + several hours of runtime
- **Description:** Batch script that caches one `uint8 [T,96,96]` `.npy` file per utterance.
- **Checklist:**
  - [x] Resumable: skip utterances already processed
  - [x] Run in parallel across CPU cores
  - [x] Save to `data/processed/mouths/s<N>/<utt_id>.npy`
  - [x] Log the per-utterance Face Mesh success rate
  - [x] Add the Face Mesh ≥ 90 % check to `verify_dataset.py`
  - [x] Record the MediaPipe version and crop parameters in the manifest
  - [x] Check total disk use (expected ~23 GB)
- **Done when:** Every utterance has a crop file or is listed as excluded.

### W2-03 · Crop quality review (contact sheets)
- **Owner:** Wakisa Isabelle
- **Reviewer:** Ansasire Nehum Naamaye
- **Labels:** Visual, Testing
- **Due:** Fri 30 Oct 2026
- **Depends on:** W2-02
- **Estimate:** 3 h
- **Description:** Visually inspect crops from every speaker before training starts.
- **Checklist:**
  - [x] Script `scripts/contact_sheet.py`: 10 random utterances per speaker, frames 0/25/50/74
  - [x] Inspect every speaker for mis-centred, cut-off or blurred crops
  - [x] List problem utterances or speakers on this card
  - [x] Agree fixes with Cole (margin, smoothing) if there are systematic problems
- **Done when:** All speakers are reviewed and problems are fixed or documented.

### W2-04 · Audio standardisation (`prepare_audio.py`)
- **Owner:** Lisa Lusinga Namugenyi
- **Reviewer:** Ainomugisha Faith
- **Labels:** Audio, Data
- **Due:** Tue 27 Oct 2026
- **Depends on:** W1-05
- **Estimate:** 3 h
- **Description:** Convert all GRID audio to 16 kHz, 16-bit, mono WAV with a band-limited resampler.
- **Checklist:**
  - [x] Read with `soundfile`, average to mono, resample to 16 kHz
  - [x] Save to `data/processed/audio16k/s<N>/<utt_id>.wav`
  - [x] Shared function `load_audio()` in `lipscribe/preprocessing/audio.py` (also used by the demo)
  - [x] Check that durations still match the video
- **Done when:** All files are converted and spot-checked by listening to 5 of them.

### W2-05 · Noise mixer with SNR verification
- **Owner:** Lisa Lusinga Namugenyi
- **Reviewer:** Charis Opol
- **Labels:** Audio, Testing
- **Due:** Wed 28 Oct 2026
- **Depends on:** W1-06, W2-04
- **Estimate:** 5 h
- **Description:** Implement Algorithm 1 (mixing noise at a target SNR) from the implementation document.
- **Checklist:**
  - [x] Random noise segment from the correct pool; loop the file if shorter than the speech
  - [x] Compute powers over the whole clip; scale noise by α = √(Ps / (Pw·10^(SNR/10)))
  - [x] Verify the measured SNR is within 0.01 dB of the target (assert)
  - [x] If peak > 1, scale the whole mixture (this preserves SNR)
  - [x] "clean" returns the original audio
  - [x] Unit tests for all five conditions; output length equals input length; peak ≤ 1
  - [x] Transcript is unchanged by mixing (documented in the docstring)
- **Done when:** Tests pass in CI.

### W2-06 · Fixed noisy validation and test sets (`make_noisy_test.py`)
- **Owner:** Lisa Lusinga Namugenyi
- **Reviewer:** Ansasire Nehum Naamaye
- **Labels:** Audio, Data
- **Due:** Fri 30 Oct 2026
- **Depends on:** W2-05, W1-08
- **Estimate:** 3 h
- **Description:** Mix every validation and test utterance once per SNR with a fixed seed, so all models are evaluated on identical audio.
- **Checklist:**
  - [x] Use only the val/test noise pools
  - [x] Save to `data/processed/noisy_test/<snr>/<utt_id>.wav`
  - [x] Log the noise file, offset and measured SNR for every mixture
  - [x] Listen to examples at each SNR and confirm −5 dB is very noisy but speech is present
- **Done when:** All five conditions exist for every val/test utterance.

### W2-07 · Feature extraction module (log-Mel and MFCC)
- **Owner:** Charis Opol
- **Reviewer:** Lisa Lusinga Namugenyi
- **Labels:** Audio
- **Due:** Wed 28 Oct 2026
- **Depends on:** W2-04
- **Estimate:** 4 h
- **Description:** `torchaudio` transforms for feature configurations A1 (80 log-Mel), A2 (39 MFCC + Δ + ΔΔ) and A3 (log-Mel ⊕ MFCC).
- **Checklist:**
  - [x] 25 ms Hann window, 10 ms hop, 512-point FFT, 80 Mel bands, 20 Hz–8 kHz, log(x + 1e-6)
  - [x] Trim to 4 × number of video frames (300 for a 3 s clip)
  - [x] RMS normalisation to −25 dBFS (gain limited to +20 dB), applied after noise mixing
  - [x] Config switch `audio_features: logmel80 | mfcc39 | logmel_mfcc`
  - [x] Unit test: 3 s → [300, 80]; matches `librosa` within 1e-3
- **Done when:** Tests pass; the same function is importable by training and the app.

### W2-08 · Normalisation statistics (`compute_stats.py`)
- **Owner:** Ainomugisha Faith
- **Reviewer:** Lisa Lusinga Namugenyi
- **Labels:** Audio, Visual, Data
- **Due:** Thu 29 Oct 2026
- **Depends on:** W2-02, W2-05, W2-07
- **Estimate:** 3 h
- **Description:** Mean and standard deviation from the **training partition only**: per Mel band (over the training SNR mixture) and for mouth-crop pixels.
- **Checklist:**
  - [x] Sample training utterances under the training SNR distribution
  - [x] Save `data/stats/feature_norm.npz` (audio mean/std per band, pixel mean/std)
  - [x] Unit test: normalised training features have mean within 0.05 of 0 and std within 0.05 of 1
- **Done when:** The stats file is committed (it is small) and loaded by the data loader.

### W2-09 · Manifest, dataset class and collate function
- **Owner:** Charis Opol
- **Reviewer:** Cole Rodney Migisha
- **Labels:** Data, Training
- **Due:** Fri 30 Oct 2026
- **Depends on:** W1-08, W1-09, W2-02, W2-05
- **Estimate:** 8 h
- **Description:** `manifest.csv` (one row per utterance) and the PyTorch `Dataset`/collate that every model uses.
- **Checklist:**
  - [x] `build_manifest.py`: utterance id, speaker, transcript, frame count, duration, Face Mesh success rate, paths
  - [x] Dataset filters the manifest by `splits.csv`; raises an error if a speaker is in two partitions
  - [x] Training items: crops + clean audio mixed on the fly at an SNR drawn uniformly from {clean, 10, 5, 0, −5}
  - [x] Val/test items: pre-mixed noisy files for a given SNR
  - [x] Collate: pad waveforms and crops, feature lengths, encoder lengths ⌈T_a/4⌉, padding mask [B,T], concatenated targets
  - [x] Assert the CTC condition T ≥ L + repeats for every item
  - [ ] Optional bucketing by length
- **Done when:** One batch of 16 loads with consistent shapes (integration test).

### W2-10 · Train/inference parity test
- **Owner:** Ansasire Nehum Naamaye
- **Reviewer:** Wakisa Isabelle
- **Labels:** Testing
- **Due:** Sun 1 Nov 2026
- **Depends on:** W2-02, W2-07, W2-09
- **Estimate:** 3 h
- **Description:** Check that the live preprocessing path gives the same tensors as the cached training features for the same GRID file.
- **Checklist:**
  - [x] Process a fixture clip through the "live" path (frames → crop → features)
  - [x] Compare with the cached crop and the training feature path
  - [x] Assert max error < 1e-5 for features and identical crops (or document the expected difference for causal smoothing)
- **Done when:** The test passes in CI.

### W2-11 · Audio encoder module
- **Owner:** Cole Rodney Migisha
- **Reviewer:** Charis Opol
- **Labels:** Model
- **Due:** Thu 29 Oct 2026
- **Depends on:** W1-11
- **Estimate:** 4 h
- **Description:** Two strided Conv1d layers (80→256→256, k=3, s=2, BN, ReLU) and a 2-layer BiLSTM (256 per direction, dropout 0.3) with packed sequences.
- **Checklist:**
  - [x] Implement in `lipscribe/models/audio_encoder.py`
  - [x] Length function using ⌊(L + 2p − k)/s⌋ + 1 twice
  - [x] `pack_padded_sequence` with true lengths
  - [x] Shape test: [2,300,80] → [2,75,512]
- **Done when:** Shape and length tests pass.

### W2-12 · CTC head, greedy decoder and metrics
- **Owner:** Wakisa Isabelle
- **Reviewer:** Lisa Lusinga Namugenyi
- **Labels:** Model, Evaluation
- **Due:** Thu 29 Oct 2026
- **Depends on:** W1-09
- **Estimate:** 4 h
- **Description:** Shared output layer and decoding/metric functions.
- **Checklist:**
  - [x] CTC head: sequence BiLSTM (128 per direction) → Linear(256→28) → log-softmax, output [T,B,28]
  - [x] `collapse()` (merge repeats, remove blanks) and greedy decoder in `lipscribe/decoding/`
  - [x] Unit test: `[b,b,0,i,0,n,n] → "bin"`, `[e,0,e] → "ee"`
  - [x] WER/CER with `jiwer` at corpus level (sum errors, then divide)
  - [x] Unit test: WER("bin blue at f two now", "bin blue at s two now") = 1/6
- **Done when:** Tests pass in CI.

### W2-13 · Training script (`train.py`)
- **Owner:** Charis Opol
- **Reviewer:** Cole Rodney Migisha
- **Labels:** Training
- **Due:** Sat 31 Oct 2026
- **Depends on:** W2-09, W2-12
- **Estimate:** 8 h
- **Description:** One training script driven by a YAML config (Section 8 of the implementation document).
- **Checklist:**
  - [x] Load config; set Python/NumPy/PyTorch seeds; create run directory; save config and Git commit hash
  - [x] AdamW (lr 3e-4, weight decay 1e-2, not on biases/norms); 1-epoch warm-up + cosine decay
  - [x] Mixed precision (AMP); unscale, then clip gradient norm at 5.0
  - [x] `nn.CTCLoss(blank=0, zero_infinity=True)`; skip and log non-finite losses
  - [x] Validation each epoch: loss, greedy WER/CER per SNR and their mean
  - [x] Checkpoints: latest and best by mean validation WER (model, optimiser, scheduler, scaler, epoch, config, vocab)
  - [x] Early stopping (patience 8); resume from checkpoint
  - [x] TensorBoard logging; append a summary row to `experiments/results.csv`
  - [x] LSTM forget-gate bias initialised to 1
- **Done when:** A 2-epoch run on a small subset completes, saves, resumes and logs.

### W2-14 · GPU memory test and B1 overfit check
- **Owner:** Charis Opol
- **Reviewer:** Ansasire Nehum Naamaye
- **Labels:** Training, GPU
- **Due:** Sun 1 Nov 2026
- **Depends on:** W2-11, W2-13
- **Estimate:** 3 h
- **Description:** Confirm batch size 16 fits with AMP, and that B1 can overfit 8 utterances (a sanity check for bugs).
- **Checklist:**
  - [x] Record peak GPU memory at batch size 16; use gradient accumulation if it does not fit
  - [x] B1 overfit: CTC loss < 0.05 and CER = 0 on 8 utterances within 300 steps
  - [x] Record time per epoch for the full training set (for planning)
  - [x] Launch the full B1 training run
- **Done when:** The overfit test passes and the full B1 run is training.

### W2-M · MILESTONE M2 — Preprocessing & alignment complete
- **Owner:** Charis Opol
- **Reviewer:** All
- **Labels:** Milestone, PM
- **Due:** Sun 1 Nov 2026
- **Depends on:** W2-01 … W2-14
- **Checklist:**
  - [x] Crop contact sheets reviewed
  - [x] Noise-mixing tests pass
  - [x] One batch loads with consistent lengths
  - [x] Train/inference parity test passes
  - [x] B1 full run started
  - [x] Re-plan Week 3; move unfinished cards with a reason

---

## Week 3 — Baselines (Mon 2 Nov – Sun 8 Nov 2026)
**Milestones:** M3 Audio-only baseline complete · M4 Visual encoder, B2 and B3
**Week goal:** All three baselines are trained or training, and evaluation, beam search and the first demo page work.

### W3-01 · Complete and evaluate B1 (audio-only)
- **Owner:** Charis Opol
- **Reviewer:** Lisa Lusinga Namugenyi
- **Labels:** Training, Evaluation, GPU
- **Due:** Wed 4 Nov 2026
- **Depends on:** W2-14, W3-02
- **Estimate:** 3 h of work + runtime
- **Description:** Finish the B1 run and evaluate it at all five SNRs. It is the reference for every fused model.
- **Checklist:**
  - [x] Check training curves for over/underfitting
  - [x] Restore the best checkpoint
  - [x] Evaluate at clean, 10, 5, 0, −5 dB
  - [x] Post the WER/CER table and curves on the card
- **Done when:** B1 results are in `results.csv` with their config and checkpoint path.

### W3-02 · Evaluation script (`evaluate.py`)
- **Owner:** Lisa Lusinga Namugenyi
- **Reviewer:** Wakisa Isabelle
- **Labels:** Evaluation
- **Due:** Tue 3 Nov 2026
- **Depends on:** W2-06, W2-12
- **Estimate:** 6 h
- **Description:** Standard evaluation used for every model, on identical pre-mixed test audio.
- **Checklist:**
  - [x] Load a checkpoint and its config
  - [x] For each SNR: decode all test utterances (greedy or beam via config)
  - [x] Corpus-level WER and CER: per SNR, per speaker, overall
  - [x] Save per-utterance hypotheses to CSV (for error analysis)
  - [x] Hook to record the mean fusion gate per utterance (used from Week 4)
  - [x] Option to replace one modality with `e_miss` at test time (for Week 5 ablations)
  - [x] Append to `experiments/results.csv`
- **Done when:** Re-running it on the same checkpoint gives identical numbers.

### W3-03 · Visual encoder V1 (3D-CNN + BiLSTM)
- **Owner:** Cole Rodney Migisha
- **Reviewer:** Charis Opol
- **Labels:** Model, Visual
- **Due:** Tue 3 Nov 2026
- **Depends on:** W1-11
- **Estimate:** 6 h
- **Description:** Implement Table 6 of the implementation document.
- **Checklist:**
  - [x] Conv3d 1→32 (3×5×5, stride 1×2×2) + BN + ReLU + MaxPool 1×2×2
  - [x] Conv3d 32→64 (3×5×5) + BN + ReLU + MaxPool 1×2×2
  - [x] Conv3d 64→96 (3×3×3) + BN + ReLU + MaxPool 1×2×2
  - [x] Temporal padding 1, temporal stride 1 (T stays 75)
  - [x] Flatten 96·6·6 = 3456 → Linear 512 + ReLU + Dropout 0.3 → 2-layer BiLSTM (256 per direction)
  - [x] Shape test: [2,1,75,96,96] → [2,75,512], finite values
  - [x] Report the parameter count on the card: 5,246,368 trainable parameters
- **Done when:** Tests pass and the reviewer approves.

### W3-04 · Visual data augmentation
- **Owner:** Ainomugisha Faith
- **Reviewer:** Cole Rodney Migisha
- **Labels:** Visual, Data
- **Due:** Wed 4 Nov 2026
- **Depends on:** W2-09
- **Estimate:** 3 h
- **Description:** Training-only augmentation that keeps the motion pattern intact.
- **Checklist:**
  - [x] Random crop jitter (same offset for every frame of a clip)
  - [x] Random horizontal flip (same for every frame of a clip)
  - [x] Disabled for validation, test and inference
  - [x] Unit test: all frames of an augmented clip get the same transform
- **Done when:** Tests pass; augmentation is switchable in the config.

### W3-05 · Train B2 (visual-only)
- **Owner:** Cole Rodney Migisha
- **Reviewer:** Charis Opol
- **Labels:** Training, GPU
- **Due:** Sat 7 Nov 2026
- **Depends on:** W3-03, W3-04, W2-13
- **Estimate:** 3 h of work + runtime
- **Description:** Visual-only CTC can be slow to start; check it early.
- **Checklist:**
  - [x] Overfit test on 8 utterances (loss < 0.05)
  - [x] Full run with `configs/b2_visual_only.yaml`
  - [x] Monitor validation CER; it must fall clearly below that of an untrained model
  - [x] If it has not improved after several epochs, raise it at the review (the curriculum option in W4-05)
  - [x] Evaluate (performance should be roughly constant across SNR by design)
- **Done when:** B2 results are logged.

### W3-06 · Implement and train B3 (concatenation fusion)
- **Owner:** Ainomugisha Faith
- **Reviewer:** Charis Opol
- **Labels:** Model, Training, GPU
- **Due:** Sun 8 Nov 2026
- **Depends on:** W2-11, W3-03
- **Estimate:** 4 h of work + runtime
- **Description:** Concatenate audio and visual features per frame → Linear to 256 → the same sequence BiLSTM and CTC head. No attention and no modality dropout.
- **Checklist:**
  - [x] `lipscribe/models/fusion_concat.py`
  - [x] Config `configs/b3_concat.yaml` (same optimiser, schedule and epochs as B1)
  - [x] Overfit test, then full run (book GPU time with Charis)
  - [x] Evaluate at all five SNRs
- **Done when:** B3 results are logged.

### W3-07 · CTC prefix beam search
- **Owner:** Wakisa Isabelle
- **Reviewer:** Lisa Lusinga Namugenyi
- **Labels:** Model, Evaluation
- **Due:** Thu 5 Nov 2026
- **Depends on:** W2-12
- **Estimate:** 6 h
- **Description:** Implement Algorithm 3 from the implementation document in about 60 lines of Python, in log space.
- **Checklist:**
  - [x] Track blank and non-blank probabilities per prefix
  - [x] Repeated symbol needs a blank between copies
  - [x] Beam width configurable (default 10)
  - [x] Unit test: beam width 1 equals greedy on random posteriors
  - [x] Measure decode time per utterance
- **Done when:** Tests pass; `evaluate.py` can use it through the config.

### W3-08 · Experiment X5: audio feature configurations
- **Owner:** Lisa Lusinga Namugenyi
- **Reviewer:** Charis Opol
- **Labels:** Training, Evaluation, GPU
- **Due:** Sun 8 Nov 2026
- **Depends on:** W3-01
- **Estimate:** 2 h of work + runtime
- **Description:** Train B1 with A2 (MFCC) and A3 (log-Mel + MFCC) to decide whether MFCCs add anything. Run only if GPU time allows; otherwise move to Week 4.
- **Checklist:**
  - [x] Configs `b1_mfcc39.yaml`, `b1_logmel_mfcc.yaml`
  - [x] Train and evaluate on the same protocol as B1
  - [x] Decision on the card: keep A1 unless another configuration improves validation WER
- **Done when:** The decision is recorded.

### W3-09 · Experiment tracking conventions and plotting scripts
- **Owner:** Ansasire Nehum Naamaye
- **Reviewer:** Lisa Lusinga Namugenyi
- **Labels:** Evaluation, Docs
- **Due:** Fri 6 Nov 2026
- **Depends on:** W2-13
- **Estimate:** 5 h
- **Description:** Make every result traceable and every figure reproducible from logs.
- **Checklist:**
  - [x] Run-naming rule: `<experiment>_<seed>_<date>`
  - [x] `docs/experiments.md`: how to start, resume and log a run
  - [x] `scripts/plot_wer_vs_snr.py` (one line per model, from `results.csv`)
  - [x] `scripts/plot_training_curves.py` (from TensorBoard logs)
  - [x] Shared view of TensorBoard or an exported results sheet for the team
- **Done when:** Plots are generated from the B1 results.

### W3-10 · Gradio Mode A skeleton
- **Owner:** Wakisa Isabelle
- **Reviewer:** Ansasire Nehum Naamaye
- **Labels:** App
- **Due:** Sun 8 Nov 2026
- **Depends on:** W3-01, W3-07
- **Estimate:** 6 h
- **Description:** First working web demo using the B1 checkpoint. It is upgraded to the fused model later.
- **Checklist:**
  - [x] Webcam/microphone recorder and file-upload input
  - [x] Extract frames (OpenCV) and audio (`ffmpeg`); resample to 25 fps and 16 kHz
  - [x] Run the shared preprocessing, the model and beam search
  - [x] Show the caption beside the video
  - [x] Handle errors (no audio track, no face) with a friendly message
- **Done when:** A GRID test clip uploaded to the page gives the same caption as `evaluate.py`.

### W3-M · MILESTONES M3 & M4 — Baselines
- **Owner:** Charis Opol
- **Reviewer:** All
- **Labels:** Milestone, PM
- **Due:** Sun 8 Nov 2026
- **Checklist:**
  - [x] B1 evaluated at all five SNRs
  - [x] Encoder shape tests pass
  - [x] B2 validation CER clearly below an untrained model
  - [x] B3 evaluated at all SNRs (or running, with an expected finish date)
  - [x] Beam search and `evaluate.py` merged
  - [x] Risk check: Is GPU time enough? Is B2 learning? Re-plan Week 4.

---

## Week 4 — Multimodal Fusion (Mon 9 Nov – Sun 15 Nov 2026)
**Milestone:** M5 Multimodal fusion (M6 tuning starts)
**Week goal:** The proposed model P trains stably, ablation configs are ready, and the inference script and live-mode foundations exist.

### W4-01 · Cross-modal attention module
- **Owner:** Cole Rodney Migisha
- **Reviewer:** Charis Opol
- **Labels:** Model
- **Due:** Tue 10 Nov 2026
- **Depends on:** W3-03, W2-11
- **Estimate:** 6 h
- **Description:** Two multi-head attention blocks (audio→video and video→audio) with residual connections and LayerNorm (Section 7.3 of the implementation document).
- **Checklist:**
  - [x] Projections 512→256 + LayerNorm for each stream
  - [x] `nn.MultiheadAttention`, 4 heads, d = 256, dropout 0.1, `batch_first=True`
  - [x] Padding mask for keys
  - [x] Return attention weights for analysis
  - [x] Truncate the longer stream if the lengths differ by 1–2 frames
- **Done when:** Shape tests pass ([B,T,256] for each stream).

### W4-02 · Gated fusion, missing-modality vector and modality dropout
- **Owner:** Cole Rodney Migisha
- **Reviewer:** Charis Opol
- **Labels:** Model
- **Due:** Wed 11 Nov 2026
- **Depends on:** W4-01
- **Estimate:** 6 h
- **Description:** Per-frame gate g_t = σ(w·[ã_t; ṽ_t] + b), fused f_t = g_t·ã_t + (1−g_t)·ṽ_t; learned `e_miss` vector; modality dropout (p_audio = 0.25, p_visual = 0.15, never both).
- **Checklist:**
  - [x] Gate module returning g for logging
  - [x] Learned `e_miss` (256-d) that replaces a dropped stream's encoding (not an attention mask; masking all keys gives NaN)
  - [x] Modality dropout active only in `model.train()`
  - [x] Inference: if a face is missing in > 50 % of the window's frames, use `e_miss` for the visual stream
  - [x] Config keys for both dropout rates
- **Done when:** The NaN test in W4-03 passes.

### W4-03 · Fusion unit tests
- **Owner:** Ansasire Nehum Naamaye
- **Reviewer:** Cole Rodney Migisha
- **Labels:** Testing, Model
- **Due:** Thu 12 Nov 2026
- **Depends on:** W4-02
- **Estimate:** 3 h
- **Description:** Tests that catch the common fusion bugs.
- **Checklist:**
  - [x] Output shapes for full model forward pass (Algorithm 2)
  - [x] No NaN when audio is dropped, when video is dropped, and with padded batches
  - [x] Gate values lie in (0, 1)
  - [x] Modality dropout is inactive in eval mode
  - [x] Both streams are never dropped together (statistical test over 1000 samples)
- **Done when:** All tests pass in CI.

### W4-04 · Assemble proposed model P and first full run
- **Owner:** Charis Opol
- **Reviewer:** Cole Rodney Migisha
- **Labels:** Training, Model, GPU
- **Due:** Sun 15 Nov 2026
- **Depends on:** W4-02, W4-03
- **Estimate:** 4 h of work + runtime
- **Description:** Full model: encoders → attention fusion → gate → sequence BiLSTM → CTC.
- **Checklist:**
  - [x] Config `configs/p_proposed.yaml` (matches Listing 2 of the implementation document)
  - [x] Overfit test on 8 utterances
  - [x] Full run; log mean gate value per SNR on validation
  - [x] Check training is stable (no loss spikes or NaNs)
  - [x] First evaluation at all five SNRs
- **Done when:** P trains stably and first results are logged.

### W4-05 · Visual pre-training curriculum (only if B2 learned slowly)
- **Owner:** Cole Rodney Migisha
- **Reviewer:** Charis Opol
- **Labels:** Training, Model, GPU
- **Due:** Sat 14 Nov 2026
- **Depends on:** W3-05
- **Estimate:** 3 h
- **Description:** Conditional card. If B2 was slow to learn, initialise P's visual stream from the B2 visual encoder weights. Archive this card if it isn't needed.
- **Checklist:**
  - [x] Decide at the Week 3 review whether it is needed
  - [x] Weight-loading option `init_visual_from: <checkpoint>`
  - [x] Compare P with and without initialisation on validation WER
- **Decision:** Keep `init_visual_from: checkpoints/b2_visual_only_best.pt` enabled in `configs/p_proposed.yaml` (reduces 5-epoch validation WER from 97.10% to 35.05% and CER from 74.29% to 16.44%; see `reports/w4_05_curriculum_comparison.md`).
- **Done when:** A decision is recorded on the card.

### W4-06 · Gate and attention analysis script
- **Owner:** Lisa Lusinga Namugenyi
- **Reviewer:** Cole Rodney Migisha
- **Labels:** Evaluation
- **Due:** Fri 13 Nov 2026
- **Depends on:** W3-02, W4-02
- **Estimate:** 4 h
- **Description:** Shows whether the fusion actually relies more on the lips when the audio is noisy. This is not guaranteed; the analysis tests it.
- **Checklist:**
  - [x] Distribution of mean gate ḡ per utterance at each SNR (box plot)
  - [x] Expected if it works: ḡ decreases as SNR falls
  - [x] Plot of per-frame gate for example utterances
  - [x] Optional: attention-weight heatmap for one utterance
- **Status (16 Nov):** Done on 16 Nov: `scripts/analyse_gate.py`; results in `reports/gate_analysis/` (mean gate 0.804 clean → 0.691 at −5 dB, monotonic, Spearman ρ 0.47 over 3,998 test utterances).
- **Done when:** Plots are generated from the first P checkpoint.

### W4-07 · Ablation configurations X1–X3
- **Owner:** Ainomugisha Faith
- **Reviewer:** Charis Opol
- **Labels:** Model, Training
- **Due:** Sat 14 Nov 2026
- **Depends on:** W4-04
- **Estimate:** 5 h
- **Description:** Prepare and smoke-test the three required ablations so they can run in Week 5.
- **Checklist:**
  - [x] X1: P with `modality_dropout: {audio: 0, visual: 0}`
  - [x] X2: P with concatenation in place of attention, keeping modality dropout
  - [x] X3: P with a 2D per-frame CNN in place of the 3D front-end (implement the 2D variant: same channels, kernels 1×k×k)
  - [x] Each config runs 50 steps without error
  - [x] Everything else identical to P (same data, seeds, schedule)
- **Status (16 Nov):** Done on 16 Nov: `configs/x1_no_modality_dropout.yaml`, `x2_concat_fusion.yaml`, `x3_2d_frontend.yaml`; each ran a 50-step epoch on the Alienware without error. The full runs are F-02.
- **Done when:** All three configs are merged and smoke-tested.

### W4-08 · Inference script (`transcribe.py`)
- **Owner:** Wakisa Isabelle
- **Reviewer:** Ansasire Nehum Naamaye
- **Labels:** App
- **Due:** Thu 12 Nov 2026
- **Depends on:** W3-07, W3-10
- **Estimate:** 4 h
- **Description:** The "clean inference script" promised in the proposal.
- **Checklist:**
  - [x] CLI: `python transcribe.py --checkpoint X --video clip.mp4 [--snr 0]`
  - [x] Uses only the shared preprocessing and stored normalisation statistics
  - [x] Prints the caption and timing (preprocessing, model, decoding)
  - [x] Works on CPU and GPU
  - [x] End-to-end test: test clip gives a non-empty lower-case string
- **Status (16 Nov):** Done on 16 Nov: `--snr`, `--noise-splits` and a timing line; end-to-end CLI test in `tests/test_app.py`. `--model grid|real` stays in F-07.
- **Done when:** The test passes and usage is documented.

### W4-09 · Gradio Mode A upgrades
- **Owner:** Wakisa Isabelle
- **Reviewer:** Ansasire Nehum Naamaye
- **Labels:** App
- **Due:** Sun 15 Nov 2026
- **Depends on:** W4-04, W4-08
- **Estimate:** 5 h
- **Description:** Make the demo show *why* the fused model helps.
- **Checklist:**
  - [x] Switch to the P checkpoint (keep a B1 option for comparison)
  - [x] Show the mouth-crop strip
  - [x] Plot the per-frame fusion gate
  - [x] Noise control: add MUSAN noise at a chosen SNR to the uploaded clip
  - [x] Side-by-side captions: audio-only vs. audio-visual
- **Status (16 Nov):** Done on 16 Nov in the Gradio app (`lipscribe/app/demo.py`): P with an optional B1, side-by-side captions, per-frame gate plot and a noise dropdown. The browser demo's versions are F-05 and F-06.
- **Done when:** The team can demonstrate clean vs. noisy captions on the same clip.

### W4-10 · Live Mode B: capture threads and ring buffers
- **Owner:** Ansasire Nehum Naamaye
- **Reviewer:** Wakisa Isabelle
- **Labels:** App
- **Due:** Sun 15 Nov 2026
- **Depends on:** W2-10
- **Estimate:** 8 h
- **Description:** Foundations of the live captioning loop (Figure 3 of the implementation document).
- **Checklist:**
  - [x] Camera thread (OpenCV) storing frames with timestamps from one monotonic clock
  - [x] `sounddevice` callback at 16 kHz into an audio ring buffer
  - [x] Both buffers hold the last 4 s
  - [x] Run Face Mesh on each new frame as it arrives (causal EMA smoothing), storing crops not full frames
  - [x] Window extraction: 75 frames nearest to 40 ms targets + matching 48,000 samples
  - [x] Display window in OpenCV
- **Status (16 Nov):** Done on 16 Nov: `lipscribe/app/live.py` and `scripts/live_capture.py`, tested with fake camera and microphone sources (`tests/test_live.py`); not yet run on real hardware.
- **Done when:** The loop runs for 2 minutes and outputs correctly shaped windows.

### W4-11 · Final report skeleton and methods drafts
- **Owner:** Lisa Lusinga Namugenyi
- **Reviewer:** Charis Opol
- **Labels:** Docs
- **Due:** Sun 15 Nov 2026
- **Depends on:** —
- **Estimate:** 4 h (editor) + 2 h per section author
- **Description:** Start the report early so Week 6 is for results, not writing from scratch. Section authors write their own parts.
- **Checklist:**
  - [x] Report skeleton (reuse the IEEE Access template from the proposal)
  - [x] Dataset & noise section — Ainomugisha Faith
  - [x] Visual and audio pipelines — Cole Rodney Migisha
  - [x] Model and training — Charis Opol
  - [x] Application — Wakisa Isabelle
  - [x] Testing — Ansasire Nehum Naamaye
- **Status (16 Nov):** Done on 16 Nov: skeleton in `documentation/5_submissions/final_report/` with first drafts of all five methods sections; each author checks and extends theirs in F-11.
- **Done when:** All methods sections have a first draft.

### W4-M · MILESTONE M5 — Multimodal fusion
- **Owner:** Charis Opol
- **Reviewer:** All
- **Labels:** Milestone, PM
- **Due:** Sun 15 Nov 2026
- **Status (16 Nov):** Closed on 16 Nov: X1–X3 configs ready (W4-07); GPU plan in `documentation/6_compute/compute.md`, Section 4.
- **Checklist:**
  - [x] Fusion unit tests pass
  - [x] No NaN under full-modality dropout
  - [x] P trains stably and has first results
  - [x] X1–X3 configs ready
  - [x] `transcribe.py` merged
  - [x] Plan GPU time for Week 5 (sweeps + final runs + ablations)

---

## Week 5 — Final Week: Real-World Model, Report & Submission (Mon 16 Nov – Sun 22 Nov 2026)

**Milestones:** M6–M9 combined (evaluation, application, testing and demonstration)
**Week goal:** Two frozen models, one for each question the project answers, and a submitted report, demo and release by **Sun 22 Nov, 18:00**.

The team re-planned on Mon 16 Nov to finish one week early. Two models are now frozen and nothing is retrained except the X1–X3 ablations:

| Role | Model | Checkpoint | Test WER (greedy) |
|---|---|---|---|
| **GRID benchmark** (core result) | P4: Model P initialised from B1 + B2 v2 | `checkpoints/p_proposed_best.pt` | 1.02 % clean, 5.34 % at −5 dB (0.71 % / 3.95 % with the GRID grammar) |
| **Real-world model** (films and TED, open vocabulary) | ws_v5: Flix v5 warm start, WavLM + Auto-AVSR lips, reliability-gated fusion | `runs/flix_v5/flix_v5_fusion_gated_ws_20261116-093641/best.pt` (Alienware) | TED 18.2 % clean, 30.6 % at −10 dB; films 30.1 % clean |

The audio-only twin twin_v5 (`runs/flix_v5/flix_v5_fusion_wp_audio_20261116-000004/best.pt`) is the real-world model's baseline: it scores 17.3 % on clean TED but 43.8 % at −10 dB, so the lips cut noisy-speech errors by 13 points. Flix v6, which retrained the warm start on audio that does not belong to the face, is reported as a negative result. It stays near lips-only level when the voice is someone else's (TED 44.1 % against ws_v5's 103.3 %) but loses 17 points on clean speech. ws_v5 therefore remains the real-world model, and its weakness with mismatched audio goes into the limitations. Sources: `reports/flix_v5/decision_log.md` and `reports/flix_v6/decision_log.md`.

**Roles this week:** as in Part B, except that Wakisa Isabelle now works only on the demo's frontend (F-06). Her earlier duties moved by role: `transcribe.py` and the demo video to Ainomugisha Faith (`transcribe.py` reuses her preprocessing), and the slides to Charis Opol.

**Daily rhythm:** a 15-minute check-in at 18:00 each day (done / doing / blocked), and a card moves to `Done` only after its Reviewer approves it, as before.

### F-00 · Re-plan and freeze both models
- **Owner:** Charis Opol
- **Reviewer:** Lisa Lusinga Namugenyi
- **Labels:** PM, Milestone
- **Due:** Mon 16 Nov 2026
- **Depends on:** —
- **Estimate:** 1 h
- **Description:** Agree this plan, freeze P4 and ws_v5, and book the Alienware GPU for the week.
- **Checklist:**
  - [ ] Team agrees the plan and the owners below
  - [ ] Record P4 as the final GRID model (chosen on validation in Week 3; no further tuning)
  - [ ] Record ws_v5 as the real-world model, with twin_v5 as its baseline and v6 as a negative result
  - [ ] GPU schedule: X1–X3 (Mon–Tue nights), ws_v5 decoder tuning (Tue day), demo and latency (Wed–Thu)
- **Done when:** The plan is merged to `main` and announced in the group chat.

### F-01 · Merge the open branches into `main`
- **Owner:** Charis Opol
- **Reviewer:** Ansasire Nehum Naamaye
- **Labels:** PM, Testing
- **Due:** Tue 17 Nov 2026
- **Depends on:** F-00
- **Estimate:** 3 h
- **Description:** `main` lacks two weeks of work: Flix v3–v6, `demo/` and the Week 4 report live on `cole`, and the progress ticks and the `FEATURE_DIMS` fix are in PR #58.
- **Checklist:**
  - [ ] Review and merge PR #58 (`alienware`: this plan, progress ticks and the `FEATURE_DIMS` import fix)
  - [ ] Merge `main` into `cole` and resolve conflicts (`cole` predates the caption app, `lipscribe/app/`)
  - [ ] Open and merge the PR `cole` → `main`
  - [ ] Full test suite passes on `main`
- **Done when:** `main` holds all work and `pytest` passes on it.

### F-02 · Ablations X1–X3 on GRID
- **Owner:** Ainomugisha Faith
- **Reviewer:** Cole Rodney Migisha
- **Labels:** Training, Evaluation, GPU
- **Due:** Wed 18 Nov 2026
- **Depends on:** F-00
- **Estimate:** 4 h of work + two overnight GPU runs
- **Description:** The ablations that REF-03 never drops. Each changes one thing in P4's config (P4 = `p_init_b1_b2_v2`) and keeps the data, seed, schedule and initialisation.
- **Checklist:**
  - [x] X1: P4 with `modality_dropout: {audio: 0, visual: 0}`
  - [x] X2: P4 with concatenation fusion in place of cross-modal attention, keeping modality dropout
  - [x] X3: P4 with a 2D per-frame front-end in place of the 3D one (Cole Rodney Migisha implements it: same channels, kernels 1×k×k)
  - [x] Each config runs 50 steps without error before the overnight run
  - [x] Run the three on the Alienware, Mon and Tue nights (the pipeline's retry and watchdog apply)
  - [x] Evaluate each at the five SNRs with `scripts/evaluate.py`; rows in `experiments/results.csv`
- **Done when:** X1–X3 have test WER/CER at five SNRs beside P4 in `results.csv`.
- **Status (16 Nov):** Done on 16 Nov, 19:33. All three ran on P4's real budget (6 epochs, one seed). The first X1 run used 40 epochs, so it was stopped and rerun. Test WER %, clean / −5 dB, greedy (grammar): P4 1.02 / 5.34 (0.71 / 3.95); X1 1.57 / 6.33 (1.11 / 4.93); X2 1.23 / 5.89 (0.73 / 3.96); X3 1.40 / 7.50 (0.98 / 5.60). Modality dropout and the 3D front end both help. Gated attention beats concatenation with greedy decoding, but the two tie with the grammar decoder; with one seed, call them comparable. Reports are in `reports/x*_2026111*`, and the tables are in `analysis/01_grid/tables`.

### F-03 · Final results and fusion analysis (GRID, films, TED)
- **Owner:** Lisa Lusinga Namugenyi
- **Reviewer:** Ainomugisha Faith
- **Labels:** Evaluation
- **Due:** Wed 18 Nov 2026
- **Depends on:** F-02
- **Estimate:** 5 h
- **Description:** Completes the GRID evaluation of the Week 3 report. It takes over the old cards W5-05 and W5-06 (done items ticked below) and the rest of W4-06.
- **Checklist:**
  - [x] WER and CER table per model × SNR (Week 3 report)
  - [x] WER-vs-SNR plot (`snr_comparison_curves.png`)
  - [x] Clean-condition target checked: P4 1.02 % < 10 %
  - [x] P4 audio-only and video-only with `e_miss` at all SNRs (Week 3 missing-modality table)
  - [ ] Per-speaker WER for the four test speakers (P4, B1, B3 v2)
  - [ ] Paired bootstrap 95 % CI for P4 vs B1 at each SNR (resample sentences)
  - [x] Box plot of the mean gate ḡ per utterance at each SNR, for P4
  - [x] Per-frame gate plot for three example utterances (clean and −5 dB)
  - [ ] Ablation table: P4 vs X1, X2, X3 at five SNRs
  - [ ] One paragraph per finding for the report
  - [ ] Films: the analysis listed in `analysis/02_films/README.md` (lip gain with CIs, per-film spread, edge words, wrong audio)
  - [ ] TED: the analysis listed in `analysis/03_ted/README.md` (the same, per talk)
  - [ ] Cross-source figure and paragraph (`analysis/04_cross_source/README.md`)
- **Where to look:** `analysis/README.md`. Tables for GRID, films and TED are already generated (`python analysis/scripts/build_tables.py` rebuilds them), and each section's README lists what is left.
- **Done when:** Every table and figure for the report (GRID, films, TED) is generated from `analysis/` or the evaluation CSVs.

### F-04 · Real-world model: decoder and final numbers
- **Owner:** Cole Rodney Migisha
- **Reviewer:** Charis Opol
- **Labels:** Evaluation, GPU
- **Due:** Tue 17 Nov 2026
- **Depends on:** F-00
- **Estimate:** 4 h
- **Description:** Pick ws_v5's decoder on validation, then score the locked v5 test sets once more with it. Beam + 4-gram LM gained 0.5–1 point for the v4 models (`reports/flix_v4/beam.md`).
- **Checklist:**
  - [x] Tune beam width and the LM's alpha/beta on `film_val` + `ted_val` (`scripts/flix_v4_beam.py`, `lm/flix_train_4gram.arpa`)
  - [x] Score twin_v5 and ws_v5 on the v5 test sets once: greedy and beam + LM at clean, 0, −5, −10 dB
  - [x] Table of the v6 wrong-audio conditions (silent, talk-over, replace) for the limitations
  - [x] Copy the twin_v5 and ws_v5 checkpoints and configs to the release folder
- **Done when:** `reports/flix_v5/final_decoder.md` holds the decoder settings and the final real-world table.
- **Status (17 Nov):** Done on 17 Nov, 07:50 (`scripts/flix_v5_beam.py`, run by `scripts/flix_v5_beam_run.sh`). On validation, the LibriSpeech + own-transcript 4-gram won for every model. ws_v5 uses alpha 0.2, beta 1.0, width 32; twin_v5 uses alpha 0.2, beta 0.5, width 64. Test WER % with beam + LM, ws_v5 (twin_v5): TED clean 16.9 (15.7), −10 dB 28.5 (42.3); films clean 27.3 (27.7), −10 dB 49.2 (64.5). The decoder gains 1–3 points over greedy. The lips cut 14–15 points at −10 dB (CIs exclude zero) but cost 1.3 points on clean TED (CI [+0.6, +1.8]). The wrong-audio table is unchanged in kind: ws_v5 still fails on replaced audio (TED 103 %). Checkpoints, configs and the decoder settings are in `~/Desktop/lipscribe_release/real_world/` on the Alienware, with `SHA256SUMS`.

### F-05 · Demo backend on the real-world model
- **Owner:** Cole Rodney Migisha
- **Reviewer:** Wakisa Isabelle
- **Labels:** App, Model, GPU
- **Due:** Wed 18 Nov 2026
- **Depends on:** F-04
- **Estimate:** 6 h
- **Description:** The browser demo (`demo/`) is the live application: record a clip on a phone or laptop and get a caption in 2–3 s. Its "Any speech" toggle still runs the old v2 film model. Switch `demo/server.py` to ws_v5 and return everything the new frontend (F-06) shows. The "GRID sentences" toggle stays P4 + grammar.
- **Checklist:**
  - [x] "Any speech" loads ws_v5 with its own preprocessing (Auto-AVSR mouth crop, WavLM audio, word pieces) and the F-04 decoder
  - [x] Both captions for every recording: twin_v5 (audio only) and ws_v5 (audio + lips), over the page's WebSocket (`/ws/live`) instead of `POST /caption`
  - [x] It returns the per-frame gate and accepts an SNR, adding MUSAN noise to the clip before decoding
  - [x] Clear error codes for no face, no audio and too-short clips
  - [x] Agree the message format with Wakisa Isabelle (her WebSocket format in `demo/README.md`, extended with `snr`, uploads and new error codes, with her agreement)
  - [x] Parity: 32 v5 test clips (16 TED, 16 film), batched as in the evaluation, give the same captions as `scripts/flix_v5_eval.py` (greedy, 64 of 64 model outputs) and `scripts/flix_v5_beam.py` (beam + LM, 64 of 64). Run one at a time, as the demo runs them, 37 of 40 match: WavLM's normalisation sees a batch's padding.
  - [x] Update the backend part of `demo/README.md`
- **Done when:** The endpoint serves both models with the gate and noise option, and the parity check passes.
- **Status (17 Nov):** Done. `demo/server.py` now serves `WS /ws/live` for three runs: Live (partials about every 0.5 s), Clip, and **Upload**, which was added at Cole's request. Upload takes a file of any length (a whole film); the backend asks for the 4 MB pieces it needs, so an MP4 with its index at the end starts at once, and it sends captions back in batches as subtitle cues while the file uploads. Each upload cue says whether the lips may be used (the `scripts/mine_flix.py` training-clip rules), and the page's "Best of both" track uses ws_v5 only there. Code: `lipscribe/app/{session,stream,realworld}.py`; tests: `tests/test_stream.py` (281 passed on the Alienware). A 10 min test-film excerpt is captioned in 54 s. Cole also added the Upload section, the test-noise control and `.srt` export to the page, with Wakisa Isabelle's agreement.

### F-06 · Demo frontend
- **Owner:** Wakisa Isabelle
- **Reviewer:** Cole Rodney Migisha
- **Labels:** App
- **Due:** Thu 19 Nov 2026
- **Depends on:** F-05 (response format agreed on Tue)
- **Estimate:** 8 h
- **Description:** The page people see in the demo, `demo/index.html` (one static page, no build step). It replaces the Gradio Mode A upgrades of W4-09.
- **Checklist:**
  - [ ] Model toggle: "GRID sentences" (P4) and "Any speech" (ws_v5)
  - [ ] Side-by-side captions: "Ears only" (twin_v5) and "Ears + lips" (ws_v5)
  - [ ] Per-frame gate plot: how much the model trusted the sound vs the lips
  - [x] Noise control: an SNR slider (clean, 10, 5, 0, −5, −10 dB) sent with the clip (added by Cole with F-05, as a menu)
  - [ ] Mouth-crop strip, recording timer and a loading state while the server works
  - [ ] Friendly messages for the backend's error codes (no face, no audio, clip too short)
  - [ ] Works on a phone and a laptop browser (Chrome, Firefox, Safari)
  - [ ] Update the user part of `demo/README.md`
- **Done when:** A phone recording shows both captions, the gate plot and the noise control, on two browsers.

### F-07 · `transcribe.py` for both models
- **Owner:** Ainomugisha Faith
- **Reviewer:** Ansasire Nehum Naamaye
- **Labels:** App, Testing
- **Due:** Thu 19 Nov 2026
- **Depends on:** F-05
- **Estimate:** 3 h
- **Description:** Finishes W4-08.
- **Checklist:**
  - [ ] `--model grid|real` (P4 + grammar, or ws_v5 + F-04 decoder) besides `--checkpoint`
  - [x] `--snr` adds MUSAN noise at that SNR before decoding
  - [x] Prints the caption and the time for preprocessing, the model and decoding
  - [x] End-to-end CLI test: a test clip gives a non-empty lower-case string on CPU
- **Done when:** The test passes and the README documents the usage.

### F-08 · Latency and real-time factor
- **Owner:** Ansasire Nehum Naamaye
- **Reviewer:** Cole Rodney Migisha
- **Labels:** App, Evaluation
- **Due:** Thu 19 Nov 2026
- **Depends on:** F-05, F-06
- **Estimate:** 3 h
- **Description:** Replaces W5-09 for the browser demo.
- **Checklist:**
  - [ ] Per 3 s clip on the Alienware GPU and on a laptop CPU: mouth cropping, model and decoding time, for P4 and ws_v5
  - [ ] Real-time factor (target ≤ 0.5)
  - [ ] End-to-end demo latency: recording uploaded → caption shown
- **Done when:** A latency table is ready for the report.

### F-09 · Tests and fresh-clone reproducibility
- **Owner:** Ansasire Nehum Naamaye
- **Reviewer:** Ainomugisha Faith
- **Labels:** Testing
- **Due:** Fri 20 Nov 2026
- **Depends on:** F-01, F-10
- **Estimate:** 4 h
- **Description:** Combines the old cards W6-03 (fresh clone) and the testing parts of W6-01 and W6-02.
- **Checklist:**
  - [ ] Full test suite passes on `main`; keep the output for the release
  - [ ] Fresh clone on a clean machine, installed from the README only
  - [ ] Reproduce P4's GRID test WER from the released checkpoint
  - [ ] Demo failure cases: no face, mouth covered, microphone denied, background noise
  - [ ] Report every unclear README step to its owner
- **Done when:** A fresh clone reproduces P4's numbers and the demo failure cases give clear messages.

### F-10 · README and setup guide
- **Owner:** Ansasire Nehum Naamaye
- **Reviewer:** Ainomugisha Faith
- **Labels:** Docs
- **Due:** Thu 19 Nov 2026
- **Depends on:** F-01
- **Estimate:** 4 h
- **Description:** Replaces W6-04.
- **Checklist:**
  - [ ] Installation (environment files, MediaPipe and PyTorch notes)
  - [ ] GRID: data download, preprocessing, training and evaluation commands
  - [ ] Real-world model: a short summary of the Flix data pipeline and how to run ws_v5
  - [ ] `transcribe.py` and demo usage
  - [ ] Where checkpoints, configs and results live
- **Done when:** F-09's fresh-clone test passes using only the README.

### F-11 · Final technical report
- **Owner:** Lisa Lusinga Namugenyi (editor)
- **Reviewer:** Charis Opol
- **Labels:** Docs
- **Due:** Draft Thu 19 Nov; final Sat 21 Nov 2026
- **Depends on:** F-03, F-04, F-08
- **Estimate:** 6 h editing + 2–3 h per section author
- **Description:** Replaces W4-11, W5-10 and W6-05. Use the IEEE Access template of the proposal. The report answers two questions: does lip-assisted fusion help on GRID (P4), and does it carry over to real speech (ws_v5)?
- **Checklist:**
  - [ ] Introduction and related work — Charis Opol
  - [ ] Data and noise: GRID, MUSAN, the Flix film and TED pipeline — Ainomugisha Faith
  - [ ] Visual and audio pipelines; the real-world model and the v4–v6 story — Cole Rodney Migisha
  - [ ] Model P and training — Charis Opol
  - [ ] GRID results, ablations and fusion analysis (F-03) — Lisa Lusinga Namugenyi
  - [ ] Real-world results (F-04), including v6 as a negative result — Cole Rodney Migisha
  - [ ] Application and latency (F-05, F-06, F-07, F-08) — Ansasire Nehum Naamaye, with the frontend paragraph from Wakisa Isabelle
  - [ ] Testing (F-09) — Ansasire Nehum Naamaye
  - [ ] Limitations and future work: GRID's closed vocabulary, edge words, voices that do not match the face (an audio-visual sync signal), synthetic noise
  - [ ] Every number traced to `experiments/results.csv`, `reports/flix_v5/` or `reports/flix_v6/`
  - [ ] Full-team proofread (Sat)
- **Done when:** The PDF is final and every number matches its source.

### F-12 · Presentation slides
- **Owner:** Charis Opol
- **Reviewer:** Lisa Lusinga Namugenyi
- **Labels:** Docs
- **Due:** Fri 20 Nov 2026
- **Depends on:** F-03, F-04
- **Estimate:** 5 h
- **Description:** Replaces W6-06; reuse the 12 Nov progress presentation where it still holds.
- **Checklist:**
  - [ ] Problem and motivation
  - [ ] Architecture of P4 and of ws_v5
  - [ ] GRID WER vs SNR, and the ablations
  - [ ] Gate behaviour plot
  - [ ] Real-world results: twin_v5 vs ws_v5 in noise
  - [ ] Demo section and limitations
  - [ ] Speakers assigned per slide (all six members)
- **Done when:** The deck is reviewed and every member knows their slides.

### F-13 · Recorded fallback demo video
- **Owner:** Ainomugisha Faith
- **Reviewer:** Ansasire Nehum Naamaye
- **Labels:** App, Docs
- **Due:** Fri 20 Nov 2026
- **Depends on:** F-06
- **Estimate:** 2 h
- **Checklist:**
  - [ ] GRID sentences mode: the same sentence clean and at 0 dB
  - [ ] Any speech mode: twin_v5 vs ws_v5 captions with the noise control on
  - [ ] Only team members appear, with consent; no third-party recordings
- **Done when:** A 2–3 minute video is in the submission folder.

### F-14 · Core freeze and release package
- **Owner:** Charis Opol
- **Reviewer:** Cole Rodney Migisha
- **Labels:** PM, Docs
- **Due:** Freeze Thu 19 Nov; release Sat 21 Nov 2026
- **Depends on:** F-02, F-04, F-05, F-06
- **Estimate:** 3 h
- **Description:** Replaces the W5-M core freeze and W6-09.
- **Checklist:**
  - [ ] Tag `v1.0-core` on Thu once F-02, F-04, F-05 and F-06 are merged; afterwards only bug fixes
  - [ ] Release folder: B1, B2, B3, P4, X1–X3, twin_v5 and ws_v5 checkpoints with their configs
  - [ ] `experiments/results.csv`, the Flix decision logs and the pytest output from F-09
  - [ ] Tag `v1.0-final` on Sat
  - [ ] Delete personal recordings that are no longer needed
- **Done when:** Both tags exist and the release folder is complete.

### F-15 · Demo rehearsal
- **Owner:** Charis Opol
- **Reviewer:** Lisa Lusinga Namugenyi
- **Labels:** PM
- **Due:** Sat 21 Nov 2026
- **Depends on:** F-12, F-13
- **Estimate:** 2 h (all six members)
- **Checklist:**
  - [ ] Full run-through with timing
  - [ ] Live demo on the presentation laptop: browser, Alienware tunnel, room lighting and microphone
  - [ ] Agree who switches to the fallback video, and when
  - [ ] Answers ready: why GRID, why CTC, does fusion help, why ws_v5 and not v6
- **Done when:** The team has run the full presentation once without stopping.

### F-16 · Final submission
- **Owner:** Charis Opol
- **Reviewer:** Lisa Lusinga Namugenyi
- **Labels:** PM
- **Due:** Sun 22 Nov 2026, 18:00
- **Depends on:** F-11, F-12, F-13, F-14
- **Estimate:** 1 h
- **Checklist:**
  - [ ] Report PDF
  - [ ] Code repository link (tag `v1.0-final`)
  - [ ] Slides
  - [ ] Demo video
  - [ ] Confirm the submission was received
- **Done when:** The submission is confirmed.

### F-M · MILESTONE M6–M9 — Evaluation, application, testing and demonstration
- **Owner:** Charis Opol
- **Labels:** Milestone
- **Due:** Sun 22 Nov 2026
- **Checklist:**
  - [x] All GRID models evaluated on identical test audio (fixed noisy test sets, W2-06)
  - [x] GRID results traceable to a config and checkpoint (`experiments/results.csv`)
  - [ ] Both models frozen; every real-world number traceable to a config and checkpoint
  - [ ] X1–X3 evaluated at five SNRs
  - [ ] Demo runs ws_v5 and P4 and passes its parity check
  - [ ] Fresh clone reproduces P4; all automated tests pass
  - [ ] Report, slides, video and release submitted
  - [ ] Retrospective: what went well / what to improve
- **Done when:** Every item above is ticked.

### F-DROP · Dropped in the 16 Nov re-plan
- **Owner:** Charis Opol
- **Labels:** PM
- **Description:** Work dropped to finish by Sun 22 Nov, following REF-03's drop order. Each item goes into the report's future-work section where relevant.
- **Checklist:**
  - [x] Optional extensions OPT-01 (EfficientNet-B0 trunk), OPT-02 (SNR auxiliary head), OPT-04 (emotion) and OPT-05 (language identification); OPT-03 (grammar decoder) is done
  - [x] Learning-rate and modality-dropout sweeps for P (W5-01, W5-02): P4 was chosen on validation in Week 3
  - [x] Extra seeds and final reruns of B1–B3 and P (W5-03): the Week 3 runs are final
  - [x] Live Mode B (W4-10, W5-07, W5-08 and its W6-01 tests): the browser demo replaces it
  - [x] Gradio Mode A upgrades beyond W3-10: moved into the browser demo (F-05, F-06)
  - [x] A Flix v7 with an audio-visual sync signal: future work, not this week

---

## Part D — Workload summary (core cards, excluding milestones)

| Member | W1 | W2 | W3 | W4 | Final week | Total |
|---|---|---|---|---|---|---|
| Charis Opol | 2 | 4 | 1 | 1 | 6 (F-00, F-01, F-12, F-14, F-15, F-16) | 14 |
| Cole Rodney Migisha | 1 | 2 | 2 | 3 | 2 (F-04, F-05) | 10 |
| Ainomugisha Faith | 3 | 2 | 2 | 1 | 3 (F-02, F-07, F-13) | 11 |
| Lisa Lusinga Namugenyi | 2 | 3 | 2 | 2 | 2 (F-03, F-11) | 11 |
| Wakisa Isabelle | 1 | 2 | 2 | 2 | 1 (F-06) | 8 |
| Ansasire Nehum Naamaye | 3 | 1 | 1 | 2 | 3 (F-08, F-09, F-10) | 10 |

Card counts are not hours. The heaviest final-week cards are F-02 (ablations), F-05 and F-06 (the demo) and F-11 (the report), and every member except Wakisa Isabelle also writes a report section. Charis Opol's cards are short coordination tasks. Rebalance at the daily check-in if anyone falls behind.
