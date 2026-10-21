# LipScribe — Work Breakdown Structure for Trello

**Project:** LipScribe: Lip-Assisted Speech Recognition for Robust Live Captioning
**Course:** Computer Vision and Deep Learning — Group 4, Uganda Christian University
**Source:** *LipScribe: Technical Implementation and System Development Document*, Section 12 (milestones M1–M9)
**Duration:** 6 weeks. Dates assume Week 1 starts **Monday 19 October 2026**. If the real start date is different, shift every due date by the same number of days.

---

## Part A — Instructions for Claude (board setup)

Use this file to create one Trello board.

1. **Board name:** `LipScribe — CVDL Group 4`
2. **Lists, left to right:**
   `📌 Team & Reference` · `Week 1 — Setup & Data` · `Week 2 — Preprocessing & Pipeline` · `Week 3 — Baselines` · `Week 4 — Multimodal Fusion` · `Week 5 — Tuning, Evaluation & Live Demo` · `Week 6 — Testing, Report & Demo` · `Optional Extensions (after core freeze)` · `In Progress` · `In Review` · `Done`
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

| Member | Primary role | Also responsible for |
|---|---|---|
| **Charis Opol** | Project coordinator & training lead | Trello board, weekly reviews, `train.py`, training runs, GPU booking, final submission |
| **Cole Rodney Migisha** | Model lead | Mouth cropping, visual encoder, audio encoder, attention fusion, modality dropout |
| **Ainomugisha Faith** | Data lead | GRID download, integrity checks, speaker splits, manifest, crop extraction, augmentation, ablation runs |
| **Lisa Lusinga Namugenyi** | Audio & evaluation lead | MUSAN noise pools, noise mixing, transcripts/tokeniser, `evaluate.py`, statistics, final-report editor |
| **Wakisa Isabelle** | Application lead | Decoding (greedy & beam search), metrics, `transcribe.py`, Gradio Mode A, slides, demo video |
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

### REF-04 · Experiment IDs
- **Labels:** Training, Evaluation
- **Description:** B1 audio-only · B2 visual-only · B3 concatenation fusion · P proposed attention fusion + modality dropout · X1 P without modality dropout · X2 P with concatenation in place of attention (keeping dropout) · X3 P with 2D per-frame CNN in place of 3D front-end · X4 P with EfficientNet-B0 trunk (optional) · X5 B1 with MFCC / log-Mel+MFCC features · X6 P with SNR auxiliary head (optional). SNR conditions: clean, 10, 5, 0, −5 dB.

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
  - [ ] Walk the team through the implementation document (architecture, schedule, experiment list)
  - [ ] Confirm or adjust the role allocation
  - [ ] Agree the weekly review slot and the chat channel
  - [ ] Create the board, add members, attach REF cards
  - [ ] Record decisions in a `docs/meeting_notes.md` file
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
  - [ ] Create the repo and add all six members as collaborators
  - [ ] Package `lipscribe/` with sub-packages `data`, `preprocessing`, `models`, `decoding`, `evaluation`, `app`
  - [ ] Folders `configs/`, `scripts/`, `tests/`, `tests/fixtures/`, `experiments/`, `docs/`
  - [ ] `.gitignore` excluding `data/`, `runs/`, `*.npy`, `*.pt`, `*.wav`, recordings
  - [ ] `pyproject.toml` so `pip install -e .` works
  - [ ] Protect `main`: pull requests with one approval
  - [ ] Pull request template with a "how I tested this" section
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
  - [ ] Check that MediaPipe has a wheel for Python 3.11 on every member's operating system
  - [ ] Install matching `torch`, `torchaudio`, `torchvision` (CPU build for laptops; CUDA build matching the GPU driver)
  - [ ] Install opencv-python, mediapipe, numpy, pandas, soundfile, librosa, sounddevice, scikit-learn, jiwer, matplotlib, tensorboard, pyyaml, pytest, gradio
  - [ ] Generate `requirements.txt` and `environment.yml` from the working environment (do not hand-write versions)
  - [ ] Write `scripts/check_env.py`: imports every library, prints versions and whether CUDA is available
  - [ ] Each member runs `check_env.py` and posts the output on this card
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
  - [ ] Ask the department/lecturer about a university GPU workstation
  - [ ] List team members' own GPUs (model, memory)
  - [ ] Check a cloud notebook option as a fallback (session limits, storage)
  - [ ] Confirm at least 100 GB of free disk where the data will live
  - [ ] Write `docs/compute.md`: machine, GPU, memory, how to log in, booking rules
  - [ ] Set up a shared GPU booking schedule (who trains when)
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
  - [ ] Download video, audio and alignment archives for every speaker
  - [ ] Extract to `data/raw/grid/s<N>/{video,audio,align}/`
  - [ ] Record source URL, download date and archive checksums in `data/raw/SOURCES.md`
  - [ ] Note any speakers with missing or incomplete video
  - [ ] Copy 2 utterances (video, audio, align) into `tests/fixtures/` for the tests
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
  - [ ] Download MUSAN into `data/raw/musan/`; record it in `SOURCES.md`
  - [ ] List all noise-subset files
  - [ ] Randomly assign files to train/val/test (suggested 70/15/15) with a fixed seed
  - [ ] Save `data/noise_splits.csv` (file path, pool) and commit it
  - [ ] Copy one short noise file into `tests/fixtures/`
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
  - [ ] Check each utterance has video, audio and alignment files
  - [ ] Check video decodes and has 75 ± 2 frames
  - [ ] Check audio duration matches video duration to within 40 ms
  - [ ] Check transcript has exactly six words from the GRID word list
  - [ ] Check audio is not silent or clipped throughout (RMS and peak)
  - [ ] Leave a hook for the Face Mesh success-rate check (≥ 90 % of frames), added in W2-02
  - [ ] Write `reports/integrity_report.csv` plus a summary of pass counts per check and per speaker
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
  - [ ] Exclude speakers with missing or badly corrupted video (from the integrity report)
  - [ ] Test: 4 speakers (2 female, 2 male)
  - [ ] Validation: 3 speakers
  - [ ] Training: all remaining speakers (~27)
  - [ ] Check gender balance in each partition
  - [ ] Write `data/splits.csv` (speaker → partition) and commit it
  - [ ] Add a unit test asserting the three speaker sets are pairwise disjoint
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
  - [ ] Parse `.align` files; drop silence (`sil`) and short-pause (`sp`) tokens
  - [ ] Lower-case; letters as single characters ("b"), digits as words ("four")
  - [ ] `lipscribe/data/tokenizer.py`: `encode(text) -> list[int]`, `decode(ids) -> str`, blank index 0
  - [ ] Unit test: `decode(encode(s)) == s` for all GRID word strings; `encode` never outputs blank
  - [ ] Record the longest transcript length (needed for the CTC length check)
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
  - [ ] Read frames with OpenCV, convert BGR → RGB, keep frame indices
  - [ ] Face Mesh in video mode, `max_num_faces=1`
  - [ ] Lip landmark indices from `FACEMESH_LIPS` (not hard-coded)
  - [ ] Square box around the lips + 25 % margin
  - [ ] Smooth centre and size with a centred 5-frame moving average
  - [ ] Crop with edge padding, resize to 96×96 (area interpolation), convert to grey `uint8`
  - [ ] Return a per-frame validity flag for failed detections
  - [ ] Unit test: output shape (96, 96), dtype `uint8`
  - [ ] Save a contact sheet of crops for 5 clips and attach it to the card
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
  - [ ] Preprocessing: `crop_mouths(video) -> uint8[T,96,96], valid[T]`, `load_audio(path) -> float32[N] @16 kHz`
  - [ ] Features: `logmel(wave) -> [T_a, 80]`
  - [ ] Encoders: visual `[B,1,T,96,96] -> [B,T,512]`, audio `[B,T_a,80] -> [B,T,512]`
  - [ ] Fusion `-> [B,T,256]`; CTC head `-> [T,B,28]` log-probabilities
  - [ ] Decoder `-> str`; `transcribe(video_path) -> str`
  - [ ] Batch dictionary keys returned by the data loader
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
  - [ ] `tests/conftest.py` with fixtures pointing to `tests/fixtures/`
  - [ ] GitHub Actions workflow: install CPU environment, run `pytest`
  - [ ] Add a status badge to the README
  - [ ] Write `docs/testing.md`: how to run tests locally
- **Done when:** CI runs green on a test pull request.

### W1-M · MILESTONE M1 — Environment & dataset verified
- **Owner:** Charis Opol
- **Reviewer:** All
- **Labels:** Milestone, PM
- **Due:** Sun 25 Oct 2026
- **Depends on:** W1-02 … W1-12
- **Description:** Weekly review. Check the completion criteria and re-plan Week 2.
- **Checklist:**
  - [ ] Every member runs the test suite successfully
  - [ ] Integrity report lists usable utterances per speaker
  - [ ] `splits.csv` committed; speaker sets disjoint
  - [ ] GPU access confirmed
  - [ ] Mouth-crop prototype approved
  - [ ] Unfinished cards moved to Week 2 with a comment explaining why

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
  - [ ] Failed frame → interpolate the box from the nearest successful frames; mark invalid in the mask
  - [ ] More than 10 % failed frames → flag the utterance for exclusion
  - [ ] Pad or truncate to the audio-derived frame count by repeating the last valid frame; mark padding in the mask
  - [ ] Never drop frames (this keeps audio–video alignment)
  - [ ] Unit test: a frame with no face gives an interpolated box and an invalid flag
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
  - [ ] Resumable: skip utterances already processed
  - [ ] Run in parallel across CPU cores
  - [ ] Save to `data/processed/mouths/s<N>/<utt_id>.npy`
  - [ ] Log the per-utterance Face Mesh success rate
  - [ ] Add the Face Mesh ≥ 90 % check to `verify_dataset.py`
  - [ ] Record the MediaPipe version and crop parameters in the manifest
  - [ ] Check total disk use (expected ~23 GB)
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
  - [ ] Script `scripts/contact_sheet.py`: 10 random utterances per speaker, frames 0/25/50/74
  - [ ] Inspect every speaker for mis-centred, cut-off or blurred crops
  - [ ] List problem utterances or speakers on this card
  - [ ] Agree fixes with Cole (margin, smoothing) if there are systematic problems
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
  - [ ] Read with `soundfile`, average to mono, resample to 16 kHz
  - [ ] Save to `data/processed/audio16k/s<N>/<utt_id>.wav`
  - [ ] Shared function `load_audio()` in `lipscribe/preprocessing/audio.py` (also used by the demo)
  - [ ] Check that durations still match the video
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
  - [ ] Random noise segment from the correct pool; loop the file if shorter than the speech
  - [ ] Compute powers over the whole clip; scale noise by α = √(Ps / (Pw·10^(SNR/10)))
  - [ ] Verify the measured SNR is within 0.01 dB of the target (assert)
  - [ ] If peak > 1, scale the whole mixture (this preserves SNR)
  - [ ] "clean" returns the original audio
  - [ ] Unit tests for all five conditions; output length equals input length; peak ≤ 1
  - [ ] Transcript is unchanged by mixing (documented in the docstring)
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
  - [ ] Use only the val/test noise pools
  - [ ] Save to `data/processed/noisy_test/<snr>/<utt_id>.wav`
  - [ ] Log the noise file, offset and measured SNR for every mixture
  - [ ] Listen to examples at each SNR and confirm −5 dB is very noisy but speech is present
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
  - [ ] 25 ms Hann window, 10 ms hop, 512-point FFT, 80 Mel bands, 20 Hz–8 kHz, log(x + 1e-6)
  - [ ] Trim to 4 × number of video frames (300 for a 3 s clip)
  - [ ] RMS normalisation to −25 dBFS (gain limited to +20 dB), applied after noise mixing
  - [ ] Config switch `audio_features: logmel80 | mfcc39 | logmel_mfcc`
  - [ ] Unit test: 3 s → [300, 80]; matches `librosa` within 1e-3
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
  - [ ] Sample training utterances under the training SNR distribution
  - [ ] Save `data/stats/feature_norm.npz` (audio mean/std per band, pixel mean/std)
  - [ ] Unit test: normalised training features have mean within 0.05 of 0 and std within 0.05 of 1
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
  - [ ] `build_manifest.py`: utterance id, speaker, transcript, frame count, duration, Face Mesh success rate, paths
  - [ ] Dataset filters the manifest by `splits.csv`; raises an error if a speaker is in two partitions
  - [ ] Training items: crops + clean audio mixed on the fly at an SNR drawn uniformly from {clean, 10, 5, 0, −5}
  - [ ] Val/test items: pre-mixed noisy files for a given SNR
  - [ ] Collate: pad waveforms and crops, feature lengths, encoder lengths ⌈T_a/4⌉, padding mask [B,T], concatenated targets
  - [ ] Assert the CTC condition T ≥ L + repeats for every item
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
  - [ ] Process a fixture clip through the "live" path (frames → crop → features)
  - [ ] Compare with the cached crop and the training feature path
  - [ ] Assert max error < 1e-5 for features and identical crops (or document the expected difference for causal smoothing)
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
  - [ ] Implement in `lipscribe/models/audio_encoder.py`
  - [ ] Length function using ⌊(L + 2p − k)/s⌋ + 1 twice
  - [ ] `pack_padded_sequence` with true lengths
  - [ ] Shape test: [2,300,80] → [2,75,512]
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
  - [ ] CTC head: sequence BiLSTM (128 per direction) → Linear(256→28) → log-softmax, output [T,B,28]
  - [ ] `collapse()` (merge repeats, remove blanks) and greedy decoder in `lipscribe/decoding/`
  - [ ] Unit test: `[b,b,0,i,0,n,n] → "bin"`, `[e,0,e] → "ee"`
  - [ ] WER/CER with `jiwer` at corpus level (sum errors, then divide)
  - [ ] Unit test: WER("bin blue at f two now", "bin blue at s two now") = 1/6
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
  - [ ] Load config; set Python/NumPy/PyTorch seeds; create run directory; save config and Git commit hash
  - [ ] AdamW (lr 3e-4, weight decay 1e-2, not on biases/norms); 1-epoch warm-up + cosine decay
  - [ ] Mixed precision (AMP); unscale, then clip gradient norm at 5.0
  - [ ] `nn.CTCLoss(blank=0, zero_infinity=True)`; skip and log non-finite losses
  - [ ] Validation each epoch: loss, greedy WER/CER per SNR and their mean
  - [ ] Checkpoints: latest and best by mean validation WER (model, optimiser, scheduler, scaler, epoch, config, vocab)
  - [ ] Early stopping (patience 8); resume from checkpoint
  - [ ] TensorBoard logging; append a summary row to `experiments/results.csv`
  - [ ] LSTM forget-gate bias initialised to 1
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
  - [ ] Record peak GPU memory at batch size 16; use gradient accumulation if it does not fit
  - [ ] B1 overfit: CTC loss < 0.05 and CER = 0 on 8 utterances within 300 steps
  - [ ] Record time per epoch for the full training set (for planning)
  - [ ] Launch the full B1 training run
- **Done when:** The overfit test passes and the full B1 run is training.

### W2-M · MILESTONE M2 — Preprocessing & alignment complete
- **Owner:** Charis Opol
- **Reviewer:** All
- **Labels:** Milestone, PM
- **Due:** Sun 1 Nov 2026
- **Depends on:** W2-01 … W2-14
- **Checklist:**
  - [ ] Crop contact sheets reviewed
  - [ ] Noise-mixing tests pass
  - [ ] One batch loads with consistent lengths
  - [ ] Train/inference parity test passes
  - [ ] B1 full run started
  - [ ] Re-plan Week 3; move unfinished cards with a reason

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
  - [ ] Check training curves for over/underfitting
  - [ ] Restore the best checkpoint
  - [ ] Evaluate at clean, 10, 5, 0, −5 dB
  - [ ] Post the WER/CER table and curves on the card
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
  - [ ] Load a checkpoint and its config
  - [ ] For each SNR: decode all test utterances (greedy or beam via config)
  - [ ] Corpus-level WER and CER: per SNR, per speaker, overall
  - [ ] Save per-utterance hypotheses to CSV (for error analysis)
  - [ ] Hook to record the mean fusion gate per utterance (used from Week 4)
  - [ ] Option to replace one modality with `e_miss` at test time (for Week 5 ablations)
  - [ ] Append to `experiments/results.csv`
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
  - [ ] Conv3d 1→32 (3×5×5, stride 1×2×2) + BN + ReLU + MaxPool 1×2×2
  - [ ] Conv3d 32→64 (3×5×5) + BN + ReLU + MaxPool 1×2×2
  - [ ] Conv3d 64→96 (3×3×3) + BN + ReLU + MaxPool 1×2×2
  - [ ] Temporal padding 1, temporal stride 1 (T stays 75)
  - [ ] Flatten 96·6·6 = 3456 → Linear 512 + ReLU + Dropout 0.3 → 2-layer BiLSTM (256 per direction)
  - [ ] Shape test: [2,1,75,96,96] → [2,75,512], finite values
  - [ ] Report the parameter count on the card
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
  - [ ] Random crop jitter (same offset for every frame of a clip)
  - [ ] Random horizontal flip (same for every frame of a clip)
  - [ ] Disabled for validation, test and inference
  - [ ] Unit test: all frames of an augmented clip get the same transform
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
  - [ ] Overfit test on 8 utterances (loss < 0.05)
  - [ ] Full run with `configs/b2_visual_only.yaml`
  - [ ] Monitor validation CER; it must fall clearly below that of an untrained model
  - [ ] If it has not improved after several epochs, raise it at the review (the curriculum option in W4-05)
  - [ ] Evaluate (performance should be roughly constant across SNR by design)
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
  - [ ] `lipscribe/models/fusion_concat.py`
  - [ ] Config `configs/b3_concat.yaml` (same optimiser, schedule and epochs as B1)
  - [ ] Overfit test, then full run (book GPU time with Charis)
  - [ ] Evaluate at all five SNRs
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
  - [ ] Track blank and non-blank probabilities per prefix
  - [ ] Repeated symbol needs a blank between copies
  - [ ] Beam width configurable (default 10)
  - [ ] Unit test: beam width 1 equals greedy on random posteriors
  - [ ] Measure decode time per utterance
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
  - [ ] Configs `b1_mfcc39.yaml`, `b1_logmel_mfcc.yaml`
  - [ ] Train and evaluate on the same protocol as B1
  - [ ] Decision on the card: keep A1 unless another configuration improves validation WER
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
  - [ ] Run-naming rule: `<experiment>_<seed>_<date>`
  - [ ] `docs/experiments.md`: how to start, resume and log a run
  - [ ] `scripts/plot_wer_vs_snr.py` (one line per model, from `results.csv`)
  - [ ] `scripts/plot_training_curves.py` (from TensorBoard logs)
  - [ ] Shared view of TensorBoard or an exported results sheet for the team
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
  - [ ] Webcam/microphone recorder and file-upload input
  - [ ] Extract frames (OpenCV) and audio (`ffmpeg`); resample to 25 fps and 16 kHz
  - [ ] Run the shared preprocessing, the model and beam search
  - [ ] Show the caption beside the video
  - [ ] Handle errors (no audio track, no face) with a friendly message
- **Done when:** A GRID test clip uploaded to the page gives the same caption as `evaluate.py`.

### W3-M · MILESTONES M3 & M4 — Baselines
- **Owner:** Charis Opol
- **Reviewer:** All
- **Labels:** Milestone, PM
- **Due:** Sun 8 Nov 2026
- **Checklist:**
  - [ ] B1 evaluated at all five SNRs
  - [ ] Encoder shape tests pass
  - [ ] B2 validation CER clearly below an untrained model
  - [ ] B3 evaluated at all SNRs (or running, with an expected finish date)
  - [ ] Beam search and `evaluate.py` merged
  - [ ] Risk check: Is GPU time enough? Is B2 learning? Re-plan Week 4.

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
  - [ ] Projections 512→256 + LayerNorm for each stream
  - [ ] `nn.MultiheadAttention`, 4 heads, d = 256, dropout 0.1, `batch_first=True`
  - [ ] Padding mask for keys
  - [ ] Return attention weights for analysis
  - [ ] Truncate the longer stream if the lengths differ by 1–2 frames
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
  - [ ] Gate module returning g for logging
  - [ ] Learned `e_miss` (256-d) that replaces a dropped stream's encoding (not an attention mask; masking all keys gives NaN)
  - [ ] Modality dropout active only in `model.train()`
  - [ ] Inference: if a face is missing in > 50 % of the window's frames, use `e_miss` for the visual stream
  - [ ] Config keys for both dropout rates
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
  - [ ] Output shapes for full model forward pass (Algorithm 2)
  - [ ] No NaN when audio is dropped, when video is dropped, and with padded batches
  - [ ] Gate values lie in (0, 1)
  - [ ] Modality dropout is inactive in eval mode
  - [ ] Both streams are never dropped together (statistical test over 1000 samples)
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
  - [ ] Config `configs/p_proposed.yaml` (matches Listing 2 of the implementation document)
  - [ ] Overfit test on 8 utterances
  - [ ] Full run; log mean gate value per SNR on validation
  - [ ] Check training is stable (no loss spikes or NaNs)
  - [ ] First evaluation at all five SNRs
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
  - [ ] Decide at the Week 3 review whether it is needed
  - [ ] Weight-loading option `init_visual_from: <checkpoint>`
  - [ ] Compare P with and without initialisation on validation WER
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
  - [ ] Distribution of mean gate ḡ per utterance at each SNR (box plot)
  - [ ] Expected if it works: ḡ decreases as SNR falls
  - [ ] Plot of per-frame gate for example utterances
  - [ ] Optional: attention-weight heatmap for one utterance
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
  - [ ] X1: P with `modality_dropout: {audio: 0, visual: 0}`
  - [ ] X2: P with concatenation in place of attention, keeping modality dropout
  - [ ] X3: P with a 2D per-frame CNN in place of the 3D front-end (implement the 2D variant: same channels, kernels 1×k×k)
  - [ ] Each config runs 50 steps without error
  - [ ] Everything else identical to P (same data, seeds, schedule)
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
  - [ ] CLI: `python transcribe.py --checkpoint X --video clip.mp4 [--snr 0]`
  - [ ] Uses only the shared preprocessing and stored normalisation statistics
  - [ ] Prints the caption and timing (preprocessing, model, decoding)
  - [ ] Works on CPU and GPU
  - [ ] End-to-end test: test clip gives a non-empty lower-case string
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
  - [ ] Switch to the P checkpoint (keep a B1 option for comparison)
  - [ ] Show the mouth-crop strip
  - [ ] Plot the per-frame fusion gate
  - [ ] Noise control: add MUSAN noise at a chosen SNR to the uploaded clip
  - [ ] Side-by-side captions: audio-only vs. audio-visual
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
  - [ ] Camera thread (OpenCV) storing frames with timestamps from one monotonic clock
  - [ ] `sounddevice` callback at 16 kHz into an audio ring buffer
  - [ ] Both buffers hold the last 4 s
  - [ ] Run Face Mesh on each new frame as it arrives (causal EMA smoothing), storing crops not full frames
  - [ ] Window extraction: 75 frames nearest to 40 ms targets + matching 48,000 samples
  - [ ] Display window in OpenCV
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
  - [ ] Report skeleton (reuse the IEEE Access template from the proposal)
  - [ ] Dataset & noise section — Ainomugisha Faith
  - [ ] Visual and audio pipelines — Cole Rodney Migisha
  - [ ] Model and training — Charis Opol
  - [ ] Application — Wakisa Isabelle
  - [ ] Testing — Ansasire Nehum Naamaye
- **Done when:** All methods sections have a first draft.

### W4-M · MILESTONE M5 — Multimodal fusion
- **Owner:** Charis Opol
- **Reviewer:** All
- **Labels:** Milestone, PM
- **Due:** Sun 15 Nov 2026
- **Checklist:**
  - [ ] Fusion unit tests pass
  - [ ] No NaN under full-modality dropout
  - [ ] P trains stably and has first results
  - [ ] X1–X3 configs ready
  - [ ] `transcribe.py` merged
  - [ ] Plan GPU time for Week 5 (sweeps + final runs + ablations)

---

## Week 5 — Tuning, Evaluation & Live Demo (Mon 16 Nov – Sun 22 Nov 2026)
**Milestones:** M6 Training & tuning · M7 Evaluation & ablations · M8 Application integration
**Week goal:** Final models are trained and evaluated, ablations are run, live mode works, and the core is frozen on Sunday.

### W5-01 · Learning-rate sweep for P
- **Owner:** Charis Opol
- **Reviewer:** Cole Rodney Migisha
- **Labels:** Training, GPU
- **Due:** Tue 17 Nov 2026
- **Depends on:** W4-04
- **Estimate:** 2 h of work + runtime
- **Description:** Tune the most sensitive hyperparameter on validation data only.
- **Checklist:**
  - [ ] Learning rates {1e-4, 3e-4, 1e-3} (shortened runs if time is tight)
  - [ ] Compare mean validation WER over all five SNRs
  - [ ] Record the chosen value on the card
- **Done when:** The learning rate is chosen and committed to the config.

### W5-02 · Modality-dropout rate sweep
- **Owner:** Cole Rodney Migisha
- **Reviewer:** Charis Opol
- **Labels:** Training, GPU
- **Due:** Wed 18 Nov 2026
- **Depends on:** W5-01
- **Estimate:** 2 h of work + runtime
- **Description:** Tune p_audio and p_visual over {0, 0.1, 0.25, 0.4} with p_audio + p_visual ≤ 0.5, as far as compute allows (prioritise a few combinations).
- **Checklist:**
  - [ ] Choose the combinations to run (at least 3)
  - [ ] Compare mean validation WER, and validation WER at 0 and −5 dB
  - [ ] Record the chosen rates
- **Done when:** Rates are chosen and committed.

### W5-03 · Final training runs (B1, B2, B3, P)
- **Owner:** Charis Opol
- **Reviewer:** Lisa Lusinga Namugenyi
- **Labels:** Training, GPU
- **Due:** Sat 21 Nov 2026
- **Depends on:** W5-01, W5-02
- **Estimate:** 3 h of work + runtime
- **Description:** Train final versions of all core models under identical conditions; 3 seeds (1, 2, 3) if compute allows, otherwise 1.
- **Checklist:**
  - [ ] Confirm all configs share data, noise, schedule and epochs
  - [ ] Run B1, B2, B3, P final (seed 1 first)
  - [ ] Additional seeds if time allows
  - [ ] Save best checkpoints to the release folder
- **Done when:** Final checkpoints exist for all four core models.

### W5-04 · Run ablations X1–X3
- **Owner:** Ainomugisha Faith
- **Reviewer:** Charis Opol
- **Labels:** Training, Evaluation, GPU
- **Due:** Sat 21 Nov 2026
- **Depends on:** W4-07, W5-02
- **Estimate:** 2 h of work + runtime
- **Description:** Run the three required ablations with the tuned P settings and evaluate them with the same per-SNR protocol.
- **Checklist:**
  - [ ] X1 (no modality dropout)
  - [ ] X2 (concatenation in place of attention, with dropout)
  - [ ] X3 (2D per-frame CNN in place of 3D)
  - [ ] Evaluate all at five SNRs; log to `results.csv`
- **Done when:** Ablation results are logged.

### W5-05 · Final evaluation and results tables
- **Owner:** Lisa Lusinga Namugenyi
- **Reviewer:** Wakisa Isabelle
- **Labels:** Evaluation
- **Due:** Sun 22 Nov 2026
- **Depends on:** W5-03, W5-04
- **Estimate:** 6 h
- **Description:** Evaluate every model with beam search on the fixed test audio of the 4 held-out speakers. No results are reported unless they come from this script.
- **Checklist:**
  - [ ] WER and CER table per model × SNR (fills Table 12 of the implementation document)
  - [ ] Per-speaker breakdown
  - [ ] Mean ± std over seeds where available
  - [ ] Paired bootstrap confidence interval for P vs. B1 at each SNR
  - [ ] WER-vs-SNR plot (using W3-09 scripts)
  - [ ] Check whether the clean-condition target (WER < 10 %) is met, and report it either way
- **Done when:** Tables and plots are in `experiments/final/`, traceable to configs and checkpoints.

### W5-06 · Fusion behaviour analysis and test-time modality ablations
- **Owner:** Lisa Lusinga Namugenyi
- **Reviewer:** Cole Rodney Migisha
- **Labels:** Evaluation
- **Due:** Sun 22 Nov 2026
- **Depends on:** W4-06, W5-03
- **Estimate:** 4 h
- **Description:** Shows how P uses each stream.
- **Checklist:**
  - [ ] Gate distribution per SNR for the final P
  - [ ] P with video replaced by `e_miss` (audio-only operation) at all SNRs
  - [ ] P with audio replaced by `e_miss` (visual-only operation) at all SNRs
  - [ ] If P is worse than B1 anywhere, work through the diagnosis checklist in Section 9.3 of the implementation document and write up the findings
- **Done when:** Analysis notes and plots are attached to the card.

### W5-07 · Live Mode B: windowing, VAD and captions
- **Owner:** Ansasire Nehum Naamaye
- **Reviewer:** Wakisa Isabelle
- **Labels:** App
- **Due:** Fri 20 Nov 2026
- **Depends on:** W4-10, W4-08
- **Estimate:** 8 h
- **Description:** Complete the windowed pseudo-streaming captioner.
- **Checklist:**
  - [ ] Inference worker: every 1 s, process the latest 3 s window
  - [ ] Energy-based voice-activity detection
  - [ ] Grey partial caption updated every second; black final caption after ≥ 500 ms silence
  - [ ] Caption history (last few final captions)
  - [ ] "Audio only" fallback with `e_miss` when the face is missing
  - [ ] Error handling: missing camera/microphone, worker exceptions skip the window and are logged
- **Done when:** Live captions appear for GRID-style sentences spoken to the webcam.

### W5-08 · Audio–video sync calibration (clap test)
- **Owner:** Wakisa Isabelle
- **Reviewer:** Ansasire Nehum Naamaye
- **Labels:** App, Testing
- **Due:** Thu 19 Nov 2026
- **Depends on:** W4-10
- **Estimate:** 2 h
- **Description:** Measure and correct the constant offset between camera and microphone on the demo machine.
- **Checklist:**
  - [ ] Record hand claps; find the frame where the hands meet and the audio onset
  - [ ] Average the offset over 5 claps
  - [ ] Store it in the demo config; subtract it during window extraction
- **Done when:** The offset is measured and applied.

### W5-09 · Latency and real-time factor measurement
- **Owner:** Ansasire Nehum Naamaye
- **Reviewer:** Charis Opol
- **Labels:** Evaluation, App
- **Due:** Sat 21 Nov 2026
- **Depends on:** W5-07
- **Estimate:** 3 h
- **Description:** Measure on the actual demo machine after a warm-up, over at least 100 windows (median and 95th percentile).
- **Checklist:**
  - [ ] Per-frame Face Mesh + crop time
  - [ ] Model forward time and beam-search time per window
  - [ ] End-to-end caption latency
  - [ ] Real-time factor (target ≤ 0.5)
  - [ ] If too slow: hop 1.5 s, smaller beam, GPU, or quantisation (in that order)
- **Done when:** A latency table is posted on the card.

### W5-10 · Results and discussion drafts for the report
- **Owner:** Lisa Lusinga Namugenyi
- **Reviewer:** Charis Opol
- **Labels:** Docs
- **Due:** Sun 22 Nov 2026
- **Depends on:** W5-05
- **Estimate:** 4 h
- **Description:** Write up results as they come in. Report outcomes faithfully, including any where fusion did not help.
- **Checklist:**
  - [ ] Results section with final tables and plots
  - [ ] Ablation discussion (inputs from Faith and Cole)
  - [ ] Limitations (GRID closed vocabulary, synthetic noise, frontal faces)
- **Done when:** First drafts of results and limitations exist.

### W5-M · MILESTONES M6 & M7 — Core freeze
- **Owner:** Charis Opol
- **Reviewer:** All
- **Labels:** Milestone, PM
- **Due:** Sun 22 Nov 2026
- **Checklist:**
  - [ ] Hyperparameter sweeps logged; final configuration chosen on validation data only
  - [ ] All core models evaluated on identical test audio
  - [ ] Every number traceable to a config and checkpoint
  - [ ] Tag the repository `v1.0-core` (core freeze); after this, only bug fixes on core code
  - [ ] Decide which optional extensions (if any) to attempt in Week 6

---

## Week 6 — Testing, Report & Demo (Mon 23 Nov – Sun 29 Nov 2026)
**Milestones:** M8 Application integration (complete) · M9 Testing, documentation and demonstration
**Week goal:** Everything is tested, reproducible, documented, rehearsed and submitted.

### W6-01 · Live demo stability and failure-case testing
- **Owner:** Ansasire Nehum Naamaye
- **Reviewer:** Wakisa Isabelle
- **Labels:** Testing, App
- **Due:** Tue 24 Nov 2026
- **Depends on:** W5-07
- **Estimate:** 3 h
- **Checklist:**
  - [ ] Mode B runs 5 minutes without error
  - [ ] Cover mouth, turn head, leave frame → "audio only" captions, no crash
  - [ ] Unplug or deny the microphone/camera → clear message
  - [ ] Test in a quiet room and with background noise played from a speaker
- **Done when:** All cases behave as expected; issues are fixed or documented.

### W6-02 · Application parity test (Mode A and B vs. `evaluate.py`)
- **Owner:** Wakisa Isabelle
- **Reviewer:** Lisa Lusinga Namugenyi
- **Labels:** Testing, App
- **Due:** Tue 24 Nov 2026
- **Depends on:** W4-09, W5-07
- **Estimate:** 3 h
- **Description:** Confirms the demo shows the evaluated model's real behaviour.
- **Checklist:**
  - [ ] Run 10 held-out GRID test clips through Mode A
  - [ ] Feed the same clips to Mode B buffers in real time
  - [ ] Captions match `evaluate.py` output for the same clips
- **Done when:** All match, or differences are explained (e.g. window boundaries in Mode B).

### W6-03 · Fresh-clone reproducibility test
- **Owner:** Ainomugisha Faith
- **Reviewer:** Ansasire Nehum Naamaye
- **Labels:** Testing, Docs
- **Due:** Wed 25 Nov 2026
- **Depends on:** W6-04
- **Estimate:** 3 h
- **Description:** Done by a member not involved in the environment setup.
- **Checklist:**
  - [ ] Fresh clone on a clean machine/environment
  - [ ] Install from the README only
  - [ ] Run the test suite
  - [ ] Reproduce one evaluation from a released checkpoint (same WER)
  - [ ] Report every unclear README step on the card
- **Done when:** Evaluation reproduces and README gaps are fixed.

### W6-04 · README, setup guide and data download instructions
- **Owner:** Ansasire Nehum Naamaye
- **Reviewer:** Ainomugisha Faith
- **Labels:** Docs
- **Due:** Tue 24 Nov 2026
- **Depends on:** W5-M
- **Estimate:** 4 h
- **Checklist:**
  - [ ] Installation (environment files, MediaPipe/PyTorch notes)
  - [ ] Data download and preprocessing command sequence
  - [ ] Training, evaluation and `transcribe.py` usage
  - [ ] Running the demo (Mode A and Mode B)
  - [ ] Where checkpoints and results live
- **Done when:** It passes the fresh-clone test (W6-03).

### W6-05 · Complete the final technical report
- **Owner:** Lisa Lusinga Namugenyi
- **Reviewer:** Charis Opol
- **Labels:** Docs
- **Due:** Fri 27 Nov 2026
- **Depends on:** W4-11, W5-10
- **Estimate:** 8 h (editor) + 2 h per member
- **Checklist:**
  - [ ] Introduction and related work (reuse the proposal, update)
  - [ ] Methods (from W4-11 drafts, updated to what was actually built)
  - [ ] Results, ablations and fusion analysis
  - [ ] Latency results (from Nehum)
  - [ ] Limitations and future work
  - [ ] Check that every number matches `experiments/final/`
  - [ ] References and figure/table numbering
  - [ ] Full-team proofread
- **Done when:** The report is final and exported to PDF.

### W6-06 · Presentation slides
- **Owner:** Wakisa Isabelle
- **Reviewer:** Cole Rodney Migisha
- **Labels:** Docs
- **Due:** Thu 26 Nov 2026
- **Depends on:** W5-05
- **Estimate:** 5 h
- **Checklist:**
  - [ ] Problem and motivation
  - [ ] Architecture diagram
  - [ ] WER-vs-SNR result and ablations
  - [ ] Gate behaviour plot
  - [ ] Demo section and limitations
  - [ ] Assign speakers per slide (all six members)
- **Done when:** Slides are approved by the team.

### W6-07 · Recorded fallback demo video
- **Owner:** Wakisa Isabelle
- **Reviewer:** Ansasire Nehum Naamaye
- **Labels:** App, Docs
- **Due:** Thu 26 Nov 2026
- **Depends on:** W6-01
- **Estimate:** 3 h
- **Description:** A backup in case the live demo fails on the day.
- **Checklist:**
  - [ ] Record Mode A: same clip clean vs. 0 dB, audio-only vs. audio-visual captions
  - [ ] Record Mode B live captioning
  - [ ] Only team members appear, with consent; no third-party recordings
- **Done when:** The video is saved in the shared drive and linked on the card.

### W6-08 · Demo rehearsal
- **Owner:** Charis Opol
- **Reviewer:** All
- **Labels:** PM
- **Due:** Fri 27 Nov 2026
- **Depends on:** W6-01, W6-06, W6-07
- **Estimate:** 2 h (all members)
- **Checklist:**
  - [ ] Full run-through with timing
  - [ ] Test on the presentation laptop, room lighting and microphone
  - [ ] Prepare answers to likely questions (why GRID, why CTC, does fusion help, limitations)
  - [ ] Agree who switches to the fallback video and when
- **Done when:** The rehearsal fits the time limit.

### W6-09 · Release package
- **Owner:** Charis Opol
- **Reviewer:** Ansasire Nehum Naamaye
- **Labels:** PM, Docs
- **Due:** Sat 28 Nov 2026
- **Depends on:** W6-03, W6-05
- **Checklist:**
  - [ ] Final checkpoints (B1, B2, B3, P, ablations) with configs
  - [ ] Experiment logs and `results.csv`
  - [ ] Test report (pytest output)
  - [ ] Tag `v1.0-final`
  - [ ] Delete any personal recordings that are no longer needed (privacy)
- **Done when:** The tag is pushed and the release folder is complete.

### W6-10 · Final submission
- **Owner:** Charis Opol
- **Reviewer:** Lisa Lusinga Namugenyi
- **Labels:** PM, Milestone
- **Due:** Sun 29 Nov 2026 (check the course deadline)
- **Depends on:** W6-05, W6-09
- **Checklist:**
  - [ ] Report PDF
  - [ ] Code repository link
  - [ ] Slides
  - [ ] Demo video
  - [ ] Confirm submission received
- **Done when:** Everything is submitted.

### W6-M · MILESTONE M9 — Testing, documentation & demonstration
- **Owner:** Charis Opol
- **Reviewer:** All
- **Labels:** Milestone, PM
- **Due:** Sun 29 Nov 2026
- **Checklist:**
  - [ ] Fresh clone installs and reproduces one evaluation
  - [ ] Demo rehearsed with a recorded fallback clip
  - [ ] All automated tests pass
  - [ ] Report and release submitted
  - [ ] Retrospective: what went well / what to improve

---

## Optional Extensions (after core freeze) — list `Optional Extensions`
Start these only after W5-M, and only if Week 6 core cards are on track.

### OPT-01 · X4: EfficientNet-B0 visual trunk
- **Owner:** Cole Rodney Migisha
- **Reviewer:** Charis Opol
- **Labels:** Optional, Model, GPU
- **Due:** Thu 26 Nov 2026
- **Depends on:** W5-M
- **Checklist:**
  - [ ] Single Conv3d stem → 3-channel map per frame
  - [ ] ImageNet-pretrained EfficientNet-B0 (torchvision) per frame, global pool 1280 → Linear 512 → BiLSTM
  - [ ] Train and evaluate like P; compare cost and WER
- **Done when:** Result logged or the card is archived for lack of compute.

### OPT-02 · X6: SNR auxiliary head (learned noise estimate)
- **Owner:** Charis Opol
- **Reviewer:** Lisa Lusinga Namugenyi
- **Labels:** Optional, Model, GPU
- **Due:** Thu 26 Nov 2026
- **Depends on:** W5-M
- **Checklist:**
  - [ ] Linear head on time-averaged audio encoding regresses training SNR (clean = 30 dB)
  - [ ] Append prediction to the gate input; loss weight 0.1
  - [ ] Compare with P, especially gate behaviour vs. SNR
- **Done when:** Result logged or archived.

### OPT-03 · Grammar-constrained decoder (labelled upper reference)
- **Owner:** Wakisa Isabelle
- **Reviewer:** Lisa Lusinga Namugenyi
- **Labels:** Optional, Evaluation
- **Due:** Thu 26 Nov 2026
- **Depends on:** W5-05
- **Description:** Report separately and clearly labelled, never as the headline result, because GRID's fixed grammar inflates accuracy.
- **Checklist:**
  - [ ] Restrict beam prefixes to the six-slot GRID grammar
  - [ ] Evaluate P with it at all SNRs
- **Done when:** Separate labelled table produced.

### OPT-04 · Emotion detection from facial landmarks
- **Owner:** Ainomugisha Faith
- **Reviewer:** Lisa Lusinga Namugenyi
- **Labels:** Optional
- **Due:** Sat 28 Nov 2026
- **Depends on:** W5-M
- **Description:** Separate module; GRID has no emotion labels. Needs an emotion-labelled audio-visual dataset (e.g. CREMA-D or RAVDESS), with licence terms checked first.
- **Checklist:**
  - [ ] Check dataset licence and download a subset
  - [ ] Reuse Face Mesh; geometric features (mouth width/opening, brow–eye distance, eye aperture, angles) after pose normalisation
  - [ ] Temporal statistics per utterance; scikit-learn classifier
  - [ ] Speaker-disjoint split; per-class accuracy and confusion matrix
  - [ ] State limitations (articulation confound, acted emotions)
- **Done when:** Results reported as an experimental extension.

### OPT-05 · Language identification feasibility study
- **Owner:** Cole Rodney Migisha
- **Reviewer:** Wakisa Isabelle
- **Labels:** Optional
- **Due:** Sat 28 Nov 2026
- **Depends on:** W5-M
- **Description:** GRID is English-only. Needs multilingual audio-visual data with language labels (e.g. MuAViC). At this stage, a written feasibility plan is a realistic outcome.
- **Checklist:**
  - [ ] Identify dataset, size and licence
  - [ ] Design: time-pooled fused representation → linear language classifier
  - [ ] Note the speaker-identity confound and the need for speaker-disjoint evaluation
  - [ ] Optional: small consented recording set (English + a Ugandan language) for qualitative testing
- **Done when:** A feasibility note is added to the report's future-work section.

---

## Part D — Workload summary (core cards, excluding milestones and optional work)

| Member | W1 | W2 | W3 | W4 | W5 | W6 | Total |
|---|---|---|---|---|---|---|---|
| Charis Opol | 2 | 4 | 1 | 1 | 2 | 3 | 13 |
| Cole Rodney Migisha | 1 | 2 | 2 | 3 | 1 | 0 | 9 |
| Ainomugisha Faith | 3 | 2 | 2 | 1 | 1 | 1 | 10 |
| Lisa Lusinga Namugenyi | 2 | 3 | 2 | 2 | 3 | 1 | 13 |
| Wakisa Isabelle | 1 | 2 | 2 | 2 | 1 | 3 | 11 |
| Ansasire Nehum Naamaye | 3 | 1 | 1 | 2 | 2 | 2 | 11 |

Card counts don't equal hours: Cole's and Faith's cards include long, hands-on model and data work, and Cole and Faith also own most optional extensions. Rebalance at the weekly reviews if anyone is overloaded.
