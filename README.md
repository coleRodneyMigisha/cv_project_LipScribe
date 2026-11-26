# LipScribe: Lip-Assisted Speech Recognition for Robust Live Captioning

![tests](https://github.com/coleRodneyMigisha/cv_project_LipScribe/actions/workflows/tests.yml/badge.svg)

LipScribe is an audio-visual speech recognition system. It produces live captions by combining what a speaker says with how their lips move. Speech recognition that relies only on audio degrades sharply in noisy rooms, while lip movements are unaffected by acoustic noise. LipScribe uses the visual signal to keep captions usable when the audio is degraded. The intended users are deaf and hard-of-hearing people who rely on captions in lectures, meetings and public spaces.

This is the capstone project of **Group 4** in the Computer Vision and Deep Learning course, Department of Computer Science, Uganda Christian University.

> **Status (17 November 2026):** both models are trained and frozen, and the final report is being
> written for submission on 22 November.
> - **GRID benchmark (P4):** 1.02 % test WER on clean audio and 5.34 % at −5 dB (greedy decoding;
>   0.71 % and 3.95 % with the GRID grammar), on speakers held out from training.
> - **Real speech (ws_v5, films and TED talks):** 16.9 % WER on clean TED speech and 28.5 % at −10 dB,
>   where audio alone scores 42.3 % (beam search with a 4-gram language model;
>   [`reports/flix_v5/final_decoder.md`](reports/flix_v5/final_decoder.md)).

## Try the demo

The demo runs the real models on the team's GPU machine (the Alienware) and works on any phone or
laptop browser. It can caption your camera live, a 3-second clip, an uploaded video of any length
(even a whole film), or a Zoom meeting, where the captions appear inside the meeting.

**How to open it this week:** ask the team for the current link. It has the form
`https://<address>.trycloudflare.com/?code=<access code>` and opens the current page, already
connected to the models. The link is kept out of this repository because the access code is in it.

**What does not work yet:**

- **`https://lipscribe.pages.dev` on its own does not use the real models yet.** It still shows an
  older version of the page, which runs on a built-in pretend backend with sample captions. It will
  use the real models once three things are done:
  1. pull request #69 is merged;
  2. the site is redeployed, either by hand (`frontend/deploy-pages.ps1`) or automatically once the
     Cloudflare secrets are added (see [`frontend/README.md`](frontend/README.md));
  3. the backend has a fixed address (on Coolify, see [`DEPLOYMENT.md`](DEPLOYMENT.md)), written
     into `frontend/config.js`.
- **The link above can change.** The backend is reached through a free Cloudflare quick tunnel,
  whose address changes whenever the tunnel restarts, for example after a reboot. The backend
  restarts itself within 5 minutes, and `bash backend/run_local.sh` on the Alienware prints the new
  link.
- **A fixed address waits for the Alienware's clustering,** planned for the weekend of
  21–22 November 2026. After that, the backend gets a permanent address (Tailscale Funnel or a
  Cloudflare named tunnel). Then `lipscribe.pages.dev` alone will open the real demo.
- **It is a demo, not a service.** It shares one GPU with training, and takes at most 3 users at
  once and one upload at a time.

How it is built and run: [`frontend/README.md`](frontend/README.md) (the page, Cloudflare Pages)
and [`backend/README.md`](backend/README.md) (the models server). Deploying both, step by step:
[`DEPLOYMENT.md`](DEPLOYMENT.md).

## How it will work

1. Capture video and audio from a camera and microphone, or from a recorded clip.
2. Detect the face with MediaPipe Face Mesh and crop a 96×96 region around the mouth in every frame.
3. Convert the audio to log-Mel spectrogram features.
4. Encode the mouth sequence with a 3D-CNN and a bidirectional LSTM, and the audio with a bidirectional LSTM.
5. Fuse the two streams with cross-modal attention. Modality dropout during training stops the model from ignoring the lips.
6. Decode the fused sequence into text with CTC and beam search, and display it as a caption.

The model will be trained on the GRID audio-visual corpus, with MUSAN noise mixed into the audio at clean, 10, 5, 0 and −5 dB SNR. It will be evaluated with word and character error rates (WER and CER) on speakers held out from training, at each noise level, and compared against audio-only, visual-only and simple-fusion baselines.

## Team

| Member | Role |
|---|---|
| Charis Opol | Project coordinator & training lead |
| Cole Rodney Migisha | Model lead |
| Ainomugisha Faith | Data lead |
| Lisa Lusinga Namugenyi | Audio & evaluation lead |
| Wakisa Isabelle | Application lead |
| Ansasire Nehum Naamaye | QA & live-systems lead |

These roles come from the work breakdown structure and were confirmed at the project kick-off.

## Repository structure

```
cv_project_LipScribe/
├── README.md
├── DEPLOYMENT.md                  step-by-step deployment: frontend to Cloudflare Pages, backend to Coolify
├── frontend/                      the demo page (static; deployed to Cloudflare Pages)
├── backend/                       the demo server (FastAPI + Dockerfile; deployed with Coolify on the Alienware)
├── lipscribe/                     Python package: preprocessing, data, models, decoding, evaluation, app engine
├── scripts/                       data preparation, verification, training, evaluation and Flix pipelines
├── configs/                       experiment configurations
├── tests/                         automated tests
├── runs/                          training histories and curves (checkpoints are not in git)
├── reports/                       per-run evaluation reports
├── analysis/                      result tables and the scripts that build them
├── experiments/                   early notebooks and crop experiments
├── data/, audio_processing/       speaker and noise splits, audio conversion log
├── aligns/                        GRID word alignments
└── docs/
    ├── *.md, mouth_crop/          interfaces, testing guide, retrain guide, crop review sheets
    ├── notes/                     working notes (B2 overfit test, audio features guide)
    ├── 1_proposal/                final proposal PDF, proposal slides, presenter guide, talking points
    ├── 2_implementation/
    │   ├── implementation_document/   LaTeX source + compiled PDF of the implementation plan
    │   └── presentation/              implementation slides (full and 5-min) + presenter notes
    ├── 3_project_management/      work breakdown structure for the Trello board, milestones
    ├── 4_templates/               course PowerPoint template
    ├── 5_submissions/             progress reports, the final report (LaTeX, planning/, papers/ kept
    │                              locally) and the final presentation tracker
    └── 6_compute/                 GPU plan
```

## Key documents

| Document | What it covers |
|---|---|
| [Project proposal](docs/1_proposal/cv_lip_reading_project_proposal.pdf) | Problem, objectives, research questions, literature review, methodology |
| [Implementation document](<docs/2_implementation/implementation_document/Implementation plan 1.pdf>) | Architecture, technology stack, data pipeline, model, training, evaluation, live captioning, schedule, risks |
| [Work breakdown structure](docs/3_project_management/LipScribe_WBS_Trello.md) | Week-by-week tasks with owners, reviewers, checklists and due dates, written for building the Trello board |
| [Implementation presentation](docs/2_implementation/presentation/) | Slides and presenter notes for presenting the implementation plan |
| [Testing guide](docs/testing.md) | How to set up and run the test suite |

## Rebuilding the implementation document

The implementation document is written in LaTeX. To rebuild it, run this from `docs/2_implementation/implementation_document/`:

```sh
latexmk -pdf main.tex      # or: tectonic main.tex
```

The [document's README](docs/2_implementation/implementation_document/README.md) has the full requirements and alternatives.

## Project management

The team tracks work on a Trello board built from the [work breakdown structure](docs/3_project_management/LipScribe_WBS_Trello.md). It covers six weeks and nine milestones (M1–M9), from environment setup and dataset verification to the final report and demonstration. Emotion detection and language identification, the proposal's additional objectives, are optional extensions scheduled only after the core speech-recognition system is complete.
