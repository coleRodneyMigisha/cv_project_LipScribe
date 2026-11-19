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
  2. the site is redeployed, either by hand (`demo/deploy-pages.ps1`) or automatically once the
     Cloudflare secrets are added (see [`demo/README.md`](demo/README.md));
  3. the backend has a fixed address, written into `DEFAULT_BACKEND` in `demo/app.js`.
- **The link above can change.** The backend is reached through a free Cloudflare quick tunnel,
  whose address changes whenever the tunnel restarts, for example after a reboot. The backend
  restarts itself within 5 minutes, and `bash scripts/demo_up.sh` on the Alienware prints the new
  link.
- **A fixed address waits for the Alienware's clustering,** planned for the weekend of
  21–22 November 2026. After that, the backend gets a permanent address (Tailscale Funnel or a
  Cloudflare named tunnel). Then `lipscribe.pages.dev` alone will open the real demo.
- **It is a demo, not a service.** It shares one GPU with training, and takes at most 3 users at
  once and one upload at a time.

How it is built and run: [`demo/README.md`](demo/README.md).

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
├── lipscribe/                     Python package: preprocessing, data, models, decoding, evaluation
├── scripts/                       data preparation, verification and training entry points
├── configs/                       experiment configurations
├── tests/                         automated tests
├── docs/                          interfaces, testing guide and crop review sheets
└── documentation/
    ├── 1_proposal/                final proposal PDF, proposal slides, presenter guide, talking points
    ├── 2_implementation/
    │   ├── implementation_document/   LaTeX source + compiled PDF of the implementation plan
    │   └── presentation/              implementation slides (full and 5-min) + presenter notes
    ├── 3_project_management/      work breakdown structure for the Trello board
    ├── 4_templates/               course PowerPoint template
    └── 5_submissions/             weekly milestone submissions and progress reports
```

## Key documents

| Document | What it covers |
|---|---|
| [Project proposal](documentation/1_proposal/cv_lip_reading_project_proposal.pdf) | Problem, objectives, research questions, literature review, methodology |
| [Implementation document](<documentation/2_implementation/implementation_document/Implementation plan 1.pdf>) | Architecture, technology stack, data pipeline, model, training, evaluation, live captioning, schedule, risks |
| [Work breakdown structure](documentation/3_project_management/LipScribe_WBS_Trello.md) | Week-by-week tasks with owners, reviewers, checklists and due dates, written for building the Trello board |
| [Implementation presentation](documentation/2_implementation/presentation/) | Slides and presenter notes for presenting the implementation plan |
| [Testing guide](docs/testing.md) | How to set up and run the test suite |

## Rebuilding the implementation document

The implementation document is written in LaTeX. To rebuild it, run this from `documentation/2_implementation/implementation_document/`:

```sh
latexmk -pdf main.tex      # or: tectonic main.tex
```

The [document's README](documentation/2_implementation/implementation_document/README.md) has the full requirements and alternatives.

## Project management

The team tracks work on a Trello board built from the [work breakdown structure](documentation/3_project_management/LipScribe_WBS_Trello.md). It covers six weeks and nine milestones (M1–M9), from environment setup and dataset verification to the final report and demonstration. Emotion detection and language identification, the proposal's additional objectives, are optional extensions scheduled only after the core speech-recognition system is complete.


## Installation

Tested with **Python 3.11** (the Alienware runs 3.11.17). Newer Pythons such as 3.14 may not have
wheels for `torch`, `mediapipe` or `kenlm`.

```bash
git clone https://github.com/coleRodneyMigisha/cv_project_LipScribe.git
cd cv_project_LipScribe

# Option A: conda
conda env create -f environment.yml
conda activate lipscribe

# Option B: venv
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .

python check_env.py                # checks the install
python -m pytest tests             # optional
```

Notes:

- **PyTorch.** `requirements.txt` pins `torch==2.14.1`. For a GPU, install the build that matches
  your CUDA driver from pytorch.org first (the Alienware uses `2.14.1+cu130`), then run
  `pip install -r requirements.txt`. Without a GPU, everything runs on CPU, only slower.
- **MediaPipe.** `mediapipe==1.1.1` finds the face for mouth cropping. The face model file
  `models/face_landmarker.task` is downloaded the first time you run
  `python -m scripts.extract_mouths`. The real-world model reads it from the same place.
- **kenlm** (needed for the real-world model's 4-gram language model) has no Windows wheel and must
  be compiled. On Windows install the "Desktop development with C++" build tools first, or use
  WSL or Linux. Without it, GRID training and evaluation still work. Only ws_v5 beam search needs it.
- **ffmpeg.** `imageio-ffmpeg` is in `requirements.txt` and supplies it. No system install is needed.
- **Not in the repository.** Checkpoints (`*.pt`), `runs/`, `Data/`, `lm/` and `models/` are
  git-ignored because they are large. See [Where things live](#where-things-live).

## GRID: data, preprocessing, training and evaluation

Run every command from the repository root. Every step except downloading can be re-run safely.

```bash
# 1. Download GRID speakers (interactive: asks for the first and last speaker, 1 to 34)
python downlaod_data.py

# 2. Check the data, then make the speaker-independent split (done once; do not regenerate)
python -m scripts.verify_dataset
python -m scripts.make_splits

# 3. Audio to 16 kHz mono WAV, and cached 96x96 mouth crops
python scripts/prepare_audio.py --workers 8
python -m scripts.extract_mouths --workers 8

# 4. One index of the whole dataset
python -m scripts.build_manifest

# 5. Noise for the noisy test sets (needs the MUSAN corpus), then normalisation statistics
python scripts/make_noise_splits.py
python scripts/compute_norm_stats.py --noise-splits Data/Noise_splits/splits.csv

# 6. Train one model from one config (the frozen P4 model uses configs/p_proposed.yaml)
python -m scripts.train --config configs/p_proposed.yaml
#    quick smoke test:  python -m scripts.train --config configs/b1_audio_only.yaml --limit 64 --max-epochs 2

# 7. Evaluate on the held-out test speakers at every noise level
python -m scripts.evaluate --checkpoint runs/<run>/best.pt
python -m scripts.evaluate --checkpoint runs/<run>/best.pt --decode grammar
```

Each training run writes `runs/<experiment>_<timestamp>/` (config, Git commit, `best.pt`, logs) and
appends a row to `experiments/results.csv`. Evaluation writes to `reports/<run>/`. Configs for every
model are in `configs/` (B1 audio-only, B2 visual-only, B3 concat, P1 to P4 and the X ablations).

## Real-world model (ws_v5)

P4 is trained on GRID, whose speakers are filmed face-on in a studio. **ws_v5** is the model for
ordinary speech. It was fine-tuned on "Flix" data, which is clips mined from films and TED talks.

**Data pipeline** (scripts in `scripts/`): `mine_flix.py` and `ted_harvest.py` find clips that show
one large, frontal, speaking face with no camera cut. `flix_v5_recut.py` re-cuts them and
`flix_recrop.py` produces the mouth crops in the same way as the live demo
(25 fps, at most 640 px wide, MediaPipe landmarks, Auto-AVSR 96x96 crop). A 4-gram language model
is built by `flix_lm.py`.

**Models.** ws_v5 (audio and lips, gated fusion) and its audio-only twin twin_v5. Both are decoded
with beam search and the 4-gram LM; the settings chosen on validation are in
[`reports/flix_v5/final_decoder.md`](reports/flix_v5/final_decoder.md).

**Run ws_v5** on prepared segments through `lipscribe.app.realworld.RealWorld`. To score the
models on the Flix test sets:

```bash
PYTHONPATH=. python scripts/flix_v5_eval.py --run twin_v5=<run dir> ws_v5=<run dir>
PYTHONPATH=. python scripts/flix_v5_beam.py dump --run twin_v5=<run dir> ws_v5=<run dir>
```

ws_v5 only learned from clips that pass the filter above. On other footage, such as reaction
shots, profiles or a second face, it can do worse than audio alone, so the demo falls back to twin_v5
there. See [`demo/README.md`](demo/README.md).

## Transcribe a clip and run the demo

Caption one clip from the command line (works with the GRID models P4 and B1):

```bash
python -m scripts.transcribe --checkpoint checkpoints/p_proposed_best.pt --video clip.mp4
python -m scripts.transcribe --checkpoint checkpoints/p_proposed_best.pt --video clip.mpg --audio clip.wav
```

Options: `--decode {beam,greedy,grammar}` (default `beam`), `--device {cpu,cuda}`,
`--snr <dB>` (mix in MUSAN test noise), `--noise-splits <csv>`. It prints the caption, and the
preprocess, model and decode times on stderr.

Run the web demo backend (needs the checkpoints, `lm/` and `models/` in place):

```bash
DEMO_CODE=<secret> PYTHONPATH=.:asr_pkgs python -m uvicorn demo.server:app \
    --host 127.0.0.1 --port 8800 --ws-max-size 16777216
```

Check `http://127.0.0.1:8800/health`, then open `http://127.0.0.1:8800/?code=<secret>`. Test
without a browser: `python scripts/demo_ws_client.py "ws://127.0.0.1:8800/ws/live?code=<secret>" clip any clip.mp4`.
Full details are in [`demo/README.md`](demo/README.md).

## Where things live

| What | Where | In Git? |
|---|---|---|
| Library code | `lipscribe/` (`data`, `preprocessing`, `models`, `decoding`, `evaluation`, `app`) | yes |
| Commands | `scripts/` | yes |
| Training configs | `configs/` | yes |
| Speaker split, noise split | `data/splits.csv`, `data/noise_splits.csv` | yes |
| Experiment table | `experiments/results.csv` | yes |
| Results, plots, reports | `reports/` (final decoder and Flix numbers: `reports/flix_v5/`) | yes |
| Frozen GRID checkpoints | `checkpoints/` (`p_proposed_best.pt` = P4, `b1_audio_only_best.pt` = B1) | **no** (`*.pt`) |
| Training runs | `runs/<experiment>_<timestamp>/` (ws_v5, twin_v5: `runs/flix_v5/`) | **no** |
| GRID, MUSAN, cached crops | `Data/`, `data/processed/` | **no** |
| 4-gram LM, face model | `lm/flix_libri_4gram.arpa`, `models/face_landmarker.task` | **no** |
| Latency benchmarks | `reports/latency/` | yes |

**Getting the files that are not in Git:** TODO: link to the shared drive or release.