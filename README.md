# LipScribe: Lip-Assisted Speech Recognition for Robust Live Captioning

LipScribe is an audio-visual speech recognition system. It produces live captions by combining what a speaker says with how their lips move. Speech recognition that relies only on audio degrades sharply in noisy rooms, while lip movements are unaffected by acoustic noise. LipScribe uses the visual signal to keep captions usable when the audio is degraded. The intended users are deaf and hard-of-hearing people who rely on captions in lectures, meetings and public spaces.

This is the capstone project of **Group 4** in the Computer Vision and Deep Learning course, Department of Computer Science, Uganda Christian University.

> **Status:** planning and documentation. The proposal and the implementation plan are complete. No code has been written yet, and no results exist. Implementation follows the six-week schedule in the implementation document.

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

These roles are proposed in the work breakdown structure and are confirmed at the project kick-off.

## Repository structure

```
cv_project_LipScribe/
├── README.md
├── lipScribe_proposal/            original proposal bundle as first submitted
│   ├── cv_lip_reading_project_proposal.pdf   (earlier version of the proposal)
│   ├── Capstone_Project_Proposal_Requirements.docx
│   ├── IEEE_Access_LaTeX_template.pdf
│   ├── lipScribe.zip
│   └── sources/                   papers cited in the proposal
└── documentation/
    ├── 1_proposal/                final proposal PDF, proposal slides, presenter guide, talking points
    ├── 2_implementation/
    │   ├── implementation_document/   LaTeX source + compiled PDF of the implementation plan
    │   └── presentation/              implementation slides (full and 5-min) + presenter notes
    ├── 3_project_management/      work breakdown structure for the Trello board
    ├── 4_templates/               course PowerPoint template
    ├── 5_submissions/             submitted milestone materials
    └── 6_compute/                 compute access and GPU/runtime guidance
```

## Key documents

| Document | What it covers |
|---|---|
| [Project proposal](documentation/1_proposal/cv_lip_reading_project_proposal.pdf) | Problem, objectives, research questions, literature review, methodology |
| [Implementation document](<documentation/2_implementation/implementation_document/Implementation plan 1.pdf>) | Architecture, technology stack, data pipeline, model, training, evaluation, live captioning, schedule, risks |
| [Work breakdown structure](documentation/3_project_management/LipScribe_WBS_Trello.md) | Week-by-week tasks with owners, reviewers, checklists and due dates, written for building the Trello board |
| [Compute access and usage](documentation/6_compute/compute.md) | SSH access, available notebook runtime limits, hardware requirements and GPU booking guidance |
| [Implementation presentation](documentation/2_implementation/presentation/) | Slides and presenter notes for presenting the implementation plan |
| [Reference papers](lipScribe_proposal/sources/) | The papers cited in the proposal |

## Rebuilding the implementation document

The implementation document is written in LaTeX. To rebuild it, run this from `documentation/2_implementation/implementation_document/`:

```sh
latexmk -pdf main.tex      # or: tectonic main.tex
```

The [document's README](documentation/2_implementation/implementation_document/README.md) has the full requirements and alternatives.

## Project management

The team tracks work on a Trello board built from the [work breakdown structure](documentation/3_project_management/LipScribe_WBS_Trello.md). It covers six weeks and nine milestones (M1–M9), from environment setup and dataset verification to the final report and demonstration. Emotion detection and language identification, the proposal's additional objectives, are optional extensions scheduled only after the core speech-recognition system is complete.
