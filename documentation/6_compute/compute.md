# LipScribe Compute Access and Usage

**Project:** LipScribe — Lip-Assisted Speech Recognition for Robust Live Captioning  
**Team:** Computer Vision and Deep Learning — Group 4, Uganda Christian University  
**WBS reference:** W1-04 — Secure GPU access and storage (due Wed 21 Oct 2026)  
**Status:** SSH access details and notebook runtime limits are recorded. The SSH host's GPU and free disk space, and the team's booking schedule, still need verification.

## Compute options

### Shared SSH host

The team has SSH access information for the following Ubuntu host:

| Property | Available information |
|---|---|
| Host | `10.4.18.17` |
| Login account | `Lipscribe-team` |
| Operating system | Ubuntu 26.04.1 LTS |
| Reported kernel / architecture | Linux 7.0.0-38-generic, x86_64 |
| GPU model and VRAM | Not yet verified |
| Free disk space | Not yet verified; confirm at least 100 GB is available for project data and cached crops |

Connect from a terminal:

```sh
ssh Lipscribe-team@10.4.18.17
```

Enter the account password only at the interactive SSH prompt. Do not put passwords in this document, shell commands, source control, or shared chat.

To copy the repository from the local directory containing `cv_project_LipScribe`:

```sh
scp -r "cv_project_LipScribe" Lipscribe-team@10.4.18.17:~/Desktop/
```

Then, in the SSH session:

```sh
cd ~/Desktop/cv_project_LipScribe
```

Before treating this host as the training machine, verify its resources while logged in:

```sh
nvidia-smi
df -h ~
free -h
```

Record the GPU model and memory, available disk space, and whether the installed driver supports the project's selected PyTorch/CUDA build. If `nvidia-smi` is unavailable or no CUDA GPU is shown, ask the host administrator or lecturer whether GPU access is enabled. Keep raw datasets, cached crops, checkpoints, and recordings out of Git.

### Notebook runtime fallback

The following limits and hardware options were supplied for a cloud notebook runtime. The provider/account and persistent-storage allocation have not been confirmed; availability may vary by session.

| Runtime | Continuous session limit | Interactive idle timeout |
|---|---:|---:|
| CPU | 12 hours | 20–40 minutes |
| GPU | 12 hours | 20–40 minutes |
| TPU | 9 hours | 20–40 minutes |

**Weekly accelerator quotas:** up to 30 GPU hours and 20 TPU hours per week; quotas reset Friday night / Saturday midnight CET. CPU runtime is unlimited.

**Reported hardware:**

| Runtime option | CPU and RAM | Accelerator |
|---|---|---|
| CPU-only | 4 CPU cores, approximately 30 GB RAM | None |
| GPU | 4 CPU cores, approximately 29 GB RAM | Either 1 × NVIDIA Tesla P100 (16 GB VRAM) or 2 × NVIDIA Tesla T4 (16 GB VRAM each) |
| TPU | 96 CPU cores, approximately 330 GB RAM | Google TPU v5e-8 or v3-8 (128 GB high-speed HBM) |

The two T4 devices provide 32 GB of physical VRAM in total, but memory is not automatically pooled: each device has 16 GB, and using both requires the training code to distribute work across devices.

Notebook runtime limits make it important to save checkpoints and logs to persistent storage throughout a run. Confirm where the provider stores persistent files and how much space is available before downloading the corpus or cached crops. Do not rely on an interactive browser session remaining connected for a long training job.

## Project requirements and operating guidance

- Full-model training should use a CUDA-capable GPU with about 8 GB or more of VRAM. The implementation plan's starting point is batch size 16 with automatic mixed precision; verify memory use in the W2 GPU memory test and use gradient accumulation if necessary.
- Plan for at least 100 GB of free disk space. The cached mouth crops alone are estimated at approximately 23 GB, in addition to the raw GRID corpus and MUSAN.
- CPU is suitable for preprocessing, tests, and the demonstration, but full-model CPU training is unlikely to fit the six-week schedule.
- Match the PyTorch, torchaudio, and torchvision versions, and use a PyTorch CUDA build compatible with the selected machine's driver.
- Save training checkpoints and logs regularly, especially before a notebook session limit. Store large data and run artifacts outside the Git repository.

## Access and booking

No team booking calendar or host-specific booking policy has been provided yet. Until the team agrees on a shared schedule, use this proposed process:

1. Record the operator, machine/runtime, start time, expected end time, and purpose in the team's shared GPU booking schedule before starting a long run.
2. Keep runs within the runtime's continuous-session and weekly quota limits; save a checkpoint before the session ends.
3. Release the booking when the run finishes or the resource is no longer in use, and note any interruption or resource issue for the next user.
4. Confirm the final booking rules and access permissions with the team/host administrator; do not share account passwords in the schedule.

| Operator | Machine/runtime | Start (local time) | End (local time) | Run / purpose |
|---|---|---|---|---|
| _To be scheduled_ | _To be confirmed_ | — | — | — |

## W1-04 follow-up checklist

- [ ] Confirm the SSH host's GPU model, VRAM, CUDA/driver compatibility, and at least 100 GB of free disk.
- [ ] Confirm the notebook provider/account and persistent storage capacity if it will be used as fallback.
- [ ] Agree and publish the shared GPU booking schedule and host rules.
- [ ] Escalate to the lecturer if no suitable, accessible GPU machine is confirmed.
