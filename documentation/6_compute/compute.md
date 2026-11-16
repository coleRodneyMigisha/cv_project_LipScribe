# LipScribe Compute Access and Cross-Network SSH Guide

**Project:** LipScribe — Lip-Assisted Speech Recognition for Robust Live Captioning  
**Team:** Computer Vision and Deep Learning — Group 4, Uganda Christian University  
**WBS reference:** W1-04 — Secure GPU access and storage  
**Status:** Verified and active on the Alienware Aurora R16 GPU Workstation.

---

## 1. Workstation Specifications & Local (LAN) SSH Access

| Property | Verified Information |
| :--- | :--- |
| **Machine** | Alienware Aurora R16 (`cse-robotics-alienware-aurora-r16`) |
| **Login Account** | `Lipscribe-team` |
| **Operating System** | Ubuntu 26.04.1 LTS (Linux `7.0.0-38-generic`, `x86_64`) |
| **GPU Model & VRAM** | **NVIDIA GeForce RTX 4080 SUPER (16 GB VRAM)** |
| **Campus LAN IP (`enp3s0`)** | `10.4.18.17` |
| **Tailscale IP (`tailscale0`)** | `100.115.172.88` (hostname: `cse-robotics-alienware-aurora-r16`) |
| **Project Directory** | `/home/Lipscribe-team/Desktop/cv_project_LipScribe` |

### Connecting When on the Same Campus Network (LAN)
```bash
ssh Lipscribe-team@10.4.18.17
```

---

## 2. How to SSH into This Machine from Anywhere (Across Different Networks)

Because `10.4.18.17` is a private campus LAN address behind NAT, it is only reachable when your laptop is on the same local network. To connect from **home Wi-Fi, a mobile hotspot, or any external network**, use one of the three methods below.

---

### Method A: Tailscale Mesh VPN (Already Running on This Machine — Recommended)

Tailscale (`tailscaled`) is **already installed and active** on this workstation with the permanent virtual IP **`100.115.172.88`**.

1. **Install Tailscale on your laptop**:
   * **Linux**: `curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up`
   * **Windows / macOS**: Download from [https://tailscale.com/download](https://tailscale.com/download) and sign in.
2. **Connect your laptop to this machine's Tailnet**:
   * Right now, this machine is registered to the `snydemorgan@` tailnet (`100.115.172.88`).
   * **If you have access to that Tailscale account** (or ask the lab admin/owner):
     * Either log into the same Tailscale account on your laptop, **OR**
     * Open [https://login.tailscale.com/admin/machines](https://login.tailscale.com/admin/machines), click **`...` -> Share...** on `cse-robotics-alienware-aurora-r16`, and share it with your personal Tailscale email. (Sharing lets you connect from your own free Tailscale account without affecting other machines on the lab tailnet.)
3. **SSH from anywhere on any network**:
   ```bash
   ssh Lipscribe-team@100.115.172.88
   ```

---

### Method B: VS Code Remote Tunnel (Zero Network/Firewall Config, Works Anywhere, No `sudo` Needed)

If you use **VS Code** (or Cursor/Antigravity) and want to connect from any network using only your **GitHub account** (without touching the lab's Tailscale account or needing `sudo`):

1. **On this Alienware machine**, start a persistent background tunnel inside `tmux`:
   ```bash
   tmux new -s tunnel
   code tunnel --accept-server-license-terms --name lipscribe-alienware
   ```
   *(Follow the one-time GitHub device login prompt printed in the terminal, then detach with `Ctrl+B` then `D`.)*
2. **On your laptop (from any network anywhere)**:
   * Open VS Code $\to$ install the **Remote - Tunnels** extension.
   * Click the bottom-left green corner $\to$ **Connect to Tunnel...** $\to$ **GitHub** $\to$ select **`lipscribe-alienware`**.
   * You get full terminal + file explorer + GPU access from anywhere in the world.

---

### Method C: Passwordless SSH Key Setup + `~/.ssh/config` (One-Command Login)

Once you can reach the machine (via `10.4.18.17` on LAN or `100.115.172.88` on Tailscale):

1. **On your laptop**, copy your SSH public key to this workstation so you never have to type the password again:
   ```bash
   # Generate a key on your laptop if you don't have one yet:
   ssh-keygen -t ed25519

   # Copy your key to the Alienware (use 10.4.18.17 on LAN or 100.115.172.88 on Tailscale):
   ssh-copy-id Lipscribe-team@100.115.172.88
   ```
2. **On your laptop**, add this block to `~/.ssh/config`:
   ```sshconfig
   Host alienware
       HostName 100.115.172.88
       User Lipscribe-team
       IdentityFile ~/.ssh/id_ed25519
       ServerAliveInterval 60
       ServerAliveCountMax 5

   Host alienware-lan
       HostName 10.4.18.17
       User Lipscribe-team
       IdentityFile ~/.ssh/id_ed25519
       ServerAliveInterval 60
       ServerAliveCountMax 5
   ```
3. Now, from **any network**, simply run:
   ```bash
   ssh alienware
   ```
   Or open VS Code $\to$ **Remote-SSH: Connect to Host...** $\to$ `alienware`.

---

## 3. Keeping Training Runs Alive When Disconnecting (`tmux`)

Always launch training inside `tmux` so that closing your laptop or switching Wi-Fi networks never kills a run:

```bash
# Start a named session:
tmux new -s train

# Detach (leave running in background):
# Press Ctrl+B, release, then press D

# Re-attach later from any SSH session:
tmux attach -t train
```

## 4. GPU plan for the final week (16–22 Nov 2026)

The Alienware's RTX 4080 SUPER (16 GB) runs one job at a time. Book a slot in the group chat before starting a run, and say when it ends.

| When | Job | Owner | GPU time |
|---|---|---|---|
| Mon 16 Nov, evening | Ablations X1, X2, X3 (`configs/x1_*.yaml`, `x2_*.yaml`, `x3_*.yaml`), one after another; P4 took 27 min | Ainomugisha Faith | ≈ 1.5–2 h |
| Tue 17 Nov, morning | Evaluate X1–X3 at five SNRs (`scripts/evaluate.py`); gate analysis of P4 (`scripts/analyse_gate.py`) | Ainomugisha Faith, Lisa Lusinga Namugenyi | ≈ 1 h |
| Tue 17 Nov, day | ws_v5 decoder tuning on validation, then one test pass (F-04) | Cole Rodney Migisha | ≈ 2 h |
| Wed 18 – Thu 19 Nov | Demo backend and latency measurements (F-05, F-08); the demo server shares the GPU, one clip at a time | Cole Rodney Migisha, Ansasire Nehum Naamaye | short bursts |
| Fri 20 Nov | Fresh-clone reproduction of P4's test WER (F-09) | Ansasire Nehum Naamaye | ≈ 30 min |

Long runs go in `tmux` (Section 3), so a dropped SSH connection does not stop them.
