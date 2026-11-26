# Deploying LipScribe: frontend on Cloudflare Pages, backend on Coolify

This guide takes the demo from the current setup (a backend started by cron on the Alienware and
reached through a Cloudflare quick tunnel whose address changes) to a fixed one:

```
 browser ──https──▶ lipscribe.pages.dev            (frontend/, Cloudflare Pages, static)
    │
    └──wss──▶ lipscribe-api.ucu-dct.org/ws/live   (Cloudflare, then Coolify's proxy)
                    │
                    ▼
              Alienware: Docker container "backend" (backend/Dockerfile), port 8800, GPU
                         models mounted read-only from /srv/lipscribe/assets
```

`lipscribe-api.ucu-dct.org` is a placeholder used throughout this guide. Replace it with the
address you are given in step 6.

Work through the steps in order. Each step ends with a **Check** that should pass before you move
on. The old quick-tunnel demo keeps running the whole time and is switched off only in step 10, so
the demo is never down.

| Step | What | Where |
|---|---|---|
| 0 | Gather what you need | anywhere |
| 1 | Docker and the GPU on the Alienware | Alienware (SSH) |
| 2 | Record the working package versions | Alienware |
| 3 | Put the model files in one folder | Alienware |
| 4 | Join the Coolify team and look around | coolify.ucu-dct.org |
| 5 | Create the backend in Coolify | Coolify |
| 6 | Give the backend its address | Coolify and Cloudflare (domain owner) |
| 7 | Deploy and check the backend | Coolify |
| 8 | Point the frontend at the backend and deploy it | your laptop, GitHub, Cloudflare |
| 9 | Test everything end to end | phone and laptop |
| 10 | Switch off the old quick-tunnel demo | Alienware |
| 11 | Updating, rolling back, troubleshooting | reference |

---

## 0. What you need before you start

- [ ] SSH access to the Alienware with `sudo`, and the folder `~/Desktop/lipscribe_demo` that the
      current demo runs from (`backend/README.md`, "Running it").
- [ ] An account in the Coolify team at https://coolify.ucu-dct.org (the invitation).
- [ ] Contact with whoever manages `ucu-dct.org` in Cloudflare and set up Coolify there. Step 6
      needs them to create a DNS record or tunnel hostname. Ask early, because it can take a day.
- [ ] GitHub access to `coleRodneyMigisha/cv_project_LipScribe`. Coolify builds from a branch on
      GitHub, so this reorganisation must be pushed (and ideally merged to `main`) before step 7.
- [ ] For the frontend: access to the Cloudflare account that owns the `lipscribe` Pages project,
      or permission to add two repository secrets on GitHub (step 8).
- [ ] A new access code for the demo. Generate one on any machine:
      ```bash
      python3 -c "import secrets; print(secrets.token_urlsafe(9))"
      ```
      Keep it out of git and out of public chats. It goes into Coolify in step 5.

---

## 1. Docker and the GPU on the Alienware

Coolify runs every app as a Docker container, and the backend needs the GPU inside its container.
That needs the NVIDIA driver (already there, since training runs) and the NVIDIA Container Toolkit.

```bash
ssh alienware
nvidia-smi                                  # note the "Driver Version" (needed in step 2)
docker --version                            # Coolify installs Docker on servers it manages
docker info | grep -i runtimes              # look for "nvidia" in the list
```

If `nvidia` is not listed, install the toolkit (Ubuntu; from NVIDIA's install guide):

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Restarting Docker restarts every container on the machine, including Coolify's own and anything
other users run there. Agree a time with whoever manages the Alienware before you do it.

**Check:** a throwaway container sees the GPU.

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

It prints the same GPU table as `nvidia-smi` on the host. If it fails with "could not select
device driver", the toolkit is not configured; repeat the last two commands above.

---

## 2. Record the package versions that work today

The image installs packages from `backend/requirements.txt`. Four of them (`transformers`,
`sentencepiece`, `pyctcdecode`, `kenlm`) are not pinned in the repository, and torch must be a CUDA
build that suits the driver. Copy the versions from the venv the demo runs on now, so that the
container gives the same captions.

```bash
cd ~/Desktop/lipscribe_demo
.venv/bin/python --version
.venv/bin/pip freeze | grep -iE "^(torch|torchaudio|transformers|sentencepiece|pyctcdecode|kenlm|mediapipe|silero-vad|av|numpy)=="
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Then, in the repository on your laptop:

1. In `backend/requirements.txt`, replace the four lines marked `check` with `name==version` from
   the output.
2. In `backend/Dockerfile`, set:
   - `PYTHON_VERSION` to the venv's Python (for example `3.11`);
   - `TORCH_VERSION` and `TORCHAUDIO_VERSION` to the versions without the `+cuXXX` suffix;
   - `TORCH_INDEX` to `https://download.pytorch.org/whl/cuXXX`, where `cuXXX` comes from
     `torch.version.cuda` (12.8 gives `cu128`, 12.6 gives `cu126`).
3. If any other version in the output differs from `backend/requirements.txt`, use the Alienware's.

**Check:** `kenlm` shows a version. If it shows nothing, the demo is running without the language
model; tell Cole before going further, because the "Any speech" captions would then differ from
the report.

---

## 3. Put the model files in one folder

The image holds code only. The weights stay on the Alienware and are mounted into the container
read-only. They are too large for git, and keeping them out of the image makes rebuilds quick.
`backend/docker-compose.yml` expects one folder laid out like the repository, at
`/srv/lipscribe/assets`. (To use another folder, change that path on every volume line of the
compose file. Coolify does not accept a variable there; see step 11.)

The demo folder uses symlinks into other folders, and a symlink does not work inside a container
unless its target is mounted too. So copy the real files (`cp -L` follows the links):

```bash
A=/srv/lipscribe/assets
D=~/Desktop/lipscribe_demo
sudo mkdir -p $A && sudo chown "$USER" $A
mkdir -p $A/checkpoints $A/runs/flix_v5 $A/lm $A/models $A/data $A/Data

# GRID models (P4 and B1)
cp -L $D/checkpoints/p_proposed_best.pt $D/checkpoints/b1_audio_only_best.pt $A/checkpoints/
# Any-speech models (ws_v5 and twin_v5), whole run folders
cp -rL $D/runs/flix_v5/flix_v5_fusion_gated_ws_20261116-093641 $A/runs/flix_v5/
cp -rL $D/runs/flix_v5/flix_v5_fusion_wp_audio_20261116-000004 $A/runs/flix_v5/
# 4-gram language model, face landmarker
cp -L $D/lm/flix_libri_4gram.arpa $A/lm/
cp -L $D/models/face_landmarker.task $A/models/
# pretrained encoders: Auto-AVSR weights, word pieces, WavLM in the Hugging Face cache
cp -rL $D/pretrained $A/
# feature normalisation statistics (GRID models)
cp -rL $D/data/stats $A/data/
# test noise (the page's "Add test noise" option): the split file and the test-pool files only
mkdir -p $A/Data/Noise_splits
cp -L $D/Data/Noise_splits/splits.csv $A/Data/Noise_splits/
python3 - "$D/Data" "$A/Data" <<'EOF'
import csv, shutil, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
n = 0
for row in csv.DictReader(open(src / "Noise_splits/splits.csv", newline="")):
    if row["pool"].strip().lower() != "test":
        continue
    rel = Path(row["file"])
    if rel.is_absolute():
        sys.exit(f"absolute path in splits.csv: {rel}; mount that folder at the same path instead (step 11)")
    (dst / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2((src / rel).resolve(), dst / rel); n += 1
print("copied", n, "test noise files")
EOF

# the container runs as uid 1000 and only reads
chmod -R a+rX $A
```

These are the paths the code reads (`backend/server.py`, `lipscribe/app/realworld.py`,
`lipscribe/models/pretrained.py`, `lipscribe/data/subword.py`,
`lipscribe/preprocessing/norm_stats.py`). If a copy fails because a file is not where this guide
says, find it with `readlink -f $D/<path>` and copy it from there, keeping the same path under
`$A`.

**Check:**

```bash
ls -la $A/checkpoints $A/runs/flix_v5/*/best.pt $A/lm $A/models $A/data/stats
ls $A/pretrained            # vsr_trlrs3vox2_base.pth, unigram5000.model, unigram5000_units.txt, hf/
du -sh $A
```

Every file is there, none is a symlink (`ls -la` shows no `->`), and none is 0 bytes.

---

## 4. Join the Coolify team and look around

1. Accept the invitation and log in at https://coolify.ucu-dct.org.
2. Open **Servers**. Find the Alienware.
   - If it is listed: open it and check that its status is *Reachable* and *Validated*
     (Docker checked).
   - If it is not listed: an admin of the team must add it (Servers → Add, with an SSH key that
     Coolify uses to reach the Alienware), or give you the rights to do so. The backend must run
     on the Alienware, because the GPU and the model files are there.
3. Open **Projects** and create a project named `LipScribe` (or use the one an admin made for you).
   Inside it, the *production* environment is fine.
4. Note your role in the team (Team → Members). *Member* can deploy; adding servers and keys
   usually needs *Admin*.

**Letting Claude help in Coolify (optional).** Claude cannot accept an email invitation or log in
through the browser. It can work through Coolify's API with a token:

1. In Coolify: **Keys & Tokens → API tokens → Create**, with the narrowest permission that allows
   deploying (read and deploy; avoid *root*).
2. Give the token to Claude in the session, never in a file in the repository.
3. Delete the token in Coolify when the work is done.

Without a token, this guide is enough to do every step by hand.

**Check:** you can see the Alienware under Servers, and you have a project to put the backend in.

---

## 5. Create the backend in Coolify

1. In the project: **+ New → Resource**.
   - If the GitHub repository is public: **Public Repository**, and paste
     `https://github.com/coleRodneyMigisha/cv_project_LipScribe`.
   - If it is private: **Private Repository (with GitHub App)**. The repository owner (Cole)
     installs the Coolify GitHub App on it once.
2. **Branch:** `main` once this reorganisation is merged. Until then, the branch that has the
   `backend/` folder (for example `charis`).
3. **Build pack:** `Docker Compose`.
4. **Base Directory:** `/`
   **Docker Compose Location:** `/backend/docker-compose.yml`
   The build context must be the repository root, because the backend imports `lipscribe/` and
   `scripts/train.py`. The compose file already says `context: ..`.
5. **Server / Destination:** the Alienware.
6. Save. Coolify reads the compose file and shows one service, `backend`.
7. **Environment Variables** (resource → *Environment Variables*; Coolify lists the three it found
   in the compose file). `backend/.env.example` has the same list to copy from:

   | Variable | Value | Why |
   |---|---|---|
   | `DEMO_CODE` | the access code from step 0 | without it, anyone with the address can use the GPU |
   | `DEMO_ORIGINS` | `https://lipscribe.pages.dev,https://lipscribe-api.ucu-dct.org` | the sites allowed to call the backend over HTTP (the WebSocket checks the code instead) |
   | `DEMO_MODELS` | `any,grid` | both model pairs; `any` or `grid` loads one and saves GPU memory |

   Tick **Is Secret** (or *Locked*) for `DEMO_CODE`, so that it is hidden in the UI and the logs.

**Check:** the resource page shows the `backend` service and the three variables, and the
**Docker Compose** tab shows the `deploy.resources.reservations.devices` block (the GPU) and the
seven volumes.

---

## 6. Give the backend its address

The page is served over https, so the browser only allows a secure WebSocket (`wss://`) to a
domain with a valid certificate. `ucu-dct.org` is on Cloudflare (`coolify.ucu-dct.org` answers
from Cloudflare), so the new name is made there by the domain's manager.

1. Agree the name with the domain manager. This guide uses `lipscribe-api.ucu-dct.org`.
2. Ask them how `coolify.ucu-dct.org` reaches the server, and follow the matching option:

   **Option A: a Cloudflare Tunnel** (likely, if the university network has no open ports).
   - They add a **public hostname** to the existing tunnel: `lipscribe-api.ucu-dct.org`, service
     `http://localhost:80` (Coolify's proxy on the Alienware), or whatever the Coolify hostname
     already points to.
   - In Coolify, set the `backend` service's **Domains** to `http://lipscribe-api.ucu-dct.org:8800`.
     Use `http`: Cloudflare provides https to the browser, and an https domain in Coolify behind a
     tunnel causes a redirect loop. `:8800` tells Coolify's proxy which container port to use; it
     is not part of the public address.

   **Option B: a DNS record to the Alienware's public IP** (ports 80 and 443 open).
   - They add an `A` record `lipscribe-api` pointing to the Alienware's public IP, proxied
     (orange cloud), with SSL/TLS mode **Full**.
   - In Coolify, set **Domains** to `https://lipscribe-api.ucu-dct.org:8800`. Coolify obtains the
     certificate from Let's Encrypt.

3. Save.

**Check:**

```bash
nslookup lipscribe-api.ucu-dct.org        # returns Cloudflare addresses (104.x / 172.x)
```

---

## 7. Deploy and check the backend

1. Press **Deploy**. The first build takes 10 to 20 minutes: it downloads the CUDA build of torch
   (several GB) and compiles kenlm. Later builds reuse the cached layers and take a few minutes.
2. Watch **Deployments → the running deployment → logs**. A build error names the failing line of
   `backend/Dockerfile` (see step 11).
3. When the container is up, open **Logs** for the `backend` service. About 30 s after start you
   should see `models loaded`. `could not load the models: ...` names the missing file; fix the
   path in `/srv/lipscribe/assets` (step 3) and restart.

**Check, from your laptop:**

```bash
curl -s https://lipscribe-api.ucu-dct.org/health
```

Expected: `{"ok": true, "loading": false, "error": null, "models": {"any": [...], "grid": [...]}, ...}`.
While the models load it says `"loading": true`; wait 30 s and repeat.

Then on the Alienware, confirm that the container uses the GPU:

```bash
nvidia-smi            # a python process from the container appears in the process list
```

Finally, open `https://lipscribe-api.ucu-dct.org/?code=<code>` in a browser. The backend serves
the page itself, so this already works on any device before the Pages site is updated.

---

## 8. Point the frontend at the backend and deploy it

The Pages site is the four files in `frontend/`. Only `config.js` changes.

1. In `frontend/config.js`, set:
   ```js
   window.LIPSCRIBE_BACKEND = window.LIPSCRIBE_BACKEND || 'wss://lipscribe-api.ucu-dct.org/ws/live';
   ```
2. Open a pull request, have it reviewed, and merge it to `main`.
3. Deploy, in one of two ways:
   - **Automatic (recommended).** Once, the owner of the Cloudflare account creates an API token
     (Cloudflare → My Profile → API Tokens → Create, permission *Account → Cloudflare Pages →
     Edit*) and notes the **Account ID** (Cloudflare dashboard, right-hand column). In GitHub:
     repository → Settings → Secrets and variables → Actions → New repository secret, add
     `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID`. From then on,
     `.github/workflows/deploy-pages.yml` uploads `frontend/` on every merge to `main` that
     changes it. To deploy now: Actions → deploy-pages → Run workflow.
   - **By hand**, from the repository root on Windows:
     ```powershell
     npx wrangler login
     powershell -ExecutionPolicy Bypass -File frontend\deploy-pages.ps1
     ```
4. Optional: a name on the university domain for the page. Cloudflare → Workers & Pages →
   `lipscribe` → Custom domains → add `lipscribe.ucu-dct.org`. Then add that origin to
   `DEMO_ORIGINS` in Coolify and redeploy the backend.

**Check:** open https://lipscribe.pages.dev/?code=<code>. The footer shows *Connected* once you
start a recording, and the captions come from the real models (not the sample sentences of the
pretend backend).

A browser that was once given an old `?backend=...trycloudflare.com` address remembers it and
keeps using it. Open `https://lipscribe.pages.dev/?backend=mock` once in that browser to forget
it, then open the normal link again.

---

## 9. Test everything end to end

Do these on a phone (on mobile data, not the university Wi-Fi) and on a laptop.

- [ ] `https://lipscribe-api.ucu-dct.org/health` says `"ok": true` with both model pairs.
- [ ] `https://lipscribe.pages.dev/?code=<code>` opens, asks for the camera and microphone, and
      shows *Connected*.
- [ ] A wrong code (`?code=wrong`) gives the "wrong access code" message.
- [ ] **Clip, GRID sentences:** say "bin blue at f two now". Both captions arrive within about
      3 s.
- [ ] **Clip, GRID sentences, Add test noise −5 dB:** Ears + lips is still close; Ears only is
      worse.
- [ ] **Live, Any speech:** words appear while you talk (about 3 s behind).
- [ ] **Upload:** a short MP4 (1 to 2 minutes). Cues arrive in batches and an `.srt` downloads.
- [ ] **Upload of a long file** (10 minutes or more). It keeps going to the end. This tests that
      nothing between the browser and the container cuts a long WebSocket.
- [ ] Two devices at once: both get captions.
- [ ] Restart test: in Coolify, **Restart** the backend. `/health` says `"loading": true`, then
      `"ok": true` within about a minute.
- [ ] Reboot test (only when nobody is training): after `sudo reboot`, the container comes back
      by itself (`restart: unless-stopped`) and `/health` is `ok` again.

Record the results, with the date, in `backend/README.md` (in the same style as the 17 Nov tests).

---

## 10. Switch off the old quick-tunnel demo

Only after step 9 passes. The Coolify container does not open a port on the host, so it never
clashed with the old backend on `127.0.0.1:8800`; the old one is simply no longer needed.

```bash
cd ~/Desktop/lipscribe_demo
crontab -l > ~/crontab.backup             # keep a copy
crontab -e                                # delete the line that runs demo_up.sh (or run_local.sh) every 5 minutes
bash scripts/demo_up.sh stop 2>/dev/null || bash backend/run_local.sh stop
```

`scripts/flix_boot.sh` (crontab `@reboot`) also starts the old demo at boot, in its last two
lines. Delete those two lines (in the repository, through a pull request) so a reboot does not
bring the quick tunnel back. Leave the training lines in that file alone.

**Check:** `curl -s http://127.0.0.1:8800/health` on the Alienware fails (nothing listens there
any more), and `https://lipscribe-api.ucu-dct.org/health` still answers. Then update the "Try the
demo" section of the root `README.md`: the demo now lives at `https://lipscribe.pages.dev`, and the
code is given on request.

---

## 11. Updating, rolling back and troubleshooting

### Updating

- **Backend code** (`backend/`, `lipscribe/`, `scripts/`, `configs/`): merge to the branch Coolify
  deploys, then press **Deploy**. To deploy on every push, turn on **Auto Deploy** in the
  resource's settings (it uses the GitHub App's webhook).
- **New model weights:** copy them into `/srv/lipscribe/assets` at the same path and **Restart**.
  No rebuild is needed. For a new run folder, also change `WS_V5` or `TWIN_V5` in
  `lipscribe/app/realworld.py`, then deploy.
- **Frontend:** merge changes in `frontend/` to `main` (automatic upload), or run
  `frontend\deploy-pages.ps1`.
- **Access code:** change `DEMO_CODE` in Coolify and **Restart**. Send the new link to the team.

### Rolling back

- Backend: Coolify → Deployments → pick an earlier successful deployment → **Redeploy**.
- Frontend: Cloudflare → Workers & Pages → `lipscribe` → Deployments → an earlier one →
  **Rollback**.

### Troubleshooting

| What you see | Likely cause | What to do |
|---|---|---|
| Build fails at `pip install torch==...` | that version is not on that CUDA index | check https://download.pytorch.org/whl/torch/ for the version and `cuXXX`; fix `TORCH_*` in `backend/Dockerfile` (step 2) |
| Build fails building `kenlm` | a missing build library | read the last 30 lines of the build log; add the named `-dev` package to the `apt-get install` line |
| Build fails with `ResolutionImpossible` | two pinned versions conflict | use the exact versions from the Alienware's `pip freeze` (step 2) |
| Container exits with `could not select device driver "nvidia"` | NVIDIA Container Toolkit missing | step 1 |
| `/health` shows `"error": "FileNotFoundError: ..."` | a model file is missing under `/srv/lipscribe/assets` | copy it to the path named in the error, then Restart |
| `/health` shows `"error": "... CUDA out of memory"` | training is using the GPU | set `DEMO_MODELS=any` (or `grid`), or wait for the training job; the demo shares one GPU |
| `/health` works, the page says *Not connected* | wrong address in `config.js`, or the browser remembers an old one | check `config.js` on the live site; open `?backend=mock` once, then the normal link |
| The page says *wrong or missing access code* | `?code=` missing or old | use the current code from Coolify |
| `502 Bad Gateway` from the domain | Coolify's proxy cannot reach the container | the domain in Coolify must end in `:8800`; check that the container is *Running (healthy)* |
| Redirect loop (`ERR_TOO_MANY_REDIRECTS`) | https domain in Coolify behind a Cloudflare Tunnel, or Cloudflare SSL mode *Flexible* | step 6: `http://` domain with a tunnel; SSL mode *Full* with a DNS record |
| Long uploads stop after a few minutes | a proxy timeout on long WebSockets | note the time it stops after and ask the Coolify or domain admin to raise the proxy's idle timeout for this host |
| "Add test noise" gives an error | the test noise files were not copied | step 3, the noise part |
| Deployment fails with `invalid spec: ...:: empty section between colons` | a `${VAR:-default}` in a volume line of `backend/docker-compose.yml` (Coolify splits volume lines on `:`) | write the host path out in full, as the compose file now does |
| `splits.csv` has absolute paths | the noise files are referenced by full path | add a volume to `backend/docker-compose.yml` that mounts that folder at the same full path, read-only |

---

## What was checked when this guide was written (26 Nov 2026)

Checked:

- After the reorganisation, the full test suite passes (290 passed, 1 skipped).
- `backend/server.py` serves the page, `app.js`, `config.js` and `mock-backend.js` from
  `frontend/`, and `/health` answers. The page loads `config.js` before the server's `"self"`
  setting, so the backend's own link still points the page at itself.
- `backend/docker-compose.yml` is valid (`docker compose config`).
- `coolify.ucu-dct.org` is served through Cloudflare.

Not checked:

- The Docker image has not been built yet. Docker was not running on the machine this was written
  on, so step 7 is the first build.
- The package versions on the Alienware and the CUDA wheel versions (step 2).
- Whether the Alienware is already a server in the Coolify team (step 4), and how `ucu-dct.org`
  reaches it (step 6).
- Whether a long WebSocket upload survives Cloudflare and Coolify's proxy (step 9).
