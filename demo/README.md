# LipScribe web demo

You record a short clip with the camera and microphone, and the page shows the caption
from Model P. It also shows **how much the model trusted the sound vs the lips**
(from P's fusion gate) and the mouth crops it actually used.

```
 phone / laptop browser                Alienware (backend)
 ┌───────────────────────┐   HTTPS   ┌───────────────────────────────────────────────┐
 │ demo/index.html       │ ────────▶ │ demo/server.py (FastAPI, 127.0.0.1:8800)        │
 │ record (≤ 6 s, webm/  │  clip +   │   lipscribe.app.inference.LipScribe              │
 │ mp4) → POST /caption  │  access   │   (25 fps → MediaPipe mouth crops → audio →      │
 │ ◀ caption, trust,     │  code     │    Model P → decoder; same chain as evaluate.py)│
 │   mouth strip         │           └───────────────────────────────────────────────┘
 └───────────────────────┘     ▲ reached through an SSH tunnel (private) or a
                               └ Cloudflare Tunnel (public; no ports opened)
```

## Files

| File | What it is |
|---|---|
| `demo/index.html` | The whole frontend: one static page, no build step. Camera preview, Record button, model toggle, caption, sound/lips trust bar, mouth strip. Works on phones. |
| `demo/server.py` | Backend: `POST /caption`, `GET /health`, and `GET /` (serves the page). Binds to localhost only. Needs the `X-Demo-Code` header. Handles one clip at a time, because the GPU is shared with training. |
| `scripts/flix_boot.sh` | Started by `crontab @reboot` on the Alienware. Restarts the demo server, and the public tunnel too if you opted in (see below). |

### The two models (the "GRID sentences" / "Any speech" toggle)

| Toggle | Checkpoint | Decoder | Use it for |
|---|---|---|---|
| **GRID sentences** (default) | Model P4, `cv_project_LipScribe/checkpoints/p_proposed_best.pt` (read only) | GRID grammar | The demo. Say a GRID sentence, e.g. *"place blue at f two now"*, *"bin red by g nine soon"*. |
| **Any speech (experimental)** | Best seed of the film-trained model (`reports/flix_v2/lm/lm_summary.json`) | Word list + bigram LM (`data/flix_lm.json`) | Showing where open vocabulary stands. Expect rough results (≈96% WER on unseen films). |

To use a newer film model (e.g. v3, due around Monday), set
`DEMO_FILMS=<path to best.pt>` before starting the server.

## Running it (on the Alienware)

Everything lives in `~/Desktop/lipscribe_flix`, the Flix checkout. That folder has the film
model, `data/flix_lm.json`, `models/face_landmarker.task`, and `asr_pkgs/`, which holds
`imageio-ffmpeg`. That package is the only ffmpeg on the machine.

```bash
cd ~/Desktop/lipscribe_flix
# access code: generate once and keep it out of git (.demo_code is not committed)
[ -f .demo_code ] || python3 -c "import secrets; print(secrets.token_urlsafe(9))" > .demo_code
DEMO_CODE="$(cat .demo_code)" PYTHONPATH=asr_pkgs nohup setsid .venv/bin/python -m uvicorn demo.server:app \
    --host 127.0.0.1 --port 8800 >> logs/demo_server.log 2>&1 &
curl -s http://127.0.0.1:8800/health       # {"ok":true,"models":{"grid":...,"films":...}}
```

Stop it with `pkill -f "uvicorn demo.server:app"`.

### Open it privately (SSH tunnel, nothing public)

From a laptop that can `ssh alienware`:

```bash
ssh -N -L 8800:localhost:8800 alienware
# then open http://localhost:8800/?code=<the code in .demo_code>
```

Browsers only allow the camera on **https** or **localhost**, so this works on that laptop
only. To use another device, use the public tunnel below.

### Open it from any device (Cloudflare quick tunnel, public)

`cloudflared` is installed in `~/bin` on the Alienware. It is the official release, and its
sha256 was checked against the release notes. It connects **out** to Cloudflare, so no port is
opened on the university network. No Cloudflare account is needed.

```bash
cd ~/Desktop/lipscribe_flix
nohup setsid ~/bin/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8800 > logs/cloudflared.log 2>&1 &
grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" logs/cloudflared.log | head -1
# share:  https://<random>.trycloudflare.com/?code=<code>
```

- **Stop the tunnel:**
  ```
  pkill -f "^$HOME/bin/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8800"
  ```
  Keep this exact pattern. The Alienware also runs **another user's** `cloudflared`
  (`--url http://web:8081`), and that one must be left alone.
- **Restart the tunnel after a reboot:** run `touch .demo_tunnel` and the boot script will
  restart it too. It currently does **not**; the tunnel was switched off on 12 Nov.
- **Address changes:** the address changes every time the tunnel starts.
- **Access code:** anyone with the address still needs the code. Don't post the link with the
  code in it publicly.
- **Permission:** it is a shared university machine. Agree with whoever runs it before you
  leave a public tunnel up.

### A fixed address + the page on Cloudflare Pages (not done yet)

1. Create a free Cloudflare account. A domain on Cloudflare is needed for a named tunnel
   hostname.
2. On the Alienware, create a named tunnel to the backend:
   ```
   ~/bin/cloudflared tunnel login
   ~/bin/cloudflared tunnel create lipscribe
   ~/bin/cloudflared tunnel route dns lipscribe api.<domain>
   ~/bin/cloudflared tunnel run --url http://127.0.0.1:8800 lipscribe
   ```
3. Deploy the frontend: `npx wrangler pages deploy demo/ --project-name lipscribe`, or drag
   `demo/index.html` into *Workers & Pages → Create → Pages → Upload*.
4. Open `https://lipscribe.pages.dev/?api=https://api.<domain>&code=<code>`. The page keeps
   the server address and code in localStorage, and they can also be changed under
   **Settings**. CORS is already open on the backend; the access code is the protection.

## API

```
POST /caption   multipart: clip=<webm|mp4 with audio>, model=grid|films ; header X-Demo-Code
  200 {"caption": "bin blue at j five please", "seconds": 2.08, "model": "p_init_b1_b2_v2 (grammar)",
       "face_fraction": 1.0, "lips_trust": 0.23, "warnings": [], "mouth_strip": "data:image/png;base64,…"}
  401 wrong code · 413 > ~10 s · 422 friendly input problem (e.g. no face and video-only model)
GET  /health    {"ok": true, "models": {...}}
```

A test you can run on the Alienware. It converts a GRID test-speaker clip to WebM the way
Chrome records, then posts it:

```bash
FF=$(PYTHONPATH=asr_pkgs .venv/bin/python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
M=Data/grip_corpus/grid-corpus-dataset-for-training-lipnet/data/s24_processed/bbaj5p.mpg   # "bin blue at j five please"
$FF -y -i $M -c:v libvpx -b:v 1M -c:a libopus -ac 1 /tmp/t.webm
curl -s -H "X-Demo-Code: $(cat .demo_code)" -F clip=@/tmp/t.webm -F model=grid http://127.0.0.1:8800/caption
```

Tested on 12 Nov, through the public tunnel:

- Speaker s24, *"bin blue at j five please"*: **exact**, in 2.1 s.
- Speaker s4, *"bin blue at e nine now"*: **exact**, in 2.8 s.
- Wrong code: refused with 401.

## How to continue

1. **Test on real devices:** phones and laptops; Safari (sends MP4) and Chrome (WebM); no face
   in view; silence; noisy rooms (the lips share of the trust bar should rise); two uploads at
   once (the second waits for the first).
2. **Live captions (next feature):** the model reads a whole window both ways in time
   (BiLSTM), so it can't caption word by word. Plan:
   - add a WebSocket endpoint;
   - have the page send `MediaRecorder` chunks every ~1 s (`rec.start(1000)`);
   - on the server, keep the last ~3 s, decode it with `LipScribe.transcribe`, and push the
     caption back.
   Expect ~1–2 s delay. Roughly 2–3 hours of work.
3. **Fixed address / Cloudflare Pages:** see the section above.
4. **Better open-vocabulary model:** when v3 (film + Whisper + TED clips) finishes, point
   `DEMO_FILMS` at its best checkpoint.

## Don't disturb (long jobs on the Alienware, Nov 2026)

Training runs for days in `~/Desktop/lipscribe_flix`. See the PR description and
`scripts/flix_chain_run.sh`.

- **The jobs:** `scripts/ted_harvest.py` (TED clip harvest) and `scripts/flix_chain_run.sh`
  (ResNet tuning, then v3 on the full data; ~3–4 days).
- **After a reboot:** both restart by themselves (`crontab -l` → `scripts/flix_boot.sh`). If
  needed, restart by hand with `BOOT_DELAY=0 bash scripts/flix_boot.sh`; it never starts a
  second copy.
- **Logs:** `logs/flix_chain.log`, `logs/ted_harvest.log`, `logs/boot.log`,
  `logs/demo_server.log`.
- **The demo shares the GPU:** a caption takes a few seconds even while training runs.
