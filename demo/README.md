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
The demo page is one static page with no build step: `demo/index.html`, plus `demo/app.js` (the page logic) and `demo/mock-backend.js` (a pretend backend for testing).

## Using the page

1. Open the page and press **Turn on camera & mic**. Allow both when the browser asks.
2. Pick how you want captions:
   - **Live**: press the glowing button and talk normally. Words appear while you speak. Press it again (or Esc) to stop.
   - **Clip**: press the button and say one sentence. It stops by itself after 3 seconds, and the captions arrive 2–3 seconds later.
3. Pick the model:
   - **GRID sentences**: the six-word GRID commands, e.g. "bin blue at j five please".
   - **Any speech**: open vocabulary (experimental).
4. Read the results:
   - **Ears only** (sound alone) and **Ears + lips** (sound and lips) are shown side by side. Words where Ears only disagrees with Ears + lips are underlined in pink.
   - **Room noise** is measured automatically. The louder the room, the more the model leans on the lips.
   - **Who the model trusted**: each bar is one frame (25 a second). Teal is the sound's share of the model's trust and pink is the lips' share.
   - **What the eyes saw**: the 96×96 mouth crops the model used.

The page works on phones and laptops in Chrome, Edge, Firefox and Safari. Browsers only give a page the camera when it is opened on `https://…` or on `localhost`.

### Light and dark

The page follows the device's theme. Use the moon/sun button to switch, or add `?theme=dark` or `?theme=light` to the address.

## Pretend backend vs real backend

| Address | What it talks to |
|---|---|
| `…/` (nothing added) | The **pretend backend** in `mock-backend.js`. It shows sample captions, and the mouth crops come from your own camera. |
| `…/?backend=wss://HOST/ws/live` | The **real backend** on the Alienware |

Once the backend address stops changing, set `DEFAULT_BACKEND` at the top of `app.js`.

Testing switches for the pretend backend:

- `?mock_error=no_face` (also `no_audio`, `clip_too_short`, `server_busy`) shows each error message.
- `?mock_noise=quiet` or `?mock_noise=noisy` fixes the room noise. In a quiet room both captions match, which is expected.
- `?run=clip` and `?mode=any` open the page in Clip mode or Any speech mode.

## Message format (for the backend)

The page opens **one WebSocket** per recording.

### Page → backend

1. A text message to start:

   ```json
   {"type":"start","run":"live","mode":"grid","mime":"video/webm;codecs=vp8,opus","chunk_ms":1000,"fps":25,"clip_seconds":null}
   ```

   - `run` is `live` or `clip`.
   - `mode` is `grid` (GRID sentences) or `any` (Any speech).
   - `clip_seconds` is `3` in Clip mode and `null` in Live mode.
   - Safari sends `video/mp4` instead of WebM, so read `mime`.

2. **Binary messages**: MediaRecorder chunks of audio and video together, one about every `chunk_ms`. The first chunk carries the container header, so join the chunks in order to decode them.

3. A text message `{"type":"stop"}` after the last chunk.

### Backend → page

The page only reads the fields listed here. Fields can be left out.

```json
{"type":"ready"}
```

```json
{"type":"progress","step":"sent|face|listen|write"}
```

Clip mode only. It drives the "Reading your lips…" steps.

```json
{"type":"partial",
 "t": 2.4,
 "ears": {"text":"bin blue at", "pending":"d", "model":"twin_v5"},
 "lips": {"text":"bin blue at", "pending":"j", "model":"ws_v5"},
 "gate": [0.21, 0.24],
 "noise_db": -4.8,
 "mouth_box": [0.36, 0.56, 0.28, 0.2],
 "crops": [{"t": 2.4, "src": "data:image/jpeg;base64,..."}]}
```

| Field | Meaning |
|---|---|
| `t` | Seconds since the recording started. |
| `ears`, `lips` | Each has `text` (the words decided so far) and `pending` (the words that may still change, shown faded), plus the model's name. Send the **whole** transcript each time, not only the new words. |
| `gate` | The lips' share of the model's trust, from 0 to 1, for each **new** frame since the last message, at 25 per second. Sound's share is 1 minus this. |
| `noise_db` | The estimated signal-to-noise ratio in dB. Higher means quieter. The page reads roughly ≥15 as quiet, 5–15 as some noise, −2 to 5 as noisy and below −2 as very noisy. |
| `mouth_box` | The mouth's position in the camera frame as `[x, y, width, height]`, each a fraction from 0 to 1 of the un-mirrored frame. `null` means no face was found. |
| `crops` | New 96×96 mouth crops, each with its time and a data URL. The page keeps the last 8. |

```json
{"type":"final", ...same fields..., "processing_s": 2.6, "face_found": 1.0}
```

- Live mode: the transcript after the stop.
- Clip mode: the whole result, where `gate` and `crops` cover the **whole** clip.
- `processing_s` is the time from stop to result.
- `face_found` is the fraction of frames in which a face was found.

```json
{"type":"error","code":"no_face","message":"optional detail for the console"}
```

The page has friendly text for these codes: `no_face`, `no_audio`, `clip_too_short`, `server_busy`. Any other code shows a general message.

## Run it on your laptop

```bash
cd demo
python -m http.server 8000
```

Then open http://localhost:8000.

## Put it online (Cloudflare Pages)

Run this once:

```bash
npx wrangler login
```

Then deploy, from the repository root, on Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File demo\deploy-pages.ps1
```

The page goes live at https://lipscribe.pages.dev. Only `index.html`, `app.js` and `mock-backend.js` are uploaded, never the backend code.

