# LipScribe web demo

The page records you with the camera and microphone, or plays a video file you choose, and shows
two captions side by side: **Ears only** (sound alone) and **Ears + lips** (sound and lips). It also
shows how much the model trusted the sound and the lips, frame by frame, and the mouth crops it used.

```
 phone / laptop browser                         Alienware (backend)
 ┌─────────────────────────────┐  WebSocket   ┌──────────────────────────────────────────────┐
 │ demo/index.html + app.js    │ ───────────▶ │ demo/server.py  (FastAPI, 127.0.0.1:8800)    │
 │ Live / Clip: MediaRecorder  │ chunks, or   │   lipscribe.app.session   one per recording  │
 │   chunks every 1 s          │ the pieces   │   lipscribe.app.stream    bytes → PyAV →     │
 │ Upload: the file plays from │ of a file    │                           16 kHz + 25 fps    │
 │   disk; only bytes go up    │ ◀─────────── │   Silero VAD → speech segments ≤ 4 s         │
 │ ◀ captions, gate, crops     │ captions     │   lipscribe.app.realworld ws_v5 + twin_v5    │
 └─────────────────────────────┘              │   lipscribe.app.inference P4 + B1 (GRID)     │
          ▲ SSH tunnel (private) or Cloudflare Tunnel (public; no port opened)            │
                                              └──────────────────────────────────────────────┘
```

## Files

| File | What it is |
|---|---|
| `demo/index.html`, `demo/app.js` | The page (one static page, no build step). Owner: Wakisa Isabelle. |
| `demo/mock-backend.js` | A pretend backend for building the page without the Alienware. |
| `demo/server.py` | The backend: `WS /ws/live`, `GET /health`, and `GET /` (serves the page too). |
| `lipscribe/app/session.py` | One recording or upload: segments, captions and the messages to the page. |
| `lipscribe/app/stream.py` | The growing upload file, the PyAV reader, the timeline and Silero VAD. |
| `lipscribe/app/realworld.py` | ws_v5 and twin_v5 with the Flix v5 preprocessing and the F-04 decoder. |
| `lipscribe/app/inference.py` | P4 and B1 (the `evaluate.py` chain), used for GRID sentences. |

### The models (the "GRID sentences" / "Any speech" toggle)

| Toggle | Ears + lips | Ears only | Decoder |
|---|---|---|---|
| **GRID sentences** | P4, `checkpoints/p_proposed_best.pt` | B1, `checkpoints/b1_audio_only_best.pt` | GRID grammar |
| **Any speech** | ws_v5, `runs/flix_v5/flix_v5_fusion_gated_ws_20261116-093641/best.pt` | twin_v5, `runs/flix_v5/flix_v5_fusion_wp_audio_20261116-000004/best.pt` | beam + 4-gram LM, the F-04 settings (`reports/flix_v5/final_decoder.md`) |

"Any speech" makes its inputs exactly as the Flix v5 clips were made: frames at 25 fps and at most
640 px wide, MediaPipe landmarks, the Auto-AVSR mouth crop, and 16 kHz audio over the same window,
with 0.32 s added on each side of the speech. On 32 v5 test clips (16 TED, 16 film), batched as in
the evaluation, the backend gives exactly the evaluation's captions: 64 of 64 model outputs, both
greedy and beam + LM. The demo runs one segment at a time, and then 37 of 40 outputs match; the
other 3 differ by a word, because WavLM's normalisation sees a batch's padding.

### Runs

| Run | Input | What comes back |
|---|---|---|
| **Live** | MediaRecorder chunks every 1 s | `partial` about every 0.5 s: finished stretches of speech (beam + LM) and the stretch still going on as `pending` (greedy, replaced when it ends) |
| **Clip** | a 3 s recording | one `final` with both captions, the gate over the whole clip and 4 mouth crops |
| **Upload** | a file of any length, up to 8 GB or 3 hours (MP4, MKV, MOV, WebM; any codec PyAV reads) | `cues` in batches while the file uploads, then `final` |

**How uploads work.** The page does not send the file in one go. The backend asks for 4 MB pieces
(`need`), at most 3 at a time, so no request comes near Cloudflare's 100 MB limit. It asks first for
the bytes the decoder is waiting on. An MP4 whose index (`moov`) is at the end of the file can
therefore start at once: the backend asks for the last piece, then reads from the start. Speech is
cut into stretches of at most 4 s with Silero VAD, and each stretch becomes one subtitle cue.

**Which caption to trust.** ws_v5 learned only from clips that showed one large, frontal face with
moving lips, no camera cut, and the face in at least 90 % of frames (the rules in
`scripts/mine_flix.py`). On other footage the visible lips may belong to someone who is not
speaking, and then ws_v5 does much worse than ears alone (F-04: 103 % WER when the voice is someone
else's). So every upload cue says whether its segment passes those rules (`lips_used`, with the
`reason` when it does not). The page's **Best of both** track shows ws_v5 for those cues and twin_v5
for the rest.

Tested on 17 Nov 2026 on two held-out test films (excerpts):

- *Challengers* (10 min MP4, index at the end): captioned in 54 s, 116 cues. Lips were used for 1 %
  of the speech, because the film is mostly cutaways and profiles. Against the film's subtitles the
  WER is about 40 % for each track: ears 39.4 %, ears + lips 40.6 %, best of both 39.6 %. Subtitles
  paraphrase, so treat this as a rough check.
- *Little Women* (5 min MKV): captioned in 40 s, 68 cues; lips used for 10 % of the speech.
- Live mode in Chrome (camera and microphone simulated from a film clip): the first words appear
  about 3 s after you start talking.

## Running it (on the Alienware)

The demo runs from a clean copy of the repository, `~/Desktop/lipscribe_demo`. It has symlinks to
the large assets in `~/Desktop/lipscribe_flix` and `~/Desktop/cv_project_LipScribe`: `.venv`,
`runs/`, `lm/`, `models/face_landmarker.task`, `pretrained/`, `Data/` (MUSAN), `checkpoints/`,
`data/stats/` and `asr_pkgs/` (imageio-ffmpeg). The environment also needs `websockets`, `av` and
`silero-vad`. They were installed on 17 Nov, `silero-vad` with `--no-deps` so that torch stays as
it is.

Since 17 Nov the demo is **always on**: `scripts/demo_up.sh` starts the backend on 127.0.0.1:8800
and a public Cloudflare quick tunnel to it, and starts only what is not already running. It runs
at boot (`scripts/flix_boot.sh`, crontab `@reboot`) and every 5 minutes from cron, so a crash or a
dropped tunnel is repaired within 5 minutes.

```bash
cd ~/Desktop/lipscribe_demo
bash scripts/demo_up.sh          # prints the share links
bash scripts/demo_up.sh stop     # stops both (cron starts them again unless you also edit crontab)
touch .demo_no_tunnel            # keep it private: no public tunnel from now on
curl -s http://127.0.0.1:8800/health
```

- **Access code:** `.demo_code` (generated once, not in git).
- **Current public address:** `.demo_url`. It changes only when the tunnel restarts (a reboot, or a
  tunnel crash).
- **Logs:** `logs/demo_server.log`, `logs/demo_tunnel.log`, `logs/demo_up.log`.

The models take about 30 s to load; until then `/health` says `"loading": true`. `DEMO_MODELS=any`
or `DEMO_MODELS=grid` loads only one pair. The server takes up to 3 recordings at once and one
upload at a time; the GPU runs one segment at a time, so a camera clip still gets through while a
film is being captioned.

**Share link.** The backend serves the page itself and points it at itself, so this works on any
device, phone or laptop, anywhere:

```
https://<address in .demo_url>/?code=<code>
```

The Pages site works too, once it has the current frontend:
`https://lipscribe.pages.dev/?backend=wss://<address>/ws/live&code=<code>`. It remembers the address
and the code in that browser.

### Open it privately (SSH tunnel, nothing public)

```bash
ssh -N -L 8800:localhost:8800 alienware
# then open http://localhost:8800/?backend=self&code=<the code in .demo_code>
```

Browsers only allow the camera on **https** or **localhost**, so this works on that laptop only.
Uploads work from any browser.

### Open it from any device (Cloudflare quick tunnel, public)

`cloudflared` is in `~/bin` on the Alienware (the official release, sha256 checked). It connects
**out** to Cloudflare, so no port is opened on the university network, and no account is needed.

`demo_up.sh` starts and stops it (it tracks the process by its PID).

- **Other tunnel:** the Alienware also runs **another user's** `cloudflared` (`--url http://web:8081`).
  Never stop cloudflared by name; leave that one alone.
- **A fixed address** needs Tailscale Funnel enabled by the tailnet admin, or a Cloudflare account
  with a domain (a named tunnel). Neither is set up.
- **Address changes:** the address changes every time the tunnel starts.
- **Access code:** anyone with the address still needs the code. Do not post the link with the
  code in it publicly.
- **Permission:** it is a shared university machine. Agree with whoever runs it before you leave a
  public tunnel up.

### Test it without a browser

`scripts/demo_ws_client.py` drives the WebSocket exactly as the page does (start message, binary
chunks or `need` replies, stop) and prints every message with its arrival time:

```bash
.venv/bin/python scripts/demo_ws_client.py "ws://127.0.0.1:8800/ws/live?code=$(cat .demo_code)" clip grid clip.webm --snr -5
.venv/bin/python scripts/demo_ws_client.py "ws://127.0.0.1:8800/ws/live?code=$(cat .demo_code)" upload any film.mp4
```
 The 17 Nov tests used held-out GRID speaker s4 and test-film
excerpts. At −5 dB of MUSAN noise, B1 heard *"bin blue at l five please"* as "bin blue at **f** five
please", and P4 got it exactly right.

`legacy.html`, the first page, used the old `POST /caption` endpoint, which has been removed.

## Captions in a Zoom meeting

In **Live** mode, choose **Caption from: A meeting**. The host shares the meeting's tab, window or
screen, and everyone in the Zoom meeting sees LipScribe's lines as the meeting's own closed captions.

1. **Once, in the Zoom web settings** (Settings → Meeting → In meeting (advanced)): turn on
   *Closed captioning* and *Allow use of caption API token to integrate with 3rd-party closed
   captioning services*. Whether a free Zoom account offers this was not checked.
2. **In the meeting:** CC → *Set up manual captioner* → *Copy the API token*, and paste it into
   **Zoom caption API token** on the page. The token stops working if the meeting ends or restarts.
3. **Meeting sound from:**
   - *The shared tab or screen*: Zoom in a Chrome or Edge tab (tick "Share tab audio"), or the
     Zoom app on Windows (share the screen and tick "Share system audio");
   - *an audio input*: the Zoom app on a Mac, with its sound routed through a virtual device such
     as BlackHole; on Linux, the "Monitor of …" input.
4. Press the button and pick the meeting when the browser asks. Tell everyone in the meeting that
   it is being captioned.

The backend sends each finished line (the **Best of both** line: ws_v5 where the lips pass the
training-clip rules, twin_v5 elsewhere) with a rising sequence number. It posts only to
`https://….zoom.us` caption links (`lipscribe/app/zoom_captions.py`). The page shows how many lines
reached Zoom, or why Zoom refused them. Lines appear in Zoom about 2–4 s after they are spoken,
because only finished lines are sent. Meeting tiles are small and often show someone other than
the speaker, so most lines come from the ears alone.

Tested on 17 Nov in Chrome, with a tab playing a film clip and its sound from an audio input: live
captions on the page, and the backend's posts reached Zoom's caption server, which refused the test
token (403) as expected. Not yet tested in a real Zoom meeting.

## Using the page

1. Open the page and press **Turn on camera & mic**. Allow both when the browser asks.
2. Pick how you want captions:
   - **Live**: press the glowing button and talk normally. Words appear while you speak. Press it again (or Esc) to stop.
   - **Clip**: press the button and say one sentence. It stops by itself after 3 seconds, and the captions arrive 2–3 seconds later.
   - **Upload** (in Clip mode): choose a video file instead, of any length, even a whole film. It plays in the page while it uploads, and captions arrive in batches a little ahead of the picture. If playback catches up with them, the video pauses until they are ahead again. Pick the subtitle track (**Best of both**, **Ears only** or **Ears + lips**) and download it as an `.srt` file.
3. Pick the model:
   - **GRID sentences**: the six-word GRID commands, e.g. "bin blue at j five please".
   - **Any speech**: open vocabulary (experimental).
4. Read the results:
   - **Ears only** (sound alone) and **Ears + lips** (sound and lips) are shown side by side. Words where Ears only disagrees with Ears + lips are underlined in pink.
   - **Room noise** is measured automatically. The louder the room, the more the model leans on the lips.
   - **Add test noise before captioning** mixes MUSAN noise into the sound at the chosen level (10 to −10 dB) before the models hear it, the same noise as the test sets. It shows what the lips add when the sound gets bad.
   - In an upload, the small label under **Ears + lips** says whether the lips were used for the line playing now, and why not when they were not.
   - **Who the model trusted**: each bar is one frame (25 a second). Teal is the sound's share of the model's trust and pink is the lips' share.
   - **What the eyes saw**: the 96×96 mouth crops the model used.

The page works on phones and laptops in Chrome, Edge, Firefox and Safari. Browsers only give a page the camera when it is opened on `https://…` or on `localhost`.

### Light and dark

The page follows the device's theme. Use the moon/sun button to switch, or add `?theme=dark` or `?theme=light` to the address.

## Pretend backend vs real backend

| Address | What it talks to |
|---|---|
| `…/` (nothing added) | The **pretend backend** in `mock-backend.js`. It shows sample captions, and the mouth crops come from your own camera. |
| `…/?backend=wss://HOST/ws/live&code=CODE` | The **real backend** on the Alienware. The code is remembered in this browser. |
| `…/?backend=self&code=CODE` | The backend that served the page (e.g. `http://localhost:8800` through the SSH tunnel) |

Once the backend address stops changing, set `DEFAULT_BACKEND` at the top of `app.js`.

Testing switches for the pretend backend:

- `?mock_error=no_face` (also `no_audio`, `clip_too_short`, `server_busy`) shows each error message.
- `?mock_noise=quiet` or `?mock_noise=noisy` fixes the room noise. In a quiet room both captions match, which is expected.
- `?run=clip` and `?mode=any` open the page in Clip mode or Any speech mode.

## Message format (for the backend)

The page opens **one WebSocket** per recording or upload, at `/ws/live?code=<access code>`. A wrong
code gets `{"type":"error","code":"bad_code"}` and the socket closes.

### Page → backend

1. A text message to start:

   ```json
   {"type":"start","run":"live","mode":"grid","mime":"video/webm;codecs=vp8,opus","chunk_ms":1000,"fps":25,"clip_seconds":null}
   ```

   - `run` is `live`, `clip` or `upload`.
   - `mode` is `grid` (GRID sentences) or `any` (Any speech).
   - `clip_seconds` is `3` in Clip mode and `null` in Live mode.
   - Safari sends `video/mp4` instead of WebM, so read `mime`.
   - `snr` (optional): `null` for the sound as recorded, or a number in dB; the backend mixes MUSAN test noise in at that level before captioning.
   - Upload only: `"name"` and `"size"` (bytes) of the file. GRID sentences takes uploads of up to 10 s.

2. **Binary messages**: MediaRecorder chunks of audio and video together, one about every `chunk_ms`. The first chunk carries the container header, so join the chunks in order to decode them.

3. A text message `{"type":"stop"}` after the last chunk.

**Upload:** instead of chunks, the page answers each `need` message with one binary message: the
8-byte big-endian `offset`, then exactly `length` bytes of the file from that offset.
`{"type":"cancel"}` stops the upload.

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

The page has friendly text for these codes: `no_face`, `no_audio`, `clip_too_short`, `server_busy`, `bad_code`, `bad_file`, `file_too_large`, `grid_too_long`, `model_unavailable`, `server_error`. Any other code shows a general message.

### Upload messages (backend → page)

```json
{"type":"ready","ears_model":"twin_v5","lips_model":"ws_v5","chunk_bytes":4194304}
{"type":"need","offset":88080384,"length":4194304}
{"type":"progress","uploaded":12582912,"size":89389751}
{"type":"progress","captioned_s":131.2,"duration_s":600.2}
{"type":"cues","captioned_s":152.3,"duration_s":600.2,
 "cues":[{"i":3,"start":149.67,"end":151.36,"ears":"know shes a pillar of the community",
          "lips":"i know shes a pill of the community","lips_used":false,"reason":"not frontal",
          "face":0.95,"gate":[0.31,0.29],"crops":[{"t":149.7,"src":"data:image/jpeg;base64,..."}]}]}
{"type":"final","processing_s":54.2,"cues":116,"duration_s":600.2,"face_found":0.36,"lips_share":0.006}
```

| Field | Meaning |
|---|---|
| `start`, `end` | Where the speech is, in seconds of the file. The model heard 0.32 s more on each side. |
| `ears`, `lips` | twin_v5 and ws_v5 (or B1 and P4) for this stretch. |
| `lips_used`, `reason` | Whether the segment passes the training-clip rules, so that ws_v5's line can be trusted; if not, why (`camera cut`, `no face`, `not frontal`, `face too small`, `2nd face`, `face lost`, `lips still`). |
| `gate`, `crops` | The lips' share of trust for each frame of the speech, and 4 mouth crops. |
| `captioned_s` | Everything before this time has been captioned. |
| `lips_share` | The fraction of the speech time whose line came from ws_v5. |

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

