# LipScribe demo page (frontend)

One static page, no build step: `index.html`, `app.js`, `config.js` and `mock-backend.js`. It is
deployed to Cloudflare Pages (https://lipscribe.pages.dev). The backend it talks to is in
[`../backend/`](../backend/README.md), which also documents the WebSocket message format.
Deployment, step by step: [`../DEPLOYMENT.md`](../DEPLOYMENT.md).

| File | What it is |
|---|---|
| `index.html`, `app.js` | The page. Owner: Wakisa Isabelle. |
| `config.js` | The backend address the Pages copy uses by default (`window.LIPSCRIBE_BACKEND`). |
| `mock-backend.js` | A pretend backend for building the page without the Alienware. |
| `legacy.html` | The first page; it used the old `POST /caption` endpoint, which has been removed. |
| `deploy-pages.ps1` | Manual upload to Cloudflare Pages. |

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

The default address (used when the link has no `?backend=`) is set in `config.js`. A browser that
was once given a `?backend=` address remembers it and uses it instead; open the page once with
`?backend=mock` to forget it.

Testing switches for the pretend backend:

- `?mock_error=no_face` (also `no_audio`, `clip_too_short`, `server_busy`) shows each error message.
- `?mock_noise=quiet` or `?mock_noise=noisy` fixes the room noise. In a quiet room both captions match, which is expected.
- `?run=clip` and `?mode=any` open the page in Clip mode or Any speech mode.

## Run it on your laptop

```bash
cd frontend
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
powershell -ExecutionPolicy Bypass -File frontend\deploy-pages.ps1
```

The page goes live at https://lipscribe.pages.dev. Only `index.html`, `app.js`, `config.js` and `mock-backend.js` are uploaded, never the backend code.

**Automatic deploys.** `.github/workflows/deploy-pages.yml` does the same upload whenever a merge to `main` changes a file in `frontend/`, so an approved and merged pull request goes live by itself. It needs two repository secrets, set once by the owner of the Cloudflare account: `CLOUDFLARE_API_TOKEN` (a token with *Cloudflare Pages: Edit*) and `CLOUDFLARE_ACCOUNT_ID`. Until both are set, the workflow skips the upload, and the manual script above still works.
