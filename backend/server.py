"""LipScribe demo backend (FastAPI, F-05). Runs on the Alienware; see backend/README.md.

WS   /ws/live?code=<access code>   one WebSocket per recording or upload (message format: backend/README.md)
GET  /health                       loaded models and busy state
GET  /, /app.js, /config.js, /mock-backend.js  the page itself (so the backend can also serve it)

Models (the page's "mode"):
  grid  "GRID sentences": P4 + GRID grammar (ears + lips) beside B1 + grammar (ears only)
  any   "Any speech": ws_v5 (ears + lips) beside its audio-only twin twin_v5, beam + 4-gram LM (F-04)

Runs: live (camera, captions while you talk), clip (3 s camera clip), upload (a file of any length,
captioned in batches while it uploads). The work is done by lipscribe.app.session.

  DEMO_CODE=<secret> PYTHONPATH=.:asr_pkgs .venv/bin/python -m uvicorn backend.server:app \
      --host 127.0.0.1 --port 8800 --ws-max-size 16777216
Locally it binds to localhost only (backend/run_local.sh). In the Coolify container it listens on
0.0.0.0:8800 behind Coolify's proxy (backend/Dockerfile, DEPLOYMENT.md).

Environment (all optional):
  DEMO_CODE         access code the page must send (?code=...); empty means no code
  DEMO_MODELS       "any,grid" (default), "any" or "grid"
  DEMO_ORIGINS      comma-separated origins allowed by CORS, e.g. https://lipscribe.pages.dev (default "*")
  DEMO_P4, DEMO_B1  GRID checkpoints (default checkpoints/p_proposed_best.pt, checkpoints/b1_audio_only_best.pt)
  DEMO_FACE         MediaPipe face landmarker (default models/face_landmarker.task)
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from lipscribe.app.session import Engines, Session, SessionError

REPO = Path(__file__).resolve().parents[1]
CODE = os.environ.get("DEMO_CODE", "")
P4 = Path(os.environ.get("DEMO_P4", REPO / "checkpoints/p_proposed_best.pt"))
B1 = Path(os.environ.get("DEMO_B1", REPO / "checkpoints/b1_audio_only_best.pt"))
FACE = Path(os.environ.get("DEMO_FACE", REPO / "models/face_landmarker.task"))
FRONTEND = REPO / "frontend"
ORIGINS = [o.strip() for o in os.environ.get("DEMO_ORIGINS", "*").split(",") if o.strip()]
MODELS = set(os.environ.get("DEMO_MODELS", "any,grid").split(","))
CHUNK = 4 << 20                 # upload piece size (well under Cloudflare's 100 MB request limit)
IN_FLIGHT = 3                   # upload pieces requested at once
MAX_UPLOAD = 8 << 30            # 8 GB
MAX_SESSIONS = 3

app = FastAPI(title="LipScribe demo")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"])
_engines: Engines | None = None
_load_error: str | None = None
_sessions: dict[str, Session] = {}
_slock = threading.Lock()


@app.on_event("startup")
def _load() -> None:
    def load():
        global _engines, _load_error
        try:
            _engines = Engines(REPO, P4, B1, FACE, load_grid="grid" in MODELS, load_any="any" in MODELS)
            print("models loaded", flush=True)
        except Exception as e:                           # noqa: BLE001 - shown by /health
            _load_error = f"{type(e).__name__}: {e}"
            print("could not load the models:", _load_error, flush=True)
    threading.Thread(target=load, daemon=True).start()


@app.get("/health")
def health() -> dict:
    e = _engines
    models = {}
    if e is not None:
        if e.rw is not None:
            models["any"] = list(e.names("any"))
        if e.p4 is not None:
            models["grid"] = list(e.names("grid"))
    return {"ok": e is not None, "loading": e is None and _load_error is None, "error": _load_error,
            "models": models, "sessions": len(_sessions),
            "upload_busy": any(s.run == "upload" for s in _sessions.values())}


@app.get("/")
def page() -> HTMLResponse:
    """The page, told to talk to this server (so the tunnel link alone works on any device)."""
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    html = html.replace('<script src="mock-backend.js"></script>',
                        '<script>window.LIPSCRIBE_BACKEND = "self";</script>\n<script src="mock-backend.js"></script>', 1)
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


@app.get("/app.js")
def app_js() -> FileResponse:
    return FileResponse(FRONTEND / "app.js", media_type="text/javascript")


@app.get("/mock-backend.js")
def mock_js() -> FileResponse:
    return FileResponse(FRONTEND / "mock-backend.js", media_type="text/javascript")


@app.get("/config.js")
def config_js() -> FileResponse:
    return FileResponse(FRONTEND / "config.js", media_type="text/javascript")


async def _error(ws: WebSocket, code: str, message: str = "") -> None:
    with contextlib.suppress(Exception):
        await ws.send_text(json.dumps({"type": "error", "code": code, "message": message}))
        await ws.close()


def _snr(v) -> float | None:
    if v in (None, "", "clean"):
        return None
    v = float(v)
    if not -20 <= v <= 40:
        raise SessionError("bad_request", "snr must be between -20 and 40 dB")
    return v


@app.websocket("/ws/live")
async def live(ws: WebSocket, code: str = "") -> None:
    await ws.accept()
    if CODE and code != CODE:
        await _error(ws, "bad_code", "wrong or missing access code (add ?code=... to the page address)")
        return
    if _engines is None:
        await _error(ws, "server_busy" if _load_error is None else "server_error",
                     "the models are still loading" if _load_error is None else _load_error)
        return
    try:
        start = json.loads(await asyncio.wait_for(ws.receive_text(), timeout=15))
    except (asyncio.TimeoutError, json.JSONDecodeError, WebSocketDisconnect, KeyError):
        await _error(ws, "bad_request", "expected a start message")
        return
    run, mode = start.get("run", "clip"), start.get("mode", "grid")
    size = start.get("size")
    with _slock:
        busy = len(_sessions) >= MAX_SESSIONS or (run == "upload" and any(s.run == "upload" for s in _sessions.values()))
    if busy:
        await _error(ws, "server_busy", "another upload is being captioned" if run == "upload" else "too many recordings at once")
        return
    if run == "upload" and (not isinstance(size, int) or not 0 < size <= MAX_UPLOAD):
        await _error(ws, "file_too_large" if isinstance(size, int) and size > MAX_UPLOAD else "bad_request",
                     f"size must be 1 byte to {MAX_UPLOAD >> 30} GB")
        return

    loop = asyncio.get_running_loop()
    out: asyncio.Queue = asyncio.Queue()

    def emit(msg: dict) -> None:
        loop.call_soon_threadsafe(out.put_nowait, msg)

    sid = uuid.uuid4().hex[:8]
    try:
        s = Session(_engines, run, mode, emit, snr=_snr(start.get("snr")), size=size,
                    mime=start.get("mime", ""), name=start.get("name", ""), sid=sid,
                    source=start.get("source", "camera"), zoom_url=start.get("zoom_url") or None)
    except SessionError as e:
        await _error(ws, e.code, e.message)
        return
    with _slock:
        _sessions[sid] = s
    s.start()
    print(f"{time.strftime('%H:%M:%S')} session {sid} start run={run} mode={mode} size={size} snr={start.get('snr')} "
          f"source={start.get('source', 'camera')} zoom={'yes' if start.get('zoom_url') else 'no'}", flush=True)
    names = _engines.names(mode)
    await ws.send_text(json.dumps({"type": "ready", "ears_model": names[0], "lips_model": names[1],
                                   **({"chunk_bytes": CHUNK} if run == "upload" else {})}))
    asked: dict[int, float] = {}                     # upload piece -> time requested

    async def receive() -> None:
        while True:
            m = await ws.receive()
            if m["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()
            if m.get("bytes") is not None:
                b = m["bytes"]
                if run == "upload":
                    off = int.from_bytes(b[:8], "big")
                    s.feed_at(off, b[8:])
                    asked.pop(off // CHUNK, None)
                else:
                    s.feed(b)
            elif m.get("text"):
                with contextlib.suppress(json.JSONDecodeError):
                    t = json.loads(m["text"])
                    if t.get("type") == "stop":
                        s.stop()
                    elif t.get("type") == "cancel":
                        s.cancel()
                        return

    async def request_pieces() -> None:
        """Ask the page for the upload pieces the decoder needs first, then the rest in order."""
        n = (size + CHUNK - 1) // CHUNK
        last_progress = 0.0
        while True:
            src = s.src
            have = {k for k in range(n) if src.covered(k * CHUNK, min(size, (k + 1) * CHUNK))}
            now = time.time()
            for k, t in list(asked.items()):
                if now - t > 60:                     # lost: ask again
                    asked.pop(k)
            want = []
            if src.wanted is not None:
                want.append(src.wanted // CHUNK)
            want += [k for k in range(n) if k not in have]
            for k in want:
                if len(asked) >= IN_FLIGHT:
                    break
                if k in have or k in asked:
                    continue
                asked[k] = now
                a = k * CHUNK
                await ws.send_text(json.dumps({"type": "need", "offset": a, "length": min(CHUNK, size - a)}))
            if now - last_progress > 1.0:
                await ws.send_text(json.dumps({"type": "progress", "uploaded": src.received(), "size": size}))
                last_progress = now
            if len(have) == n:
                await ws.send_text(json.dumps({"type": "progress", "uploaded": size, "size": size}))
                return
            await asyncio.sleep(0.05)

    async def send() -> None:
        while True:
            msg = await out.get()
            await ws.send_text(json.dumps(msg))
            if msg.get("type") in ("final", "error"):
                return

    tasks = [asyncio.create_task(receive()), asyncio.create_task(send())]
    if run == "upload":
        tasks.append(asyncio.create_task(request_pieces()))
    try:
        # ends when the result is sent (sender) or the page goes away (receiver)
        await asyncio.wait(tasks[:2], return_when=asyncio.FIRST_COMPLETED)
    finally:
        for t in tasks:
            t.cancel()
        s.cancel()
        with _slock:
            _sessions.pop(sid, None)
        with contextlib.suppress(Exception):
            await ws.close()
        print(f"{time.strftime('%H:%M:%S')} session {sid} end ({time.time() - s.started:.1f} s)", flush=True)
