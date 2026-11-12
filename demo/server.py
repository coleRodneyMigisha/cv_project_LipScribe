"""LipScribe web demo backend (FastAPI), runs on the Alienware. See demo/README.md.

POST /caption  multipart: clip (a browser recording, webm/mp4 with audio), model ("grid"|"films"),
               header X-Demo-Code -> {caption, seconds, face_fraction, lips_trust, warnings, mouth_strip}
GET  /health   -> loaded models
GET  /         -> the demo page (demo/index.html)

Models:
  grid   Model P4 (p_proposed_best.pt, read-only) + GRID grammar decoder: accurate for GRID
         sentences ("place blue at f two now") - the trained task.
  films  experimental open-vocabulary model retrained on film clips + lexicon/bigram LM.
The caption chain is lipscribe.app.inference.LipScribe (same as evaluate.py, PR #55).

  DEMO_CODE=<secret> PYTHONPATH=asr_pkgs .venv/bin/python -m uvicorn demo.server:app \
      --host 127.0.0.1 --port 8800
Binds to localhost only; reach it through an SSH tunnel or a Cloudflare Tunnel.
"""
from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from lipscribe.app.inference import FriendlyError, LipScribe, mouth_strip

REPO = Path(__file__).resolve().parents[1]
CODE = os.environ.get("DEMO_CODE", "")
P4 = Path(os.environ.get("DEMO_P4", REPO.parent / "cv_project_LipScribe/checkpoints/p_proposed_best.pt"))
FILMS = os.environ.get("DEMO_FILMS")          # checkpoint of the film model (default: best v2 seed)
LM = REPO / "data/flix_lm.json"

app = FastAPI(title="LipScribe demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_engines: dict[str, LipScribe] = {}
_lock = threading.Lock()                      # one clip at a time on the shared GPU


def _films_checkpoint() -> tuple[Path, dict]:
    if FILMS:
        return Path(FILMS), {"alpha": 1.5, "beta": 4.0}
    summ = json.loads((REPO / "reports/flix_v2/lm/lm_summary.json").read_text())
    best = min((k for k in summ if k.startswith("sweep_s")), key=lambda k: summ[k]["val_wer"])
    return Path(summ[best]["checkpoint"]), {"alpha": summ[best]["alpha"], "beta": summ[best]["beta"]}


def engine(name: str) -> LipScribe:
    if name not in _engines:
        if name == "grid":
            _engines[name] = LipScribe(P4, decode="grammar", face_model=REPO / "models/face_landmarker.task")
        elif name == "films":
            from lipscribe.decoding import get_decoder
            ck, lm = _films_checkpoint()
            e = LipScribe(ck, decode="greedy", face_model=REPO / "models/face_landmarker.task")
            e.decode_fn = get_decoder({"method": "lexicon_lm", "lm_path": str(LM), "beam_width": 32, **lm})
            e.decode_method = "word grammar"
            _engines[name] = e
        else:
            raise HTTPException(400, "model must be 'grid' or 'films'")
    return _engines[name]


@app.on_event("startup")
def _warm() -> None:
    for n in ("grid", "films"):
        try:
            engine(n)
        except Exception as e:                # the page still works with the other model
            print(f"could not load {n}: {e}", flush=True)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "models": {k: v.name for k, v in _engines.items()}}


@app.get("/")
def page() -> FileResponse:
    return FileResponse(REPO / "demo/index.html")


def _png(a: np.ndarray) -> str:
    from PIL import Image
    buf = io.BytesIO(); Image.fromarray(a).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.post("/caption")
async def caption(clip: UploadFile = File(...), model: str = Form("grid"),
                  x_demo_code: str = Header("")) -> dict:
    if CODE and x_demo_code != CODE:
        raise HTTPException(401, "wrong access code")
    data = await clip.read()
    if len(data) > 30_000_000:
        raise HTTPException(413, "clip too large (max ~10 s)")
    suffix = ".mp4" if "mp4" in (clip.content_type or "") else ".webm"
    with tempfile.TemporaryDirectory(prefix="lipscribe_demo_") as tmp:
        path = Path(tmp) / f"clip{suffix}"
        path.write_bytes(data)
        try:
            with _lock:
                t = engine(model).transcribe(video_path=path)
        except FriendlyError as e:
            raise HTTPException(422, str(e))
    strip = mouth_strip(t.crops, 8) if t.crops is not None else None
    lips = float(1 - np.mean(t.gate)) if t.gate is not None and np.size(t.gate) else None
    return {"caption": t.caption, "seconds": round(t.seconds, 2), "model": engine(model).name,
            "face_fraction": t.face_fraction, "lips_trust": lips, "warnings": t.warnings,
            "mouth_strip": _png(strip) if strip is not None else None}
