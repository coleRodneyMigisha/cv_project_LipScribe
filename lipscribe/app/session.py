"""One demo recording or upload, from bytes to caption events (F-05). Used by demo/server.py.

Runs (the page's ``run``):
  live    MediaRecorder chunks while the person talks; every finished stretch of speech is captioned
          once, the stretch still going on is sent as ``pending``.
  clip    a 3 s camera recording, captioned as one piece when it stops.
  upload  a file of any length (up to 3 h), e.g. a film. Speech is cut into <= 4 s pieces and the
          captions are sent in batches as subtitle cues, while the file is still uploading.

Models (the page's ``mode``):
  any     ws_v5 (ears + lips) and twin_v5 (ears only), lipscribe.app.realworld.
  grid    P4 + GRID grammar (ears + lips) and B1 + grammar (ears only), lipscribe.app.inference.

For uploads every cue says whether the lips may be used (``lips_used``): only when the segment
passes the training-clip rules (realworld.face_check). The page's "routed" track shows ws_v5 for
those cues and twin_v5 for the rest.
"""
from __future__ import annotations

import base64
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

from lipscribe.app import realworld as rwm
from lipscribe.app.stream import (
    FPS,
    PAD_S,
    SPF,
    SR,
    VAD,
    GrowingFile,
    LiveStream,
    MediaReader,
    Span,
    Timeline,
    noise_estimate_db,
    window,
)

GRID_MAX_S = 10.0
CLIP_MIN_S = 0.6
CLIP_MIN_FACE = 0.5      # a camera clip with a face in fewer frames gets the no_face error


class SessionError(Exception):
    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code, self.message = code, message


def jpeg_url(img: np.ndarray, q: int = 80) -> str:
    import cv2
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def thumbs(crops: np.ndarray, t0: float, k: int) -> list[dict]:
    if crops is None or not len(crops):
        return []
    idx = np.linspace(0, len(crops) - 1, num=min(k, len(crops))).round().astype(int)
    return [{"t": round(t0 + i / FPS, 2), "src": jpeg_url(crops[i])} for i in idx]


class Engines:
    """Models shared by all sessions; the GPU lock lets one batch run at a time."""

    def __init__(self, root: Path, p4: Path, b1: Path, face_model: Path, load_grid: bool = True,
                 load_any: bool = True):
        self.root = root
        self.gpu = threading.Lock()
        self.landmarker = rwm.Landmarker(face_model)
        self.vad = VAD()
        self.rw = rwm.RealWorld(root) if load_any else None
        self.p4 = self.b1 = None
        if load_grid:
            from lipscribe.app.inference import LipScribe
            self.p4 = LipScribe(p4, decode="grammar", face_model=face_model)
            self.b1 = LipScribe(b1, decode="grammar", face_model=face_model)
        self.pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="segments")
        self.lm_pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="landmarks")

    def names(self, mode: str) -> tuple[str, str]:
        return ("twin_v5", "ws_v5") if mode == "any" else ("B1 + grammar", "P4 + grammar")


class Session:
    """Feed bytes with ``feed``/``feed_at``; events go to ``emit(dict)`` from a worker thread."""

    def __init__(self, engines: Engines, run: str, mode: str, emit: Callable[[dict], None],
                 snr: float | None = None, size: int | None = None, mime: str = "", name: str = "",
                 sid: str = "s"):
        if run not in ("live", "clip", "upload"):
            raise SessionError("bad_request", f"unknown run {run!r}")
        if mode not in ("any", "grid"):
            raise SessionError("bad_request", f"unknown mode {mode!r}")
        if mode == "any" and engines.rw is None or mode == "grid" and engines.p4 is None:
            raise SessionError("model_unavailable", f"the {mode} model is not loaded")
        self.E, self.run, self.mode, self.emit, self.snr, self.sid = engines, run, mode, emit, snr, sid
        self.src = GrowingFile(size) if run == "upload" else LiveStream()
        fmt = None
        if run != "upload":
            fmt = "mp4" if "mp4" in (mime or "") else "webm"
        self.tl = Timeline()
        self.reader = MediaReader(self.src, self.tl, fmt=fmt, ahead_s=25.0 if run == "upload" else 1e9)
        self.thread = threading.Thread(target=self._main, daemon=True)
        self.stopped_at: float | None = None
        self.started = time.time()
        self.cancelled = False
        self.cues: list[dict] = []
        self._faces: dict[int, list] = {}          # frame index -> landmarks (computed once)
        self._flock = threading.Lock()

    # -------------------------------------------------------------- input
    def start(self) -> None:
        self.reader.start()
        self.thread.start()

    def feed(self, data: bytes) -> None:                # live / clip: chunks in order
        self.src.append(data)

    def feed_at(self, offset: int, data: bytes) -> None:  # upload: any order
        self.src.write_at(offset, data)

    def stop(self) -> None:                             # live / clip: no more chunks
        self.stopped_at = time.time()
        self.src.finish()

    def cancel(self) -> None:
        self.cancelled = True
        self.src.abort()
        with self.tl.cond:
            self.tl.consumed_s = 1e12
            self.tl.cond.notify_all()

    # -------------------------------------------------------------- main
    def _main(self) -> None:
        try:
            if self.run == "clip" or (self.run == "upload" and self.mode == "grid"):
                self._whole()
            else:
                self._stream()
        except SessionError as e:
            self.emit({"type": "error", "code": e.code, "message": e.message})
        except Exception as e:                           # noqa: BLE001
            if not self.cancelled:
                self.emit({"type": "error", "code": "server_error", "message": f"{type(e).__name__}: {e}"[:300]})

    def _check_media(self) -> None:
        tl = self.tl
        if tl.error and not self.cancelled:
            raise SessionError("bad_file", tl.error)
        if not tl.has_audio:
            raise SessionError("no_audio", "the file has no audio track")

    # -------------------------------------------------------------- one piece (clip, short GRID upload)
    def _whole(self) -> None:
        tl = self.tl
        while not tl.eof:
            tl.wait_until(1e12, timeout=0.5)
            if self.cancelled:
                return
        self._check_media()
        dur = tl.end_s
        if self.mode == "grid" and dur > GRID_MAX_S + 0.5:
            raise SessionError("grid_too_long", f"GRID sentences mode takes clips up to {GRID_MAX_S:.0f} s")
        if dur < CLIP_MIN_S:
            raise SessionError("clip_too_short", f"{dur:.2f} s")
        n = int(dur * FPS)
        wave = tl.get_audio(0.0, n * SPF)
        if float(np.abs(wave).max(initial=0.0)) < 1e-4:
            raise SessionError("no_audio", "the audio is silent")
        self.emit({"type": "progress", "step": "face"})
        span = Span(0.0, n / FPS, 0.0, n)
        self.emit({"type": "progress", "step": "listen"})
        cue = self._caption(span, strict=False)
        self.emit({"type": "progress", "step": "write"})
        if self.run == "clip" and tl.has_video and cue["face"] < CLIP_MIN_FACE:
            raise SessionError("no_face", f"face found in {cue['face']:.0%} of frames")
        speech = self.E.vad.speech(wave)
        noise = self.snr if self.snr is not None else noise_estimate_db(wave, speech)
        msg = {"type": "final", "t": round(dur, 2),
               "ears": {"text": cue["ears"], "pending": "", "model": self.E.names(self.mode)[0]},
               "lips": {"text": cue["lips"], "pending": "", "model": self.E.names(self.mode)[1]},
               "gate": cue["gate"], "crops": cue["crops"], "mouth_box": None,
               "processing_s": round(time.time() - (self.stopped_at or self.started), 2),
               "face_found": cue["face"]}
        if noise is not None:
            msg["noise_db"] = round(noise, 1)
        if self.run == "upload":
            self.emit({"type": "cues", "cues": [dict(cue, i=0)], "captioned_s": round(dur, 2),
                       "duration_s": round(dur, 2)})
        self.emit(msg)

    # -------------------------------------------------------------- streams (live, long upload)
    def _stream(self) -> None:
        tl, E = self.tl, self.E
        upload = self.run == "upload"
        W = 12.0 if upload else 0.0
        max_s, min_sil = (4.0, 250) if self.mode == "any" else (6.0, 500)
        cursor, last_partial, last_end = 0.0, 0.0, -1.0
        committed = {"ears": [], "lips": []}
        pend = {"ears": "", "lips": ""}
        face_w = face_n = lips_t = speech_t = 0.0
        sent_progress = 0.0
        while True:
            if self.cancelled:
                return
            if upload:
                tl.wait_until(cursor + W, timeout=1.0)
            else:
                tl.wait_until(last_end + 0.5, timeout=0.5)
            with tl.cond:
                end, eof = tl.end_s, tl.eof
            if tl.error and not eof:
                continue
            if eof:
                self._check_media()
            if not eof and (end < cursor + W if upload else end < last_end + 0.5):
                continue
            last_end = end
            a_t0 = cursor
            a = tl.get_audio(cursor, max(0, int((end - cursor) * SR)))
            sp = [(cursor + s, cursor + e) for s, e in E.vad.speech(a, max_s, min_sil)]
            ready = [x for x in sp if eof or x[1] + PAD_S + 0.6 <= end]
            rest = [x for x in sp if x not in ready]
            limit = tl.duration if eof else None
            spans = [window(s, e, limit if limit else end) for s, e in ready]
            spans = [s for s in spans if s.end - s.start >= 0.3]
            cues = self._caption_many(spans, strict=upload)
            for c in cues:
                c["i"] = len(self.cues)
                self.cues.append(c)
                committed["ears"].append(c["ears"]); committed["lips"].append(c["lips"])
                dur = c["end"] - c["start"]
                face_w += c["face"] * dur; face_n += dur; speech_t += dur
                lips_t += dur if c["lips_used"] else 0.0
            if ready:
                cursor = ready[-1][1]
            elif not sp:
                cursor = max(cursor, end - 0.5)
            tl.drop_before(max(0.0, cursor - PAD_S - 0.2))
            with self._flock:
                k_old = int((cursor - PAD_S - 0.2) * FPS)
                for k in [k for k in self._faces if k < k_old]:
                    del self._faces[k]
            noise = self.snr if self.snr is not None else noise_estimate_db(a, sp, a_t0)
            if upload:
                if cues:
                    self.emit({"type": "cues", "cues": cues, "captioned_s": round(cursor if not eof else (tl.duration or end), 2),
                               "duration_s": round(tl.duration or 0, 2)})
                elif time.time() - sent_progress > 2:
                    self.emit({"type": "progress", "captioned_s": round(cursor, 2), "duration_s": round(tl.duration or 0, 2)})
                    sent_progress = time.time()
            else:
                if not rest or eof or cues:                 # nothing open, or its words were just committed
                    pend = {"ears": "", "lips": ""}
                if rest and not eof and time.time() - last_partial > 0.7:
                    s, e = rest[-1]
                    # provisional words: greedy (fast); the line is decoded with beam + LM once it closes
                    p = self._caption(window(s, min(e, end - 0.05), end), strict=False, thumbs_k=0, greedy=True)
                    pend = {"ears": p["ears"], "lips": p["lips"]}
                    last_partial = time.time()
                msg = {"type": "final" if eof else "partial", "t": round(end, 2),
                       "ears": {"text": " ".join(x for x in committed["ears"] if x), "pending": pend["ears"],
                                "model": E.names(self.mode)[0]},
                       "lips": {"text": " ".join(x for x in committed["lips"] if x), "pending": pend["lips"],
                                "model": E.names(self.mode)[1]},
                       "gate": [g for c in cues for g in c["gate"]],
                       "crops": [x for c in cues for x in c["crops"]][-8:],
                       "mouth_box": self._mouth_box()}
                if noise is not None:
                    msg["noise_db"] = round(noise, 1)
                if eof:
                    msg["processing_s"] = round(time.time() - (self.stopped_at or self.started), 2)
                    msg["face_found"] = round(face_w / face_n, 3) if face_n else 0.0
                    if not self.cues and not committed["ears"]:
                        raise SessionError("clip_too_short", "no speech was found")
                self.emit(msg)
            if eof:
                break
        if upload:
            self.emit({"type": "final", "processing_s": round(time.time() - self.started, 1),
                       "face_found": round(face_w / face_n, 3) if face_n else 0.0,
                       "lips_share": round(lips_t / speech_t, 3) if speech_t else 0.0,
                       "cues": len(self.cues), "duration_s": round(tl.duration or 0, 2)})

    def _mouth_box(self):
        tl = self.tl
        with tl.cond:
            f = tl.last_frame
        if f is None:
            return None
        faces = self.E.landmarker.faces(f)
        if not faces:
            return None
        from lipscribe.preprocessing import avsr_crop as ac
        h, w = f.shape[:2]
        m = ac.mesh_points(faces[0], w, h)[rwm.MOUTH_POS]
        x0, y0 = m.min(0); x1, y1 = m.max(0)
        mw, mh = x1 - x0, y1 - y0
        return [round(float((x0 - 0.25 * mw) / w), 4), round(float((y0 - 0.35 * mh) / h), 4),
                round(float(1.5 * mw / w), 4), round(float(1.7 * mh / h), 4)]

    # -------------------------------------------------------------- captioning
    def _inputs(self, span: Span, strict: bool) -> tuple[np.ndarray, rwm.FaceTrack | None, list | None]:
        tl = self.tl
        wave = tl.get_audio(span.w0, span.n * SPF)
        k0 = int(round(span.w0 * FPS))
        frames = tl.get_frames(k0, span.n) if tl.has_video else None
        if frames is None or self.mode == "grid":
            return wave, None, frames
        return wave, rwm.track_face(frames, self.E.landmarker, strict=strict, faces=self._landmarks(k0, frames)), frames

    def _landmarks(self, k0: int, frames: list) -> list:
        """Landmarks for frames k0.., from the cache where a previous tick already found them.

        New frames are spread over the face pool (MediaPipe releases the GIL while it runs).
        """
        with self._flock:
            todo = [i for i in range(len(frames)) if (k0 + i) not in self._faces]
        if todo:
            found = list(self.E.lm_pool.map(lambda i: self.E.landmarker.faces(frames[i]), todo))
            with self._flock:
                for i, f in zip(todo, found):
                    self._faces[k0 + i] = f
        with self._flock:
            return [self._faces[k0 + i] for i in range(len(frames))]

    def _caption_many(self, spans: list[Span], strict: bool) -> list[dict]:
        if not spans:
            return []
        prepared = list(self.E.pool.map(lambda s: self._inputs(s, strict), spans))
        return [self._caption(s, strict, prepared=p) for s, p in zip(spans, prepared)]

    def _caption(self, span: Span, strict: bool, prepared=None, thumbs_k: int = 4, greedy: bool = False) -> dict:
        wave, track, frames = prepared if prepared is not None else self._inputs(span, strict)
        key = f"{self.sid}:{span.w0:.2f}"
        if self.mode == "grid":
            return self._caption_grid(span, wave, frames, thumbs_k)
        if track is None:
            crops, valid, ok, reason, face = np.zeros((span.n, 96, 96), np.uint8), np.zeros(span.n, bool), False, "no video", 0.0
        else:
            crops, valid, ok, reason, face = track.crops, track.valid, track.lips_ok, track.reason, track.face_frac
        seg = rwm.Segment(wave, crops, valid, snr_db=self.snr, key=key)
        with self.E.gpu:
            r = self.E.rw.caption([seg], greedy=greedy)[0]
        n_lead = int(round((span.start - span.w0) * FPS))
        n_speech = max(1, int(round((span.end - span.start) * FPS)))
        gate = r["gate"][n_lead:n_lead + n_speech] if len(r["gate"]) > n_lead else r["gate"]
        return {"start": round(span.start, 2), "end": round(span.end, 2), "ears": r["ears"], "lips": r["lips"],
                "lips_used": bool(ok), "reason": reason, "face": round(float(face), 3),
                "gate": [round(float(g), 2) for g in gate],
                "crops": thumbs(crops[n_lead:n_lead + n_speech] if track is not None else None, span.start, thumbs_k)
                if thumbs_k else []}

    def _caption_grid(self, span: Span, wave: np.ndarray, frames: list | None, thumbs_k: int) -> dict:
        """P4 and B1 through lipscribe.app.inference (the evaluate.py chain), via a temporary clip."""
        import cv2
        import soundfile as sf
        E = self.E
        with tempfile.TemporaryDirectory(prefix="lipscribe_grid_") as tmp:
            wav = Path(tmp) / "a.wav"
            sf.write(wav, wave, SR, subtype="FLOAT")
            mp4 = None
            if frames is not None:
                mp4 = Path(tmp) / "v.mp4"
                h, w = frames[0].shape[:2]
                vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
                for f in frames:
                    vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                vw.release()
            with E.gpu:
                tp = E.p4.transcribe(video_path=mp4, audio_path=wav, snr_db=self.snr)
                tb = E.b1.transcribe(video_path=None, audio_path=wav, snr_db=self.snr)
        gate = tp.gate
        if gate is not None:
            gate = np.asarray(gate, dtype=np.float32)
            if gate.ndim > 1:
                gate = gate.mean(-1)
            gate = np.clip(1.0 - gate, 0, 1)
        face = tp.face_frac if hasattr(tp, "face_frac") else (tp.face_fraction or 0.0)
        return {"start": round(span.start, 2), "end": round(span.end, 2), "ears": tb.caption, "lips": tp.caption,
                "lips_used": bool(face >= 0.5), "reason": "ok" if face >= 0.5 else "face lost",
                "face": round(float(face), 3),
                "gate": [round(float(g), 2) for g in gate] if gate is not None else [],
                "crops": thumbs(tp.crops, span.start, thumbs_k) if thumbs_k and tp.crops is not None else []}

