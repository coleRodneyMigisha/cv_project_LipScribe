"""The real-world model for the demo (F-05): ws_v5 (ears + lips) beside its audio-only twin twin_v5.

The inputs are made exactly as the Flix v5 clips were made, so the demo sees what the test sets saw:

* frames at 25 fps, downscaled to at most 640 px wide (INTER_AREA), as ``scripts.flix_recrop``;
* MediaPipe face landmarks (IMAGE mode, up to 2 faces), the face tracked by its mouth position
  (``avsr_crop.pick_face``), gaps interpolated, then the Auto-AVSR warp to 96x96 grey crops;
* 16 kHz mono audio covering the same window, RMS-normalised as ``GridDataset`` does;
* the decoder chosen on validation in F-04 (``reports/flix_v5/final_decoder.md``):
  beam search + the LibriSpeech/own-transcript 4-gram.

``face_check`` applies the rules ``scripts.mine_flix`` used to keep a training clip (one frontal
face, large enough, found in >= 90 % of frames, no camera cut, lips moving). The model only ever
learned from such clips; on other footage (reaction shots, two people, profiles) the lips can
belong to someone who is not speaking, and ws_v5 is then far worse than ears alone
(F-04: 103 % WER on replaced audio). Callers use ``lips_ok`` to route those segments to twin_v5.
"""
from __future__ import annotations

import random
import threading
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from lipscribe.data.dataset import collate
from lipscribe.preprocessing import avsr_crop as ac
from lipscribe.preprocessing.features import rms_normalise

SR, FPS, SPF = 16000, 25, 640
MAX_WIDTH = 640
REPO = Path(__file__).resolve().parents[2]
WS_V5 = "runs/flix_v5/flix_v5_fusion_gated_ws_20261116-093641/best.pt"
TWIN_V5 = "runs/flix_v5/flix_v5_fusion_wp_audio_20261116-000004/best.pt"
LM = "lm/flix_libri_4gram.arpa"
# F-04, chosen on film_val + ted_val only
DECODE = {"ws_v5": {"alpha": 0.2, "beta": 1.0, "width": 32},
          "twin_v5": {"alpha": 0.2, "beta": 0.5, "width": 64}}
MOUTH_POS = [ac._POS[k] for k in ac.MOUTH]


# ------------------------------------------------------------------ faces

def downscale(rgb: np.ndarray) -> np.ndarray:
    """At most 640 px wide, as the Flix crops were made."""
    import cv2
    h0, w0 = rgb.shape[:2]
    if w0 <= MAX_WIDTH:
        return rgb
    sc = MAX_WIDTH / w0
    return cv2.resize(rgb, (int(w0 * sc), int(h0 * sc)), interpolation=cv2.INTER_AREA)


class Landmarker:
    """MediaPipe FaceLandmarker, one instance per thread (IMAGE mode, up to 2 faces, as mine_flix)."""

    def __init__(self, model_path: str | Path = "models/face_landmarker.task"):
        self.model_path = str(Path(model_path).resolve())
        self._local = threading.local()

    def _get(self):
        if not hasattr(self._local, "lm"):
            import mediapipe as mp_
            from mediapipe.tasks.python import BaseOptions, vision
            opts = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=vision.RunningMode.IMAGE, num_faces=2,
                min_face_detection_confidence=0.5, min_face_presence_confidence=0.5)
            self._local.lm = (mp_, vision.FaceLandmarker.create_from_options(opts))
        return self._local.lm

    def faces(self, rgb: np.ndarray) -> list:
        mp_, lm = self._get()
        res = lm.detect(mp_.Image(image_format=mp_.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        return res.face_landmarks or []


def _frontal_ok(f, w: int) -> tuple[bool, float]:
    """scripts.mine_flix._frontal_ok: nose between the cheeks (ratio 0.3-0.7), and the face width."""
    xs = np.array([p.x for p in f]) * w
    nose, left, right = f[1].x * w, f[234].x * w, f[454].x * w
    lo, hi = min(left, right), max(left, right)
    if hi - lo < 1:
        return False, float(xs.max() - xs.min())
    return 0.3 <= (nose - lo) / (hi - lo) <= 0.7, float(xs.max() - xs.min())


def _aperture(f) -> float:
    return abs(f[13].y - f[14].y) / (abs(f[152].y - f[10].y) + 1e-6)


@dataclass
class FaceTrack:
    crops: np.ndarray                  # uint8 [T, 96, 96]
    valid: np.ndarray                  # bool [T]: a face was found in that frame
    lips_ok: bool                      # passes the training-clip rules (see module docstring)
    reason: str                        # why not, for the log ("ok" otherwise)
    mouth_box: list | None = None      # last frame's mouth [x, y, w, h], fractions of the frame
    face_frac: float = 0.0


def face_check(rgbs: list[np.ndarray], faces: list[list], strict: bool = True) -> tuple[bool, str]:
    """The rules scripts.mine_flix.mine_cue used to keep a clip. strict=False: only 'a face is there'."""
    import cv2
    n = len(rgbs)
    found = sum(1 for f in faces if f) / max(1, n)
    if not strict:
        return found >= 0.5, "ok" if found >= 0.5 else "face lost"
    w = rgbs[0].shape[1]
    small = [cv2.resize(cv2.cvtColor(r, cv2.COLOR_RGB2GRAY), (64, 36)).astype(np.float32) for r in rgbs]
    if n > 1 and max(float(np.mean(np.abs(x - y))) for x, y in zip(small, small[1:])) > 25:
        return False, "camera cut"
    mid = faces[n // 2]
    if len(mid) != 1:
        return False, f"faces={len(mid)}"
    ok, width = _frontal_ok(mid[0], w)
    if not ok or width < 0.12 * w:
        return False, "not frontal" if not ok else "face too small"
    multi = sum(1 for f in faces if len(f) > 1)
    aps = [_aperture(f[0]) for f in faces if f]
    if found < 0.9:
        return False, "face lost"
    if multi > 0.1 * n:
        return False, "2nd face"
    if len(aps) < 5 or np.std(aps) < 0.01:
        return False, "lips still"
    return True, "ok"


def track_face(rgbs: list[np.ndarray], lm: Landmarker, strict: bool = True, faces: list | None = None) -> FaceTrack:
    """Frames (RGB, <= 640 wide, 25 fps) -> Auto-AVSR crops + the routing decision.

    faces: landmarks per frame if already known (the live demo caches them between ticks).
    """
    n = len(rgbs)
    h, w = rgbs[0].shape[:2]
    if faces is None:
        faces = [lm.faces(r) for r in rgbs]
    picked, prev = [], None
    for fs in faces:
        p = ac.pick_face(fs, w, h, prev)
        picked.append(p)
        prev = p if p is not None else prev
    valid = np.array([p is not None for p in picked])
    full = ac.interpolate(picked)
    if full is None:
        return FaceTrack(np.zeros((n, 96, 96), np.uint8), valid, False, "no face", None, 0.0)
    crops, _ = ac.crop_sequence(rgbs, full)
    ok, reason = face_check(rgbs, faces, strict)
    last = next((p for p in reversed(picked) if p is not None), None)
    box = None
    if last is not None:
        m = last[MOUTH_POS]
        x0, y0 = m.min(0); x1, y1 = m.max(0)
        mw, mh = x1 - x0, y1 - y0
        x0, y0, mw, mh = x0 - 0.25 * mw, y0 - 0.35 * mh, 1.5 * mw, 1.7 * mh
        box = [round(float(x0 / w), 4), round(float(y0 / h), 4), round(float(mw / w), 4), round(float(mh / h), 4)]
    return FaceTrack(crops.astype(np.uint8), valid, ok, reason, box, float(valid.mean()))


# ------------------------------------------------------------------ models

def load_flix_model(path: str | Path, device: torch.device):
    from lipscribe.models import build_model
    ck = torch.load(Path(path), map_location="cpu", weights_only=False)
    cfg = dict(ck["config"]["model"])
    for k in ("init_heads_from", "init_model_from", "init_streams_from", "init_from"):
        cfg.pop(k, None)
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


def _labels() -> list[str]:
    """Word-piece labels for pyctcdecode: blank = '', <eos> (last column) dropped (as flix_v4_beam)."""
    from lipscribe.data import subword
    tl = subword._tables()[1]
    return [""] + [t.lower() for t in tl[1:-1]]


@dataclass
class Segment:
    """One stretch of speech: n frames of crops and exactly n * 640 audio samples."""
    wave: np.ndarray                   # float32 [n * 640], 16 kHz, before RMS normalisation
    crops: np.ndarray                  # uint8 [n, 96, 96]
    valid: np.ndarray                  # bool [n]
    snr_db: float | None = None        # add MUSAN test noise at this SNR (None: as recorded)
    key: str = ""                      # seeds the noise draw
    extra: dict = field(default_factory=dict)


class RealWorld:
    """twin_v5 and ws_v5, loaded once; ``caption`` runs a batch of segments through both."""

    def __init__(self, root: str | Path = REPO, device: str | None = None, lm: str | None = LM,
                 noise_splits: str | Path | None = "Data/Noise_splits/splits.csv"):
        root = Path(root)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ws = load_flix_model(root / WS_V5, self.device)
        self.twin = load_flix_model(root / TWIN_V5, self.device)
        self.noise_splits = root / noise_splits if noise_splits else None
        self._pool = None
        self._dec = None
        self._dec_lock = threading.Lock()
        if lm:
            from pyctcdecode import build_ctcdecoder
            self._dec = build_ctcdecoder(_labels(), kenlm_model_path=str(root / lm), alpha=0.2, beta=1.0)

    # -------------------------------------------------------------- noise

    def _noise(self):
        if self._pool is None:
            from lipscribe.data.dataset import read_noise_pool
            from lipscribe.preprocessing.noise import NoisePool
            files = [f for f in read_noise_pool(self.noise_splits, "test") if Path(f).exists()]
            if not files:
                raise RuntimeError(f"no MUSAN test noise files (looked in {self.noise_splits})")
            self._pool = NoisePool(files)
        return self._pool

    def prepare_wave(self, seg: Segment) -> np.ndarray:
        wave = np.asarray(seg.wave, dtype=np.float32)
        if seg.snr_db is not None and np.isfinite(seg.snr_db):
            rng = random.Random(zlib.crc32(f"{seg.key}|{seg.snr_db:g}".encode()))
            wave = np.asarray(self._noise().mix(wave, float(seg.snr_db), rng)[0], dtype=np.float32)
        return rms_normalise(wave)

    # -------------------------------------------------------------- decoding

    def decode(self, name: str, lp: np.ndarray, greedy: bool = False) -> str:
        """lp: [T, V] log-probabilities of one segment."""
        if greedy or self._dec is None:
            from lipscribe.decoding import greedy_decode
            t = torch.from_numpy(lp)[:, None]
            return greedy_decode(t, torch.tensor([lp.shape[0]]))[0]
        p = DECODE[name]
        with self._dec_lock:
            self._dec.reset_params(alpha=p["alpha"], beta=p["beta"])
            text = self._dec.decode(np.ascontiguousarray(lp[:, :-1], dtype=np.float32), beam_width=p["width"])
        return " ".join(text.split())

    # -------------------------------------------------------------- models

    @torch.no_grad()
    def forward(self, segs: list[Segment]) -> list[dict]:
        """Both models on a batch of segments -> per segment: log-probs of each and ws_v5's gate."""
        items = [{"utt_id": s.key or f"seg{i}", "speaker": "demo", "video": s.crops, "video_valid": s.valid,
                  "audio": self.prepare_wave(s), "snr_db": float("inf"), "text": "",
                  "target": np.zeros(0, dtype=np.int64)} for i, s in enumerate(segs)]
        batch = collate(items)
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = []
        with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
            o_ws = self.ws(batch)
            o_tw = self.twin(batch)
        for i in range(len(segs)):
            nw, nt = int(o_ws["lengths"][i]), int(o_tw["lengths"][i])
            out.append({"ws": o_ws["log_probs"][:nw, i].float().cpu().numpy(),
                        "twin": o_tw["log_probs"][:nt, i].float().cpu().numpy(),
                        "gate_audio": o_ws["gate"][i, :nw].float().cpu().numpy()})
        return out

    def caption(self, segs: list[Segment], greedy: bool = False) -> list[dict]:
        """-> per segment {ears, lips, gate (the lips' share per frame, 0-1)}."""
        res = []
        for r in self.forward(segs):
            res.append({"ears": self.decode("twin_v5", r["twin"], greedy),
                        "lips": self.decode("ws_v5", r["ws"], greedy),
                        "gate": np.clip(1.0 - r["gate_audio"], 0.0, 1.0)})
        return res
