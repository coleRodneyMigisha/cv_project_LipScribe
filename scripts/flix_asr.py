"""X6 'Flix': Whisper pseudo-labels for films without subtitles (run on the Alienware).

For every <audio-dir>/<key>.ogg|.flac (16 kHz mono): Whisper (large-v3-turbo, English,
word timestamps) -> <out>/<key>.json (raw words) and <out>/<key>.srt with short cues built
from the word timings: a new cue starts at a pause > --gap s or when a cue would pass 3 s;
cues shorter than 0.8 s or with mean word probability < --min-prob are dropped (music,
mumbling, hallucinations). The .srt files feed scripts/mine_flix.py --extra-titles.

Whisper lives in ./asr_pkgs (pip --target, so the shared main .venv is untouched):
  PYTHONPATH=asr_pkgs .venv/bin/python -m scripts.flix_asr --audio-dir asr_audio --out asr_srt
Audio is passed as an array, so Whisper's ffmpeg loader is never used.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def fmt(t: float) -> str:
    h, r = divmod(t, 3600); m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}"


def cues_from_words(words: list[dict], gap: float, max_len: float, min_len: float, min_prob: float):
    cues, cur = [], []

    def flush():
        if cur:
            a, b = cur[0]["start"], cur[-1]["end"]
            p = float(np.mean([w["probability"] for w in cur]))
            if min_len <= b - a <= max_len and p >= min_prob:
                cues.append((a, b, " ".join(w["word"].strip() for w in cur), p))
        cur.clear()

    for w in words:
        if cur and (w["start"] - cur[-1]["end"] > gap or w["end"] - cur[0]["start"] > max_len):
            flush()
        cur.append(w)
    flush()
    return cues


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", type=Path, default=Path("asr_audio"))
    ap.add_argument("--out", type=Path, default=Path("asr_srt"))
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--gap", type=float, default=0.35)
    ap.add_argument("--min-prob", type=float, default=0.5)
    a = ap.parse_args()
    import whisper
    a.out.mkdir(parents=True, exist_ok=True)
    model = whisper.load_model(a.model, device="cuda")
    files = sorted(p for p in a.audio_dir.iterdir() if p.suffix in (".ogg", ".flac"))
    summary = {}
    for f in files:
        js = a.out / f"{f.stem}.json"
        if not js.exists():
            t0 = time.time()
            audio, sr = sf.read(str(f), dtype="float32", always_2d=False)
            if audio.ndim == 2:
                audio = audio.mean(1)
            assert sr == 16000, f"{f}: {sr} Hz"
            res = whisper.transcribe(model, audio, language="en", word_timestamps=True,
                                     condition_on_previous_text=False, verbose=None)
            words = [{"word": w["word"], "start": float(w["start"]), "end": float(w["end"]),
                      "probability": float(w["probability"])} for s in res["segments"] for w in s.get("words", [])]
            js.write_text(json.dumps({"duration_s": len(audio) / sr, "words": words}))
            print(f"{f.stem}: {len(words)} words in {time.time() - t0:.0f}s", flush=True)
        words = json.loads(js.read_text())["words"]
        cues = cues_from_words(words, a.gap, 3.0, 0.8, a.min_prob)
        with open(a.out / f"{f.stem}.srt", "w") as fh:
            for i, (s, e, t, _) in enumerate(cues, 1):
                fh.write(f"{i}\n{fmt(s)} --> {fmt(e)}\n{t}\n\n")
        summary[f.stem] = {"words": len(words), "cues": len(cues),
                           "cue_minutes": round(sum(e - s for s, e, _, _ in cues) / 60, 1)}
        print(f"{f.stem}: {summary[f.stem]}", flush=True)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
