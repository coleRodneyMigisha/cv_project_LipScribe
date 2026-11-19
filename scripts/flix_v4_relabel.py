"""Flix v4: relabel every training / TED clip with Whisper from the clip's OWN audio.

Why: TED subtitle cues are not aligned with the speech inside our clips (Whisper on a
ted_val clip's audio disagrees with its cue text far more than on film clips, see
reports/flix_v4/decision_log.md). The clip's audio and mouth crops are cut from the same
window, so they are in sync with each other; only the text was wrong. Relabelling the
clip from its own audio fixes that without downloading anything again.

Relabelled: data/flix_v3/{train_mix, ted_val, ted_test}.csv -> data/flix_v4/ (same columns).
Copied unchanged: film_val.csv, flix_test.csv, grid_test.csv, splits.csv (film references stay).
A clip is dropped if Whisper hears no speech, the text has digits/symbols our 28-symbol
tokenizer cannot spell, looks like a hallucination, or does not fit the CTC length limit.
Resumable through data/flix_v4/whisper_cache.jsonl.

  PYTHONPATH=asr_pkgs:. .venv/bin/python -m scripts.flix_v4_relabel
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from lipscribe.data.dataset import ctc_min_frames
from lipscribe.data.tokenizer import encode

SRC, OUT = Path("data/flix_v3"), Path("data/flix_v4")
RELABEL = ["train_mix", "ted_val", "ted_test"]
COPY = ["film_val", "flix_test", "grid_test", "splits"]


def norm(text: str) -> str | None:
    t = text.lower().replace("’", "'").replace("'", "")
    t = re.sub(r"[-—–/]", " ", t)
    t = re.sub(r"[.,!?;:\"()\[\]“”…]", " ", t)
    if re.search(r"[^a-z ]", t):          # digits, %, $, & ... cannot be spelled by the tokenizer
        return None
    t = " ".join(t.split())
    return t or None


def main() -> None:
    import torch
    import whisper

    OUT.mkdir(parents=True, exist_ok=True)
    for name in COPY:
        shutil.copy2(SRC / f"{name}.csv", OUT / f"{name}.csv")
    cache_path = OUT / "whisper_cache.jsonl"
    cache = {}
    if cache_path.exists():
        for ln in cache_path.read_text().splitlines():
            if ln.strip():
                d = json.loads(ln); cache[d["utt_id"]] = d
    model = whisper.load_model("large-v3-turbo", device="cuda")
    stats = {}
    with open(cache_path, "a") as cf:
        for name in RELABEL:
            rows = list(csv.DictReader(open(SRC / f"{name}.csv")))
            keep, drop = [], {"no_speech": 0, "unspellable": 0, "hallucination": 0, "ctc_len": 0}
            for i, r in enumerate(rows):
                u = r["utt_id"]
                if u not in cache:
                    a, sr = sf.read(r["audio_path"], dtype="float32")
                    if a.ndim > 1:
                        a = a.mean(1)
                    res = model.transcribe(a, language="en", fp16=True, temperature=0.0,
                                           condition_on_previous_text=False, without_timestamps=True)
                    segs = res.get("segments") or [{}]
                    d = {"utt_id": u, "text": res["text"].strip(),
                         "no_speech": max(s.get("no_speech_prob", 0.0) for s in segs),
                         "compression": max(s.get("compression_ratio", 0.0) for s in segs),
                         "avg_logprob": min(s.get("avg_logprob", 0.0) for s in segs),
                         "n_frames": int(np.load(r["mouths_path"])["valid"].shape[0])}
                    cache[u] = d
                    cf.write(json.dumps(d) + "\n")
                    if i % 500 == 0:
                        cf.flush(); print(f"{name}: {i}/{len(rows)}", flush=True)
                d = cache[u]
                t = norm(d["text"])
                if d["no_speech"] > 0.6 or not d["text"]:
                    drop["no_speech"] += 1; continue
                if t is None:
                    drop["unspellable"] += 1; continue
                if d["compression"] > 2.4 or d["avg_logprob"] < -1.0 or len(t) > 22 * d["n_frames"] / 25 + 10:
                    drop["hallucination"] += 1; continue
                if ctc_min_frames(encode(t)) > d["n_frames"]:
                    drop["ctc_len"] += 1; continue
                keep.append({**r, "transcript": t})
            with open(OUT / f"{name}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(keep)
            from lipscribe.evaluation import wer as _wer
            old = {r["utt_id"]: r["transcript"] for r in rows}
            stats[name] = {"before": len(rows), "kept": len(keep), "dropped": drop,
                           "old_label_wer_vs_whisper": _wer([old[k["utt_id"]] for k in keep],
                                                            [k["transcript"] for k in keep]) if keep else None}
            print(name, stats[name], flush=True)
    (OUT / "relabel_stats.json").write_text(json.dumps(stats, indent=2))
    del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
