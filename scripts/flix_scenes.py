"""X6 'Flix': pick test scenes and grab a still of each, for the presentation (run on the laptop).

For each chosen Flix *test* clip it saves one frame from the middle of the line
(<out>/scene_<n>.jpg) and writes <out>/scenes.json with the real subtitle and every
model's caption, so the slide can set the text as live text.

Choice (honest): the --best clips where the LAST --hyps model has the lowest CER,
plus --typical clips around its median CER. Each scene records which group it is in.

Usage: python -m scripts.flix_scenes --manifest ~/flix_mine/manifest_local.csv \
         --hyps reports/flix/base_flix_hyps.csv:"Model P as is" \
         --hyps reports/flix_sweep/sweep_s1_flix_hyps.csv:"Retrained on films" --out scenes/
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import scripts.mine_flix as mf


def read_hyps(path: str) -> dict[str, dict]:
    with open(path, newline="") as fh:
        return {r["utt_id"]: r for r in csv.DictReader(fh) if r["snr"] == "clean"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--hyps", action="append", required=True, help="CSV:label (last one picks the scenes)")
    ap.add_argument("--best", type=int, default=2)
    ap.add_argument("--typical", type=int, default=1)
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("scenes"))
    ap.add_argument("--videos", type=Path, default=mf.V)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    mf.V = a.videos
    videos = {key: Path(v) for key, v, _ in mf._titles()}
    with open(a.manifest, newline="") as fh:
        meta = {f"flix_{r['utt_id']}": r for r in csv.DictReader(fh)}
    models = [(h.rsplit(":", 1)[1], read_hyps(h.rsplit(":", 1)[0])) for h in a.hyps]
    ranker = models[-1][1]
    ids = [u for u in ranker if u in meta and len(ranker[u]["ref"].split()) >= a.min_words
           and all(u in m for _, m in models)]
    ids.sort(key=lambda u: float(ranker[u]["cer"]))
    mid = len(ids) // 2
    chosen = [(u, "best") for u in ids[: a.best]]
    chosen += [(u, "typical") for u in ids[mid: mid + a.typical] if u not in dict(chosen)]
    scenes = []
    for n, (u, group) in enumerate(chosen, 1):
        r = meta[u]
        t = float(r["start_s"]) + float(r["dur_s"]) / 2
        jpg = a.out / f"scene_{n}.jpg"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(videos[r["title"]]),
                        "-frames:v", "1", "-vf", "scale=960:-2", "-q:v", "3", str(jpg)], check=True)
        scenes.append({"image": jpg.name, "group": group, "title": r["speaker"], "time_s": round(t, 1),
                       "subtitle": ranker[u]["ref"],
                       "captions": [{"model": lab, "text": m[u]["hyp"], "wer": float(m[u]["wer"]),
                                     "cer": float(m[u]["cer"])} for lab, m in models]})
        print(f"[{group}] {r['speaker']} @ {t:.1f}s  ref: {ranker[u]['ref']}")
        for lab, m in models:
            print(f"     {lab}: {m[u]['hyp']!r}  (CER {float(m[u]['cer']):.2f})")
    (a.out / "scenes.json").write_text(json.dumps({"n_test_clips": len(ids), "scenes": scenes}, indent=2))


if __name__ == "__main__":
    main()
