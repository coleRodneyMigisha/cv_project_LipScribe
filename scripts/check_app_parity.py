"""W3-10 "Done when": the demo gives the same caption as evaluate.py for GRID test clips.

For each of N clean test utterances it captions the clip twice:
  evaluate path: GridDataset("test", clean) item with the cached crops -> collate -> model -> decoder
  app path:      LipScribe.transcribe(video, 16 kHz audio, speech start) with live Face Mesh crops
and prints both. Exit code 0 only if every caption matches.

    python -m scripts.check_app_parity --checkpoint runs/b1_audio_only_20261103-193743/best.pt
    python -m scripts.check_app_parity --checkpoint <best.pt> --n 20 --decode beam
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lipscribe.app.inference import LipScribe


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--n", type=int, default=10, help="number of test clips")
    p.add_argument("--decode", choices=["beam", "greedy", "grammar"], default="beam")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    from lipscribe.data.dataset import GridDataset

    engine = LipScribe(args.checkpoint, device=args.device, decode=args.decode)
    d = engine.cfg["data"]
    test = GridDataset(d["manifest"], d["splits"], "test", snr_db=float("inf"),
                       noise_splits=d.get("noise_splits"))
    step = max(1, len(test) // args.n)
    print(f"{engine.name}: comparing {args.n} of {len(test)} clean test clips\n")

    mismatches = 0
    for k in range(args.n):
        idx = k * step
        row, item = test.rows[idx], test[idx]
        evaluate_caption = engine.caption_items([item])[0][0]          # evaluate.py's path
        app = engine.transcribe(row["video_path"], row["audio_path"],    # the demo's path
                                speech_start_s=row.get("speech_start_s"), n_frames=int(row["n_frames"]))
        same = app.caption == evaluate_caption.strip()
        mismatches += not same
        print(f"{'OK  ' if same else 'DIFF'} {row['utt_id']:<14} ref: {row['transcript']}")
        print(f"     evaluate: {evaluate_caption}")
        print(f"     app:      {app.caption}  (face {app.face_fraction:.0%}, {app.seconds:.1f} s)\n")

    print(f"{args.n - mismatches}/{args.n} captions identical")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
