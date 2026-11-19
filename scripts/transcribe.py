"""Caption one clip from the command line (W3-10).

    python -m scripts.transcribe --checkpoint runs/<run>/best.pt --video clip.mp4
    python -m scripts.transcribe --checkpoint runs/<run>/best.pt --video clip.mpg --audio clip.wav
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lipscribe.app.inference import FriendlyError, LipScribe


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--video", type=Path)
    p.add_argument("--audio", type=Path)
    p.add_argument("--decode", choices=["beam", "greedy", "grammar"], default="beam")
    p.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    p.add_argument("--snr", type=float, default=None,
                   help="mix MUSAN test-pool noise in at this SNR in dB (omitted = clean)")
    p.add_argument("--noise-splits", type=Path, default=None,
                   help="noise split CSV (default Data/Noise_splits/splits.csv)")
    args = p.parse_args(argv)
    try:
        engine = LipScribe(args.checkpoint, device=args.device, decode=args.decode)
        result = engine.transcribe(args.video, args.audio, snr_db=args.snr,
                                   noise_splits=args.noise_splits)
    except FriendlyError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    print(result.caption)
    for w in result.warnings:
        print(f"warning: {w}", file=sys.stderr)
    t = result.timings
    print(f"timing: preprocess {t['preprocess_s']:.2f} s, model {t['model_s']:.2f} s, "
          f"decode {t['decode_s']:.2f} s (total {result.seconds:.2f} s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
