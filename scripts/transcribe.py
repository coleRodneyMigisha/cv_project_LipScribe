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
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    try:
        engine = LipScribe(args.checkpoint, device=args.device, decode=args.decode)
        result = engine.transcribe(args.video, args.audio)
    except FriendlyError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    print(result.caption)
    for w in result.warnings:
        print(f"warning: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
