"""Live capture check (W4-10): show the current 3 s window from the camera and microphone.

    python -m scripts.live_capture --seconds 20      # q closes the window earlier

No model runs here; this checks capture, timing and mouth tracking only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lipscribe.app.live import (DEFAULT_FACE_MODEL, AudioCapture, AudioRingBuffer, CameraThread,
                                CausalLipTracker, FrameRingBuffer, LiveError, extract_window,
                                make_face_detector, run_display)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=None, help="stop after this long (default: until q)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--face-model", type=Path, default=DEFAULT_FACE_MODEL)
    p.add_argument("--alpha", type=float, default=0.5, help="EMA weight of the newest lip box")
    p.add_argument("--no-audio", action="store_true")
    args = p.parse_args(argv)

    frames, audio = FrameRingBuffer(), AudioRingBuffer()
    try:
        tracker = CausalLipTracker(make_face_detector(args.face_model), alpha=args.alpha)
    except LiveError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    camera = CameraThread(frames, tracker, index=args.camera)
    mic = AudioCapture(audio)
    try:
        camera.start()
    except LiveError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    mic_on = False
    if not args.no_audio:
        try:
            mic.start()
            mic_on = True
        except LiveError as err:
            print(f"Warning: {err} Carrying on with video only.", file=sys.stderr)
    try:
        stats = run_display(camera, frames, audio, seconds=args.seconds)
    finally:
        camera.stop()
        if mic_on:
            mic.stop()
        import cv2
        cv2.destroyAllWindows()

    newest = frames.latest_time()
    item = extract_window(frames, audio, newest) if newest is not None else None
    print(f"Ran for {stats.elapsed:.1f} s; {camera.frames} frames captured, "
          f"{camera.dropped} dropped, {camera.failed_reads} failed reads.")
    if item is not None:
        print(f"Last window: {int(item['video_valid'].sum())}/{len(item['video_valid'])} frames with a face, "
              f"{item['audio_real_samples']}/{len(item['audio'])} audio samples.")
    if mic_on:
        print(f"Microphone overflows: {mic.status_flags}.")
    if camera.error:
        print(f"Error: {camera.error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
