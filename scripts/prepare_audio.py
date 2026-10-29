#!/usr/bin/env python3
"""Convert all GRID audio to 16 kHz, 16-bit, mono WAV (W2-04).

Resumable: skips utterances already converted. Parallel across CPU cores.
Run from the repo root:  python3 scripts/prepare_audio.py --workers 4
"""
import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf

sys.path.insert(0, ".")
from lipscribe.preprocessing.audio import load_audio, TARGET_SR

RAW_DIR = Path("/Users/apple/Desktop/cv_project_LipScribe/aligns")
OUT_DIR = Path("data/processed/audio16k")
LOG = Path("data/processed/audio16k/conversion_log.csv")
MAX_DUR_DIFF = 0.04  # 40 ms tolerance (one video frame), per plan


def convert_one(wav: Path) -> dict:
    spk = wav.parent.name            # s1, s2, ...
    utt_id = wav.stem                 # s1-bbafn9s.mpg-style id
    out = OUT_DIR / spk / f"{utt_id}.wav"
    if out.exists():
        return {"utt_id": utt_id, "status": "skipped"}

    audio = load_audio(wav)           # mono, 16 kHz float
    # duration check: original vs resampled
    info = sf.info(wav)
    dur_in = info.frames / info.samplerate
    dur_out = len(audio) / TARGET_SR
    if abs(dur_in - dur_out) > MAX_DUR_DIFF:
        return {"utt_id": utt_id, "status": "FAIL", "reason": f"duration {dur_in:.3f}->{dur_out:.3f}"}

    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, audio, TARGET_SR, subtype="PCM_16")   # 16-bit
    return {"utt_id": utt_id, "status": "ok"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    wavs = sorted(RAW_DIR.glob("s*/audio/*.wav"))
    if not wavs:
        raise SystemExit(f"No audio under {RAW_DIR}/s*/audio/")
    print(f"{len(wavs)} files, {args.workers} workers")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(convert_one, w): w for w in wavs}
        for i, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if i % 2000 == 0:
                print(f"  {i}/{len(wavs)}")

    ok = sum(r["status"] in ("ok", "skipped") for r in results)
    fails = [r for r in results if r["status"] == "FAIL"]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["utt_id", "status", "reason"], extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"Done: {ok} ok/skipped, {len(fails)} failed -> {LOG}")
    for r in fails[:10]:
        print("  FAIL:", r)


if __name__ == "__main__":
    main()