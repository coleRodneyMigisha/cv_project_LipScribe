#!/usr/bin/env python3
"""Convert all GRID audio to 16 kHz, 16-bit, mono WAV (W2-04).

Resumable: skips utterances already converted. Parallel across CPU cores.

    python scripts/prepare_audio.py --workers 8 [--raw-dir <path>]
"""
import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, ".")
from lipscribe.preprocessing.audio import load_audio, TARGET_SR

OUT_DIR = Path("data/processed/audio16k")
LOG = Path("data/processed/audio16k/conversion_log.csv")
MAX_DUR_DIFF = 0.04          # 40 ms tolerance (one video frame), per plan
RAW_DIR = None               # set in main() from --raw-dir


def convert_one(wav: Path) -> dict:
    rel = wav.relative_to(RAW_DIR)     # s1/audio/xxx.wav  or  s1/xxx.wav
    spk = rel.parts[0]                 # "s1" — first level is always the speaker
    utt_id = wav.stem
    out = OUT_DIR / spk / f"{utt_id}.wav"
    if out.exists():
        return {"utt_id": f"{spk}/{utt_id}", "status": "skipped"}

    try:
        audio = load_audio(wav)
        info = sf.info(str(wav))
        dur_in = info.frames / info.samplerate
        dur_out = len(audio) / TARGET_SR
        if abs(dur_in - dur_out) > MAX_DUR_DIFF:
            return {"utt_id": f"{spk}/{utt_id}", "status": "FAIL",
                    "reason": f"duration {dur_in:.3f}->{dur_out:.3f}"}
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), np.clip(audio, -1.0, 1.0), TARGET_SR, subtype="PCM_16")
        return {"utt_id": f"{spk}/{utt_id}", "status": "ok"}
    except Exception as e:
        return {"utt_id": f"{spk}/{utt_id}", "status": "FAIL", "reason": str(e)}


def main():
    global RAW_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--raw-dir", default="data/raw/grid")
    args = ap.parse_args()
    RAW_DIR = Path(args.raw_dir).expanduser()

    # handles both s1/audio/x.wav and s1/x.wav layouts
    wavs = sorted(RAW_DIR.glob("s*/audio/*.wav")) or sorted(RAW_DIR.glob("s*/*.wav"))
    if not wavs:
        raise SystemExit(f"No audio under {RAW_DIR} (expected s*/audio/*.wav or s*/*.wav)")
    print(f"{len(wavs)} files | raw={RAW_DIR} | {args.workers} workers", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(convert_one, w): w for w in wavs}
        for i, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if i % 2000 == 0:
                print(f"  {i}/{len(wavs)}", flush=True)

    ok = sum(r["status"] in ("ok", "skipped") for r in results)
    fails = [r for r in results if r["status"] == "FAIL"]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["utt_id", "status", "reason"], extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"Done: {ok} ok/skipped, {len(fails)} FAIL -> {LOG}")
    for r in fails[:10]:
        print("  FAIL:", r)


if __name__ == "__main__":
    main()