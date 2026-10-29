"""Build data/manifest.csv: one row per GRID utterance (W2-09).

The manifest is the single index of the dataset (implementation plan, Section 4).
Every script selects data by filtering it, never by listing directories.

It joins three sources:
  * the integrity report (reports/integrity_report.csv, from verify_dataset.py):
    transcript, frame count, durations, pass/fail checks, original paths
  * the cached mouth crops written by extract_mouths.py (W2-02)
  * the 16 kHz audio written by the audio standardisation script (W2-04)
and, if given, data/splits.csv (W1-08) to add each utterance's partition.

Safe to re-run at any time (13:30 rebuild after extraction): it only reads,
and it overwrites the manifest. Missing crops or audio are recorded, not fatal.

Example (on the Alienware, from the repo root):
    python scripts/build_manifest.py \
        --integrity reports/integrity_report.csv \
        --mouths-dir data/processed/mouths \
        --audio-dir data/processed/audio16k \
        --splits data/splits.csv \
        --out data/manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

MIN_FACE_RATE = 0.90  # clips with >10 % failed detections are dropped (Section 5)

MANIFEST_COLUMNS = [
    "utt_id", "speaker", "utterance", "partition", "transcript", "n_chars",
    "n_frames", "audio_duration_s", "speech_start_s", "speech_end_s",
    "audio16k_duration_s", "face_mesh_rate", "integrity_pass", "has_mouths", "has_audio", "usable",
    "reason", "mouths_path", "valid_path", "audio_path", "video_path",
]

# File-name patterns tried, in order, for the crop and validity files.
# {spk} = speaker (s1), {utt} = utterance (bbaf2n), {uid} = s1_bbaf2n
MOUTH_PATTERNS = ["{spk}/{utt}.npy", "{spk}/{utt}_crops.npy", "{uid}.npy", "{uid}_crops.npy"]
VALID_PATTERNS = ["{spk}/{utt}_valid.npy", "{uid}_valid.npy"]
NPZ_PATTERNS = ["{spk}/{utt}.npz", "{uid}.npz"]
AUDIO_PATTERNS = ["{spk}/{utt}.wav", "{uid}.wav"]


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _first_existing(root: Path | None, patterns: list[str], **names: str) -> Path | None:
    if root is None:
        return None
    for pattern in patterns:
        path = root / pattern.format(**names)
        if path.exists():
            return path
    return None


def read_splits(path: Path | None) -> dict[str, str]:
    """speaker -> partition. Accepts columns speaker + partition/split/set."""
    if path is None:
        return {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        cols = {c.lower().strip(): c for c in (reader.fieldnames or [])}
        spk_col = cols.get("speaker")
        part_col = next((cols[c] for c in ("partition", "split", "set") if c in cols), None)
        if spk_col is None or part_col is None:
            raise ValueError(f"{path} needs a 'speaker' and a 'partition' column, found {reader.fieldnames}")
        mapping: dict[str, str] = {}
        for row in reader:
            spk = row[spk_col].strip().lower()
            part = row[part_col].strip().lower()
            if spk in mapping and mapping[spk] != part:
                raise ValueError(f"Speaker {spk} is in two partitions: {mapping[spk]} and {part}")
            mapping[spk] = part
    return mapping


def face_rate_for(mouths: Path | None, valid: Path | None) -> tuple[float, int | None]:
    """Fraction of frames with a successful Face Mesh detection, and crop frame count."""
    if mouths is None:
        return math.nan, None
    if mouths.suffix == ".npz":
        with np.load(mouths) as z:
            n = int(z["crops"].shape[0]) if "crops" in z else None
            if "valid" in z:
                v = z["valid"].astype(bool)
                return (float(v.mean()) if v.size else 0.0), n
        return math.nan, n
    n = int(np.load(mouths, mmap_mode="r").shape[0])
    if valid is not None:
        v = np.load(valid).astype(bool)
        return (float(v.mean()) if v.size else 0.0), n
    return math.nan, n


def build_rows(integrity: Path, mouths_dir: Path | None, audio_dir: Path | None,
               splits: dict[str, str]) -> list[dict]:
    rows = []
    with open(integrity, newline="") as fh:
        for rec in csv.DictReader(fh):
            spk = rec["speaker"].strip().lower()
            utt = rec["utterance_id"].strip().lower()
            uid = f"{spk}_{utt}"
            names = {"spk": spk, "utt": utt, "uid": uid}

            mouths = _first_existing(mouths_dir, MOUTH_PATTERNS, **names)
            valid = _first_existing(mouths_dir, VALID_PATTERNS, **names)
            if mouths is None:
                mouths = _first_existing(mouths_dir, NPZ_PATTERNS, **names)
                valid = None
            audio = _first_existing(audio_dir, AUDIO_PATTERNS, **names)

            audio_dur = round(sf.info(str(audio)).duration, 3) if audio else ""
            rate, crop_frames = face_rate_for(mouths, valid)
            if math.isnan(rate) and rec.get("face_mesh_success_rate"):
                rate = _float(rec["face_mesh_success_rate"])
            n_frames = crop_frames if crop_frames is not None else int(_float(rec.get("video_frames")) or 0)
            transcript = rec.get("transcript", "").strip().lower()

            reasons = []
            if not _truthy(rec.get("overall_pass", "")):
                reasons.append("integrity_fail")
            if mouths_dir is not None and mouths is None:
                reasons.append("no_mouths")
            if not math.isnan(rate) and rate < MIN_FACE_RATE:
                reasons.append("face_mesh_below_90pct")
            if audio_dir is not None and audio is None:
                reasons.append("no_audio16k")
            if not transcript:
                reasons.append("no_transcript")
            part = splits.get(spk, "")
            if splits and not part:
                reasons.append("speaker_not_in_splits")

            rows.append({
                "utt_id": uid, "speaker": spk, "utterance": utt, "partition": part,
                "transcript": transcript, "n_chars": len(transcript),
                "n_frames": n_frames,
                "audio_duration_s": rec.get("audio_duration_seconds", ""),
                "speech_start_s": rec.get("aligned_speech_start_seconds", ""),
                "speech_end_s": rec.get("aligned_speech_end_seconds", ""),
                "audio16k_duration_s": audio_dur,
                "face_mesh_rate": "" if math.isnan(rate) else round(rate, 4),
                "integrity_pass": _truthy(rec.get("overall_pass", "")),
                "has_mouths": mouths is not None, "has_audio": audio is not None,
                "usable": not reasons, "reason": ";".join(reasons),
                "mouths_path": str(mouths) if mouths else "",
                "valid_path": str(valid) if valid else "",
                "audio_path": str(audio) if audio else "",
                "video_path": rec.get("video_path", ""),
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--integrity", type=Path, default=Path("reports/integrity_report.csv"))
    p.add_argument("--mouths-dir", type=Path, default=None, help="extract_mouths.py output folder")
    p.add_argument("--audio-dir", type=Path, default=None, help="16 kHz audio folder (W2-04 output)")
    p.add_argument("--splits", type=Path, default=None, help="data/splits.csv (speaker -> partition)")
    p.add_argument("--out", type=Path, default=Path("data/manifest.csv"))
    args = p.parse_args(argv)

    if not args.integrity.exists():
        print(f"ERROR: integrity report not found: {args.integrity}", file=sys.stderr)
        return 1
    for name in ("mouths_dir", "audio_dir"):
        folder = getattr(args, name)
        if folder is not None and not folder.is_dir():
            print(f"ERROR: --{name.replace('_', '-')} folder not found: {folder}", file=sys.stderr)
            return 1

    splits = read_splits(args.splits)
    rows = build_rows(args.integrity, args.mouths_dir, args.audio_dir, splits)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(args.out)  # atomic: a crashed run never leaves a half-written manifest

    usable = [r for r in rows if r["usable"]]
    print(f"Wrote {args.out}: {len(rows)} utterances, {len(usable)} usable")
    print("Not usable, by reason:", dict(Counter(
        reason for r in rows for reason in r["reason"].split(";") if reason)))
    short = [r for r in rows if r["audio16k_duration_s"] != "" and r["n_frames"]
             and r["audio16k_duration_s"] < 0.9 * r["n_frames"] * 0.04]
    if short:
        print(f"WARNING: {len(short)} clips have 16 kHz audio much shorter than their video "
              f"(e.g. {short[0]['utt_id']}: {short[0]['audio16k_duration_s']} s audio vs "
              f"{short[0]['n_frames'] * 0.04:.2f} s video). GRID 25 kHz audio is end-pointed "
              "(trimmed to speech); it must be padded with silence by speech_start_s to sit "
              "on the video's time grid, or audio and lips will be out of sync.")
    if splits:
        print("Usable by partition:", dict(Counter(r["partition"] for r in usable)))
        print("Usable speakers by partition:", {
            part: len({r["speaker"] for r in usable if r["partition"] == part})
            for part in sorted({r["partition"] for r in usable})})
    return 0


if __name__ == "__main__":
    sys.exit(main())
