"""Verify GRID utterance integrity and write per-utterance and summary reports.

The input may use the planned ``sN/{video,audio,align}`` layout, separate
``video/sN`` / ``audio/sN`` / ``align/sN`` directories, or GRID speaker WAV
TAR files plus an alignment ZIP. Face Mesh results can be merged later with
``--face-mesh-report``; its ``success_rate`` is a fraction between 0 and 1.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import tarfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

import cv2
import numpy as np
import soundfile as sf


SPEAKER_RE = re.compile(r"s\d+", re.IGNORECASE)
VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg"}
SILENCE_WORDS = {"sil", "sp"}
GRID_WORDS_BY_SLOT = (
    {"bin", "lay", "place", "set"},
    {"blue", "green", "red", "white"},
    {"at", "by", "in", "with"},
    set("abcdefghijklmnopqrstuvxyz"),  # GRID's letter slot excludes w.
    {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"},
    {"again", "now", "please", "soon"},
)

CHECK_COLUMNS = {
    "files_present": "files_present",
    "video_decodes": "video_decodes",
    "frame_count": "frame_count_ok",
    "audio_video_sync": "audio_video_duration_ok",
    "transcript": "transcript_words_valid",
    "audio_level": "audio_ok",
}


@dataclass(frozen=True)
class FileRef:
    path: Path
    member: str | None = None

    def display(self) -> str:
        return f"{self.path}::{self.member}" if self.member else str(self.path)


def _speaker_in(parts: tuple[str, ...] | list[str]) -> str | None:
    for part in reversed(parts):
        if SPEAKER_RE.fullmatch(part):
            return part.lower()
    return None


def _key_for(path_parts: tuple[str, ...] | list[str], stem: str, root: Path) -> tuple[str, str] | None:
    speaker = _speaker_in(path_parts) or (root.name.lower() if SPEAKER_RE.fullmatch(root.name) else None)
    return (speaker, stem.lower()) if speaker else None


class DatasetIndex:
    """Index extracted files and audio/alignment members in GRID archives."""

    def __init__(self, root: Path):
        self.root = root
        self.files: dict[str, dict[tuple[str, str], FileRef]] = {
            "video": {}, "audio": {}, "alignment": {},
        }
        self.scan_errors: list[str] = []
        self._tar_handles: dict[Path, tarfile.TarFile] = {}
        self._zip_handles: dict[Path, zipfile.ZipFile] = {}
        self._index()

    def _add(self, kind: str, key: tuple[str, str] | None, ref: FileRef) -> None:
        if key is not None:
            self.files[kind].setdefault(key, ref)

    def _index(self) -> None:
        paths = sorted(path for path in self.root.rglob("*") if path.is_file())
        for path in paths:
            relative_parts = path.relative_to(self.root).parts
            if "__MACOSX" in relative_parts or path.name.startswith("._"):
                continue
            suffix = path.suffix.lower()
            kind = (
                "video" if suffix in VIDEO_SUFFIXES else
                "audio" if suffix == ".wav" else
                "alignment" if suffix == ".align" else None
            )
            if kind:
                key = _key_for(path.relative_to(self.root).parts[:-1], path.stem, self.root)
                self._add(kind, key, FileRef(path))

        for path in paths:
            if path.suffix.lower() == ".zip":
                self._index_zip(path)
            elif path.suffix.lower() in {".tar", ".tgz", ".gz"}:
                self._index_tar(path)

    def _index_zip(self, path: Path) -> None:
        try:
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    member = PurePosixPath(name)
                    if "__MACOSX" in member.parts or member.name.startswith("._"):
                        continue
                    if member.suffix.lower() == ".align":
                        key = _key_for(member.parts[:-1], member.stem, self.root)
                        self._add("alignment", key, FileRef(path, name))
        except (OSError, zipfile.BadZipFile) as exc:
            self.scan_errors.append(f"Could not index ZIP {path}: {exc}")

    def _index_tar(self, path: Path) -> None:
        try:
            with tarfile.open(path, mode="r:*") as archive:
                for member in archive.getmembers():
                    member_path = PurePosixPath(member.name)
                    if "__MACOSX" in member_path.parts or member_path.name.startswith("._"):
                        continue
                    if not member.isfile() or member_path.suffix.lower() != ".wav":
                        continue
                    key = _key_for(member_path.parts[:-1], member_path.stem, self.root)
                    if key is None:
                        key = _key_for((path.stem,), member_path.stem, self.root)
                    self._add("audio", key, FileRef(path, member.name))
        except (OSError, tarfile.TarError) as exc:
            self.scan_errors.append(f"Could not index TAR {path}: {exc}")

    @property
    def utterances(self) -> list[tuple[str, str]]:
        keys = set().union(*(items.keys() for items in self.files.values()))
        return sorted(keys, key=lambda item: (int(item[0][1:]), item[1]))

    def read_alignment(self, ref: FileRef) -> str:
        if ref.member is None:
            return ref.path.read_text(encoding="utf-8", errors="replace")
        archive = self._zip_handles.get(ref.path)
        if archive is None:
            archive = zipfile.ZipFile(ref.path)
            self._zip_handles[ref.path] = archive
        return archive.read(ref.member).decode("utf-8", errors="replace")

    def read_audio(self, ref: FileRef) -> tuple[np.ndarray, int]:
        if ref.member is None:
            wave, sample_rate = sf.read(ref.path, dtype="float32", always_2d=True)
            return wave, sample_rate
        archive = self._tar_handles.get(ref.path)
        if archive is None:
            archive = tarfile.open(ref.path, mode="r:*")
            self._tar_handles[ref.path] = archive
        stream: BinaryIO | None = archive.extractfile(ref.member)
        if stream is None:
            raise OSError(f"Could not read archive member {ref.member}")
        with stream:
            wave, sample_rate = sf.read(stream, dtype="float32", always_2d=True)
        return wave, sample_rate

    def close(self) -> None:
        for archive in self._tar_handles.values():
            archive.close()
        for archive in self._zip_handles.values():
            archive.close()


def _video_metrics(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return {"video_decodes": False, "video_frames": "", "video_fps": ""}
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        frames += 1
    capture.release()
    decodes = frames > 0 and (reported_frames <= 0 or frames >= reported_frames)
    return {
        "video_decodes": decodes,
        "video_frames": frames,
        "video_fps": fps if math.isfinite(fps) else "",
    }


def _alignment_transcript(text: str) -> tuple[str, bool]:
    words: list[str] = []
    try:
        for line in text.splitlines():
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 3:
                return " ".join(words), False
            start, end = float(fields[0]), float(fields[1])
            word = fields[2].lower()
            if not math.isfinite(start) or not math.isfinite(end) or end < start:
                return " ".join(words), False
            if word not in SILENCE_WORDS:
                words.append(word)
    except ValueError:
        return " ".join(words), False
    transcript = " ".join(words)
    valid = len(words) == 6 and all(word in slot for word, slot in zip(words, GRID_WORDS_BY_SLOT))
    return transcript, valid


def _audio_metrics(
    index: DatasetIndex,
    ref: FileRef,
    min_rms_dbfs: float,
    max_clipped_fraction: float,
) -> dict[str, object]:
    wave, sample_rate = index.read_audio(ref)
    if wave.size == 0 or sample_rate <= 0:
        raise ValueError("audio has no samples or an invalid sample rate")
    peak = float(np.max(np.abs(wave)))
    rms = float(np.sqrt(np.mean(np.square(wave, dtype=np.float64))))
    rms_dbfs = 20.0 * math.log10(rms) if rms > 0 else float("-inf")
    clipped_fraction = float(np.count_nonzero(np.abs(wave) >= 0.999) / wave.size)
    return {
        "audio_sample_rate": sample_rate,
        "audio_samples": wave.shape[0],
        "audio_duration_seconds": wave.shape[0] / sample_rate,
        "audio_rms_dbfs": rms_dbfs,
        "audio_peak": peak,
        "audio_clipped_fraction": clipped_fraction,
        "audio_ok": (
            rms_dbfs >= min_rms_dbfs
            and peak < 0.999
            and clipped_fraction <= max_clipped_fraction
        ),
    }


def _load_face_mesh_report(path: Path | None) -> dict[tuple[str, str], float]:
    if path is None:
        return {}
    rates: dict[tuple[str, str], float] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"speaker", "utterance_id", "success_rate"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Face Mesh report needs speaker, utterance_id, success_rate columns")
        for row in reader:
            key = (row["speaker"].lower(), row["utterance_id"].lower())
            rate = float(row["success_rate"])
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"Face Mesh success_rate must be between 0 and 1: {row}")
            rates[key] = rate
    return rates


def verify_dataset(
    data_root: Path,
    output_path: Path,
    summary_path: Path,
    *,
    duration_tolerance: float = 0.040,
    min_rms_dbfs: float = -60.0,
    max_clipped_fraction: float = 0.001,
    face_mesh_report: Path | None = None,
    min_face_mesh_rate: float = 0.90,
) -> tuple[list[dict[str, object]], str]:
    """Verify utterances, write CSV and summary, and return both results."""
    data_root = data_root.resolve()
    if not data_root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {data_root}")
    if duration_tolerance < 0 or max_clipped_fraction < 0 or not 0 <= min_face_mesh_rate <= 1:
        raise ValueError("Thresholds must be non-negative; Face Mesh rate must be in [0, 1]")

    index = DatasetIndex(data_root)
    if not index.utterances:
        raise ValueError(f"No GRID utterances found under {data_root}")
    face_rates = _load_face_mesh_report(face_mesh_report)
    rows: list[dict[str, object]] = []
    try:
        for speaker, utterance_id in index.utterances:
            key = (speaker, utterance_id)
            video = index.files["video"].get(key)
            audio = index.files["audio"].get(key)
            alignment = index.files["alignment"].get(key)
            row: dict[str, object] = {
                "speaker": speaker,
                "utterance_id": utterance_id,
                "video_path": video.display() if video else "",
                "audio_path": audio.display() if audio else "",
                "alignment_path": alignment.display() if alignment else "",
                "video_present": video is not None,
                "audio_present": audio is not None,
                "alignment_present": alignment is not None,
                "files_present": video is not None and audio is not None and alignment is not None,
                "video_decodes": False,
                "video_frames": "",
                "video_fps": "",
                "frame_count_ok": False,
                "video_duration_seconds": "",
                "audio_sample_rate": "",
                "audio_samples": "",
                "audio_duration_seconds": "",
                "audio_rms_dbfs": "",
                "audio_peak": "",
                "audio_clipped_fraction": "",
                "audio_ok": False,
                "audio_video_delta_seconds": "",
                "audio_video_duration_ok": False,
                "transcript": "",
                "transcript_word_count": 0,
                "transcript_words_valid": False,
                "face_mesh_success_rate": "",
                "face_mesh_ok": "not_checked",
                "overall_pass": False,
                "errors": [],
            }

            if video is not None and video.member is None:
                try:
                    video_values = _video_metrics(video.path)
                    row.update(video_values)
                    row["frame_count_ok"] = 73 <= int(video_values["video_frames"] or 0) <= 77
                    fps = float(video_values["video_fps"] or 0)
                    if row["video_decodes"] and fps > 0:
                        row["video_duration_seconds"] = int(video_values["video_frames"]) / fps
                except (OSError, cv2.error, ValueError) as exc:
                    row["errors"].append(f"video: {exc}")
            elif video is not None:
                row["errors"].append("video is inside an archive and must be extracted to decode")

            if audio is not None:
                try:
                    row.update(_audio_metrics(index, audio, min_rms_dbfs, max_clipped_fraction))
                except (OSError, RuntimeError, ValueError, tarfile.TarError) as exc:
                    row["errors"].append(f"audio: {exc}")

            if alignment is not None:
                try:
                    transcript, valid = _alignment_transcript(index.read_alignment(alignment))
                    row["transcript"] = transcript
                    row["transcript_word_count"] = len(transcript.split())
                    row["transcript_words_valid"] = valid
                    if not valid:
                        row["errors"].append("alignment transcript is not a valid six-word GRID sentence")
                except (OSError, UnicodeError, zipfile.BadZipFile, KeyError) as exc:
                    row["errors"].append(f"alignment: {exc}")

            if row["video_duration_seconds"] != "" and row["audio_duration_seconds"] != "":
                delta = abs(float(row["video_duration_seconds"]) - float(row["audio_duration_seconds"]))
                row["audio_video_delta_seconds"] = delta
                row["audio_video_duration_ok"] = delta <= duration_tolerance

            if key in face_rates:
                rate = face_rates[key]
                row["face_mesh_success_rate"] = rate
                row["face_mesh_ok"] = rate >= min_face_mesh_rate
            elif face_mesh_report is not None:
                row["face_mesh_ok"] = False
                row["errors"].append("Face Mesh result missing from supplied report")

            row["overall_pass"] = all(bool(row[column]) for column in CHECK_COLUMNS.values())
            if face_mesh_report is not None:
                row["overall_pass"] = bool(row["overall_pass"] and row["face_mesh_ok"])
            row["errors"] = "; ".join(row["errors"])
            rows.append(row)
    finally:
        index.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = _format_summary(
        rows,
        data_root,
        duration_tolerance,
        min_rms_dbfs,
        max_clipped_fraction,
        min_face_mesh_rate,
        face_mesh_report is not None,
        index.scan_errors,
    )
    summary_path.write_text(summary, encoding="utf-8")
    return rows, summary


def _format_summary(
    rows: list[dict[str, object]],
    data_root: Path,
    duration_tolerance: float,
    min_rms_dbfs: float,
    max_clipped_fraction: float,
    min_face_mesh_rate: float,
    face_mesh_checked: bool,
    scan_errors: list[str],
) -> str:
    check_columns = dict(CHECK_COLUMNS)
    if face_mesh_checked:
        check_columns["face_mesh"] = "face_mesh_ok"
    lines = [
        "GRID dataset integrity summary",
        f"Dataset root: {data_root}",
        f"Utterances discovered: {len(rows)}",
        f"Thresholds: video frames 73-77; duration delta <= {duration_tolerance:.3f} s; "
        f"audio RMS >= {min_rms_dbfs:.1f} dBFS; clipped fraction <= {max_clipped_fraction:.4%}; "
        f"Face Mesh >= {min_face_mesh_rate:.0%}",
        "",
        "Pass counts by check:",
    ]
    for name, column in check_columns.items():
        passed = sum(bool(row[column]) for row in rows)
        lines.append(f"  {name}: {passed}/{len(rows)}")
    if not face_mesh_checked:
        lines.append("  face_mesh: not checked (supply --face-mesh-report in a later pipeline stage)")
    passed_overall = sum(bool(row["overall_pass"]) for row in rows)
    lines.extend([f"  overall: {passed_overall}/{len(rows)}", "", "Pass counts by speaker:"])
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["speaker"])].append(row)
    for speaker in sorted(grouped, key=lambda value: int(value[1:])):
        speaker_rows = grouped[speaker]
        counts = ", ".join(
            f"{name}={sum(bool(row[column]) for row in speaker_rows)}/{len(speaker_rows)}"
            for name, column in check_columns.items()
        )
        lines.append(f"  {speaker} ({len(speaker_rows)} utterances): {counts}")
    if scan_errors:
        lines.extend(["", "Archive indexing warnings:"])
        lines.extend(f"  {error}" for error in scan_errors)
    return "\n".join(lines) + "\n"


def _default_data_root() -> Path:
    for candidate in (Path("data/raw/grid"), Path("Data/gridcorpus/raw"), Path("gridcorpus")):
        if candidate.is_dir():
            return candidate
    return Path("data/raw/grid")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=_default_data_root())
    parser.add_argument("--output", type=Path, default=Path("reports/integrity_report.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("reports/integrity_summary.txt"))
    parser.add_argument("--duration-tolerance", type=float, default=0.040, help="Maximum audio/video delta in seconds")
    parser.add_argument("--min-rms-dbfs", type=float, default=-60.0)
    parser.add_argument("--max-clipped-fraction", type=float, default=0.001)
    parser.add_argument("--face-mesh-report", type=Path, help="CSV with speaker, utterance_id, success_rate (0-1)")
    parser.add_argument("--min-face-mesh-rate", type=float, default=0.90)
    args = parser.parse_args(argv)
    try:
        rows, summary = verify_dataset(
            args.data_root,
            args.output,
            args.summary_output,
            duration_tolerance=args.duration_tolerance,
            min_rms_dbfs=args.min_rms_dbfs,
            max_clipped_fraction=args.max_clipped_fraction,
            face_mesh_report=args.face_mesh_report,
            min_face_mesh_rate=args.min_face_mesh_rate,
        )
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
        print(f"Dataset verification failed: {exc}", file=sys.stderr)
        return 2
    print(summary, end="")
    print(f"CSV report: {args.output}")
    print(f"Summary: {args.summary_output}")
    return 0 if all(bool(row["overall_pass"]) for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())