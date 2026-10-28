import csv
import io
import tarfile
import zipfile

import cv2
import numpy as np
import soundfile as sf

from scripts.verify_dataset import verify_dataset


ALIGNMENT = """0 23750 sil
23750 29500 bin
29500 34000 blue
34000 35500 at
35500 41000 f
41000 47250 two
47250 53000 now
53000 74500 sil
"""


def _write_video(path, frame_count=75):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (32, 32)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV mp4v writer is unavailable")
    for frame_index in range(frame_count):
        frame = np.full((32, 32, 3), frame_index % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _audio_samples(duration=1.19):
    time = np.arange(int(16_000 * duration), dtype=np.float32) / 16_000
    return 0.1 * np.sin(2 * np.pi * 440 * time)


def _write_extracted_utterance(root, utterance_id="bbaf2n", *, frame_count=75, transcript=ALIGNMENT):
    speaker = root / "s1"
    (speaker / "video").mkdir(parents=True)
    (speaker / "audio").mkdir()
    (speaker / "align").mkdir()
    _write_video(speaker / "video" / f"{utterance_id}.mp4", frame_count)
    sf.write(speaker / "audio" / f"{utterance_id}.wav", _audio_samples(), 16_000)
    (speaker / "align" / f"{utterance_id}.align").write_text(transcript, encoding="utf-8")


def test_verifies_extracted_utterance_and_writes_reports(tmp_path):
    data_root = tmp_path / "grid"
    _write_extracted_utterance(data_root)

    rows, summary = verify_dataset(
        data_root,
        tmp_path / "reports/integrity_report.csv",
        tmp_path / "reports/integrity_summary.txt",
    )

    assert len(rows) == 1
    assert rows[0]["overall_pass"] is True
    assert rows[0]["transcript"] == "bin blue at f two now"
    assert rows[0]["face_mesh_ok"] == "not_checked"
    assert "files_present: 1/1" in summary
    with (tmp_path / "reports/integrity_report.csv").open(newline="") as stream:
        report = next(csv.DictReader(stream))
    assert report["video_frames"] == "75"
    assert report["audio_video_sync_ok"] == "True"
    assert abs(float(report["audio_speech_delta_seconds"]) - 0.02) < 1e-6
    assert float(report["aligned_speech_start_seconds"]) == 0.95
    assert float(report["alignment_end_seconds"]) == 2.98


def test_flags_bad_frame_count_and_invalid_grid_sentence(tmp_path):
    data_root = tmp_path / "grid"
    invalid_alignment = ALIGNMENT.replace("23750 29500 bin", "23750 29500 jump")
    _write_extracted_utterance(data_root, frame_count=72, transcript=invalid_alignment)

    rows, _ = verify_dataset(data_root, tmp_path / "report.csv", tmp_path / "summary.txt")

    assert rows[0]["video_decodes"] is True
    assert rows[0]["frame_count_ok"] is False
    assert rows[0]["transcript_words_valid"] is False
    assert rows[0]["overall_pass"] is False


def test_sync_rejects_full_video_audio_for_trimmed_alignment(tmp_path):
    data_root = tmp_path / "grid"
    _write_extracted_utterance(data_root)
    sf.write(
        data_root / "s1/audio/bbaf2n.wav",
        _audio_samples(duration=3.0),
        16_000,
    )

    rows, _ = verify_dataset(data_root, tmp_path / "report.csv", tmp_path / "summary.txt")

    assert rows[0]["audio_speech_delta_seconds"] > 1.0
    assert rows[0]["audio_video_sync_ok"] is False


def test_reads_audio_tar_and_alignment_zip_and_applies_face_mesh_threshold(tmp_path):
    data_root = tmp_path / "raw"
    video_dir = data_root / "video/s1"
    video_dir.mkdir(parents=True)
    _write_video(video_dir / "bbaf2n.mp4")

    audio_bytes = io.BytesIO()
    sf.write(audio_bytes, _audio_samples(), 16_000, format="WAV")
    audio_archive = data_root / "audio/s1.tar"
    audio_archive.parent.mkdir()
    with tarfile.open(audio_archive, "w") as archive:
        audio_info = tarfile.TarInfo("s1/bbaf2n.wav")
        audio_info.size = len(audio_bytes.getvalue())
        archive.addfile(audio_info, io.BytesIO(audio_bytes.getvalue()))

    alignment_archive = data_root / "video/alignments.zip"
    with zipfile.ZipFile(alignment_archive, "w") as archive:
        archive.writestr("alignments/s1/bbaf2n.align", ALIGNMENT)

    face_mesh_report = tmp_path / "face_mesh.csv"
    face_mesh_report.write_text(
        "speaker,utterance_id,success_rate\ns1,bbaf2n,0.89\n", encoding="utf-8"
    )

    rows, _ = verify_dataset(
        data_root,
        tmp_path / "report.csv",
        tmp_path / "summary.txt",
        face_mesh_report=face_mesh_report,
    )

    assert rows[0]["files_present"] is True
    assert rows[0]["audio_ok"] is True
    assert rows[0]["transcript_words_valid"] is True
    assert rows[0]["face_mesh_success_rate"] == 0.89
    assert rows[0]["face_mesh_ok"] is False
    assert rows[0]["overall_pass"] is False


def test_flags_silent_audio_and_missing_modality(tmp_path):
    data_root = tmp_path / "grid"
    speaker = data_root / "s1"
    (speaker / "video").mkdir(parents=True)
    (speaker / "audio").mkdir()
    (speaker / "align").mkdir()
    _write_video(speaker / "video/bbaf2n.mp4")
    sf.write(speaker / "audio/bbaf2n.wav", np.zeros(48_000, dtype=np.float32), 16_000)
    (speaker / "align/bbaf2n.align").write_text(ALIGNMENT, encoding="utf-8")

    (speaker / "align/bbaf3s.align").write_text(ALIGNMENT, encoding="utf-8")

    rows, _ = verify_dataset(data_root, tmp_path / "report.csv", tmp_path / "summary.txt")
    rows_by_id = {row["utterance_id"]: row for row in rows}

    assert rows_by_id["bbaf2n"]["audio_ok"] is False
    assert rows_by_id["bbaf2n"]["files_present"] is True
    assert rows_by_id["bbaf3s"]["files_present"] is False


def test_flags_a_single_clipped_peak(tmp_path):
    data_root = tmp_path / "grid"
    _write_extracted_utterance(data_root)
    audio_path = data_root / "s1/audio/bbaf2n.wav"
    wave = _audio_samples()
    wave[100] = 1.0
    sf.write(audio_path, wave, 16_000)

    rows, _ = verify_dataset(data_root, tmp_path / "report.csv", tmp_path / "summary.txt")

    assert rows[0]["audio_peak"] >= 0.999
    assert rows[0]["audio_clipped_fraction"] < 0.001
    assert rows[0]["audio_ok"] is False


def test_indexes_lipnet_processed_speaker_folder_and_reports_missing_audio(tmp_path):
    data_root = tmp_path / "data"
    speaker = data_root / "s1_processed"
    speaker.mkdir(parents=True)
    _write_video(speaker / "bbaf2n.mpg")
    (speaker / "bbaf2n.align").write_text(ALIGNMENT, encoding="utf-8")

    rows, _ = verify_dataset(data_root, tmp_path / "report.csv", tmp_path / "summary.txt")

    assert len(rows) == 1
    assert rows[0]["speaker"] == "s1"
    assert rows[0]["video_present"] is True
    assert rows[0]["alignment_present"] is True
    assert rows[0]["audio_present"] is False
    assert rows[0]["files_present"] is False
    assert rows[0]["video_decodes"] is True
    assert rows[0]["transcript_words_valid"] is True