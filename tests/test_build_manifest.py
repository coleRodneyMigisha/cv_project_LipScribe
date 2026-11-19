import csv

from scripts.build_manifest import build_rows


def test_missing_frame_count_without_crops_defaults_to_zero(tmp_path):
    report = tmp_path / "integrity.csv"
    with report.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["speaker", "utterance_id", "overall_pass", "video_frames", "transcript"],
        )
        writer.writeheader()
        writer.writerow({
            "speaker": "s21",
            "utterance_id": "bbaf2n",
            "overall_pass": "False",
            "video_frames": "",
            "transcript": "",
        })

    rows = build_rows(report, None, None, {})

    assert rows[0]["n_frames"] == 0