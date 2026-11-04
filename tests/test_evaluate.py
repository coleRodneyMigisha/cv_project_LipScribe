import csv

from scripts.evaluate import save_hypotheses, update_results_csv


def test_save_hypotheses_creates_valid_csv(tmp_path):
    hyp_path = tmp_path / "hypotheses.csv"
    sample_data = [
        {"utt_id": "s1_bbaf2n", "snr": "clean", "speaker": "s1", "ref": "bin blue at f two now",
         "hyp": "bin blue at f two now", "wer": "0.0000", "cer": "0.0000", "gate": "0.8500"},
        {"utt_id": "s1_bgaa6n", "snr": "0", "speaker": "s1", "ref": "bin green at a six now",
         "hyp": "bin green at a six now", "wer": "0.0000", "cer": "0.0000", "gate": ""},
    ]
    save_hypotheses(hyp_path, sample_data)
    assert hyp_path.exists()
    with hyp_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[0]["utt_id"] == "s1_bbaf2n"
    assert rows[0]["gate"] == "0.8500"


def test_update_results_csv_adds_checkpoint_and_test_metrics(tmp_path):
    run_dir = tmp_path / "runs" / "b1"
    run_dir.mkdir(parents=True)
    results_path = tmp_path / "experiments" / "results.csv"
    results_path.parent.mkdir()
    with results_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("run_dir", "best_val_wer_mean"))
        writer.writeheader()
        writer.writerow({"run_dir": str(run_dir), "best_val_wer_mean": "0.2"})

    update_results_csv(results_path, run_dir, {
        "clean": {"wer": 0.1, "cer": 0.05},
        "-5": {"wer": 0.4, "cer": 0.2},
    })

    with results_path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["best_val_wer_mean"] == "0.2"
    assert row["config_path"] == str(run_dir / "config.yaml")
    assert row["checkpoint_path"] == str(run_dir / "best.pt")
    assert row["test_wer_by_snr"] == "clean=0.1000;-5=0.4000"
    assert row["test_cer_by_snr"] == "clean=0.0500;-5=0.2000"