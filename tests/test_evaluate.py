import csv

from scripts.evaluate import update_results_csv


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