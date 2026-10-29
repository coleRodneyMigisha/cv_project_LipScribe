"""W2-06: fixed noisy validation and test sets, on a tiny synthetic corpus."""
import csv

import numpy as np
import soundfile as sf

from lipscribe.data.dataset import GridDataset
from scripts.make_noisy_test import main as make_noisy_test

SPEAKERS = {"s1": "train", "s3": "val", "s4": "test"}


def _corpus(tmp_path):
    rng = np.random.default_rng(0)
    (tmp_path / "audio").mkdir()
    rows = []
    for spk in SPEAKERS:
        for k in range(3):
            uid = f"{spk}_u{k}"
            mouths = tmp_path / f"{uid}.npy"
            np.save(mouths, np.zeros((75, 96, 96), np.uint8))
            wav = tmp_path / "audio" / f"{uid}.wav"
            sf.write(wav, (0.1 * rng.standard_normal(48000)).astype(np.float32), 16000)
            rows.append({"utt_id": uid, "speaker": spk, "transcript": "bin blue at f two now",
                         "n_frames": 75, "usable": "True", "mouths_path": str(mouths),
                         "valid_path": "", "audio_path": str(wav)})
    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    splits = tmp_path / "splits.csv"
    splits.write_text("speaker,partition\n" + "".join(f"{s},{p}\n" for s, p in SPEAKERS.items()))

    # MUSAN pools in make_noise_splits.py format (paths relative to the folder above the CSV)
    (tmp_path / "musan").mkdir()
    pool_rows = []
    for pool in ("train", "val", "test"):
        for j in range(2):
            f = tmp_path / "musan" / f"{pool}{j}.wav"
            sf.write(f, (0.2 * rng.standard_normal(20000)).astype(np.float32), 16000)
            pool_rows.append(f"musan/{pool}{j}.wav,{pool}\n")
    (tmp_path / "Noise_splits").mkdir()
    noise_splits = tmp_path / "Noise_splits" / "splits.csv"
    noise_splits.write_text("file,pool\n" + "".join(pool_rows))
    return manifest, splits, noise_splits


def _run(tmp_path, manifest, splits, noise_splits, out):
    return make_noisy_test(["--manifest", str(manifest), "--splits", str(splits),
                            "--noise-splits", str(noise_splits), "--out", str(out), "--workers", "1"])


def test_writes_every_val_test_utterance_at_every_snr(tmp_path):
    manifest, splits, noise_splits = _corpus(tmp_path)
    out = tmp_path / "noisy_test"
    assert _run(tmp_path, manifest, splits, noise_splits, out) == 0

    for snr in ("10", "5", "0", "-5"):
        files = sorted(p.name for p in (out / snr).glob("*.wav"))
        assert files == [f"{s}_u{k}.wav" for s in ("s3", "s4") for k in range(3)]  # no train speakers
        y, sr = sf.read(out / snr / "s3_u0.wav")
        assert sr == 16000 and len(y) == 48000

    log = list(csv.DictReader(open(out / "mix_log.csv")))
    assert len(log) == 6 * 4
    for row in log:
        pool = "val" if row["utt_id"].startswith("s3") else "test"
        assert f"/musan/{pool}" in row["noise_file"].replace("\\", "/")   # only the matching pool
        assert abs(float(row["measured_snr_db"]) - float(row["snr_db"])) < 0.01
        assert int(row["offset"]) >= 0


def test_fixed_seed_and_resumable(tmp_path):
    manifest, splits, noise_splits = _corpus(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    _run(tmp_path, manifest, splits, noise_splits, a)
    _run(tmp_path, manifest, splits, noise_splits, b)
    assert np.array_equal(sf.read(a / "0" / "s4_u1.wav")[0], sf.read(b / "0" / "s4_u1.wav")[0])

    (a / "0" / "s4_u1.wav").unlink()                       # simulate an interrupted run
    before = len(list(csv.DictReader(open(a / "mix_log.csv"))))
    _run(tmp_path, manifest, splits, noise_splits, a)
    after = list(csv.DictReader(open(a / "mix_log.csv")))
    assert len(after) == before + 1                        # only the missing file is redone
    assert np.array_equal(sf.read(a / "0" / "s4_u1.wav")[0], sf.read(b / "0" / "s4_u1.wav")[0])


def test_dataset_reads_the_fixed_files(tmp_path):
    manifest, splits, noise_splits = _corpus(tmp_path)
    out = tmp_path / "noisy_test"
    _run(tmp_path, manifest, splits, noise_splits, out)
    item = GridDataset(manifest, splits, "val", snr_db=5.0, noisy_dir=out)[0]
    expected = sf.read(out / "5" / f"{item['utt_id']}.wav", dtype="float32")[0]
    assert item["snr_db"] == 5.0
    assert np.array_equal(item["audio"], expected)
