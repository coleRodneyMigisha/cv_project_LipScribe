"""W4-06 gate analysis: runs the core on a tiny random P model and synthetic batches."""
import csv
import json

import numpy as np
import torch

from lipscribe.models import build_model
from scripts.analyse_gate import CSV_FIELDS, run_analysis, spearman

SNRS = ["clean", "10", "5", "0", "-5"]


def _tiny_model():
    torch.manual_seed(0)
    cfg = {"name": "p_proposed", "streams": ["audio", "visual"], "visual_encoder": "conv3d_bilstm",
           "audio_features": "logmel80", "fusion": "cross_attention_gated"}
    return build_model(cfg).eval()


def _batches(seed: int):
    g = torch.Generator().manual_seed(seed)
    out = []
    for names, lengths in ((["s1_aaa", "s2_bbb"], [8, 5]), (["s3_ccc"], [6])):
        b, t = len(names), 8
        lens = torch.tensor(lengths)
        out.append({
            "utt_ids": names, "texts": [f"text {n}" for n in names],
            "features": torch.randn(b, 4 * t, 80, generator=g),
            "feat_lengths": lens * 4,
            "video": torch.rand(b, 1, t, 96, 96, generator=g),
            "video_valid": torch.ones(b, t, dtype=torch.bool),
            "lengths": lens,
            "mask": torch.arange(t)[None, :] < lens[:, None],
        })
    return out


def test_run_analysis_writes_all_outputs(tmp_path):
    loaders = {snr: _batches(i) for i, snr in enumerate(SNRS)}
    summary = run_analysis(_tiny_model(), loaders, lambda b: b, torch.device("cpu"),
                           tmp_path / "out", n_examples=2)
    out = tmp_path / "out"
    for name in ("per_utterance_gate.csv", "summary.json", "summary.md", "gate_boxplot.png",
                 "gate_per_frame_examples.png", "attention_heatmap.png"):
        assert (out / name).stat().st_size > 0, name

    with (out / "per_utterance_gate.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert tuple(reader.fieldnames) == CSV_FIELDS
    assert len(rows) == 3 * len(SNRS)
    first = next(r for r in rows if r["utt_id"] == "s2_bbb")
    assert first["speaker"] == "s2" and int(first["n_frames"]) == 5  # real frames only
    assert 0.0 < float(first["mean_gate"]) < 1.0

    saved = json.loads((out / "summary.json").read_text())
    assert set(saved["per_snr"]) == set(SNRS) and saved["snr_order"] == SNRS
    assert all(saved["per_snr"][s]["n"] == 3 for s in SNRS)
    assert isinstance(saved["monotonic_median"], bool)
    assert summary["spearman_rho_snr_vs_gate"] == saved["spearman_rho_snr_vs_gate"]


def test_spearman_handles_ties_and_direction():
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))
