"""Flix v4: probe whether a fused model uses (and benefits from) the lips.

  PYTHONPATH=. .venv/bin/python scripts/flix_v4_probe.py RUN_DIR [N_TED] [OUT_JSON]


Per test set / SNR: WER with real video, with video shuffled across the batch, with video zeroed,
and with visual -> e_miss; plus gate statistics and the share of the fused vector coming from the
visual branch.  Run from the repo root with PYTHONPATH=.
"""
import sys, json
import torch
from torch.utils.data import DataLoader, Subset
from lipscribe.data.dataset import GridDataset, collate
from lipscribe.decoding import greedy_decode
from lipscribe.evaluation import wer
from lipscribe.models import build_model

RUN = sys.argv[1]
N_TED = int(sys.argv[2]) if len(sys.argv) > 2 else 320
TESTS = {"film_test": "data/flix_v4c/flix_test.csv", "ted_test": "data/flix_v4c/ted_test.csv"}
dev = torch.device("cuda")
ck = torch.load(f"{RUN}/best.pt", map_location="cpu", weights_only=False)
model = build_model(dict(ck["config"]["model"]))
model.load_state_dict(ck["model"]); model.to(dev).eval()

# hook the gate inputs/outputs (v4_fusion only; v4_fusion_wp has no gate)
cap = {}
g = model.fusion.gate if hasattr(model, "fusion") else None
def hook(mod, inp, out):
    a_t, v_t = inp
    fused, gate = out
    cap["gate"] = gate.detach().float()
    cap["a_part"] = (gate.unsqueeze(-1) * a_t).detach().float().norm(dim=-1)
    cap["v_part"] = ((1 - gate).unsqueeze(-1) * v_t).detach().float().norm(dim=-1)
if g is not None:
    g.register_forward_hook(hook)

res = {}
for test, man in TESTS.items():
    for snr_name, snr in {"clean": float("inf"), "0": 0.0, "-5": -5.0, "-10": -10.0}.items():
        ds = GridDataset(man, "data/flix_v4/splits.csv", "test", snr_db=snr,
                         noise_splits="Data/Noise_splits/splits.csv", visual_augment=False)
        if test == "ted_test":
            ds = Subset(ds, list(range(0, len(ds), max(1, len(ds) // N_TED)))[:N_TED])
        dl = DataLoader(ds, batch_size=16, collate_fn=collate, num_workers=6)
        out = {m: [] for m in ["real", "shuffled", "zero", "e_miss"]}
        refs, gates, vshare = [], [], []
        with torch.no_grad(), torch.autocast("cuda"):
            for b in dl:
                b = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()}
                refs += b["texts"]
                for m in out:
                    bb = dict(b)
                    if m == "shuffled":
                        bb["video"] = b["video"].roll(1, dims=0)
                        if "video_valid" in b and torch.is_tensor(b["video_valid"]):
                            bb["video_valid"] = b["video_valid"].roll(1, dims=0)
                    elif m == "zero":
                        bb["video"] = torch.zeros_like(b["video"])
                    elif m == "e_miss":
                        bb["drop_modality"] = "visual"
                    o = model(bb)
                    out[m] += greedy_decode(o["log_probs"].float(), o["lengths"])
                    if m == "real" and g is None and "gate" in o:     # v4_fusion_gated: mean channel gate
                        for i in range(len(o["lengths"])):
                            gates.append(o["gate"][i, :int(o["lengths"][i])].float().mean().item())
                    if m == "real" and g is not None:
                        L = o["lengths"]
                        for i in range(len(L)):
                            n = int(L[i])
                            gates.append(cap["gate"][i, :n].mean().item())
                            vshare.append((cap["v_part"][i, :n] / (cap["a_part"][i, :n] + cap["v_part"][i, :n])).mean().item())
        r = {m: round(100 * wer(refs, h), 1) for m, h in out.items()}
        r["n"] = len(refs)
        if gates:
            gt = torch.tensor(gates)
            r["gate_mean"] = round(gt.mean().item(), 3); r["gate_min_clip"] = round(gt.min().item(), 3)
            if vshare:
                r["visual_share_of_fused"] = round(sum(vshare) / len(vshare), 3)
        res[f"{test}|{snr_name}"] = r
        print(test, snr_name, r, flush=True)
print(json.dumps(res, indent=1))
if len(sys.argv) > 3:
    open(sys.argv[3], "w").write(json.dumps(res, indent=1))
