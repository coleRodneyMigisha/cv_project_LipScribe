"""Paired bootstrap of GRID test WER differences (F-11, final report Section 08).

Resamples test utterances with replacement (2,000 times, seed 0) and recomputes the corpus-level
WER difference between two models on the same clips at each SNR. Per-utterance edits are
recovered from reports/<run>/hypotheses.csv as round(wer * number of reference words).
Also prints per-speaker WER of B1 and P4.

Usage, from the repository root:
    python analysis/scripts/grid_bootstrap.py . > analysis/01_grid/bootstrap.txt
"""
import csv, random, sys, os
root = sys.argv[1]
runs = {
 "B1": "b1_audio_only_20261103-193743",
 "B3v2": "b3_concat_v2_20261104-204404",
 "P4": "p_init_b1_b2_v2_20261105-170138",
 "X1": "x1_no_modality_dropout_20261116-180831",
 "X2": "x2_concat_fusion_20261116-183742",
 "X3": "x3_2d_frontend_20261116-190515",
}
def load(run):
    d = {}
    with open(os.path.join(root, "reports", run, "hypotheses.csv"), newline="") as f:
        for r in csv.DictReader(f):
            n = len(r["ref"].split())
            e = round(float(r["wer"]) * n)
            d[(r["utt_id"], r["snr"])] = (e, n, r["speaker"])
    return d
data = {k: load(v) for k, v in runs.items()}
snrs = ["clean", "10", "5", "0", "-5"]
pairs = [("P4","B1"),("P4","B3v2"),("X1","P4"),("X2","P4"),("X3","P4")]
B = 2000
for a, b in pairs:
    for s in snrs:
        keys = sorted(k for k in data[a] if k[1]==s and k in data[b])
        ea = [data[a][k][0] for k in keys]; eb = [data[b][k][0] for k in keys]; n = [data[a][k][1] for k in keys]
        N = sum(n); diff = 100*(sum(ea)-sum(eb))/N
        rng = random.Random(0); ds = []
        m = len(keys)
        for _ in range(B):
            idx = [rng.randrange(m) for _ in range(m)]
            ds.append(100*(sum(ea[i]-eb[i] for i in idx))/sum(n[i] for i in idx))
        ds.sort()
        print(f"{a}-{b} {s:>5} n={m} wer_a={100*sum(ea)/N:.2f} wer_b={100*sum(eb)/N:.2f} diff={diff:+.2f} CI=[{ds[int(0.025*B)]:+.2f},{ds[int(0.975*B)-1]:+.2f}] rel={100*(sum(eb)-sum(ea))/max(1,sum(eb)):.1f}%")
# per speaker P4 vs B1
print("per speaker P4 vs B1")
for s in snrs:
    out=[]
    for spk in sorted({v[2] for v in data["P4"].values()}):
        keys=[k for k,v in data["P4"].items() if k[1]==s and v[2]==spk]
        N=sum(data["P4"][k][1] for k in keys)
        out.append(f"{spk}: B1 {100*sum(data['B1'][k][0] for k in keys)/N:.2f} P4 {100*sum(data['P4'][k][0] for k in keys)/N:.2f}")
    print(s, " | ".join(out))
