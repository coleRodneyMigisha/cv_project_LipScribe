"""Turn a Flix evaluation cache (reports/flix_v5/hyps.json or flix_v6/hyps.json, written by
scripts/flix_v5_eval.py / flix_v6_eval.py on the Alienware) into a compressed CSV for analysis/data/.

    python analysis/scripts/snapshot_hyps.py <hyps.json> analysis/data/flix_v5_hyps.csv.gz

Columns: model, test_set, condition, utt_id, ref, hyp (one row per clip, model and condition).
"""
import csv, gzip, json, sys

src, dst = sys.argv[1], sys.argv[2]
data = json.load(open(src))
with gzip.open(dst, "wt", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "test_set", "condition", "utt_id", "ref", "hyp"])
    for key, v in data.items():
        model, test_set, condition = key.split("|")
        for uid, ref, hyp in zip(v["ids"], v["refs"], v["hyps"]):
            w.writerow([model, test_set, condition, uid, ref, hyp])
print("wrote", dst)
