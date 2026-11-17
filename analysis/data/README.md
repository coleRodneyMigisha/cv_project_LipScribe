# Per-clip data

| File | From | Rows |
|---|---|---|
| `flix_v5_hyps.csv.gz` | `reports/flix_v5/hyps.json` on the Alienware (`scripts/flix_v5_eval.py`) | models twin_v4, ws_v4, twin_v5, ws_v5 × sets film_v4, ted_v4, film_v5, ted_v5 × clean, 0, −5, −10 dB |
| `flix_v6_hyps.csv.gz` | `reports/flix_v6/hyps.json` on the Alienware (`scripts/flix_v6_eval.py`) | models twin_v5, ws_v5, ws_v6 (+ lips_v5 clean) × sets film, ted × clean, 0, −5, −10, silent, talkover, replace |

**Columns:** model, test_set, condition, utt_id, ref, hyp.

**Reading a file:**
- In Python: `pandas.read_csv("analysis/data/flix_v6_hyps.csv.gz")`.
- From the shell: `zcat flix_v6_hyps.csv.gz | head`.

**Grouping clips:**
- Film clip ids are `flix_<title>_<n>`.
- TED clip ids are `flix_ted_test_<talk>_t<time>`.
- `build_tables.py` (`group_of`) groups clips by film title or TED talk this way.

Made with `analysis/scripts/snapshot_hyps.py`. These are the locked test outputs, so do not
regenerate them unless the models change.
