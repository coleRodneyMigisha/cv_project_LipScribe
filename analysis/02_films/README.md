# Films and TV series

Clips of one visible, frontal speaker mined from subtitled films and series. The test set is 41 clips from 3 held-out titles (Challengers, Little Women, The Drama). It is small, so intervals are wide. See `../00_overview/` for the models (twin_v5, ws_v5, ws_v6, lips_v5) and the test sets.

## Tables (`tables/`, generated)

| File | What it shows |
|---|---|
| `main_results.md` | **Start here.** Test WER % of every v5/v6 model at clean, 0, −5, −10 dB and with wrong audio (silent, talk-over, replace) |
| `paired_bootstrap.md` | ws_v5 − twin_v5 (do the lips help?) and ws_v6 − ws_v5, with 95 % intervals; whole film titles are resampled |
| `edge_vs_inner_words.md` | Error on the first/last word vs the inner words, for the old tight clips (v4 sets) and the padded re-cut (v5 sets) |
| `per_group_wer.csv` | WER per film title for every v6-run model, clean and −10 dB |
| `test_results_long.csv` | Everything in one long table, including the v4 models and the v4 beam + LM results |

## Raw data

- **Per-clip hypotheses** (one row per clip, model and condition): `../data/flix_v5_hyps.csv.gz`
  (v5 run: twin/ws v4 and v5 on the old and re-cut sets) and `../data/flix_v6_hyps.csv.gz` (v6 run:
  twin_v5, ws_v5, ws_v6, lips_v5 with all seven conditions).
- **Decision logs**, with the spec, expectations and test results of each version:
  `reports/flix_v4/decision_log.md`, `reports/flix_v5/decision_log.md`,
  `reports/flix_v6/decision_log.md`.

## Analysis still to do (F-03, F-11)

- [ ] Do the lips help on real speech? Use `paired_bootstrap.md` (ws_v5 − twin_v5) at each SNR,
      and say where the gain starts.
- [ ] Clean-speech cost: ws_v5 is slightly worse than twin_v5 on clean audio. Say by how much,
      and whether the interval excludes 0.
- [ ] Spread across film titles (`per_group_wer.csv`): which film titles are hardest, and does the lip
      gain hold for most of them?
- [ ] Edge words (`edge_vs_inner_words.md`): did the 0.32 s padding of v5 help?
- [ ] Wrong audio (v6 conditions): ws_v5 fails when the voice is not the speaker's. Use this as
      the limitation, with ws_v6 as the attempted fix.
- [ ] Optional: the most common substitutions (ref → hyp word pairs) from the per-clip file.
