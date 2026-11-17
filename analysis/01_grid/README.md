# GRID corpus

Studio recordings of 33 speakers reading six-word commands from a 51-word grammar. The test set has
4 held-out speakers (3,999 clips at each of five noise levels). See `../00_overview/`.

## Tables (`tables/`, generated)

| File | What it shows |
|---|---|
| `test_wer_greedy.md` | Test WER % of every GRID model at clean, 10, 5, 0, −5 dB (greedy) |
| `test_wer_grammar.md` | The same with the GRID grammar decoder |
| `missing_modality.md` | B3 v2 and P1–P4 with the audio or the video removed at test time |
| `per_speaker_wer.csv` | WER per test speaker and SNR for B1, B3 v2 and P4 |
| `test_results_long.csv` | Every number above in one long table (model, decode, condition, SNR, WER, CER) |

## Other GRID results (already on `main`)

- **Gate analysis of P4** (W4-06): `reports/gate_analysis/`. It holds a box plot of the mean gate
  per SNR, per-frame gate examples, an attention heatmap, the per-utterance CSV and `summary.md`.
  The mean gate falls from 0.804 (clean) to 0.691 (−5 dB), and Spearman ρ = 0.47.
- **Per-clip hypotheses** of every run: `reports/<run>/hypotheses.csv`, with columns utt_id, snr,
  speaker, ref, hyp, wer, cer and gate. Grammar runs are in `reports/<run>_grammar/`.
- **Evaluation reports and curves:** `reports/<run>/evaluation_report.md` and
  `test_snr_curves.png`. All-model plot: `reports/all_models_snr_and_summary.png`.
- **Ablations X1–X3:** trained on 16 Nov. Their rows appear in `experiments/results.csv`; then
  rerun `python analysis/scripts/build_tables.py`.

## Analysis still to do (F-03)

- [ ] Paired bootstrap 95 % CI for P4 vs B1 at each SNR. Resample speakers (or sentences), using
      `reports/<run>/hypotheses.csv`.
- [ ] Ablation table: P4 vs X1, X2, X3 at five SNRs. Write one sentence per ablation on what the
      removed part contributes.
- [ ] Per-speaker breakdown (`per_speaker_wer.csv`): does any speaker gain much more from the
      lips than the others?
- [ ] Gate paragraph from `reports/gate_analysis/summary.md`: does the model lean on the lips as
      noise rises?
- [ ] One paragraph per finding for the report (`documentation/5_submissions/final_report/sections/08_results.tex`).
