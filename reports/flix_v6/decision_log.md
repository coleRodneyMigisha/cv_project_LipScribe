# Flix v6 decision log

## Spec (written 15 Nov 2026, before any v6 run)

- **Why.** In use, the audio may not belong to the face on screen: the speaker may only mouth
  words (silence), or mouth them while other people talk. v4/v5 never trained on that: their
  noise is MUSAN non-speech only, and nothing checks that the voice matches the lips. With the
  audio stream switched off the v4 warm start reads lips at 48 % WER (TED) / 61 % (films).
- **Change (one).** Retrain the v5 warm start (fusion_gated_ws: same config, same frozen v5 twin
  and v5 fusion_wp streams) with data.interference: per training clip, p 0.10 silent (zeros, or
  MUSAN noise alone at −50..−30 dBFS), p 0.10 replace (another talk's / film's speech alone),
  p 0.15 talk-over (other speech at SIR −5..10 dB under the usual MUSAN noise). The label stays
  the on-screen speaker's words. best.pt is picked on the usual val sets plus ted/film × silent,
  replace (data.val_conditions).
- **Locked evaluation.** v5 test sets, once: clean, 0, −5, −10 dB; silent; talk-over at 0 dB;
  replace (scripts/flix_v6_eval.py). Models: twin_v5, ws_v5, ws_v6, and lips_v5 (ws_v5 with
  its audio switched off) as the lip-only reference.
- **Baseline.** ws_v5 under the same conditions.
- **Expected.** Best: ws_v6 on silent and replace within 5 points of lips_v5, talk-over well
  below ws_v5, ordinary conditions unchanged. Realistic: replace and silent 10–20 points above
  lips_v5 (the gate has to learn from features alone that the voice is not the speaker's),
  clean +0.5–1 point. Worst: the gate cannot tell a mismatched voice from a matching one, and
  replace stays near ws_v5.
- **Success.** ws_v6 beats ws_v5 on replace and silent (TED and films, CIs excluding 0), and is
  no more than 1 point worse than ws_v5 on clean TED.
- **Stop rule.** One training run; no tuning without the user. If the gate cannot detect a
  mismatched voice, the next step is an explicit audio-visual sync signal, not more of this.
- 2026-11-15 06:01 Waiting for Flix v5 (reports/flix_v5/ALL_DONE).
- 2026-11-15 08:01 Waiting for Flix v5 (reports/flix_v5/ALL_DONE).
- 2026-11-15 10:01 Waiting for Flix v5 (reports/flix_v5/ALL_DONE).
- 2026-11-15 12:01 Waiting for Flix v5 (reports/flix_v5/ALL_DONE).
- 2026-11-15 14:01 Waiting for Flix v5 (reports/flix_v5/ALL_DONE).
- 2026-11-15 16:01 Waiting for Flix v5 (reports/flix_v5/ALL_DONE).
- 2026-11-15 18:01 Waiting for Flix v5 (reports/flix_v5/ALL_DONE).
- 2026-11-15 20:01 Waiting for Flix v5 (reports/flix_v5/ALL_DONE).
- 2026-11-15 22:01 Waiting for Flix v5 (reports/flix_v5/ALL_DONE).
- 2026-11-16 00:01 Waiting for Flix v5 (reports/flix_v5/ALL_DONE).
- 2026-11-16 02:01 Waiting for Flix v5 (reports/flix_v5/ALL_DONE).
- 2026-11-16 04:01 Waiting for Flix v5 (reports/flix_v5/ALL_DONE).
- 2026-11-16 06:01 Waiting for Flix v5 (reports/flix_v5/ALL_DONE).
- 2026-11-16 08:01 Waiting for Flix v5 (reports/flix_v5/ALL_DONE).
- 2026-11-16 09:41 Waiting for Flix v5 (reports/flix_v5/ALL_DONE).
- 2026-11-16 10:53 Config written: configs/flix_v6/fusion_gated_ws.yaml.
- 2026-11-16 10:53 Stage fusion_gated_ws started (configs/flix_v6/fusion_gated_ws.yaml).
- 2026-11-16 12:16 Stage fusion_gated_ws done: best epoch 5/10: mean val WER 51.7% (film_clean 53.2%, film_0 59.2%, film_-5 59.7%, film_-10 66.8%, ted_clean 33.8%, ted_0 34.8%, ted_-5 36.5%, ted_-10 37.6%, film_silent 76.0%, ted_silent 43.1%, film_replace 76.0%, ted_replace 44.1%). Run: runs/flix_v6/flix_v6_fusion_gated_ws_20261116-105310

## Test results, 2026-11-16 12:20

WER % (greedy). Reference = the on-screen speaker's words.

| test set | condition | twin_v5 | ws_v5 | ws_v6 | lips_v5 |
|---|---|---:|---:|---:|---:|
| film | clean | 29.3 | 30.1 | 48.4 | 64.5 |
| film | 0 | 41.0 | 35.2 | 50.8 | – |
| film | -5 | 53.9 | 43.4 | 52.7 | – |
| film | -10 | 65.6 | 50.8 | 58.2 | – |
| film | silent | 100.0 | 67.2 | 63.7 | – |
| film | talkover | 94.5 | 72.7 | 62.5 | – |
| film | replace | 123.8 | 108.2 | 70.3 | – |
| ted | clean | 17.3 | 18.2 | 35.4 | 44.0 |
| ted | 0 | 23.5 | 22.3 | 36.4 | – |
| ted | -5 | 31.1 | 26.6 | 37.2 | – |
| ted | -10 | 43.8 | 30.6 | 38.5 | – |
| ted | silent | 100.0 | 45.6 | 43.7 | – |
| ted | talkover | 82.4 | 59.8 | 40.5 | – |
| ted | replace | 117.5 | 103.3 | 44.1 | – |

Paired bootstrap ws_v6 − ws_v5 (WER points, 95 % CI; whole talks resampled for TED):

| test set | condition | diff | CI |
|---|---|---:|---|
| film | clean | +18.36 | [+11.21, +25.20] |
| film | 0 | +15.62 | [+9.77, +21.58] |
| film | -5 | +9.38 | [+2.87, +16.13] |
| film | -10 | +7.42 | [+2.80, +12.39] |
| film | silent | -3.52 | [-6.59, -0.71] |
| film | talkover | -10.16 | [-24.66, +1.74] |
| film | replace | -37.89 | [-50.22, -27.31] |
| ted | clean | +17.19 | [+13.71, +20.64] |
| ted | 0 | +14.10 | [+11.22, +17.20] |
| ted | -5 | +10.63 | [+8.13, +13.22] |
| ted | -10 | +7.96 | [+6.01, +9.93] |
| ted | silent | -1.90 | [-2.53, -1.20] |
| ted | talkover | -19.24 | [-25.82, -13.27] |
| ted | replace | -59.23 | [-63.45, -54.62] |
- 2026-11-16 12:20 Flix v6 finished.
