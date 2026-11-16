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
