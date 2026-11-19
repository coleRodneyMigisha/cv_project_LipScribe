Frame-level combination of fusion_wp (lips + audio) and fusion_wp_audio, chosen on validation: **loglinear@0.1** (weight on the audio-visual model). Test WER %:

| test | SNR | audio-only twin | fusion_wp | combined |
|---|---|---:|---:|---:|
| film_test | clean | 30.12 | 34.54 | **29.72** |
| film_test | 0 | 38.96 | 38.96 | **37.35** |
| film_test | -5 | 46.18 | 45.38 | **45.78** |
| film_test | -10 | 59.44 | 47.79 | **58.23** |
| ted_test | clean | 17.36 | 26.93 | **17.16** |
| ted_test | 0 | 23.35 | 29.87 | **22.79** |
| ted_test | -5 | 31.5 | 33.18 | **30.8** |
| ted_test | -10 | 43.83 | 36.25 | **43.53** |
