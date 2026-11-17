Greedy vs beam search (width 32) without and with the 4-gram LM `lm/flix_train_4gram.arpa`. alpha/beta chosen per model on validation. Test WER %:

| model | alpha | beta | test | SNR | greedy | beam, no LM | beam + LM |
|---|---:|---:|---|---|---:|---:|---:|
| twin | 0.3 | 1.0 | film_test | -10 | 59.44 | 58.23 | **57.43** |
| twin | 0.3 | 1.0 | film_test | -5 | 46.18 | 46.59 | **46.18** |
| twin | 0.3 | 1.0 | film_test | 0 | 38.96 | 39.36 | **39.36** |
| twin | 0.3 | 1.0 | film_test | clean | 30.12 | 29.72 | **29.72** |
| twin | 0.3 | 1.0 | ted_test | -10 | 43.83 | 43.08 | **42.38** |
| twin | 0.3 | 1.0 | ted_test | -5 | 31.5 | 30.63 | **30.29** |
| twin | 0.3 | 1.0 | ted_test | 0 | 23.35 | 22.63 | **22.3** |
| twin | 0.3 | 1.0 | ted_test | clean | 17.36 | 16.53 | **16.35** |
| av | 0.3 | 1.0 | film_test | -10 | 47.79 | 46.18 | **48.19** |
| av | 0.3 | 1.0 | film_test | -5 | 45.38 | 44.58 | **43.78** |
| av | 0.3 | 1.0 | film_test | 0 | 38.96 | 37.35 | **37.75** |
| av | 0.3 | 1.0 | film_test | clean | 34.54 | 33.33 | **34.54** |
| av | 0.3 | 1.0 | ted_test | -10 | 36.25 | 35.33 | **34.74** |
| av | 0.3 | 1.0 | ted_test | -5 | 33.18 | 32.13 | **31.68** |
| av | 0.3 | 1.0 | ted_test | 0 | 29.87 | 28.74 | **28.35** |
| av | 0.3 | 1.0 | ted_test | clean | 26.93 | 25.87 | **25.48** |
| visual | 0.3 | 1.0 | film_test | clean | 60.64 | 60.64 | **58.63** |
| visual | 0.3 | 1.0 | ted_test | clean | 48.25 | 48.22 | **48.17** |
| gated | 0.3 | 2.0 | film_test | -10 | 44.58 | 44.58 | **45.38** |
| gated | 0.3 | 2.0 | film_test | -5 | 42.97 | 41.37 | **40.96** |
| gated | 0.3 | 2.0 | film_test | 0 | 37.35 | 35.34 | **35.34** |
| gated | 0.3 | 2.0 | film_test | clean | 30.92 | 29.72 | **30.12** |
| gated | 0.3 | 2.0 | ted_test | -10 | 32.93 | 32.11 | **31.93** |
| gated | 0.3 | 2.0 | ted_test | -5 | 27.4 | 26.66 | **26.39** |
| gated | 0.3 | 2.0 | ted_test | 0 | 22.84 | 22.26 | **21.91** |
| gated | 0.3 | 2.0 | ted_test | clean | 18.51 | 17.87 | **17.87** |
