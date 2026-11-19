Greedy vs beam search (width 32) without and with the 4-gram LM `lm/flix_libri_4gram.arpa`. alpha/beta chosen per model on validation. Test WER %:

| model | alpha | beta | test | SNR | greedy | beam, no LM | beam + LM |
|---|---:|---:|---|---|---:|---:|---:|
| twin | 0.3 | 1.0 | film_test | -10 | 59.44 | 57.83 | **57.83** |
| twin | 0.3 | 1.0 | film_test | -5 | 46.18 | 46.18 | **46.59** |
| twin | 0.3 | 1.0 | film_test | 0 | 38.96 | 38.96 | **38.55** |
| twin | 0.3 | 1.0 | film_test | clean | 30.12 | 29.72 | **28.51** |
| twin | 0.3 | 1.0 | ted_test | -10 | 43.83 | 42.95 | **42.71** |
| twin | 0.3 | 1.0 | ted_test | -5 | 31.5 | 30.51 | **30.04** |
| twin | 0.3 | 1.0 | ted_test | 0 | 23.35 | 22.41 | **21.65** |
| twin | 0.3 | 1.0 | ted_test | clean | 17.36 | 16.36 | **15.9** |
| av | 0.3 | 2.0 | film_test | -10 | 47.79 | 46.18 | **46.99** |
| av | 0.3 | 2.0 | film_test | -5 | 45.38 | 44.58 | **41.77** |
| av | 0.3 | 2.0 | film_test | 0 | 38.96 | 37.35 | **38.96** |
| av | 0.3 | 2.0 | film_test | clean | 34.54 | 33.33 | **33.33** |
| av | 0.3 | 2.0 | ted_test | -10 | 36.25 | 35.25 | **34.57** |
| av | 0.3 | 2.0 | ted_test | -5 | 33.18 | 32.01 | **31.3** |
| av | 0.3 | 2.0 | ted_test | 0 | 29.87 | 28.58 | **27.69** |
| av | 0.3 | 2.0 | ted_test | clean | 26.93 | 25.78 | **24.68** |
| visual | 0.3 | 1.0 | film_test | clean | 60.64 | 60.64 | **59.44** |
| visual | 0.3 | 1.0 | ted_test | clean | 48.25 | 47.99 | **47.17** |
| gated | 0.5 | 2.0 | film_test | -10 | 44.58 | 44.58 | **43.78** |
| gated | 0.5 | 2.0 | film_test | -5 | 42.97 | 41.37 | **40.56** |
| gated | 0.5 | 2.0 | film_test | 0 | 37.35 | 35.34 | **35.34** |
| gated | 0.5 | 2.0 | film_test | clean | 30.92 | 29.72 | **30.12** |
| gated | 0.5 | 2.0 | ted_test | -10 | 32.93 | 31.93 | **31.37** |
| gated | 0.5 | 2.0 | ted_test | -5 | 27.4 | 26.54 | **25.91** |
| gated | 0.5 | 2.0 | ted_test | 0 | 22.84 | 22.04 | **21.42** |
| gated | 0.5 | 2.0 | ted_test | clean | 18.51 | 17.63 | **17.25** |
