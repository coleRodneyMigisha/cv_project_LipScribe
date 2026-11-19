# Metrics and methods

- **WER and CER:** word and character error rate, computed as total edits ÷ total reference words
  (or characters) over the whole test set, not as the mean of per-clip rates. A WER above 100 %
  means many insertions.
- **Decoding:**
  - *greedy:* the most likely symbol per frame; all headline numbers use it.
  - *grammar:* GRID only; the output must fit GRID's six-slot grammar.
  - *beam + LM:* films and TED, v4 only; prefix beam search (width 32) with a 4-gram language
    model whose weights were chosen on validation.
- **Noise:** MUSAN noise, split by file into separate pools for training, validation and test.
  The test noise is fixed, so every model hears identical audio.
- **Missing modality:** a stream is removed at test time without retraining. In P, the learned
  vector e_miss takes its place.
- **Edge vs inner words (films and TED):** edge words are the first and last reference word of a
  clip. The v4 clips were cut tightly, so their edge words were often clipped; v5 adds 0.32 s of
  padding on each side.
- **Paired bootstrap:** resample whole groups (TED talks, films, GRID speakers) with replacement
  2,000 times and recompute the WER difference between two models on the same clips. A 95 %
  interval that excludes 0 means the difference is unlikely to be luck.
- **Gate g_t (GRID P4):** the per-frame audio weight in [0, 1]. A value of 1 means the model
  trusts the sound, and 0 means it trusts the lips.
