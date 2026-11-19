# Test sets

All test sets are held out: no speaker, film or talk in them was used for training or model choice.

| Source | Test set | Clips | Held out by | Labels | Conditions |
|---|---|---:|---|---|---|
| GRID | `data/splits.csv` test | 3,999 per SNR | 4 speakers | GRID transcripts (six-word grammar) | clean, 10, 5, 0, −5 dB (fixed noisy copies) |
| Films | `film_v4` (v4 clips) | 43 | 3 titles: Challengers, Little Women, The Drama | human-checked subtitles | clean, 0, −5, −10 dB |
| Films | `film_v5` / `film` (re-cut) | 41 | the same 3 titles | Whisper word timestamps | clean, 0, −5, −10 dB; silent, talk-over, replace (v6) |
| TED | `ted_v4` (v4 clips) | 1,122 | 67 talks | Whisper large-v3-turbo on each clip's own audio | clean, 0, −5, −10 dB |
| TED | `ted_v5` / `ted` (re-cut) | 1,094 | the same 67 talks | Whisper word timestamps | clean, 0, −5, −10 dB; silent, talk-over, replace (v6) |

Notes:
- `film_v5`, `ted_v5` (in the v5 run) and `film`, `ted` (in the v6 run) are the **same clips**.
- **Labels.** TED's own subtitle timings did not match the cut clips, so on 13 Nov every TED clip was relabelled with Whisper from its own audio (`reports/flix_v4/decision_log.md`). The v4 film test is the only set with human-checked labels. For this reason no Whisper-based model is used, as it would be scored against its own output.
- **The film test set is small** (41 clips, 3 titles), so its confidence intervals are wide.
  TED (1,094 clips, 67 talks) is the main real-world result.
- **Wrong-audio conditions (v6):**
  - *silent:* the audio is zeros, as if the speaker only mouths.
  - *talk-over:* another talk's speech is mixed in at 0 dB.
  - *replace:* only another talk's speech is heard.
  - In all three, the reference is always the on-screen speaker's words.
