# Across sources

`tables/headline.md` puts the audio-only model and the audio + lips model side by side on each
source:

| Source | Audio only | Audio + lips |
|---|---|---|
| GRID | B1 | P4 |
| Films | twin_v5 | ws_v5 |
| TED | twin_v5 | ws_v5 |

**How to compare.** Absolute WERs are not comparable across sources: GRID has 51 words, while films
and TED are open vocabulary with Whisper labels. Compare the **gap** between the two models on each
source and how it grows as noise rises. That gap is the project's claim: the lips help most
when the audio is worst.

## Analysis still to do (F-11)

- [ ] A figure: WER vs SNR, one panel per source, audio-only vs audio + lips (GRID SNRs clean to
      −5 dB; films and TED clean to −10 dB).
- [ ] One paragraph for the report: where the lips help, where they cost a little (clean speech on
      real data), and where they fail (audio that does not belong to the face).
