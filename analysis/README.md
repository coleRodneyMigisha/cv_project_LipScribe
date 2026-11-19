# LipScribe analysis

This folder holds the evaluation of LipScribe on **every data source**: the GRID corpus, films and
TED talks. It is self-contained. Every table is generated from the raw results by one script, and
the per-clip outputs it needs are stored here too.

**Owner:** Lisa Lusinga Namugenyi (cards F-03 and F-11). The tables were set up on 16 Nov 2026.

## Start here

1. Read `00_overview/` (15 minutes): which models exist, which test sets, and how scoring works.
   It also covers the films and TED retraining (Flix v4–v6).
2. Open the section for a source. Each `README.md` lists the tables, what they show, and the
   **analysis still to do** for the report.
3. To rebuild every table after new results arrive (for example X1–X3), run this from the
   repository root:

   ```
   python analysis/scripts/build_tables.py
   ```

   It uses only the Python standard library and takes about 30 s.

## Layout

| Folder | Source | What is in it |
|---|---|---|
| `00_overview/` | all | `models.md` (every model and its checkpoint), `test_sets.md` (every test set), `methods.md` (metrics, decoding, noise, statistics) |
| `01_grid/` | GRID corpus | Test WER/CER per model and SNR (greedy and grammar), missing-modality test, per-speaker WER, gate analysis, ablations X1–X3 |
| `02_films/` | Films and TV series | Flix v4–v6 results; edge-word vs inner-word error; per-film WER; paired bootstrap; audio that does not belong to the face |
| `03_ted/` | TED talks | The same tables as films, for the 67 held-out TED talks |
| `04_cross_source/` | all | One headline table: the audio-only model vs audio + lips on each source |
| `data/` | films, TED | Per-clip hypotheses of the v5 and v6 test runs (compressed CSV) |
| `scripts/` | — | `build_tables.py` (every table) and `snapshot_hyps.py` (how `data/` was made) |

## Rules for the analysis

- The test sets are **locked**. Every number here was produced once, on held-out speakers, films or
  talks. Do not tune anything on them. To try something new, use the validation sets.
- **Do not edit the generated tables** (`*/tables/`). Change the script and rebuild instead.
- Compare models **within a source**, not across sources. GRID has a 51-word vocabulary; films and
  TED are open vocabulary, so absolute WERs are not comparable (see `04_cross_source/`).
- Quote a number together with its source file, so the report can be checked.
