# Final report: notes on the 23 Nov draft

Drafted with Claude Code (AI assistance) from the repository and the PDFs in `papers/` (kept locally, not in git). The
section owners should check and rewrite their sections in their own words before submission. Any
AI-use statement the course requires must say that AI assistance was used.

## What changed

| Section | Before | Now |
|---|---|---|
| 00 Abstract | stub | full draft, about 200 words |
| 01 Introduction | stub | full draft: problem, two research questions, scope change, contributions, outline |
| 02 Related work | stub | five subsections, 23 new citations |
| 03 Data | TODO on real-world data | two paragraphs: mining filters, labels, harvester, v5 split counts |
| 04 Pipelines | TODO on log-Mel 80 | why P4 keeps log-Mel 80 (P4 predates the X5 decision) |
| 05 Model | TODO on X1–X3 | ablation subsection; gate tied to Tao & Busso, Sterpu, Meutzner |
| 08 GRID results | stub | protocol, Table I (all GRID models), Fig. 1 (WER vs SNR, three sources), bootstrap CIs, ablation table, missing modality, gate figure, comparison with published GRID results |
| 09 Real world | — | one paragraph added: effective SNR gain (1.2 dB TED, 3.6 dB films) |
| 10 Limitations | stub | full draft |
| 11 Conclusion | stub | one paragraph per research question |
| references.bib | 11 entries | 34 entries; every entry is cited and every citation resolves |

Sections 06 and 07 were not touched (they wait on F-08 and F-09).

New files:
- `figures/wer_vs_snr.pdf`/`.png` and `figures/make_wer_snr.py` (Fig. 1)
- `figures/gate_boxplot.png` (copied from `reports/gate_analysis/`)
- `analysis/scripts/grid_bootstrap.py` → `analysis/01_grid/bootstrap.txt` (paired bootstrap, per-speaker WER)

## IEEE Access template (official `access.tex` sample)

Applied:
- **Abstract.** One paragraph of about 235 words. It contains no abbreviations, references or equations, and no model labels such as P4 or ws_v5.
- **Keywords** are in alphabetical order.
- **Abbreviations** are defined at first use in the body: CTC in §01, MLP and LM in §04.
- **Layout.** `\titlepgskip=-15pt`, as in the template.
- **Acknowledgment.** An unnumbered section, reusing the proposal's wording.
- **Already compliant.** "Fig." in running text, table captions above and figure captions below, `\EOD`, no footnotes.

Not applied (team decision):
- **Spelling.** The template follows American English ("acknowledgment", periods inside quotation marks). Every section of the report uses British spelling ("initialisation", "behaviour"). Choose one and apply it to the whole report.
- **Serial comma.** The template prefers the serial comma ("A, B, and C"); the drafts mostly omit it.
- **Author biographies** (`IEEEbiography`) are optional and are not included.

## Author queries

- **[AQ4] Acknowledgment.** Confirm who provided the GPU workstation, and whether the course or the department should be thanked.

- **[AQ1] §08, ablation training budget.** `experiments/results.csv` shows that X1–X3 ran 6 epochs, as P4 did. Confirm that they had the same 30-minute budget as P1–P4 (Lisa or Cole).
- **[AQ2] §08, X2 against B3 v2.** Both models concatenate the streams, but X2 is worse (for example 5.89 % against 5.21 % at −5 dB). The text says they differ in their projection layers and training length (6 against 7 epochs). Confirm what else differs (Cole).
- **[AQ3] §10, latency.** No CPU latency or real-time factor yet (F-08, Nehum).
- **§03 test-set size.** §03 gives 3,999 test utterances, but every evaluation scored 3,998 clips per SNR (`hypotheses.csv`, `gate_analysis/summary.md`). Find the dropped clip and reconcile the two numbers (Faith, Lisa).
- **§04, V1 against V2.** §04 says that V2 "learned faster and generalised better". However, lips-only B2 v2 is worse than B2 v1 on test (32.20 % against 23.04 %); V2 wins only inside the fused model (P4 beats P3). Reword (Cole).
- **§08 draft.** The tracker listed §08 at 1,624 words, but the file on `charis` was a stub. If Lisa has a draft elsewhere, merge it with this one.
- **Citations read from title only.** No PDFs were downloaded for Zhou 2019, Zhang 2019 or Chung & Zisserman 2017 (SyncNet). Their metadata was checked on Crossref, and the text claims only what their titles state. Read them before final submission.

## Verification

- Every GRID number comes from `analysis/01_grid/tables/*.md`, `analysis/01_grid/bootstrap.txt` or `reports/gate_analysis/summary.md`; every real-world number from Tables III and IV (`reports/flix_v5/`, `reports/flix_v6/`). The bootstrap reproduces the table WERs exactly.
- Effective SNR gains use linear interpolation of the beam + LM WERs in Table III.
- Each related-work finding was checked against the paper's abstract or text in `papers/`. LCANet's overlapped-speaker split was confirmed in its text (255 test sentences per speaker).
- All 23 new BibTeX entries were checked against Crossref by DOI; LipNet was checked against arXiv.
- **Compile.** `LipScribe_final_report_DRAFT.pdf` (10 pages) was built with Tectonic 0.15 (XeTeX), using the IEEE Access class from a GitHub mirror (`erwinrmendez/ieee-access-template`). The mirror's class is edited: logos are read from `images/`. The result has no errors and no undefined references or citations. It is a **preview, not the submission build**:
  - Tectonic runs XeTeX, but the class's `spotcolor` package needs pdfTeX. The scratch copy therefore swaps the Pantone header blue for its CMYK values and loads T1 font encoding. The repository files are unchanged.
  - The submission PDF should be built with pdfLaTeX, using the class files from the proposal's Overleaf project.
- **Changes the real class needs** (made in the repository):
  - Figures use the class's `\Figure` macro. A plain `figure` environment fails, because `\xfigwd` is undefined.
  - Fig. 1's caption is wrapped in a `\parbox`. The class sets full-width captions in a one-line `l` column, which cut it off.
  - `\allowbreak{}` follows each `/` and `\_` inside `\texttt` paths, in every section. Unbreakable paths overflowed the column by up to 119 pt.
  - Table II's column spacing is now 1.8 pt.
- **Remaining warning.** The running head (`\markboth`, the full title as in the proposal) is 9 pt too wide.
- No contractions, em dashes or ellipses appear in the new text.
