# Final Report Tracker (F-11)

**Owner:** Lisa Lusinga Namugenyi (editor) · **Reviewer:** Charis Opol
**Due:** Draft Thu 19 Nov, final Sat 21 Nov 2026. **Both dates have passed as of 23 Nov.**
**Depends on:** F-03, F-04, F-08 · **Replaces:** W4-11, W5-10, W6-05
**Template:** IEEE Access (`\documentclass{ieeeaccess}`)
**Source:** `documentation/5_submissions/final_report/` (`main.tex`, `sections/`, `references.bib`)

The report answers two questions:
1. Does lip-assisted fusion help on GRID (P4)?
2. Does it carry over to real speech (ws_v5)?

**Done when:** the PDF is final and every number matches its source.

Status is from the repository on 23 Nov 2026. Update it as work lands.

---

## 1. Dependencies

| Item | Status | Evidence |
|---|---|---|
| F-03 GRID results and fusion analysis | Landed | `analysis/` (feb2f585), `reports/gate_analysis/`, `reports/model_p_gate_vs_snr.png` |
| F-04 real-world results | Landed | 62117689, `reports/flix_v5/`, `reports/flix_v6/` |
| F-05 demo backend | Landed | b02c8138 |
| F-06 demo frontend | Landed | 2fc375a1 |
| F-07 | Not seen in git log | Ask Nehum |
| F-08 latency / real-time factor | Not seen in git log | Blocks §06 latency paragraph |
| F-09 testing / fresh-clone reproducibility | Not seen in git log | Blocks §07 |

- [ ] Confirm F-03 outputs are final (Lisa)
- [ ] Get F-07, F-08, F-09 status (Nehum)

---

## 2. Sections

Word counts are from 23 Nov. "Stub" means the file holds only a TODO comment.

| # | File | Section | Author | Words | Status |
|---|---|---|---|---|---|
| 00 | `00_abstract.tex` | Abstract (150–250 words) | Lisa (editor) | 77 | Stub |
| 01 | `01_introduction.tex` | Introduction | **Charis** | 58 | Stub |
| 02 | `02_related_work.tex` | Related work | **Charis** | 41 | Stub |
| 03 | `03_data.tex` | Data and noise: GRID, MUSAN, Flix film and TED pipeline | Faith | 360 | Partial; TODO on out-of-domain data |
| 04 | `04_pipelines.tex` | Visual and audio pipelines; real-world model; v4–v6 story | Cole | 1242 | Drafted; one TODO (log-Mel 80 in P4) |
| 05 | `05_model.tex` | Model P and training | **Charis** | 383 | Partial; TODO on ablations X1–X3 |
| 06 | `06_application.tex` | Application and latency (F-05 to F-08) | Nehum, frontend paragraph by Isabelle | 270 | Partial; TODOs for frontend and latency |
| 07 | `07_testing.tex` | Testing (F-09) | Nehum | 145 | Partial; TODO on final test counts |
| 08 | `08_results.tex` | GRID results, ablations, fusion analysis (F-03) | Lisa | 1624 | Drafted |
| 09 | `09_real_world.tex` | Real-world results (F-04), v6 as a negative result | Cole | 1097 | Drafted |
| 10 | `10_limitations.tex` | Limitations and future work | Lisa with Cole | 52 | Stub |
| 11 | `11_conclusion.tex` | Conclusion: answer both questions | Lisa (editor) | 19 | Stub |

### Checklist per section
- [ ] **00 Abstract.** Problem, approach, headline GRID and real-world numbers.
- [ ] **01 Introduction (Charis).** Captioning in noisy rooms and accessibility; why lips help when audio fails; the two research questions; contributions; outline.
- [ ] **02 Related work (Charis).** Lip reading; end-to-end AVSR; fusion, gating and reliability; noise-robust benchmarks; datasets. Use the reading list.
- [ ] **03 Data (Faith).** GRID, MUSAN mixing over SNR, Flix film and TED pipeline; out-of-domain data paragraphs.
- [ ] **04 Pipelines (Cole).** Resolve the log-Mel 80 TODO (`configs/p_init_b1_b2_v2.yaml`).
- [ ] **05 Model P (Charis).** One line each for ablations X1–X3 (`configs/x1_*.yaml`, `x2_*.yaml`, `x3_*.yaml`); cite fusion and gating work (Tao & Busso, Abdelaziz, Zhou, Zhang).
- [ ] **06 Application (Nehum, Isabelle).** Frontend (`demo/index.html`): recording, two captions, gate display. Latency and RTF from F-08.
- [ ] **07 Testing (Nehum).** Final test counts and the fresh-clone result from F-09.
- [ ] **08 GRID results (Lisa).** WER vs SNR, ablations, gate behaviour.
- [ ] **09 Real-world results (Cole).** twin_v5 vs ws_v5 in noise, lips-gain CIs, wrong-audio conditions, v6 negative result.
- [ ] **10 Limitations (Lisa, Cole).** GRID's closed 51-word vocabulary and studio recordings; synthetic noise; edge words; voices that don't match the face (AV sync signal, cite SyncNet); frontal faces only; CPU latency.
- [ ] **11 Conclusion.** One paragraph per research question.

---

## 3. References

`references.bib` currently has 11 entries: cooke2006, snyder2015, graves2006, stafylakis2017,
martinez2020, ma2023, chen2022wavlm, afouras2018, radford2023whisper, heafield2011, panayotov2015.

The full list is in `LipScribe_reading_list.md` (31 papers, with priority and target section).

### PDFs still to download (6)
- [ ] Shi et al. 2022, *AV-HuBERT* (ICLR): https://arxiv.org/abs/2201.02184
- [ ] Zhou et al. 2019, *Modality Attention for End-to-end AVSR* (ICASSP): https://doi.org/10.1109/ICASSP.2019.8683733
- [ ] Zhang et al. 2019, *Bimodal DFSMN with Multi-Condition Training and Dropout* (ICASSP): https://doi.org/10.1109/ICASSP.2019.8682566
- [ ] Chung & Zisserman 2017, *Out of Time: Automated Lip Sync in the Wild* (SyncNet): https://doi.org/10.1007/978-3-319-54427-4_19
- [ ] Cooke et al. 2006, *GRID corpus* (JASA): https://doi.org/10.1121/1.2229005 (already in the bib)
- [ ] Ma et al. 2022, *VSR for Multiple Languages in the Wild* (Nature MI): https://doi.org/10.1038/s42256-022-00550-z

### BibTeX
- [ ] Add BibTeX for every cited paper on the reading list (about 20 new entries)
- [ ] Use one key style, `firstauthorYEAR`, to match the existing entries
- [ ] Check that every `\cite{}` resolves and that `references.bib` has no unused entries

---

## 4. Number tracing

Every number in the PDF must trace to one of:
- `experiments/results.csv`
- `reports/flix_v5/`
- `reports/flix_v6/`

- [ ] §00 Abstract
- [ ] §04 Pipelines
- [ ] §05 Model P
- [ ] §06 Application (latency)
- [ ] §07 Testing (test counts)
- [ ] §08 GRID results
- [ ] §09 Real-world results
- [ ] §11 Conclusion
- [ ] Every table and figure caption

---

## 5. Close-out

- [ ] Lisa assembles the full draft
- [ ] Charis reviews
- [ ] Whole team proofreads (Sat)
- [ ] PDF compiles cleanly with no undefined references or citations
- [ ] Final PDF committed to `documentation/5_submissions/final_report/`
- [ ] Close W4-11, W5-10, W6-05 and F-11
