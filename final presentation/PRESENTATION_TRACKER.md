# Final Presentation Tracker

**Owner:** Charis Opol · **Reviewer:** Lisa Lusinga Namugenyi
**Due:** Fri 20 Nov 2026. **This date has passed as of 23 Nov.**
**Depends on:** F-03, F-04 · **Replaces:** W6-06 · **Estimate:** 5 h

Reuse the 12 Nov progress presentation wherever it still holds:
`documentation/2_implementation/presentation/`
(`LipScribe_Implementation_Presentation.pptx`, `_5min.pptx`, and their presenter notes).

**Done when:** the deck is reviewed and every member knows their slides.

Status is from the repository on 23 Nov 2026.

---

## 1. Inputs

| Input | Status | Source |
|---|---|---|
| F-03 GRID results, ablations, gate analysis | Landed | `analysis/`, `reports/gate_analysis/` |
| F-04 real-world results | Landed | `reports/flix_v5/`, `reports/flix_v6/` |
| 12 Nov progress deck | Available | `documentation/2_implementation/presentation/` |
| Report draft (for matching numbers) | Partly drafted | `documentation/5_submissions/final_report/` |

- [ ] Go through the 12 Nov deck and mark each slide as keep, update or drop

---

## 2. Slides

| # | Section | Content | Figures / sources | Speaker | Status |
|---|---|---|---|---|---|
| 1 | Problem and motivation | Captioning in noisy rooms; why the lips help; the two research questions | 12 Nov deck | | [ ] |
| 2 | Architecture: P4 | B1 audio, B2 lips (3D-conv + ResNet-18 + BiLSTM), cross-attention, gate, CTC | `05_model.tex`, `docs/resnet_vs_conv3d_justification.md` | | [ ] |
| 3 | Architecture: ws_v5 | Auto-AVSR `lrs3vox2_base` checkpoint, decoder chosen on validation | `04_pipelines.tex`, `reports/flix_v5/` | | [ ] |
| 4 | GRID WER vs SNR | B1, B2, B3 and P across SNR | `reports/snr_comparison_curves.png`, `reports/all_models_snr_and_summary.png`, `experiments/results.csv` | | [ ] |
| 5 | Ablations | X1–X3, no-audio and no-visual conditions | `08_results.tex`, `experiments/results.csv` | | [ ] |
| 6 | Gate behaviour | The gate shifts toward the lips as SNR drops | `reports/model_p_gate_vs_snr.png`, `reports/gate_analysis/` | | [ ] |
| 7 | Real-world results | twin_v5 vs ws_v5 in noise; lips-gain CIs; wrong-audio conditions | `reports/flix_v5/`, `09_real_world.tex` | | [ ] |
| 8 | v6 negative result | What was tried and why it didn't help | `reports/flix_v6/` | | [ ] |
| 9 | Demo | Live and clip captions in the web app | `demo/`, F-05 / F-06 | | [ ] |
| 10 | Limitations and future work | Closed vocabulary, synthetic noise, edge words, voice and face mismatch | `10_limitations.tex` | | [ ] |

Add or merge slides as needed. Keep the table in step with the deck.

---

## 3. Speakers (all six members)

| Member | Slides |
|---|---|
| Charis Opol | |
| Lisa Lusinga Namugenyi | |
| Cole Rodney Migisha | |
| Ainomugisha Faith | |
| Ansasire Nehum Naamaye | |
| Wakisa Isabelle | |

A suggested split that follows report authorship: Charis takes motivation and P4 architecture;
Lisa takes GRID results, ablations and gate; Cole takes ws_v5 and real-world results, including v6;
Faith takes data and noise; Nehum takes the demo, latency and testing; Isabelle takes the frontend
part of the demo.

- [ ] Every slide has a speaker
- [ ] Every member has at least one slide

---

## 4. Consistency and close-out

- [ ] Every number on a slide matches the report and its source (`experiments/results.csv`, `reports/flix_v5/`, `reports/flix_v6/`)
- [ ] Figures are the same files the report uses
- [ ] Demo tested on the presentation machine, with a recorded fallback video
- [ ] Lisa reviews the deck
- [ ] Each member confirms their slides
- [ ] Rehearsal with timing
- [ ] Final deck committed
- [ ] Close W6-06
