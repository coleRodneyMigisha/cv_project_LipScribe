# LipScribe: Related-Work Reading List

Papers to download for the final report (IEEE Access template). There are 24 from the Scopus export
(`Scopus_shortlist.csv`) and 7 more that are not in Scopus or that the query missed.

**Priority:** **A** = must cite · **B** = should cite · **C** = optional / use if space allows
**Section:** Intro = Introduction and related work · Model P = Model P and training ·
Data = Data and noise · RW = real-world model (ws_v5) · Lim = Limitations

---

## 1. Lip reading (visual front-end, B2)

- [ ] **[A]** Wand, Koutník, Schmidhuber. *Lipreading with Long Short-Term Memory.* ICASSP 2016. 209 cites.
  https://doi.org/10.1109/ICASSP.2016.7472852. Intro, Model P. Early end-to-end lip reading on GRID.
- [ ] **[A]** Stafylakis, Tzimiropoulos. *Combining Residual Networks with LSTMs for Lipreading.* Interspeech 2017. 201 cites.
  https://doi.org/10.21437/Interspeech.2017-85. Model P. The 3D-conv + ResNet-18 + BiLSTM front-end that V2/P4 uses.
- [ ] **[B]** Xu, Li, Cassimatis, Wang. *LCANet: End-to-End Lipreading with Cascaded Attention-CTC.* FG 2018. 131 cites.
  https://doi.org/10.1109/FG.2018.00088. Intro. Sentence-level lip reading on GRID.
- [ ] **[A]** Martinez, Ma, Petridis, Pantic. *Lipreading Using Temporal Convolutional Networks.* ICASSP 2020. 326 cites.
  https://doi.org/10.1109/ICASSP40776.2020.9053841. Model P. Lip-reading front-end baseline; cited in our encoder justification.

## 2. End-to-end AVSR (P4 and ws_v5)

- [ ] **[A]** Afouras, Chung, Senior, Vinyals, Zisserman. *Deep Audio-Visual Speech Recognition.* IEEE TPAMI 2022. 670 cites.
  https://doi.org/10.1109/TPAMI.2018.2889052. Intro. The core AVSR paper; LRS2.
- [ ] **[A]** Petridis, Stafylakis, Ma, Cai, Tzimiropoulos, Pantic. *End-to-End Audiovisual Speech Recognition.* ICASSP 2018. 315 cites.
  https://doi.org/10.1109/ICASSP.2018.8461326. Intro. First end-to-end AVSR; shows the gain in noise.
- [ ] **[B]** Petridis et al. *Audio-Visual Speech Recognition with a Hybrid CTC/Attention Architecture.* SLT 2018. 155 cites.
  https://doi.org/10.1109/SLT.2018.8639643. Model P. CTC-based AVSR.
- [ ] **[A]** Ma, Petridis, Pantic. *End-to-End Audio-Visual Speech Recognition with Conformers.* ICASSP 2021. 309 cites.
  https://doi.org/10.1109/ICASSP39728.2021.9414567. Intro, RW. Current standard architecture.
- [ ] **[A]** Ma, Haliassos, Fernandez-Lopez, Chen, Petridis, Pantic. *Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels.* ICASSP 2023. 192 cites.
  https://doi.org/10.1109/ICASSP49357.2023.10096889. RW. **ws_v5 is built on its `lrs3vox2_base` checkpoint.**
- [ ] **[B]** Shi, Hsu, Lakhotia, Mohamed. *Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction (AV-HuBERT).* ICLR 2022. 378 cites.
  https://arxiv.org/abs/2201.02184. Intro, RW. Self-supervised AV pre-training.
- [ ] **[B]** Shi, Hsu, Mohamed. *Robust Self-Supervised Audio-Visual Speech Recognition.* Interspeech 2022. 72 cites.
  https://doi.org/10.21437/Interspeech.2022-99. Intro. AV-HuBERT in babble noise.

## 3. Fusion, gating and modality reliability (gate analysis)

- [ ] **[A]** Tao, Busso. *Gating Neural Network for Large Vocabulary Audiovisual Speech Recognition.* IEEE/ACM TASLP 2018. 57 cites.
  https://doi.org/10.1109/TASLP.2018.2815268. Model P. **Closest to our gated fusion**; addresses AV hurting clean speech.
- [ ] **[B]** Meutzner, Ma, Nickel, Schymura, Kolossa. *Improving AVSR Using DNNs with Dynamic Stream Reliability Estimates.* ICASSP 2017. 35 cites.
  https://doi.org/10.1109/ICASSP.2017.7953172. Model P. Weighting each modality by its reliability.
- [ ] **[A]** Abdelaziz. *Comparing Fusion Models for DNN-Based Audiovisual Continuous Speech Recognition.* IEEE/ACM TASLP 2018. 48 cites.
  https://doi.org/10.1109/TASLP.2017.2783545. Model P. Early vs. late fusion, for the concat and X1–X3 ablations.
- [ ] **[B]** Sterpu, Saam, Harte. *Attention-Based Audio-Visual Fusion for Robust Automatic Speech Recognition.* ICMI 2018. 77 cites.
  https://doi.org/10.1145/3242969.3243014. Model P. Cross-attention fusion.
- [ ] **[A]** Zhou, Yang, Chen, Wang, Jia. *Modality Attention for End-to-End Audio-Visual Speech Recognition.* ICASSP 2019. 77 cites.
  https://doi.org/10.1109/ICASSP.2019.8683733. Model P. Modality weights that change with SNR.
- [ ] **[A]** Zhang, Lei, Ma, Xie. *Robust AVSR Using Bimodal DFSMN with Multi-Condition Training and Dropout Regularization.* ICASSP 2019. 42 cites.
  https://doi.org/10.1109/ICASSP.2019.8682566. Model P. Multi-condition (noise) training and modality dropout, as in our training.
- [ ] **[B]** Sterpu, Saam, Harte. *How to Teach DNNs to Pay Attention to the Visual Modality in Speech Recognition.* IEEE/ACM TASLP 2020. 35 cites.
  https://doi.org/10.1109/TASLP.2020.2980436. Model P. Why models learn to ignore the lips.
- [ ] **[B]** Hong, Kim, Choi, Ro. *Watch or Listen: Robust AVSR with Visual Corruption Modeling and Reliability Scoring.* CVPR 2023. 69 cites.
  https://doi.org/10.1109/CVPR52729.2023.01801. Model P, Lim. Reliability scoring per modality.
- [ ] **[A]** Lin, Harte. *Uncovering the Visual Contribution in Audio-Visual Speech Recognition.* ICASSP 2025. 8 cites.
  https://doi.org/10.1109/ICASSP49660.2025.10888423. Intro, RW. **Asks our question: do the lips actually help?** (Uses Auto-AVSR.)
- [ ] **[C]** Lim, Kim, Kim, et al. *Improving Noise Robust AVSR via Router-Gated Cross-Modal Feature Fusion.* ASRU 2025. 1 cite.
  https://doi.org/10.1109/ASRU65441.2025.11434748. Model P. Recent gated fusion on LRS3. Low citations because it is new.

## 4. Noise, sync and survey

- [ ] **[B]** Abdelaziz. *NTCD-TIMIT: A New Database and Baseline for Noise-Robust Audio-Visual Speech Recognition.* Interspeech 2017. 40 cites.
  https://doi.org/10.21437/Interspeech.2017-860. Data. Benchmarking with added noise, like our MUSAN-over-SNR setup.
- [ ] **[A]** Chung, Zisserman. *Out of Time: Automated Lip Sync in the Wild (SyncNet).* ACCV Workshops 2016 (LNCS 2017). 469 cites.
  https://doi.org/10.1007/978-3-319-54427-4_19. Lim. Background for an AV-sync signal when the voice doesn't match the face.
- [ ] **[C]** Ivanko, Ryumin, Karpov. *A Review of Recent Advances on Deep Learning Methods for Audio-Visual Speech Recognition.* Mathematics 2023. 41 cites.
  https://doi.org/10.3390/math11122665. Intro. A survey for broad statements; MDPI venue, so don't lean on it alone.

---

## 5. Not in the Scopus export, but must be added by hand

- [ ] **[A]** Cooke, Barker, Cunningham, Shao. *An Audio-Visual Corpus for Speech Perception and Automatic Speech Recognition (GRID).* JASA 120(5), 2006.
  https://doi.org/10.1121/1.2229005. Data. **The GRID dataset paper.**
- [ ] **[A]** Assael, Shillingford, Whiteson, de Freitas. *LipNet: End-to-End Sentence-Level Lipreading.* arXiv 2016.
  https://arxiv.org/abs/1611.01599. Intro, Model P. The standard GRID lip-reading baseline.
- [ ] **[A]** Snyder, Chen, Povey. *MUSAN: A Music, Speech, and Noise Corpus.* arXiv 2015.
  https://arxiv.org/abs/1510.08484. Data. **Our noise source.**
- [ ] **[A]** Afouras, Chung, Zisserman. *LRS3-TED: A Large-Scale Dataset for Visual Speech Recognition.* arXiv 2018.
  https://arxiv.org/abs/1809.00496. RW, Data. Training data behind the ws_v5 checkpoint; also relevant to our TED clips.
- [ ] **[A]** Chung, Nagrani, Zisserman. *VoxCeleb2: Deep Speaker Recognition.* Interspeech 2018.
  https://doi.org/10.21437/Interspeech.2018-1929. RW. The other half of the `lrs3vox2` pre-training data.
- [ ] **[B]** Ma, Petridis, Pantic. *Visual Speech Recognition for Multiple Languages in the Wild.* Nature Machine Intelligence 2022. 182 cites. *(In the export; added on review.)*
  https://doi.org/10.1038/s42256-022-00550-z. RW. The visual front-end that Auto-AVSR builds on.
- [ ] **[C]** Rouditchenko et al. *Whisper-Flamingo: Integrating Visual Features into Whisper for AVSR and Translation.* Interspeech 2024. 34 cites. *(In the export.)*
  https://doi.org/10.21437/Interspeech.2024-322. Intro. Brings the related work up to date with large pre-trained ASR.

---

### Download tips
- On campus or VPN, IEEE Xplore, ISCA (Interspeech) and ACM give free access through the DOI link.
- Interspeech papers are open access at isca-archive.org.
- arXiv papers are free.
- To get a BibTeX entry, open the DOI and choose *Cite → BibTeX*, or import `Scopus_shortlist.csv` into Zotero.
