# Speaker consistency under Nepali–English code-switching: research plan

v0.5 · 2026-10-18 · draft for Ishan's review.
v0.2 folded in four independent read-only referee passes (summary in §14). v0.3 adds Ishan's decisions of 2026-10-18 (`planning/DECISIONS.md`). v0.4 folds in two consistency reviews of the D2 protocol, annotation guidelines and ethics package. v0.5 replaces our own recordings with public conversation data (DECISIONS.md decision 6).
Sections 2, 6 and 7 become `PREREGISTRATION.md`, frozen together with the analysis-code commit hash in two stages: the D1 parts at milestone M2a and the D2 parts at M2b, each before its confirmatory data are scored. Later changes go to `DEVIATIONS.md` with dates.

## 1. Question and what is new

Do natural intra-speaker Nepali↔English switches (SS-DL: same speaker, different language) make neural diarization systems hypothesize speaker changes that are not there? Can that be reduced without missing real cross-lingual speaker turns (DS-DL: different speaker, different language)?

What already exists (§12): cross-lingual speaker verification shows that embeddings underestimate same-speaker similarity across languages (Thienpondt et al. 2022; LASE 2026), and DISPLACE reports aggregate DER on code-mixed conversations. Nobody has measured diarization errors located at language-switch points, separated SS-DL false splits from DS-DL misses, or evaluated language-invariance fixes on that trade-off. Nepali–English is untouched for this question. The two closest language-adversarial papers trained on TTS-generated cross-language pairs; this study uses real speech only.

The outcome is not guaranteed. LASE found that the same-voice cross-script gap shrinks for Indian-accented voices (0.006 cosine for WavLM-SV, 0.044 for ECAPA), so for L1-Nepali speakers the effect may be small. The study is powered and pre-registered so that a well-estimated small or null effect is still a publishable result.

Claims are graded. An excess of false speaker changes at switch points is called *switch-associated*. It is called *language-attributable* only if it survives the acoustic-change covariates and phrase-boundary controls (§2) and the non-adjacent embedding test H3 points the same way.

## 2. Definitions (pre-registered)

**Language tags.** Every aligned word is tagged by script: Devanagari → `ne`, Latin → `en`. A lexicon pass adds `loan` for English loanwords written in Devanagari and Nepali written in Latin; the lexicon (corpus audit lists plus a curated list) is frozen before the pilot. Digits, punctuation, mixed-script tokens and proper names are `neutral` (annotators mark names in D2; in D1 a curated name list is applied, frozen together with the loan lexicon). A *language run* is a maximal sequence of same-tag words from one speaker. `neutral` tokens do not break a run. `loan` tokens break it and may not fall inside any analysis window; a sensitivity analysis re-includes them with the language they are pronounced in.

**Pauses.** Measured with a VAD around each aligned word boundary, because CTC aligners place word ends imprecisely.

**Switch points.** For consecutive runs A (language a) and B (language b ≠ a) of one speaker, the point t is the midpoint of the pause between them (or the aligned boundary if there is no pause). There are two families:
- *Embedding family* (H1, H3): both runs last at least L = 1.0 s. This is fixed now; 0.5 s and 1.5 s are sensitivity analyses.
- *Detection family* (H2, H4, H5): no length requirement, because most English insertions are short. Points are stratified by the shorter run's length: < 0.5 s, 0.5–1.0 s, ≥ 1.0 s.

**Controls.** Both control sets are time points in the same speaker's recording, never audio edits.
- *Matched within-language controls.* Points inside a single-language run whose windows contain no `loan` or `neutral` tokens. Each is matched to a switch point on VAD pause bin (none, < 150 ms, 150–300 ms, > 300 ms), run-length stratum, and position (seconds from the clip or UEM start, binned). Up to 3 controls per point. Unmatched points are counted and reported, not silently dropped.
- *Phrase-boundary controls.* Same-language word boundaries with a pause over 150 ms. They separate "a pause or prosodic reset" from "a language change".
- *Operational rules* (decision 10): the VAD (Silero), pause rule, window containment, low-alignment exclusion and control regions are fixed in DECISIONS.md. The detection family compares each switch point with all controls in its video × pause × position cell, because 1:3 matching is infeasible there; the embedding family keeps 1:3 matching.

**Acoustic covariates.** At every point, the left-vs-right change in F0, energy, speaking rate (from the alignment), spectral tilt, SNR and a reverberation proxy (C50; SNR and C50 from the Brouhaha model, to be confirmed at setup).

**Exclusion zone (conversations).** SS-DL points and controls within 2c of any reference speaker change or overlap are excluded, so DS errors cannot leak into SS statistics.

**Reference transitions.** From the speaker RTTM plus language runs, every transition is SS-SL, SS-DL, DS-SL or DS-DL. A speaker transition is the end of speaker A's segment followed by speaker B's (B ≠ A) within 2 s; longer gaps and overlapped transitions are reported separately. Backchannel-only segments (short acknowledgements such as "हो" or "yeah", tagged by annotators) do not create transitions in the primary analysis; a sensitivity analysis includes them. Any facilitator speech is excluded from scoring. Its language type compares the last run before with the first run after. In conversations (D2) the language runs come from the same script rule applied to the transcripts, not from annotator judgement, so SS-DL means the same thing in every dataset.

**Hypothesized speaker change.** Any time the system's active speaker label changes (a segment of S1 followed by S2 ≠ S1 within 2 s). For the embedding detector P0, a peak of d(t) above threshold τ.

**Boundary metrics.** The collar c = 0.5 s is primary and 0.25 s secondary: alignment error is about ±100 ms, and pipelines add their own boundary error. FAR as a function of collar (0.1–1.0 s) is also reported as a curve.
- FAR_SS-DL(c): share of SS-DL points with at least one hypothesized speaker change within ±c.
- FAR_CTRL(c): the same for matched controls (and separately for phrase-boundary controls). The diagnostic quantities are the excess ΔFAR = FAR_SS-DL − FAR_CTRL and the risk ratio RR = FAR_SS-DL / FAR_CTRL.
- MR_DS-DL(c): share of DS-DL reference turns with no hypothesized change within ±c, using one-to-one nearest matching as in pyannote's SegmentationRecall with tolerance. MR_DS-SL(c) likewise.
- Also reported: hypothesized changes per minute and the global change rate, so a system cannot look better simply by hypothesizing fewer changes everywhere.
- P0 primary statistic (threshold-free): AUC of the distance d(t) at SS-DL points against matched controls (0.5 = no effect). P0 operating-point metrics use τ set on the pilot channels so that FAR_CTRL = 10%.
- Trade-off DET: sweep a named threshold (§7) and plot P(miss | DS-DL) against P(false alarm | SS-DL) on probit axes. Primary summary: MR_DS-DL at FAR_SS-DL = 10%, interpolated on the probit scale. Secondary: MR_DS-DL at FAR_SS-DL = 5%, and FAR_SS-DL at MR_DS-DL = 5% and 10%. DETs are also stratified by turn gap.

**Embedding metrics.**
- Cosine drift: cos(e(W_left), e(W_right)) for L-second windows on either side of a point, pause excluded. Δcos = mean over the matched controls minus the value at the SS-DL point (paired). It is standardized two ways: dz (mean paired difference over the SD of paired differences), and in units of each embedding's target-trial score SD, because raw cosine scales differ between embeddings.
- Cross-language score shift (H3): mean target-score difference (cross − mono) on lag-matched target trials, i.e. the two windows are the same time apart in both conditions. EER_cross vs EER_mono and minDCF (P_target = 0.05) are secondary. Same-session and cross-session (different video) trials are reported separately. In bootstraps a nontarget trial is kept only if both its speakers were resampled.

**Pipeline metrics.** DER and its components (false alarm, missed speech, confusion) via pyannote.metrics at collar 0 and 0.25 s, overlap scored. *Language-driven split index:* adjusted mutual information (AMI) between the hypothesized speaker label and the language-run label within one reference speaker's speech, tested against a circular-shift permutation null (language labels shifted in time). Plain NMI is biased upward in finite samples, so it is not used.

## 3. Data (real recordings only)

**Rule.** Allowed: real recordings as published; cutting windows or scoring regions out of one continuous real recording; resampling and format conversion. Not allowed: concatenating clips; adding noise, reverberation or other speakers; TTS or voice conversion; speed or pitch perturbation; simulated conversations; the corpus's `augmented` config. The third-party pretrained models we evaluate (pyannote, WavLM, ECAPA, DiariZen) were trained by their authors with augmentation and simulated mixtures. That is outside our data, unavoidable when evaluating public models, and is documented.

| ID | Source | Role | Facts (checked) | Caveats |
|---|---|---|---|---|
| D1 | `saileshbro/nepali-cs-asr`, v2 `default` config, revision pinned at download | SS-DL diagnostic; training data for M1–M3 | 20,805 clips, 58.42 h, 16 kHz WAV; 20 YouTube channels, 161 STEM-lecture videos; train 16,702 clips / 15 channels, val 2,338 / 3, test 1,765 / 2; speech_type cs 19,589, en 651, ne 565; `cmi`; per-clip `video_id`/`start_ms`/`end_ms`; CC BY-NC 4.0 | `speaker_id` is the channel (a proxy); clips 3–14 s, each loudness-normalised to −20 dBFS; transcripts mostly machine-made (777 human-verified: 586 train / 23 val / 168 test); test has 3 rows without provenance and one key shared by 3 rows (all dropped). The 3 val channels are the pilot; the other 17 are confirmatory for the diagnostic |
| D1-LF | Long-form audio of the 161 source videos | Long-form single-speaker diarization (pipeline H2, AMI split index) | Real, continuous. The clips cover 95% of each video's span, so transcripts exist for nearly all of it | Ishan decided to download it for private research analysis (DECISIONS.md, 2026-10-18), on his own computer because YouTube blocks cloud IPs. Stored privately, never redistributed; only IDs and timestamps are released |
| D2-P | Public Nepali–English multi-speaker podcasts and talk shows (YouTube), annotated | Nepali DS-DL test and SS-DL inside conversations; supporting evidence for H5, primary only if it reaches the simulated size | Real, unscripted conversation | Audio downloaded on Ishan's own computer (decision 6) and stored privately; only IDs, timestamps and annotations are released, as VoxConverse does. Annotated by Ishan and/or hired annotators following `ANNOTATION_GUIDELINES.md`. There are no close-talk channels, so speaker turns are marked by ear. Drafts from ASR or diarization services may be used to save time but are never the reference, and no draft comes from a system under test. Split by show, so no host appears in both dev and test; clustered by show. Size: as much as annotation allows, working target ≥ 5 h |
| D2-R | Own recordings (deferred) | Not run unless Ishan revives it; design kept in `D2_PROTOCOL.md` and `ethics/` | — | Unscripted 2–3-person bilingual conversations, recorded with written consent (§13). One multitrack recorder: a wired lavalier per speaker plus a separate omnidirectional room mic, all on one clock; claps at start and end only verify sync. Systems only see the room channel. The speaker reference comes from close-talk VAD with per-channel thresholds; crosstalk is removed only where another channel explains it, so real overlap is kept. Annotators correct the full transcript, so script-based language tags are right everywhere, then check speaker and word boundaries around every marked point. Markers are untyped and mixed with random filler points. The draft VAD and ASR are not components of any system under test. Participants are not told the hypothesis, and eligibility uses a neutral language-background questionnaire. A participant joins at most 2 sessions, both in the same split, and is debriefed after the last one; pilot participants go to dev. At least half the sessions are same-gender, because similar voices are where diarization misses real turns. Hired bilingual annotators. Ishan confirmed the necessary permissions on 2026-10-18 (DECISIONS.md). Working size: about 28 sessions of ~45 min (≈ 21 h), 20 test and 8 dev, with about 35–50 participants; the H5 simulation on the pilot fixes the final number (expected 600–1,000 DS-DL test turns). 2–4 h would support only a descriptive trade-off. Unedited multi-host podcasts are the fallback (Option B), with permission |
| D3 | DISPLACE 2024 (and 2023); DISPLACE-M 2026 | Primary DS-DL test for H5 (decision 6): real code-mixed conversations with human speaker and language references, scored at c = 0.5 s. SS-DL analyses on D3 stay exploratory | 2024: 158 h far-field, speaker and language RTTMs (segment-level). DISPLACE-M: Hindi–English clinical dialogues, 40 h dev + 15 h eval | Access by signed terms (Ishan). DISPLACE-M use agreed by its organisers (decision 9) and primary for H5 unless DISPLACE 2024 is also granted. No document names Nepali, so H5 is confirmed on Indian-language–English conversation and supported on Nepali by D2-P. Segment-level language labels are too coarse for ±0.25 s. Released dev/eval split kept; clustered by recording session |
| D4 | MUCS 2021 (OpenSLR 104) | Confirmatory replication of the SS-DL diagnostic on other language pairs (H7) | Hindi–English 89.86 h train / 5.18 h test; Bengali–English 46.11 h / 7.02 h; mixed-script transcripts; CC BY-SA 4.0 | Single presenters: SS-DL only |
| D5 | OpenSLR 54 | Extra Nepali speakers for M3's speaker loss and for nontarget trials | ~157k read utterances with anonymised speaker IDs; CC BY-SA 4.0 | Read, monolingual |
| D6 | IndicVoices, Nepali subset | Audit for more real Nepali speakers and possible two-party conversations | 22 scheduled languages incl. Nepali; 76% extempore, 15% conversational; CC BY 4.0; gated | Nepali hours and Nepali–English switching unknown until audited |

Dropped: OpenSLR 143 (two TTS voices, non-commercial), Shrutilipi / Vistaar / Kathbath (no Nepali), Common Voice and FLEURS (read speech, weak speaker labels).

In the rest of this plan, "D2" means the conversation data: D2-P, plus D2-R if it is revived.

## 4. Data pipeline (Phase 1)

1. **Ingest D1.** Pinned revision, v2 splits kept as published, ambiguous-provenance rows dropped, per-channel and per-video statistics.
2. **Word alignment.** Implemented in `csdiar/data/align.py`: each word is romanised with uroman (language "nep"), then the clip is aligned with torchaudio 2.8's MMS_FA bundle (the MMS-300m forced-alignment model) on the GPU, with a star token at both edges. The installed `ctc-forced-aligner` 1.0.2 is not used for this: its romaniser is actually unidecode, which drops Devanagari inherent vowels. Run on 2026-10-18 over all 20,805 clips: 98.35% of the 541,899 words aligned; the rest are digits and symbols, which are `neutral`. Of the aligned words, 3.7% have a mean token probability below 0.05. Numerals are expanded to words where the spoken language is unambiguous; otherwise they stay unaligned and `neutral`. WhisperX has no Nepali alignment model. Whisper-large-v3 word timestamps (±100–300 ms) serve only as a cross-check. The per-word probability is kept, and clips with a low mean score are excluded at a threshold fixed on train.
3. **Pauses** from Silero VAD around every aligned boundary.
4. **Tag audit.** Script tags against human judgement on a sample of the 777 verified clips; the `loan` lexicon pass.
5. **Listening check** (about an hour for a hired annotator, on a review page I generate with audio snippets and spectrograms marking the boundary; switch points and controls are mixed so the condition is hidden): 300 random SS-DL points and 150 controls. For each: is it a real language switch, is the same single speaker on both sides, and is the boundary within ±100 ms? Also every point P1 flags as a speaker change, capped at 200. The contamination rate this gives feeds a contamination-corrected ΔFAR.
6. **Speaker-purity audit.**
   - (a) Embed clips with two families that are not under test (candidates: NeMo TitaNet-L, 3D-Speaker CAM++ / ERes2NetV2). Compare each clip only with clips of similar language mix (speech_type × CMI tertile) in its video, because a language-sensitive embedding would otherwise flag English-heavy clips. Report flag rates by language mix.
   - (b) Cluster video centroids within each channel to find multi-teacher channels, and split them into speaker IDs.
   - (c) Check for the same teacher on two channels.
   - (d) Listen to a stratified sample of flagged clips (capped at 200).
   - (e) Report every result on all clips and on the purity-verified subset.
7. **Long-form mapping** (only if D1-LF is obtained). Normalised cross-correlation of each clip against the long-form audio over ±5 s. A clip is accepted if its offset is within 20 ms of the video's median offset. Unavailable videos are reported, and failed clips are removed from the UEM.
8. **D2 processing** as in §3. 20% of sessions are annotated independently by a second bilingual annotator who starts from the close-talk VAD segments without the ASR draft. Inter-annotator DER and language-boundary F1 are reported.

## 5. Systems

**Embedding extractors under test**
- E1 `speechbrain/spkrec-ecapa-voxceleb`: ECAPA-TDNN, 192-d, VoxCeleb1-O EER 0.80%, Apache-2.0.
- E2 WeSpeaker ResNet34-LM: the embedding pyannote community-1 and DiariZen cluster with (exact checkpoint confirmed at setup).
- E3 WavLM-Large + ECAPA-TDNN: Microsoft UniSpeech speaker-verification checkpoint (VoxCeleb1-O EER 0.431%; code CC BY-SA 3.0, vendored in `third_party/unispeech`; loaded strictly with s3prl 0.4.18; decision 11). Also the starting point for M3.

**Pipelines**
- P0, a transparent embedding speaker-change detector: L-second windows left and right, 0.1 s hop, cosine distance d(t), peak picking at τ. It runs with E1–E3 and every mitigation.
- P1 `pyannote/speaker-diarization-community-1` (pyannote.audio 4.0.7: powerset segmentation, WeSpeaker embeddings, VBx clustering; CC BY 4.0; gated).
- P1-AHC: P1's segmentation with embeddings from E1, E2 or E3 (or a mitigated version) and AHC-cosine clustering at a dev-tuned threshold. It is the one back end that all three mitigations plug into, so it is primary for H4–H6.
- P2 DiariZen WavLM-Large-s80-md (and -v2) (Han et al. 2025: WavLM-Large + Conformer segmentation, VBx clustering; weights CC BY-NC 4.0). It pins torch 2.1.1/cu121, which has no Blackwell kernels. Plan A: port to torch ≥ 2.8/cu128. Plan B: run P2 inference on a small Ada- or Ampere-generation G instance with its pinned stack.
- P3 DiaPer (Landini et al. 2024), optional. It needs torch 1.10 and Python 3.7. Spend at most one day on a modern-torch port, otherwise drop it and document why. It is included because Neupane et al. (2026) used it for Nepali–Hindi.

**Baselines every mitigation must beat**
- B0: the unmodified system, retuned on the same dev objective as the mitigation, so a mitigation cannot win by shifting the operating point.
- B1: language-wise mean centring (subtract each language's mean embedding, estimated on train).
- B2: a language-aware score offset in the clustering affinities, in the style of Thienpondt et al. (2022).

**Mitigations**
- **M1 SubspaceProjector** (training-free nuisance attribute projection). v_lang is the average of within-video differences between duration-matched `ne` and `en` windows of the same speaker, refit inside every cross-validation fold. e′ = e − (e·v̂)v̂, re-normalised. Secondary: a k-dimensional subspace from within-speaker cross-language difference vectors, k ∈ {1, 2, 4, 8, 16} chosen inside the folds (the variant Hosseini-Kivanani 2026 used for verification). Specificity controls: a random direction and the top PCA direction of the same k. Applied before AHC-cosine; also reported inside native VBx, where it shifts the PLDA input.
- **M2 LanguageGatedSCD.** The VoxLingua107 ECAPA language-ID model (Valk & Alumäe 2021; `ne` and `en` are in its label set) runs on sliding windows. p(ne) and p(en) are renormalised, and the language-change score λ(t) is the Jensen–Shannon divergence between left and right windows. Before any tuning, report λ's AUC for separating SS-DL points from controls and from DS-SL turns: does λ track language, pauses or speaker changes? In P0 the threshold becomes τ(t) = τ0 + α·λ(t). In the pipelines, a post-processor relabels the shorter of two adjacent segments to its neighbour's speaker (a local relabel, never a global cluster merge) when λ(t) is high and their embedding distance is below τ(t). The DET sweeps τ0 with α fixed. α and τ0 are tuned on dev to minimise FAR_SS-DL subject to the upper one-sided 80% bound of ΔMR_DS-DL (vs B0) being ≤ +1 pp. Variants: an oracle λ built from reference tags (upper bound) and a time-shuffled λ (null control).
- **M3 LanguageAdversarialFineTuner.** Starts from E3. Losses:
  - AAM-softmax over the purity-verified D1-train speakers plus the D5 speakers;
  - a supervised-contrastive loss whose positives are the same speaker across languages (`ne`, `en` and high-CMI windows), on D1 only;
  - a gradient-reversal language head (`ne` / `en` / mixed window labels from the alignment), on D1 only, because D5 is all Nepali read speech and a GRL there would learn the corpus rather than the language;
  - an L2 distillation term towards the frozen E3 embedding, weight ∈ {0, 0.1, 0.3, 1.0}, pre-registered and reported as a trade-off curve.

  No augmentation. 5 seeds, early stopping on dev. Ablations: −GRL, −contrastive, −both (plain in-domain fine-tuning), −distillation, −D5. Tag noise is measured on the 777 verified clips. Optional M3-R: the same recipe on E2 as a drop-in for VBx.

## 6. Experiments (pre-registered)

- **Pilot** (before the freeze). Baselines only, on the 3 D1 validation channels and the first recorded D2 hour: counts, variance components, P0's τ, and power simulations. No mitigation results are looked at.
- **E-A Diagnostic (RQ1).** Confirmatory on the 17 non-pilot D1 channels, on D2's SS-DL points, and on D1-LF if obtained:
  1. Δcos (H1);
  2. P0 AUC, and P1/P2 RR (H2). For P1/P2 the primary condition is long-form (D2, D1-LF), because on 3–14 s clips they almost never split; clip results are descriptive;
  3. score shift and EER (H3);
  4. the AMI language-driven split index;
  5. replication on D4 (H7) and D3 (exploratory).
- **E-B Mitigation (RQ2).**
  - (i) D1 leave-channels-out cross-validation over the 18 non-test channels (6 folds of 3), with all fitting and tuning nested inside the folds: H4 on the SS-DL excess.
  - (ii) The DISPLACE eval set (DISPLACE-M, or 2024 if granted): H4 and H5 jointly (the trade-off), primary back end P1-AHC with E3; its dev set is used for tuning.
  - (iii) D1 v2 test (2 channels): descriptive only.
  - (iv) D2-P test shows: the same trade-off on Nepali conversation, as support (confirmatory only if it reaches the simulated size).

  All mitigation configurations are frozen before any D1-test, DISPLACE-eval or D2-P-test scoring.
- **E-C Exploratory (labelled as such).** Effect against run length, pause, CMI, switch direction (ne→en vs en→ne) and speaker; layer-wise WavLM probes of speaker vs language information on a frame subsample.
- **Deliverables from the original spec.** Diagnostic table and cosine-drift plot; ablation and trade-off table; `det_ssdl_vs_dsdl.pdf`; `umap_speaker_language_invariance.pdf`. UMAP is qualitative, so it ships with two numbers: linear-probe language accuracy from the embeddings (lower = more language-invariant) and speaker-identification accuracy (must not fall).

## 7. Hypotheses and analysis plan (pre-registered draft)

**Primary cells.** Embedding family at L = 1.0 s. Detection family with run-length strata pooled (strata reported). c = 0.5 s. All clips, with the purity-verified subset as a sensitivity analysis. D1 confirmatory channels for H1, H3 and P0-H2; D1-LF for pipeline-H2 (D2-P and D3 secondary); the DISPLACE eval set for H5 (DISPLACE-M, or DISPLACE 2024 if access is granted; decision 9).

| H | Prediction | Primary test | Supported if (smallest effect of interest) |
|---|---|---|---|
| H1 | Same-speaker windows straddling an SS-DL switch are less similar than matched controls (E1–E3; m = 3) | Per-channel mean paired Δcos; exact sign-flip permutation over channels; BCa CI | Holm p < .05 and dz ≥ 0.2 |
| H2 | More hypothesized changes at SS-DL points than at controls (P0 with E1–E3, P1, P2; m = 5) | P0: per-channel AUC − 0.5. P1/P2: per-cluster log RR (ratio of sums). Sign-flip over channels or participants | Holm p < .05, and AUC ≥ 0.55 or RR ≥ 1.25 with ΔFAR ≥ 1 pp |
| H3 | Cross-language same-speaker scores are shifted down (E1–E3; m = 3) | Lag-matched target-score shift; sign-flip over channels | Holm p < .05 and shift ≥ 0.2 target-score SD (EER difference ≥ 0.5 pp or ≥ 20% relative reported as secondary) |
| H4 | Each mitigation lowers the SS-DL excess against its retuned baseline B0 (M1–M3; m = 3) | Per-cluster change in ΔFAR; on D2 also MR_DS-DL at FAR_SS-DL = 10% | Holm p < .05 and ≥ 25% relative reduction in ΔFAR |
| H5 | It does so without raising misses on real cross-lingual turns | Non-inferiority on MR_DS-DL on the DISPLACE eval set (DISPLACE-M, or 2024 if granted), clustered by recording session; D2-P test shows as support; intersection-union with H4 per mitigation | One-sided 95% upper bound below both +2 pp and +25% relative |
| H6 | M3 has the best trade-off and M1 the smallest gain (secondary) | Pairwise differences in MR_DS-DL at FAR_SS-DL = 10%, Holm (m = 3); two-stage bootstrap over clusters and seeds | Ordering significant after Holm |
| H7 | The SS-DL effect replicates on D4 | As H1 and H2, on D4 | Same direction, Holm within H7 |

**Outcome table.** Every primary hypothesis is reported as one of: supported (significant and at least the smallest effect), nonzero but trivial (significant but below it), equivalent (TOST at the smallest effect), or inconclusive.

**Analysis rules**
- *Few clusters.* Exact sign-flip permutation tests on per-cluster statistics (2^17 flips for the 17 confirmatory channels) and BCa bootstrap CIs; Bonferroni-adjusted CIs within families (98.3% for m = 3). The 2-channel D1 test split is descriptive only.
- *Sensitivity model.* Logistic GLMM: FA ~ condition + acoustic covariates + pause bin + position + (1 + condition | speaker) + (1 | video/clip) + (1 | matched set). If the fit is singular, drop in this order: the random slope, then video, then matched set.
- *Operating points.* P0: τ from the pilot (FAR_CTRL = 10%). P1-AHC: threshold grid 0.30–1.00 in steps of 0.05, tuned on dev. Native P1/P2: the VBx parameter grid is fixed at setup and written into the frozen document. H4 is evaluated at the threshold that matches the baseline's dev MR_DS-DL.
- *Conversations.* DISPLACE keeps its released dev/eval split and is clustered by recording session. D2-P is split by show and clustered by show. H5 is sized by simulation (design effect included) on DISPLACE dev; if the eval set cannot reach that size, H5 is reported as descriptive and labelled as such. D2-P supports H5 descriptively unless it reaches the simulated size.
- *Hygiene.* Every run logs config, seed, git SHA and data revision; each test set is scored once per frozen configuration; everything outside the named primary cells is exploratory.

## 8. Compute, storage and environments

Box: g7e.2xlarge, 1× RTX PRO 6000 Blackwell Server Edition (96 GB). Estimated GPU time: alignment 2–4 h; embeddings and P0 ~10 h; P1/P2 10–20 h; M3 about 35 runs (main recipe, 5 ablations and the distillation grid over 5 seeds) at 3–6 h each, ≈ 100–200 h. Total about 150–250 GPU-hours spread over weeks; the box can be stopped between phases. CPU-heavy steps (audit clustering, UMAP, permutation tests) run one at a time because the 2xlarge has fewer vCPUs.

Storage: at least 1 TB gp3 for audio, caches and checkpoints. Code goes to GitHub on a feature branch with a PR. Results and logs are mirrored to a backup host.

Separate `uv` virtual environments, with exact pins set at setup:
- `env-pyannote4`: pyannote.audio 4.0.7, torch ≥ 2.8 (cu128), torchcodec, speechbrain 1.1.2, ctc-forced-aligner, pyannote.metrics.
- `env-sv`: M3 training.
- `env-diarizen`: DiariZen with its vendored pyannote 3.1 fork, patched for Blackwell.
- `env-diaper`: optional.

System: NVIDIA driver with open kernel modules (with `nvidia-smi`), fastfetch, libsndfile, sox. Amazon Linux 2023 does not package FFmpeg, so its shared libraries come from conda-forge, or waveforms are passed to pyannote in memory.

## 9. Milestones

| M | Contents | Needs from Ishan | Rough length |
|---|---|---|---|
| M0 | Plan approved; box bootstrapped; environments smoke-tested on the GPU; consent forms, recording protocol and annotation guidelines drafted | Key permissions; HF token; gated-model terms; buy the recorder and lavaliers; start hiring annotators | days |
| M1 | D1 ingested and aligned; tags, pauses, boundaries and controls; purity audit; D1-LF converted and mapped | Long-form audio download; annotators do ~2 h of listening checks | 1–2 weeks |
| M2a | Pilot on the 3 val channels; D1 parts of PREREGISTRATION.md (H1, H3, H7 and the D1 parts of H2 and H4) frozen | Review the frozen document | ~1 week |
| M2b | DISPLACE received; first annotated D2-P hour; power simulation; conversation parts (H5 and the conversation parts of H2, H4, H6) frozen | Sign the DISPLACE terms; annotate, or hire for, the first hour | ~1 week after DISPLACE arrives |
| M3 | D2-P annotated | Annotation time (Ishan and/or hired annotators) | human-paced |
| M4 | Diagnostic results (E-A) | — | 1–2 weeks |
| M5 | Mitigations (E-B) | — | 3–5 weeks |
| M6 | Final test evaluation, figures, tables | — | ~1 week |
| M7 | Paper, with independent read-only reviews | Venue choice | 2–3 weeks |

## 10. Risks

| Risk | Mitigation |
|---|---|
| `speaker_id` is a channel, not a person | Language-stratified purity audit; channel splitting; results on the verified subset |
| Few speakers (15 train, 2 test) | Permutation tests over 17 confirmatory channels; channel cross-validation; D5 speakers and distillation for M3 |
| Machine transcripts misplace switch points | `loan` class; alignment-score filter; VAD pauses; listening check with a contamination-corrected ΔFAR; collar curve |
| Effect small or null | Long-form tests where clustering errors surface; standardized smallest effects; TOST; four-way outcome table |
| Pauses, prosody or mic distance coincide with switches | Matched and phrase-boundary controls; acoustic covariates; claim grading |
| DS-DL data scarce and slow to annotate | DISPLACE supplies human-labelled turns; D2-P sized to annotation capacity; drafts from services; H5 falls back to descriptive |
| DISPLACE access refused or delayed | DISPLACE-M (signed terms on Hugging Face); more D2-P annotation |
| A mitigation wins by shifting the operating point | Retuned B0; FAR_CTRL, change rate and matched-FAR summaries |
| Tuning leakage | Nested cross-validation; participant-disjoint D2; configurations frozen before test scoring |
| Downloading the long-form audio breaches YouTube's terms | Ishan's documented decision; audio kept private and never redistributed; D2 also supplies long-form audio |
| DiariZen or DiaPer will not run on Blackwell | Port; fallback instance; drop DiaPer |
| Non-commercial licences (D1, DiariZen weights, MMS aligner) | Non-commercial project confirmed in writing; documented per component |
| Recording ethics and privacy | §13 |
| M3 overfits 15 speakers | D5 speakers, distillation grid, early stopping, no test peeking |

## 11. Corrections to the original spec

| The spec says | The check found |
|---|---|
| WavLM-Large + Conformer is the "DiariZen architecture from Barahona et al., 2026" | DiariZen is Han et al. (BUT), ICASSP 2025. Barahona et al. (Odyssey 2026) is a DISPLACE-M system paper that used DiariZen; cite it as related work. |
| "DiaPer (Neupane et al., 2026)" | DiaPer is Landini et al., TASLP 2024. Neupane et al. (2026) applied DiaPer to Nepali–Hindi, the closest Nepali diarization work; cite both. |
| "VoxLingua107 ECAPA-TDNN (Baghel et al., 2023)" | The language-ID model is Valk & Alumäe (SLT 2021) via SpeechBrain. Baghel et al. (2023) is the DISPLACE 2023 challenge paper. |
| UMAP projections "(Shrestha et al., 2026)" | That paper is about Nepali voice cloning (UMAP of a GE2E encoder). Cite McInnes et al. (2018) for UMAP. |
| `pyannote-audio-elan` produces dual SPEAKER/LANGUAGE RTTMs | It exists but only brings pyannote speaker diarization into ELAN. The language tier and the RTTM export are ours. |
| DISPLACE "includes native Nepali L1 speakers", with word-level language RTTMs | No DISPLACE document names Nepali, and its language RTTMs are segment-level. It is an exploratory replication on other language pairs. |
| OpenSLR 143 as a monolingual speaker pool | Two TTS voices (CC BY-NC-SA): dropped. OpenSLR 54 (read speech, many speakers) is kept. |
| WavLM-Large + Conformer as an embedding model for cosine drift | It is DiariZen's segmentation model. DiariZen clusters with WeSpeaker ResNet34 embeddings (E2), so drift is measured on E2. |
| `speaker_id` as speaker identity | It is the YouTube channel; hence the purity audit. |
| DiaPer and DiariZen "for H100 / RTX PRO 6000 batched inference" | Both pin old torch builds without Blackwell kernels; see the P2 and P3 plans. |
| MMS or Whisper-Large-v3 for forced alignment | torchaudio's alignment API was deprecated in 2.8 and slated for removal; we use `ctc-forced-aligner` + MMS-300m. Whisper word timestamps (±100–300 ms) are only a cross-check. |
| Subspace projection and GRL fine-tuning as the methods | Both exist for speaker verification (projection: Hosseini-Kivanani 2026; GRL: LASE 2026 and Li et al. 2026, both trained on TTS data). Our contribution is the boundary-level SS-DL/DS-DL measurement, the trade-off evaluation, and real-speech-only training. |
| 2–4 hours of multi-speaker recordings | Enough for a descriptive trade-off. A confirmatory non-inferiority claim (H5) needs roughly 600–1,000 DS-DL test turns (about 10–20 h), fixed by simulation on the pilot hour. |
| Curating YouTube podcasts and long-form videos | YouTube's terms forbid downloading; we ask for permission first (§13). |
| A "Diagnostic Proof Table" | Framed as a test: for L1-Nepali speakers the effect may be small. |

## 12. References

Checked against the source page this session (✓), or standard references (S). Items marked "to verify" are cited from a search result and must be opened before use in the paper.

1. ✓ Han, Landini, Rohdin, Silnova, Diez, Burget. *Leveraging Self-Supervised Learning for Speaker Diarization.* ICASSP 2025. [arXiv:2409.09408](https://arxiv.org/abs/2409.09408) (DiariZen; code [BUTSpeechFIT/DiariZen](https://github.com/BUTSpeechFIT/DiariZen))
2. ✓ Landini et al. *DiaPer: End-to-End Neural Diarization with Perceiver-Based Attractors.* IEEE/ACM TASLP 2024. [arXiv:2312.04324](https://arxiv.org/abs/2312.04324)
3. ✓ Neupane, Pokhrel, Pyakurel, Joshi. *Neural Speaker Diarization via Multilingual Training: Evaluation on Low-Resource Nepali-Hindi Speech.* 2026. [arXiv:2606.26144](https://arxiv.org/abs/2606.26144)
4. ✓ Barahona, Herrera-Alarcón, Alvarez-Trejos, Lozano-Diez. *Adapting Speaker Diarization to Code-Switched Medical Conversations: AUDIAS-UAM at the DISPLACE-M Challenge.* Odyssey 2026, 75–82. [doi:10.21437/Odyssey.2026-12](https://www.isca-archive.org/odyssey_2026/barahona26_odyssey.html)
5. ✓ E, Meena, Nanivadekar, et al. *Benchmarking Speech Systems for Frontline Health Conversations: The DISPLACE-M Challenge.* 2026. [arXiv:2603.02813](https://arxiv.org/abs/2603.02813)
6. Baghel et al. *DISPLACE Challenge: DIarization of SPeaker and LAnguage in Conversational Environments.* Interspeech 2023. [arXiv:2303.00830](https://arxiv.org/abs/2303.00830); *Summary of the DISPLACE Challenge 2023.* [arXiv:2311.12564](https://arxiv.org/abs/2311.12564)
7. *The Second DISPLACE Challenge* (2024). [arXiv:2406.09494](https://arxiv.org/abs/2406.09494)
8. Valk, Alumäe. *VoxLingua107: a Dataset for Spoken Language Recognition.* SLT 2021. [arXiv:2011.12998](https://arxiv.org/abs/2011.12998)
9. ✓ Thienpondt, Desplanques, Demuynck. *Tackling the Score Shift in Cross-Lingual Speaker Verification by Exploiting Language Information.* ICASSP 2022. [arXiv:2110.09150](https://arxiv.org/abs/2110.09150)
10. ✓ Menta. *LASE: Language-Adversarial Speaker Encoding for Indic Cross-Script Identity Preservation.* 2026. [arXiv:2605.00777](https://arxiv.org/abs/2605.00777)
11. ✓ Li, Miao, Liu, Li. *Language-Invariant Multilingual Speaker Verification for the TidyVoice 2026 Challenge.* 2026. [arXiv:2603.08092](https://arxiv.org/abs/2603.08092)
12. ✓ Hosseini-Kivanani. *Simple Language Normalization Wins: Cross-Lingual Speaker Verification for the TidyVoice 2026 Challenge.* 2026. [arXiv:2607.22923](https://arxiv.org/abs/2607.22923)
13. Mishra, Prasanna. *Spoken language change detection inspired by speaker change detection.* [arXiv:2302.05265](https://arxiv.org/abs/2302.05265) (journal version in Circuits, Systems and Signal Processing, to verify); *Language vs Speaker Change: A Comparative Study.* [arXiv:2203.02680](https://arxiv.org/abs/2203.02680)
14. To verify: Reuter et al. *On the influence of language similarity in non-target speaker verification trials.* Interspeech 2025. [ISCA archive](https://www.isca-archive.org/interspeech_2025/reuter25_interspeech.pdf)
15. To verify: *Speaker and Language Change Detection using Wav2vec2 and Whisper.* [arXiv:2302.09381](https://arxiv.org/abs/2302.09381)
16. ✓ Javed et al. *IndicVoices: Towards building an Inclusive Multilingual Speech Dataset for Indian Languages.* ACL Findings 2024. [arXiv:2403.01926](https://arxiv.org/abs/2403.01926)
17. ✓ Dahal. *Nepali–English Code-Switched ASR* (dataset, v2). Hugging Face, 2026. [saileshbro/nepali-cs-asr](https://huggingface.co/datasets/saileshbro/nepali-cs-asr) (no paper yet)
18. S Chen et al. *WavLM.* IEEE JSTSP 2022. [arXiv:2110.13900](https://arxiv.org/abs/2110.13900)
19. S Desplanques, Thienpondt, Demuynck. *ECAPA-TDNN.* Interspeech 2020. [arXiv:2005.07143](https://arxiv.org/abs/2005.07143)
20. S Plaquet, Bredin. *Powerset multi-class cross entropy loss for neural speaker diarization.* Interspeech 2023.
21. S Landini et al. *Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization.* Computer Speech & Language 2022.
22. S Ganin, Lempitsky. *Unsupervised Domain Adaptation by Backpropagation.* ICML 2015 (gradient reversal).
23. S Deng et al. *ArcFace: Additive Angular Margin Loss.* CVPR 2019.
24. S Pratap et al. *Scaling Speech Technology to 1,000+ Languages* (MMS). [arXiv:2305.13516](https://arxiv.org/abs/2305.13516)
25. S McInnes, Healy, Melville. *UMAP.* [arXiv:1802.03426](https://arxiv.org/abs/1802.03426)
26. ✓ UniSpeech speaker-verification checkpoints (WavLM-Large + ECAPA-TDNN). [microsoft/UniSpeech](https://github.com/microsoft/UniSpeech/blob/main/downstreams/speaker_verification/README.md)

## 13. Ethics and data governance

- **D2-P (public conversations).** These are published shows, so no participant consent is collected. Only IDs, timestamps and annotations are released, never audio or speaker embeddings, and the paper's ethics statement says so.
- **D2-R participants (deferred).** Informed consent with tiered release (team only / data-use agreement / public), an explicit statement of voice-cloning and re-identification risk, the right to withdraw until publication, and payment. Ishan confirmed on 2026-10-18 that the necessary permissions are in place; the approving body and approval reference go into the paper's ethics statement.
- **Storage.** Raw audio encrypted at rest, team-only access, a retention date. Names and identifying details are bleeped in any released audio.
- **Release.** Code, configs, annotations (speaker RTTMs and language tiers), alignments, and IDs with timestamps, under licences compatible with the sources (CC BY-NC for anything derived from D1). No per-speaker embeddings. DISPLACE and IndicVoices data are never redistributed.
- **Shared sessions.** Everyone's voice is on the room mic and on the other lavaliers, so a session's audio is released at a tier only if everyone in it chose that tier. Released audio has names bleeped, but voices stay recognisable, and the forms say so. Participants may withdraw until publication, and nothing is shared before then. If a participant withdraws, the whole session's audio leaves the study.
- **Licences.** D1, the DiariZen weights and the MMS aligner are non-commercial. Ishan confirms in writing that the project is non-commercial, and models fine-tuned on D1 are not used commercially.
- **YouTube.** On 2026-10-18 Ishan decided to download the D1 source audio for private research analysis, accepting that YouTube's terms do not permit it (DECISIONS.md). Audio only, private storage, never redistributed, deleted at the end of the project.

## 14. Review log

2026-10-18: four independent read-only review passes (diagnostic methodology, mitigation methodology, statistics, feasibility and ethics). Changes in v0.2:
- Confound handling: acoustic covariates, phrase-boundary controls and graded claims.
- Detection: P0's AUC is primary; pipeline H2 runs on long-form audio; AMI with a permutation null; short-insertion strata.
- Tagging: a `loan` class and VAD pauses; a 450-point listening check feeding a contamination-corrected ΔFAR.
- Baselines: retuned thresholds, mean centring and score offsets.
- Tuning and splits: nested cross-validation; participant-disjoint D2.
- Statistics: sign-flip permutation tests; named primary cells and Holm families; standardized smallest effects; a four-way outcome table; H5 sized by simulation (roughly 600–1,000 DS-DL test turns).
- Mitigations: M3 applies GRL and contrastive losses to D1 only, with 5 seeds and a distillation grid.
- Data and ethics: D1-LF made optional pending permission; a new ethics section.
- Not adopted: the statistics reviewer's suggestion to treat D1 clips as the primary condition for the pipelines, because the diagnostic reviewer showed that pipelines almost never split 3–14 s clips; long-form audio is primary for pipelines instead.

2026-10-18 (v0.3): Ishan's decisions are recorded in DECISIONS.md: the full D2 set with lapel mics, hired annotators, ethics handled under permissions he already holds (decision 5), and the D1 source audio downloaded privately. The pre-registration is split into a D1 freeze (M2a) and a D2 freeze (M2b).

2026-10-18 (v0.4): two read-only consistency reviews of the D2 protocol, annotation guidelines and ethics package (`reviews/review_docs_*.md`). Changes:
- Facilitator speech is excluded from scoring.
- A participant joins at most 2 sessions of one split and is debriefed after the last; pilot participants go to dev.
- The full transcript is corrected before the point checks, so no switch is missed because the ASR used the wrong script.
- Markers are untyped and mixed with fillers, and the second annotator works without the ASR draft.
- Overlap is kept in the VAD reference.
- Backchannel-only segments create no transitions in the primary analysis.
- Proper names are `neutral`.
- Shared-session release and withdrawal rules are added.
- D2 working size is about 28 sessions (~21 h).

2026-10-18 (v0.5): at Ishan's request, public data replaces our own recordings (decision 6). DISPLACE 2024 is the primary H5 test set; annotated public Nepali podcasts (D2-P, split by show) support it on Nepali; D1-LF is the primary long-form condition for pipeline H2. Own recordings (D2-R) are deferred, with their protocol and forms kept.

## Appendix A. Repository layout

```
nepali-cs-diarization/
├── planning/PLAN.md                     # this document (living)
├── PREREGISTRATION.md                   # frozen at M2
├── DEVIATIONS.md                        # dated changes after the freeze
├── ops/                                 # ssh_config, bootstrap_al2023.sh, environment scripts
├── configs/                             # one YAML per experiment
├── csdiar/
│   ├── data/
│   │   ├── ingest_nepali_cs.py          # D1: pinned download, v2 splits, provenance checks
│   │   ├── align.py                     # ctc-forced-aligner + uroman -> word times + script/loan tags
│   │   ├── pauses.py                    # VAD pauses at word boundaries
│   │   ├── boundaries.py                # SS-DL points, both control sets, covariates, RTTM/JSON export
│   │   ├── purity_audit.py
│   │   ├── listening_sheet.py           # review page: audio snippets + spectrograms with marked boundaries
│   │   ├── longform.py                  # D1-LF offsets and UEM (if obtained)
│   │   ├── ingest_multispeaker_rttm.py  # SPEAKER + LANGUAGE RTTM/ELAN -> typed transitions
│   │   └── ingest_{displace,mucs,slr54,indicvoices}.py
│   ├── models/
│   │   ├── embeddings.py                # E1-E3, batched
│   │   ├── baselines.py                 # P0-P3, P1-AHC, B0-B2
│   │   └── mitigation.py                # SubspaceProjector, LanguageGatedSCD, LanguageAdversarialFineTuner
│   └── eval/
│       ├── evaluate_boundaries.py       # FAR_SS-DL / MR_DS-DL at ±0.25 / ±0.5 s, collar curves, DET
│       ├── der.py                       # pyannote.metrics DER components
│       ├── embedding_metrics.py         # Δcos, lag-matched score shift, EER / minDCF
│       ├── stats.py                     # sign-flip tests, BCa, GLMM, Holm, non-inferiority, TOST
│       └── figures.py                   # det_ssdl_vs_dsdl.pdf, umap_speaker_language_invariance.pdf
├── scripts/                             # run_*.sh entry points, job queue
└── tests/
```
