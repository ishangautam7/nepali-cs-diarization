# Pre-registration, stage M2a: D1 diagnostic (H1, H2-P0, H3)

Status: **draft for freeze**. The freeze record is `planning/FREEZE_M2A.json`, tagged `m2a-freeze`. It pins code hashes, feature-file checksums and the commit. Later supplements, each frozen before any of its outcomes is computed (decision 19):
- H7 (D4), once its tagging rules are fixed;
- the D1 cross-validation part of H4, once M1–M3 exist;
- stage M2b: the conversation data (H5, H6 and the conversation parts of H2 and H4).

Changes after the freeze go to `planning/DEVIATIONS.md`, with dates.

This document consolidates PLAN.md §2, §6 and §7 (v0.5) with decisions 10–22 in `planning/DECISIONS.md`. Where the two differ, the decisions take precedence; this document states the result.

## 1. Question and claim grading
Do intra-speaker Nepali↔English switches (SS-DL) lower same-speaker embedding similarity (H1), make a change detector fire (H2-P0), and shift cross-language same-speaker scores (H3)?

Claims are graded. An effect at switch points is *switch-associated*. It is *language-attributable* only if it survives three checks:
- the acoustic-change covariates;
- the phrase-boundary controls;
- H3 pointing the same way.

## 2. Data, splits and channels
- **Corpus:** D1 is `saileshbro/nepali-cs-asr`, v2 `default` config, revision `dd00ca41dd38774cd6097c372c16939d7f3746df`. It holds 20,805 real lecture clips (3–14 s, 16 kHz). The synthetic `augmented` config is excluded.
- **Speakers:** a speaker is a YouTube channel (20 channels).
- **Pilot:** the 3 validation channels. Every pilot statistic in decisions 13–17 comes from these channels only.
- **Confirmatory:** the 17 non-pilot channels (15 train + 2 test). They are analysed once, with the frozen code. Exact sign-flip tests over the 17 channels have 2^17 sign patterns.
  - PLAN.md §7 says the 2-channel test split is descriptive only. That applies to analyses of the test split on its own (E-B iii); in this diagnostic the two test channels are 2 of the 17 clusters.
  - The analysis without them is a pre-registered sensitivity analysis (§5).

## 3. Features (fixed before the freeze; checksums in the freeze record)
- **Alignment:** uroman (`nep`) + torchaudio MMS_FA; per-word start, end and mean token probability (`csdiar/data/align.py`). Clips with a mean word probability below the train 5th percentile (0.369) are flagged low-alignment and excluded from every family.
- **VAD:** Silero 6.2.3, 32 ms frames, threshold 0.5.
- **Tags** (`csdiar/data/tags.py`, decision 12):
  - Script gives the tag: Devanagari → `ne`, Latin → `en`. Digits, mixed-script and other-script words → `neutral`.
  - Loan lexicon v1: 52 entries, sha256 38be59bb…7c26 → `loan`.
  - Name list v1: 144 entries, sha256 eb474023…efc3 → `neutral`.
  - A single Latin letter other than "a" → `neutral`.
  - A run is a maximal same-language sequence. Neutral words do not break a run; loan words do.
- **Boundaries** (`csdiar/data/boundaries.py`, `lex_v1b`; decisions 10 and 16):
  - **Pause:** the longest non-speech VAD run in the aligned gap ± 0.1 s, never past the words' middles.
  - **Switch point t:** the pause midpoint, or the gap midpoint when there is no pause.
  - **Switch points:** consecutive runs of different languages with no neutral or loan word between them.
  - **Window rule:** no analysis window or region may touch a neutral or loan word. An unaligned word next to a window counts as inside it. Windows lie inside the clip.
  - **Embedding families (L = 0.5, 1.0, 1.5 s):**
    - Both runs are ≥ L.
    - W_left = [p0 − L, p0] and W_right = [p1, p1 + L]. Each touches only its own run's words, with the outer edge at most 0.1 s outside the run.
    - Controls are within-run word boundaries whose region [p0 − L, p1 + L] obeys the same rule.
    - Matching: 1:3 without replacement, in seeded random order. It is exact on video (same clip first), pause bin (none, < 150, 150–300, > 300 ms), position bin (2 s, last bin ≥ 10 s) and context length (shorter side < 2L or ≥ 2L).
  - **P0 primary cell `p0L1`:**
    - Switch points: both runs ≥ 1.0 s, with [t − 1, t] and [t, t + 1] each inside one clean run.
    - Controls: the `det_w1` controls, i.e. within-run boundaries clean over [p0 − 1, p1 + 1].
    - Comparison: cell standardisation by video × pause bin × position bin.
  - **Secondary detection families:** `det` (region ± 0.5 s, the collar c) and `det_w1` (± 1.0 s). Switch-point regions may contain both languages but no neutral or loan word.
- **Embeddings:**
  - **E1:** `speechbrain/spkrec-ecapa-voxceleb`.
  - **E2:** WeSpeaker ResNet34-LM from pyannote Community-1, revision 3533c8cf.
  - **E3:** UniSpeech WavLM-Large + ECAPA, checkpoint sha256 51f07e3b…f94b, strict loading (decision 11).
  - **Two kinds:**
    - dense 0.1 s-hop grids (`csdiar/models/embed.py`);
    - point-anchored windows (`csdiar/models/point_embed.py`).
- **P0 detector:**
  - d(t) = 1 − cos(e[t − 1 s, t], e[t, t + 1 s]) on the dense L = 1.0 s grid.
  - Peaks are local maxima within ± 0.5 s.
  - A hypothesized change is a peak above τ within ± c (c = 0.5 s).
  - **τ (frozen, calibrated on the pilot `det_w1` controls so that FAR_CTRL = 10%):** E1 0.917, E2 0.863, E3 0.929.
- **H3 units** (`csdiar/data/windows.py`, decision 14): clean monolingual 1 s grid windows (one language, at least 2 words, at least 60% speech), with starts at least 1 s apart.

## 4. Hypotheses, primary cells, tests and smallest effects

| H | Primary cell | Cluster statistic | Test | Family | Smallest effect of interest |
|---|---|---|---|---|---|
| H1 | Embedding family, L = 1.0 s, 1:3 matched sets | Per-channel mean Δcos = mean(control cos) − switch cos | Exact one-sided sign-flip (Δcos > 0) | E1–E3, Holm m = 3 | dz = 0.2 (dz defined below) |
| H2-P0 | `p0L1` against `det_w1` controls, cell-standardised | Per-channel AUC of d(t) − 0.5 | Exact one-sided sign-flip (AUC > 0.5) | P0 with E1–E3 plus P1 and P2, Holm m = 5. P1 and P2 enter as p = 1 until stage M2b | AUC = 0.55 |
| H3 | Same-session (same video), non-adjacent (starts ≥ 2 s apart), lag-matched in bins [2,4), [4,8), [8,16), [16,64), ≥ 64 s | Per-channel shift: per lag bin, mean(cross) − the mean of the monolingual means that have ≥ 20 trials (ne–ne and/or en–en). A bin counts only with ≥ 20 cross trials and at least one such monolingual mean. The channel shift is the cross-trial-weighted mean over its counted bins, divided by the target-score SD (defined below) | Exact one-sided sign-flip (shift < 0) | E1–E3, Holm m = 3 | 0.2 target-score SD |

- **H3 trials:**
  - **Lag:** |abs_t0(a) − abs_t0(b)|. abs_t0 is the window's start in its source video: the clip's `start_ms` plus the window offset. Clips without provenance have none and are not used.
  - **Pair sampling:** pairs are drawn at random with the fixed seed. For each video, min(40,000, n(n − 1)/2) index pairs are drawn with replacement, and identical windows and lags under 2 s are discarded.
- **H1 units:** a matched set counts if it has at least one control. Its Δcos uses the mean over its 1–3 controls.

- **Channels entering a test:** a channel enters a primary test only with at least 10 units. Units are H1 matched sets and H2-P0 switch points with cell controls; for H3, a channel needs at least one lag bin meeting the trial rule above.
  - Excluded channels are listed with their counts. The sign-flip and the bootstrap then run over the k included channels (2^k patterns).
  - By the pre-freeze counts, which are not outcomes, every confirmatory channel has at least 13 H1 sets and 13 eligible `p0L1` points. The number of `p0L1` points with cell controls can be lower.
- **Estimands and intervals:**
  - **H1:** dz = (unweighted mean over channels of the channel mean Δcos) / (pooled within-channel SD of per-set Δcos), with the pooled SD = √(Σ_c (n_c − 1) s_c² / Σ_c (n_c − 1)).
    - The sign-flip tests the numerator. The BCa interval is for dz, with the denominator recomputed in every resample.
    - Secondary: PLAN.md §2's plain dz (mean / SD of all per-set Δcos, ignoring channels), and Δcos in units of the target-trial score SD. Following PLAN.md §2, that SD is estimated as the SD of the control-window cosines, because each control is a same-speaker pair.
  - **H2-P0:** the unweighted mean over channels of the per-channel AUC. The interval is for that mean.
    - **Score:** d(t) at the point itself; peak strength within ± c is used only for the false alarm at τ.
    - **Per-channel AUC:** within each cell, P(switch-point d > control d) + ½ P(tie). The cells are averaged with weights equal to their number of switch points.
  - **H3:** the unweighted mean over channels of the per-channel shift divided by a single target-score SD. That SD is the SD of every same-session monolingual target-trial score with a lag of at least 2 s in the analysed data (all channels, videos and lag bins, whether or not a bin counts). It is computed once and held fixed in the bootstrap.
  - **Intervals:** BCa bootstrap over channels, 10,000 resamples of whole channels, with jackknife acceleration.
    - Two-sided intervals at 1 − 0.05/m: 98.3% for H1 and H3, 99% for H2.
    - Equivalence (TOST) intervals at 1 − 2(0.05/m): 96.7% and 98%.
- **Outcome categories** (`csdiar/analysis/stats.py`, decision 16). Four flags are computed:
  - *significant:* Holm one-sided p < .05;
  - *meets the smallest effect:* the point estimate, in the predicted direction, is at least the smallest effect;
  - *reversed:* Holm two-sided p < .05, with the estimate in the opposite direction. The two-sided p-values are Holm-adjusted over the same family as the one-sided ones (m = 3, 5, 3);
  - *equivalent:* the TOST interval lies strictly inside the equivalence bounds. These are dz ∈ (−0.2, 0.2) for H1, AUC ∈ (0.45, 0.55) for H2-P0 and shift ∈ (−0.2, 0.2) target-score SD for H3.

  The category is the first of these that applies:
  1. *supported:* significant and meets the smallest effect;
  2. *nonzero but trivial:* significant, below the smallest effect;
  3. *reversed*;
  4. *equivalent*;
  5. *inconclusive*.

  All four flags are reported, so a result that is nonzero but trivial and also equivalent shows both.
- **Holm across stages:** the H2 family has m = 5. At M2a, P1 and P2 enter as p = 1, which can only raise P0's adjusted p-values compared with Holm on the real values. The M2a P0 verdicts are therefore final. M2b applies Holm to P1 and P2 with P0's observed p-values included.
- **Power** (decision 17; pilot variance components at their upper 80% bound):

  | Test | Effect | E1 | E2 | E3 |
  |---|---|---|---|---|
  | H1 | dz = 0.2 | 1.00 | 0.68 | 0.99 |
  | H2-P0 | AUC = 0.55 | 0.76 | 0.76 | 0.70 |

  An inconclusive result near the smallest effect is therefore weak evidence of absence for H2-P0 (all embeddings) and for H1 with E2; read the TOST interval with it.
- **Secondary** (reported, not tested for support):
  - Δcos in units of the target-trial score SD (the SD of the control-window cosines), and H1's plain dz;
  - H1 at L = 0.5 and 1.5 s;
  - P0 cell-standardised ΔFAR at τ;
  - the pooled `det` AUC (the detection cell PLAN.md §7 originally named primary; see §7), reported next to the primary result in every table;
  - the `det_w1` AUC (decision 16's sensitivity cell);
  - the strata of all three cells (< 0.5, 0.5–1.0, ≥ 1.0 s);
  - H3 cross-session shift;
  - EER and minDCF (P_target = 0.05), cross vs mono.
- **Exploratory** (labelled as such): effects against run length, pause, CMI, switch direction and speaker.

## 5. Sensitivity analyses (pre-registered; descriptive)
1. Exclude points whose boundary words have an alignment probability below 0.093, the 5th percentile of the 437,410 aligned train word probabilities. A boundary word without a score also removes its point, and the counts are reported.
2. Re-tag "I" and capital "A" as English (a variant boundary run).
3. Include the low-alignment clips (a variant boundary run).
4. Restrict to the purity-verified subset (PLAN §4 step 6). The rules are fixed here, and the audit runs after the freeze.
   - **Embeddings:** one embedding per clip (mean over speech) from two families not under test: NeMo TitaNet-L (`nvidia/speakerverification_en_titanet_large`) and WeSpeaker CAM++ (`Wespeaker/wespeaker-voxceleb-campplus`). If TitaNet-L cannot be run, 3D-Speaker ERes2NetV2 replaces it.
   - **Flagged clip:** under both embeddings, the clip's mean cosine to the other clips of its video in the same language-mix stratum is more than 3.5 robust SDs (1.4826 × MAD) below that stratum's median. The stratum is speech_type × the channel's CMI tertile, and the clip needs at least 5 other clips in it.
   - **Split-off video:** under both embeddings, the video's centroid has a cosine below the 1st percentile of within-video split-half centroid cosines (pooled over all videos) to every other video centroid of its channel.
   - **The subset:** flagged clips are dropped, and split-off videos become clusters of their own.
   - **Listening:** listening to flagged clips (capped at 200) is descriptive.
5. Contamination-corrected ΔFAR, once the listening check (450 blind items, PLAN §4 step 5) has been scored.
6. The whole primary analysis (§4) without the 2 test channels (15 train channels).
7. If access to Brouhaha (`pyannote/brouhaha`) is granted, the attribution models in §6 are repeated with Brouhaha's SNR in place of the SNR proxy, and with C50 added.
8. Loan words re-included with the language they are pronounced in, as PLAN.md §2 specifies (a variant boundary run). Devanagari-script loan-lexicon entries become `en`, and Latin-script ones become `ne`.

If the purity audit (item 4) or the listening check (item 5) is never completed, that analysis is reported as not done, and no substitute is run.

Items 2, 3, 4, 7 and 8 need code or inputs created after the freeze: the variant boundary runs, the audit, the Brouhaha features. That code is committed, and its hash recorded in `planning/DEVIATIONS.md`, before it is run on a confirmatory channel. It never changes the frozen primary code or features.

## 6. Acoustic covariates and attribution (decision 18)
- **Covariates** (`csdiar/data/acoustic.py`; parselmouth 0.4.7, Praat 6.1.38). They are computed per window, for the H1 windows (W_left, W_right) and for the P0 windows ([t − 1, t], [t, t + 1]). Speech frames are VAD frames with p ≥ 0.5.
  - **F0:** median of the voiced 10 ms frames, in semitones re 100 Hz. Praat autocorrelation pitch, with a per-channel range of 0.75 Q1 to 1.5 Q3 from a 60–700 Hz first pass. That pass uses up to 200 clips per channel, drawn without replacement by numpy `default_rng(20261018)`, with channels in sorted order and clips in `clip_id` order. Missing with fewer than 10 voiced frames.
  - **Energy:** dB of the mean power over speech samples. Missing with less than 0.2 s of speech.
  - **Articulation rate:** syllable nuclei per second of speech, after de Jong & Wempe (2009). Nuclei are intensity peaks above the clip's 0.99 quantile − 25 dB, with a dip of more than 2 dB before the next peak, and voiced. Missing with less than 0.3 s of speech.
  - **Spectral tilt:** least-squares slope of the long-term average spectrum, in dB per octave, over 17 one-third-octave bands from 125 Hz to 5 kHz. Missing with fewer than 6 speech frames.
  - **SNR proxy:** window energy minus the 10th percentile of 10 ms frame levels over the 2 s on that side of the window's inner edge. That 2 s span is clipped to the clip; missing if less than 0.5 s remains.
  - **Use:** each covariate enters as |right − left|. Rows with any missing covariate are dropped (complete cases), and the counts are reported.
- **Criterion (b), covariate adjustment tested at the channel level** (`csdiar/analysis/attribution.py`):
  - **Within each channel:** the outcome is regressed by least squares on the condition and the five covariates, after demeaning within sets. Demeaning absorbs set-level differences, including the pause and position bins.
    - **H1:** the window cosine over the L = 1.0 s matched sets (H1 windows; set = matched set).
    - **H2-P0:** the false alarm at τ (an NMS peak of d(t) above τ within ± c), over the `p0L1` switch points with cell controls and the `det_w1` controls in those cells (P0 windows; set = cell). This is the operating-point estimand of PLAN.md §7's sensitivity model, not the AUC.
  - **Counting:** a set counts only with at least one switch point and one control left after the complete-case filter. A channel enters with at least 10 such switch points, from the channels the primary test used.
  - **Test:** an exact one-sided sign-flip over channels on the per-channel condition coefficients, in the predicted direction (H1 negative, P0 positive).
- **Mixed models, descriptive** (`csdiar/analysis/glmm.R`; R 4.5.3, lme4 2.0-6; exact builds in `ops/locks/r-glmm.explicit.txt`):
  - **What they are:** PLAN.md §7's mixed models, fitted on the same rows as (b) and reported as a model-based summary, with the condition coefficient with and without the covariates. They do not enter the grade. Without the random slope, which the pilot suggests will be dropped, a Wald z ignores between-speaker variation in the effect and is not a valid level-α test.
  - **Models:** H1 is a linear mixed model of the cosine; H2-P0 is a logistic GLMM of the false alarm. Both have the fixed effects condition + 5 covariates (z-scored) + pause bin + position bin.
  - **Random effects:** start from (1 + condition | speaker) + (1 | video/clip) + (1 | set). The drops are cumulative, in this order: the random slope; then (1 | video/clip) becomes (1 | clip); then (1 | set) is removed.
  - **Fitting:** bobyqa is used, then Nelder–Mead after an error or a convergence warning. The first structure that converges and is not singular (`isSingular`, tolerance 1e-4) is used. Otherwise the last fit that ran is reported and flagged.
- **Phrase-boundary contrast** (`csdiar/analysis/attribution.py`): the primary statistic is recomputed using only points with a pause > 150 ms, tested with an exact one-sided sign-flip over channels. A channel enters only with at least 5 such units. By the pre-freeze counts, about 14 of the 17 channels qualify.
  - **H1:** the matched sets whose switch point has a pause > 150 ms. Exact matching on pause bin makes their controls phrase-boundary controls.
  - **P0:** the cell-standardised AUC within the 150–300 ms and > 300 ms pause cells.
  - **Descriptive:** FAR at τ of the phrase-boundary controls, the other `det_w1` controls and the switch points.
- **Grading,** for each embedding and each of H1 and H2-P0. This is an intersection–union rule: each condition is tested at α = .05, with no further adjustment.
  - **Language-attributable:** all four hold. (a) The primary test is significant (Holm, §4). (b) The channel-level adjusted test is significant. (c) The phrase-boundary contrast is significant. (d) H3 is significant for that embedding (Holm).
  - **Switch-associated:** (a) holds, but not all of (b)–(d).
  - **Not shown:** (a) fails.
  - **Assessability:** (b) and (c) need at least 5 channels, because with k ≤ 4 an exact one-sided sign-flip cannot reach p < .05 (2^−4 = .0625). With fewer, the criterion is "not assessable": it is flagged, and the grade cannot be language-attributable.
  - **Reporting:** every grade is reported with the primary outcome category, e.g. "language-attributable (nonzero but trivial)". The grade does not require the smallest effect; the category says whether it was met.
  - **Scope:** each grade is a claim about one embedding and one hypothesis. No claim is made across embeddings, and no multiplicity adjustment is made across the six grades.
  - **The rule is conservative:** F0 and tilt changes may be partly caused by the language change itself, so adjusting for them can remove part of a real language effect.

## 7. Hygiene
- **One run:** confirmatory outcomes are computed only by `csdiar/analysis/confirm.py` and `csdiar/analysis/attribution.py`. Both refuse non-validation splits unless every hash and every input path in the freeze record matches. Each test set is scored once per frozen configuration.
- **What the freeze record covers:**
  - **Code:** the guarded code (`confirm.GUARDED`) and this document.
  - **Train and test features:** boundaries, point embeddings (all L), dense L = 1 s embeddings, H3 windows, acoustic covariates and train word alignments.
  - **τ:** the pilot JSON that holds it.
- **Logging:** every run logs its configuration, seed (20261018), git SHA and data revision.
- **What the pilot fixed:** τ, the primary P0 cell and the definitions. The pilot saw no confirmatory-channel outcome.
- **Disclosure: the primary P0 cell was chosen after the pilot** (decision 16).
  - PLAN.md §7 named the detection family, with strata pooled, as primary.
  - The pilot gave a pooled AUC of 0.49. By stratum it was 0.48 for runs under 0.5 s and 0.59–0.61 for runs of at least 1 s.
  - The review found that P0's 1 s windows reach past that family's ± 0.5 s clean region, so the comparison was not like for like.
  - The new primary cell, `p0L1`, was chosen on these construct grounds, but after seeing those numbers. Its pilot AUCs were 0.615, 0.586 and 0.592.
  - The confirmatory channels are disjoint from the pilot, which protects the error rate, but the question itself was informed by the pilot. The pooled `det` AUC is therefore reported next to the primary result in every table.
- **Scope:** everything outside the named primary cells is secondary or exploratory.
