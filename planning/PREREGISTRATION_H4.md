# Pre-registration supplement: H4 on D1, the cross-validated reduction of the SS-DL excess

Status: FROZEN at tag `h4-freeze` (record `planning/FREEZE_H4.json`, tag `h4-freeze-record`). Design reviews: round 1
(3 major) and round 2 (no major, `planning/reviews/review_h4_supplement_v{1,2}.md`). Code reviews of the analysis
and the M3 code: round 1 (3 + 1 major) and round 2 (no major, `review_h4_code_v{1,2}.md`,
`review_m3_code_v{1,2}.md`). Nothing in this document had been run on a scored channel at the freeze. It fixes
PLAN.md §6 E-B (i), which
`PREREGISTRATION.md` deferred "once M1–M3 exist". The joint H4 × H5 trade-off on the DISPLACE eval set, H6, and
the conversation parts of H2 and H4 belong to the M2b supplement.

## 1. Hypothesis and family

H4 (PLAN.md §7): each mitigation lowers the SS-DL excess against its retuned baseline B0. Family M1, M2, M3;
Holm m = 3. On D1 the excess is the cell-standardised false-alarm excess of the detector P0 at switch points
(the quantity H2-P0 found: ΔFAR +2.5 to +3.0 pp, `results/confirm/summary_m2a.md`).

A D1 H4 verdict says only that the excess at same-speaker switches went down. Whether a mitigation pays for that
with missed real speaker changes is H5, tested jointly with H4 on DISPLACE (intersection-union, PLAN.md §7).
Results are always reported with that caveat until M2b.

## 2. Back end on D1: P0 with E3

- D1 clips are 3–14 s. Clustering pipelines almost never split on clips (PLAN.md §6 E-A), and the long-form
  audio D1-LF does not exist (decision 6 and the refused YouTube downloads). P1-AHC on D1 clips is therefore
  descriptive only; it stays the primary back end on DISPLACE (M2b).
- P0 is used exactly as frozen in M2a (`csdiar/analysis/pilot.py`): L = 1 s windows, hop 0.1 s,
  d(t) = 1 − cos(left, right), NMS 0.5 s, a hit when the strongest peak within ±c = 0.5 s exceeds τ. Points are
  the frozen `boundaries/lex_v1b`; the primary cell is `p0L1` switch points against `det_w1` controls, cell
  (video, pause bin, position bin) standardised as in `pilot.h2_cell`.
- Embedding: E3 (M3 starts from it, and PLAN.md names E3 for the primary H4–H6 back end). M1, M2, B1 and B2 with
  E1 and E2 are secondary.

## 3. Channels and folds

- **Scored:** the 15 D1 train channels, in 5 folds of 3: `numpy.random.default_rng(20261018).permutation` of the
  sorted channel IDs, taken in consecutive triples. The assignment is written into the freeze record.
- **Fitting side of fold f:** the other 12 train channels. B1's means, B2's offset and M1's direction are
  estimated there, and M3 trains there.
- **Calibration channels:** the 3 pilot channels, without video `zJYS-XNDP2Q`, whose teacher is very likely the
  teacher of scored channel `UCOC6C8_U9PsAs3tQodDDmGw` (decision 27). Every τ (§5) and M3's early stopping use
  them. No system is fitted or trained on them, so no operating point is set on a system's own fitting data; M3's
  checkpoint choice on them is ordinary dev use.
- **Pilot channels are never scored.** The H4 pilot (commit `2a71d20`, running `m1_subspace.py` exactly as in
  `45f92ce`, unchanged since) computed outcomes on them, which PLAN.md §6 says the pilot does not do. Seen,
  as mean ΔFAR for B0 / a language-agnostic centring stand-in for B1 / M1: E1 .0295 / .0292 / .0317,
  E2 .0251 / .0364 / .0272, E3 .0295 / .0211 / .0424. That stand-in is not the B1 defined here.
- **Test channels (2):** descriptive only (PLAN.md §6 E-B (iii)), scored once after stage 1 by a separate
  descriptive script, with B1, B2 and M1 fitted on all 15 train channels, τ on the calibration channels, and M3's
  all-data models (5 seeds).
- **Inclusion:** a scored channel enters the test with at least 10 `p0L1` switch points that have cell controls,
  as in M2a.

## 4. Systems

All systems use the frozen E3 dense grid at L = 1 s (`emb/e3_L1`; window k covers [0.1k, 0.1k + 1] s), except
M3, which re-embeds the same grid with its own weights.

**Window language.**
- *Reference (fitting side only):* the frozen clean monolingual windows (`windows/lex_v1`, decision 14: one
  language from one run, at least 2 words, at least 60% speech, thinned to starts ≥ 1 s apart), which are already
  on this grid. B1's means and B2's δ use them.
- *Predicted (any channel, deployable):* the VoxLingua107 posteriors (`csdiar/models/lid.py`: 1 s windows, hop
  0.1 s, the same grid as the embeddings; stored in `lid/*.npz`), renormalised over (ne, en); `ne` if
  p(ne) ≥ 0.5, else `en`. Test-time systems only ever use predicted labels.

**Baselines** (PLAN.md §5):
- **B0:** E3 unchanged. Its τ (§5) follows the M2a rule on the calibration channels; without `zJYS-XNDP2Q` it can
  differ slightly from the frozen M2a P0-E3 τ (0.929), and both are reported.
- **B1 language-wise mean centring:** e′ = unit(e − μ_ℓ), with ℓ the window's predicted language and μ_ℓ the mean
  of the unit E3 vectors of reference-ℓ windows on the fitting side.
- **B2 language-aware score offset (Thienpondt-style):** d′(t) = d(t) − δ · 1[ℓ̂(window k−10) ≠ ℓ̂(window k)].
  δ is the fitting side's verification-score shift, not the switch-point excess: the mean cosine of same-video
  same-language pairs of reference windows minus that of same-video cross-language pairs, both with starts ≥ 2 s
  apart (the H3 trial rule; at most 40,000 pairs per video drawn with seed 20261018). δ is clamped to [0, 0.5].

**Mitigations:**
- **M1 SubspaceProjector**, as implemented in `csdiar/mitigations/m1_subspace.py` (`45f92ce`): v̂ is the unit mean
  over (speaker, video) of [mean en-side − mean ne-side] E3 point windows (L = 1 s) at the fitting side's SS-DL
  points; e′ = unit(e − (e·v̂)v̂). It has no tuned parameter.
- **M2 LanguageGatedSCD:** λ(t) is the base-2 Jensen–Shannon divergence between the renormalised (p_ne, p_en) of
  LID windows k−10 and k, the windows left and right of t = 0.1k (as in `csdiar/analysis/m2_lambda.py`). P0
  thresholds the gated curve d′(t) = d(t) − α·λ(t), i.e. τ(t) = τ0 + α·λ(t) written as a score; NMS and peak
  picking run on d′, exactly as P0 runs on d. B2's gated curve is handled the same way.
  - **α comes from DISPLACE dev**, tuned there for this detector (P0 with E3 and the same gate, not the pipeline
    post-processor, whose scale differs) by PLAN.md's rule (minimise FAR_SS-DL subject to the upper one-sided 80%
    bound of ΔMR_DS-DL vs B0 ≤ +1 pp), and frozen in the M2b supplement. D1 has no real cross-speaker turns, and
    any α > 0 lowers switch-point hits on D1 while costing nothing there, so a D1-tuned α would let M2 win by
    construction (review round 1). τ0 is set on D1 as in §5.
  - **Staging:** until α is frozen, M2 enters the H4 Holm family with p = 1. That can only raise M1's and M3's
    adjusted p-values, so their verdicts at this stage are final (as for P1/P2 in M2a). M2's D1 test runs once
    after the M2b freeze. If DISPLACE never arrives, M2's D1 H4 is reported as not tested.
- **M3 LanguageAdversarialFineTuner:** one model per fold and seed (§7). Each re-embeds the L = 1 s grid of all
  18 non-test channels; the all-data models also embed the 2 test channels.

## 5. Operating point

For every system, fold and M3 seed, τ (τ0 for B2 and M2) is set on the 3 calibration channels so that 10% of
their `det_w1` controls fire, with `pilot.calibrate_tau`. If that returns −inf (finite-peak share ≤ 10%) for any
fold or seed, the system is reported as not assessable. Matching the control fire rate replaces PLAN.md §7's
"threshold that matches the baseline's dev MR_DS-DL", which needs DS-DL turns (§9).

## 6. Test

- **Per scored channel c and system S:** ΔFAR_S(c) is `pilot.h2_cell`'s `delta_far_std` for c in the `p0L1`
  cell, at the τ of c's fold. For M3 it is the mean over its 5 seeds.
- **Cluster statistic:** δ_M(c) = ΔFAR_M(c) − ΔFAR_B0(c).
- **Test:** exact one-sided sign-flip over the included scored channels (mean δ < 0; 2^k patterns); Holm over
  M1–M3 (M2 staged, §4).
- **Estimand:** the mean over channels of δ_M(c), in pp. Its scale is S_B0 = mean_c ΔFAR_B0(c), computed once
  on the included channels and held fixed in the bootstrap (like H3's target-score SD). The relative reduction
  R_M = −mean δ_M / S_B0 is reported with it. BCa interval for mean δ_M over channels (10,000 whole-channel
  resamples, jackknife acceleration): two-sided 98.3%, TOST 96.7%.
- **Smallest effect:** −mean δ_M ≥ 0.25 · S_B0 (a 25% relative reduction, PLAN.md §7).
- **Outcome categories** as in `PREREGISTRATION.md` §4 (`csdiar/analysis/stats.py`): significant (Holm one-sided
  p < .05); meets the smallest effect; reversed (Holm two-sided p < .05 with mean δ > 0); equivalent (TOST
  interval of mean δ inside (−0.25 S_B0, +0.25 S_B0)). Precedence: supported, nonzero but trivial, reversed,
  equivalent, inconclusive.
- **Degenerate baseline:** if S_B0 ≤ 0 there is no excess to reduce; δ is reported and H4 is inconclusive for
  every mitigation.

## 7. M3 recipe

- **Start:** E3 (UniSpeech `wavlm_large_finetune.pth`, SHA-256 in `csdiar/models/e3.py`), 256-d output. WavLM
  stays frozen, as in E3's own inference path (`update_extract=False`); the WavLM layer weights, the ECAPA-TDNN
  and the new heads are trained. With 12 training channels, full WavLM fine-tuning would mostly add capacity to
  memorise them, at about ten times the cost (§9). The distillation target is the frozen E3 head on the same
  WavLM features.
- **Data per fold:** the fold's 12 fitting-side train channels with the audit's flagged clips dropped. Speakers are
  channels, except that a video becomes its own speaker when its best within-channel centroid cosine is below the
  99.9th percentile of cross-channel video-centroid cosines under both audit embeddings (decision 27; the frozen
  split-off clusters of `PREREGISTRATION.md` §5 item 4 also split sessions of one teacher and are used only for
  that sensitivity analysis). Added to them is the Nepali speaker pool D5 =
  IndicVoices Nepali: utterances of at least 2.0 s, speakers with at least 20 of them, 50 per speaker drawn with
  seed 20261018 (all if fewer). D1 training windows come from aligned speech in clips that are not low-alignment
  (decision 10). A window is `ne` or `en` when that language covers at least 80% of the aligned ne+en word time
  inside it (frozen alignment, `lex_v1b` tags, other tags ignored), `mixed` when both occur but neither reaches
  80%, and unused when it holds less than 0.4 s of aligned ne/en word time. Tag noise is measured on the 777
  human-verified clips (window labels from the silver transcript vs the verified one) and reported.
- **Batch (96):** 48 D5 crops (48 distinct speakers drawn uniformly, one random crop of a random utterance each)
  and 48 D1 windows: 12 distinct D1
  speakers drawn uniformly from the fold's speakers (after the purity remap) × 4 windows each (2 `ne` and 2 `en`;
  `mixed` windows substituted when a speaker lacks one language). One window length per batch, drawn uniformly
  from {1.0, 1.25, 1.5, 1.75, 2.0} s and shared by all 96 items, so a step is one forward pass (batch-norm
  statistics over both halves). The −D5 ablation drops the D5 half (batch 48).
- **No purity subset:** if the audit is not completed, M3 trains on all clips with channel IDs as speakers, and
  this is reported.
- **Losses:** L = L_AAM + L_SupCon + L_GRL + w_dist · L_dist.
  - AAM-softmax (margin 0.2, scale 30) over the D1 fitting-side speakers and the D5 speakers, on all 96 items;
  - supervised contrastive (temperature 0.1), D1 only: positives are all other windows of the same speaker in
    the batch, so cross-language and `mixed` (high-CMI) windows are positives;
  - gradient-reversal language head (ne / en / mixed; a 2-layer MLP), D1 only, reversal coefficient ramped
    0 → 1 over the first 30% of steps (Ganin et al. 2016);
  - squared L2 distance between unit embeddings and the frozen E3's, w_dist ∈ {0, 0.1, 0.3, 1.0}; **0.3 is the primary**
    configuration and the other three form the reported trade-off curve.
- **Optimisation:** AdamW; lr 2e-5 for the layer weights and the ECAPA-TDNN, 1e-3 for the AAM prototypes and the
  language head; weight decay 0.01. For the first 500 steps only the new heads train: the body's weights stay
  fixed while its batch-norm statistics adapt in train mode. The learning-rate schedule has 500 warm-up steps,
  then cosine decay to 0 at 6,000 steps. fp32 throughout, E3's own numerics (bf16 WavLM features were only 25%
  faster); gradient clipping at 5. No augmentation. The body learning rate and the warm-up were chosen on the
  dev (calibration) channels before the freeze (decision 28): the first draft's 2e-4 made the dev EER worse than
  E3's.
- **Early stopping:** every 500 steps, dev EER on the calibration (pilot) channels: cross-language same-speaker
  trials (same video, starts ≥ 2 s apart) against different-channel non-target trials, both between frozen clean
  windows (`windows/lex_v1`),
  20,000 of each drawn once with seed 20261018. Patience 3; the best checkpoint among steps ≥ 500 is kept, so the
  unmodified E3 can never be selected. The H4 statistic is never computed during training.
- **Seeds:** 0–4 for the primary configuration in every fold and for the all-data model; seed 0 for each other
  w_dist (w_dist = 0 is also the −distillation ablation) and for the ablations −GRL, −contrastive, −both and −D5,
  in every fold.

## 8. Secondary (reported, not tested for support)

- Each mitigation against B1 and B2, and B1 and B2 against B0 (same statistic, unadjusted p). B2's gate only
  lowers d where LID predicts a language change, and its τ0 is set on same-language controls, so on D1 it can
  lower switch-point hits at no D1 cost, like a D1-tuned M2. Comparisons with B2 are therefore biased against
  the mitigations on D1 and are read together with H5.
- The `p0L1` AUC of every system, FAR_SS-DL, and FAR_CTRL on the scored channels (operating-point check).
- λ's AUC for SS-DL points against controls on the scored channels, before any use of α (PLAN.md §5); the DS-SL
  half needs conversations and goes to M2b.
- M1: the k-dimensional subspace (k ∈ {1, 2, 4, 8, 16}, chosen inside the folds on the fitting side's ΔFAR) and
  its specificity controls, a random direction and the top PCA direction of the same k.
- M2: oracle λ (reference languages differ) and time-shuffled λ (permuted within each clip, a null control).
- M3: the distillation curve and the ablations; per-seed spread.
- B1, B2, M1 and M2 with E1 and E2.
- `PREREGISTRATION.md` §5 sensitivities as they apply: the purity-verified subset (item 4), the alignment-score
  filter (item 1), and the logistic GLMM of PLAN.md §7 with a system term (descriptive).
- Speaker information kept: same- and cross-language EER on the scored channels for every embedding, and the two
  UMAP companion numbers (linear-probe language accuracy, speaker-identification accuracy; PLAN.md §6).
- The 2 test channels, for everything above. P1-AHC on D1 clips: change counts at SS-DL points and controls.

## 9. Departures from PLAN.md, and what was known

1. 15 scored channels in 5 folds instead of 18 in 6, because the pilot saw M1/B1 outcomes on the 3 pilot channels.
2. P0 with E3 is the D1 back end for H4, because pipelines do not split on clips and D1-LF does not exist; P1-AHC
   stays primary on DISPLACE.
3. The operating point matches the control fire rate (10%) on the pilot channels, not the baseline's dev
   MR_DS-DL, which needs DS-DL turns.
4. M2's α is tuned on DISPLACE dev and M2 is staged in the Holm family (§4).
5. D5 is IndicVoices Nepali instead of OpenSLR 54: real conversational and extempore speech from 1,076 speakers
   rather than read speech (DECISIONS.md, IndicVoices audit).
6. B1 and B2 use LID-predicted window languages at test time; reference tags would make them oracles. B2's δ is
   the verification-score shift, so it is not fitted to the H4 statistic.
7. M3 uses 5 seeds for the primary configuration only, and one seed per fold elsewhere (GPU budget, PLAN.md §8).
8. PLAN.md §5 lists B1 and B2 among the baselines "every mitigation must beat", but its H4 row (§7) tests against
   B0. Here the comparisons with B1 and B2 are secondary and unadjusted, so the confirmatory family stays m = 3,
   and because B2 is biased on D1 (§8).
9. M3 keeps WavLM frozen and fine-tunes E3's layer weights and ECAPA head. PLAN.md says only that M3 starts from
   E3.

What was known when this was written: the M2a and H7 outcomes, including H2-P0 on these 15 channels, and the
pilot numbers in §3. E3 and the `p0L1` cell were not chosen on those outcomes: PLAN.md named E3 for H4–H6 before
any data, and `p0L1` is the cell frozen in M2a. The M3 recipe, the B2 and M2 rules and the operating point were
written after them.

## 10. Hygiene

- `planning/FREEZE_H4.json` records the code hashes (`confirm_h4.py`, `pilot.py`, `stats.py`, the mitigation and
  M3 modules), the feature hashes (E3 dense grid, LID, points, point embeddings, the purity subset), the fold
  assignment and the seeds; tags `h4-freeze` and `h4-freeze-record`.
- M3 training and re-embedding run after the freeze; their checkpoints and grids are hashed and appended to the
  record before `confirm_h4.py` runs. `confirm_h4.py` refuses to run without a matching record and runs once per
  stage (M1 and M3, then M2 after M2b).
- Changes after the freeze go to `planning/DEVIATIONS.md`.
