# Pre-registration supplement, stage M2b: the conversation data (H5, H6, H4 × H5 jointly on DISPLACE, M2's α, and the conversation parts of H2 and H4)

Status: **DRAFT v4, never frozen**. Design reviews: round 1 (4 major, 6 minor), round 2 (1 major, 6 minor), round 3 (no major, 5 minor), all applied. The archives' file listings were done (§2), but the DISPLACE 2024 password never arrived. The dev labels and counts that would have set the values marked *not determined* below were never read, and stage M2b was not run (`planning/DEVIATIONS.md` entry 7). Had it been frozen, the record `planning/FREEZE_M2B.json`
(tag `m2b-freeze`) would have pinned the code hashes, the DISPLACE feature and reference checksums, the dev/eval split
taken as released, and the commit.
Nothing in this document will have been run on a DISPLACE eval recording at the freeze. Changes after the freeze
go to `planning/DEVIATIONS.md`, with dates.

This supplement consolidates PLAN.md §2, §4, §5, §6 E-B (ii) and (iv), §7 (the H4–H6 rows, the trade-off DET and
the intersection–union test) and §8 with decisions 6, 9, 10, 15, 20, 21, 23, 24, 26 and 29 in
`planning/DECISIONS.md`. Where the two
differ, the decisions take precedence; this document states the result. It fixes what `PREREGISTRATION.md`
(M2a) and `PREREGISTRATION_H4.md` (the D1 H4 supplement) deferred to M2b: H5, H6, the joint H4 × H5 test on
DISPLACE, M2's α (named but not frozen in the H4 supplement), M2's D1 stage-2 test, and the conversation parts of
H2 and H4. The M2a and H4 documents are frozen and are not reopened here.

This document is written before any DISPLACE eval scoring, but **after** the D1 confirmatory results (M2a,
decision 23; the H4 supplement is frozen, its confirmatory run not yet executed). What was known is stated in
§12. Every number that depends on the contents of the password-protected archives (only their file names, sizes
and checksums are readable, §2), or on dev counts that need the extracted data, is marked **not determined** with the rule that would have set it.

## 1. Question and claim grading on conversations
Do the mitigations lower the SS-DL excess at same-speaker language switches **without** missing real
cross-lingual speaker turns, on real multi-speaker code-mixed conversation? H4 is the reduction of the SS-DL
excess; H5 is non-inferiority on missed DS-DL turns; H6 is the ordering of the three mitigations' trade-offs.
H4 and H5 are tested jointly, per mitigation, by an intersection–union rule (§7), because a mitigation that buys
a lower false-split rate with extra missed turns has not helped.

Claim grading (PLAN.md §1, §2) carries over: an excess at switch points is *switch-associated*, and
*language-attributable* only if it survives the acoustic-change covariates and the phrase-boundary controls and
H3 points the same way. On DISPLACE the covariate and phrase-boundary machinery of PREREGISTRATION.md §6 is run
where the references allow it (§2); where they do not, the conversation H2/H4 results are reported as
switch-associated only, and said to be so.

## 2. Data: DISPLACE 2024
- **Corpus (checked, arXiv 2406.09494, CC BY 4.0):** 158 h of mono-channel far-field recordings for the speaker-
  diarization (SD) and language-diarization (LD) tracks, of which **38 h are annotated (supervised)** and 120 h
  are unsupervised; recordings from four academic institutes under one recording setting; natural overlap,
  reverberation and noise. A separate 12 h of close-field ASR data on five Indian languages is **not used** here.
  The supervised speaker references (SD track) and segment-level language references (LD track) are the two
  tracks this study reads. Zenodo record 14097187 (version 3 of 12166687) holds
  `Track1_SD_supervised.zip` and `Track2_LD_supervised.zip`, password-protected; the password never arrived (decision 26;
  `planning/DEVIATIONS.md` entry 7). The encrypted archives were fetched in advance; nothing was
  extracted, listed beyond file names, or scored. Both archives are fetched and
  checksum-verified, and their central directories (names, lengths, CRC-32; nothing extracted) fix the layout:
  speaker RTTMs `Track1_SD_supervised/{Dev/Track1_SD_labels, Eval/SD_eval_rttms}/<ID>_SPEAKER.rttm`, language
  RTTMs `track2_LD_supervised/{Dev/Track2_LD_labels, Eval/LD_eval_rttms}/<ID>_LANGUAGE.rttm` (lower-case top
  folder); the **same 35 dev and 32 eval recording IDs** in both archives; and audio **byte-identical** across
  them for all 67 recordings (same length and CRC-32), so one copy serves both references. The reader stops if a
  split's speaker and language references cover different recordings.
- **Why DISPLACE 2024, not DISPLACE-M** (decisions 6, 29): DISPLACE-M is Hindi health conversation with verbatim
  transcripts rendered in Devanagari and **no language annotation**, so its DS-DL turns could only come from
  automatic LID. DISPLACE 2024 has a language-diarization track with **segment-level language references**, which
  is exactly what H5 needs. DISPLACE-M stays a secondary set (§8) and, only if DISPLACE 2024 is never granted, the
  fallback of §10.
- **Languages:** multiple Indian languages with English; no DISPLACE document names Nepali (decision 6,
  correction in PLAN.md §11). The H5/H6 confirmatory claim is therefore about **Indian-language–English**
  conversation; D2-P supports it on Nepali (§8, confirmatory only if it reaches the simulated size). The exact
  set of languages present and their per-recording pairs are read from the LD references at ingest and reported;
  which languages appear, and how the LD label strings map to VoxLingua107 codes, are **not determined**: the dev
  references were never extracted, and the listing shows file names only. The mapping is fixed before the freeze from
  the dev label sets and the release's language list (English, Hindi, Bengali, Kannada, Telugu); an eval label
  outside it takes the full-posterior fallback below for that recording, flagged, and the mapping is not revised
  once eval is read.
- **Units and clusters:** the released **dev/eval split is kept** (PLAN.md §7); recordings are **clustered by
  recording session** for every conversation test (PLAN.md §6, §7). The cluster is the recording session; the
  analysis unit within it is the reference DS-DL turn (H5) or the SS-DL point (H2/H4). Speaker counts, recording
  counts and per-split hours from the Track 1 file listing: **dev 35 recordings (about 19.7 h), eval 32 (about 17.9 h)**, each 26–60 min, matching the paper's 38 h supervised total; speaker counts are read from the dev and eval RTTMs at ingest. Up to 32 eval sessions enter (both archives list the same IDs); the sign-flip is Monte Carlo unless 17 or fewer pass the inclusion rule, in which case it is exact (§7).
- **Reading SS-DL and DS-DL from the two reference tracks** (`csdiar/data/ingest_displace.py`, hashed before the
  freeze; no eval statistic computed before it): the **speaker reference** is the SD-track RTTM (speaker-labelled
  segments); the **language reference** is the LD-track RTTM (language-labelled segments), which are
  **segment-level**, coarser than D1's word alignment (PLAN.md §3, §11). The two segmentations are intersected
  on one elementary timeline. An interval is *clean* when exactly one speaker and one language are active,
  *overlap* when two or more speakers are, and *ambiguous* when speech carries no language or two or more. A
  **turn** is a maximal stretch of one speaker's clean speech, joined across silences shorter than 2.0 s in
  which no other speaker talks. A **language run** is a maximal same-language stretch inside a turn; its length is
  its clean speech time. Overlap and ambiguity are excluded and counted (§4 overlap); the silent, language-free
  gap between two runs of the same speaker is the junction itself, not an ambiguity.
  A **DS-DL turn** is a reference speaker change (A→B, B ≠ A, B starting within 2 s of A's end, no overlap
  across the transition, PLAN.md §2) where A's last run and B's first run differ in language; the H5 units; turn
  time = the middle of the gap. An **SS-DL point** is the junction between two consecutive language runs of one
  turn; the conversation H2/H4 units. Its time t is the midpoint of the longest VAD non-speech stretch inside the
  junction gap widened by 0.1 s on each side (never past the runs' midpoints), else the gap's middle, as decision
  10's word-junction rule on D1. PLAN.md §2's exclusion zone applies: a point within 2c of any reference speaker
  change, overlap or ambiguous interval is dropped and counted, so DS errors cannot leak into the SS statistics.
  DS-SL turns are read the same way, for the turn-gap-stratified DETs (PLAN.md §2) and M2's λ-vs-DS-SL check
  (§8, the half decision 15 could not do on D1). `csdiar/data/ingest_displace.py` implements these rules.
- **Controls (FAR_CTRL):** D1's controls were word junctions inside a run; DISPLACE has no word alignment, so a
  control is a point inside a **clean same-speaker same-language run**, at least 2c from the run's edges and from
  any SS-DL point or turn. Candidates are (a) the midpoint of every VAD non-speech stretch inside the run, placed
  first so that each pause yields a control, then (b) a 0.5 s grid; controls are kept at least 2c apart, so their
  ±c firing regions never overlap. The **pause** at any time t, for SS-DL points, turns
  and controls alike, is the length of the longest Silero VAD non-speech stretch overlapping [t − 0.1, t + 0.1]
  (0 if none), binned as on D1 (none, < 150, 150–300, > 300 ms; decision 10). The cell is (session, pause bin),
  and ΔFAR is cell-standardised exactly as `pilot.h2_cell` does on D1: each SS-DL point is compared with every
  eligible control of its own cell, controls reused. A seeded 1:3 matching on pause bin (seed 20261018) is a
  sensitivity analysis, as on D1. Controls with a pause > 150 ms are the phrase-boundary controls of the
  attribution criterion (PREREGISTRATION.md §6).
- **Scoring a hypothesis** (PLAN.md §2): a point *fires* when at least one hypothesized speaker change lies within
  ±c of it (FAR_SS-DL, FAR_CTRL); a DS-DL turn is *missed* when no hypothesized change is matched to it within ±c
  under one-to-one nearest matching (pyannote's SegmentationRecall with tolerance; MR_DS-DL). A hypothesized
  change of P1-AHC is a boundary in its output where the dominant hypothesis speaker changes; P0's are its NMS
  peaks above τ. Changes per minute and the global change rate are reported beside every rate.
- **Collars and run lengths:** the primary collar is **c = 0.5 s** (PLAN.md §2 names 0.5 s primary; the ±0.25 s
  cell is **not** used on DISPLACE because its language references are segment-level, PLAN.md §3, correction
  §11). The primary collar is fixed at 0.5 s; FAR as a function of collar (0.1–1.0 s) is reported as a
  descriptive curve and never replaces it. A minimum-run-length rule matching D1's
  `p0L1` cell (both runs ≥ 1 s) is applied to the conversation H2/H4 SS-DL points **where the segment references
  allow it**; segment granularity may leave few sub-second runs resolvable, so the achievable minimum was to be
  read from the **dev** references' timing distribution (never eval) and reported. Run-length strata (< 0.5, 0.5–1.0, ≥ 1.0 s,
  PLAN.md §2) are reported for SS-DL points and for DS-DL turns, pooled for the primary test.
- **Overlap handling:** overlap is scored (PLAN.md §2 "overlap scored"; the pipelines are overlap-aware,
  `p1_ahc.py`). A DS-DL turn whose two speakers overlap through the transition, and an SS-DL point inside an
  overlapped region, are excluded from the primary cells and counted, because the single far-field channel cannot
  attribute overlapped speech to one reference run; a sensitivity analysis re-includes overlapped turns scored
  against the overlap-aware hypothesis (§9).
- **LID gate generalisation to DISPLACE's languages:** on D1 the gate (M2, `confirm_h4.py`) renormalised the
  VoxLingua107 posterior over (ne, en). On DISPLACE the gate **renormalises over the reference languages present
  in that recording's LD track** (all of them when the track names more than two; the share of such recordings is
  reported), giving λ(t) = base-2 JSD of the renormalised left/right windows exactly as in `m2_lambda.py` /
  `confirm_h4.py`. This treats the conversation's language inventory as deployment configuration, as (ne, en) is
  on D1, while the language at each moment is always predicted. A base-2 JSD is at most 1 bit for any number of
  languages, so the relabel gate's "λ ≥ 0.5 bit" means the same share of its range everywhere; the language count
  of each recording is reported beside its relabel rate. Reported beside it: (a) a **full-posterior** λ (JSD over the whole VoxLingua107 label set, no
  renormalisation, assuming no reference language set — the deployable fallback), and (b) **oracle** λ (from
  reference tags, upper bound) and **time-shuffled** λ (null control), as on D1. VoxLingua107 covers all five languages the
  release lists (bn, en, hi, kn, te are in the model's label encoder); a reference label that maps to none of its
  labels falls back to full-posterior for that recording, flagged.
- **DISPLACE-M as fallback only** (§10): Hindi–English, segment speaker references, **no language annotation**, so
  its DS-DL turns would be automatic-LID-derived and H5 on it is partly automatic — a limitation that bars it from
  the primary confirmatory role (decision 29).

## 3. Tuning on dev only
All fitting and tuning use the **DISPLACE dev set** (and the fitting sides of D1 where a system was already fit
there). No eval recording is scored until every operating point below is frozen into `FREEZE_M2B.json`.
- **M2's α (the quantity the H4 supplement deferred):** tuned on DISPLACE dev **for the P1-AHC post-processor**
  (the primary conversation back end, §5) and, separately, **for P0 with E3 and the same gate** (the detector the
  D1 stage-2 test uses, H4 supplement §4), because the two gates act on different score scales and PLAN.md warns a
  single α cannot serve both. Rule (PLAN.md §5, §7; H4 supplement §4): α and τ0 are tuned **jointly** on dev to
  **minimise FAR_SS-DL subject to the upper one-sided 80% bound of ΔMR_DS-DL (vs B0) ≤ +1 pp**, the bound being
  the one-sided 80% BCa bound over dev sessions (10,000 whole-session resamples). Ties go to the smaller α, then
  to the lower dev MR_DS-DL, then to the smaller τ0. The grids are fixed now:
  - **P0 gate:** α ∈ {0, 0.05, …, 1.00}; τ0 = B0's dev τ (10% of dev controls fire, §5) + {−0.10, −0.09, …, +0.10}.
  - **P1-AHC post-processor:** τ0 = 0 and α ∈ {0, 0.05, …, 1.00} on the cosine-distance scale, so the gate is
    distance < α·λ(t) and the post-processor adds only α·λ to B0, which never relabels; "λ is high" means λ ≥ 0.5
    bit. Draft v4 set τ0 to the system's clustering threshold instead. That is not a no-op at α = 0, because the
    distance between two output segments' own embeddings can fall below the threshold AHC applied to cluster
    centroids, and it broke the feasibility claim below; found while building the runner, before any DISPLACE
    data was read.
  α = 0 (with τ0 = 0 for the post-processor and B0's τ for the P0 gate) reproduces B0 exactly, so it always meets
  the bound (ΔMR = 0) and the feasible set is never empty.
  **λ has the same form in both corpora, on different label pairs:** the base-2 JSD between the left and right
  windows' posteriors renormalised over the **two languages of the conversation**. That is (ne, en) on D1,
  exactly as `confirm_h4.py` computes it, and the recording's two reference languages on DISPLACE. Renormalising
  DISPLACE audio over (ne, en) instead would throw away the posterior of the language actually spoken beside
  English (Hindi, Bengali, Kannada or Telugu) and keep a Nepali posterior that only tracks how Nepali-like that
  language sounds. The P0-gate α is tuned only on dev recordings with exactly two reference
  languages, so it multiplies a two-language JSD in bits, as on D1. The label pair still differs, so the
  transfer crosses both the corpus and the pair of languages. It assumes that α's effect depends on how much the
  two-language posterior moves, not on which languages they are. D1 cannot test this, because it has no
  cross-speaker turns; it is departure 9 in §12.
  **Not assessable:** if fewer than 5 dev sessions have at least 10 DS-DL turns each (for the P0 gate, among the
  two-language recordings), the bound cannot be estimated. α is then not frozen, M2 is reported as not assessable
  on DISPLACE, and its D1 stage-2 test as not tested. Both session counts (all dev recordings, and the
  two-language ones) are reported before the freeze.
- **tau0 and every system's operating point:** τ0 (P0) and the AHC-cosine clustering threshold (P1-AHC) are tuned
  on dev. PLAN.md §7's grid for P1-AHC is **0.30–1.00 in steps of 0.05**; the native P1/P2 VBx grid is fixed at
  setup and written into the record. B0's threshold is the grid value with the lowest dev DER
  (`pyannote.metrics`, overlap scored, no collar). Each mitigation's threshold is then the grid value whose dev
  MR_DS-DL is closest to B0's (ties to the lower dev FAR_SS-DL), so a mitigation cannot win by moving along the
  trade-off curve (PLAN.md §5 "retuned B0"). For the trade-off DET (§5) each system's DET summary is read at
  **FAR_SS-DL = 10%** (primary) and 5% (secondary).
- **H4's evaluation threshold** (PLAN.md §7): H4 is evaluated at the threshold that matches the **baseline's dev
  MR_DS-DL**; on DISPLACE this is available because DS-DL turns exist on dev (unlike D1, where the H4 supplement
  substituted the 10% control fire rate, its §5, departure 3).
- **Clustering threshold, every system, fold/seed:** frozen from dev before eval. For M3, the all-data models
  (5 seeds) are used on DISPLACE; the per-seed operating points are frozen.
- **Freeze before any eval scoring:** α (both gates), τ0, the clustering thresholds, the VBx grid, the dev-derived
  operating points and the H5 sample-size simulation (§7) are all fixed in `FREEZE_M2B.json` before one eval
  recording is scored. `csdiar/analysis/confirm_m2b.py` refuses to run unless the record matches every input and
  runs once (one-shot), in the manner of `confirm.py` and `confirm_h4.py`.

## 4. Systems
All mitigations plug into **P1-AHC with E3 as the primary back end** (PLAN.md §5; H4 supplement §2 names E3 for
H4–H6). **P0 on DISPLACE is secondary.** The pipeline is `csdiar/pipelines/p1_ahc.py`: Community-1 powerset
segmentation and overlap-aware masked embeddings, E1/E2/E3 or a mitigated embedding, AHC-cosine at the dev-tuned
threshold. B2's affinity offset and M2's local relabel act after/inside clustering and are the two hooks that
`p1_ahc.py` says are added with this supplement.
- **B0:** the unmodified system, **retuned** on the same dev objective as the mitigation (PLAN.md §5).
- **B1:** language-wise mean centring (subtract each language's mean embedding), the language of a window from the
  renormalised VoxLingua107 posterior (§2 gate): its argmax over the recording's reference languages, which is the
  primary label for B1 and B2 alike (the full-posterior label is only sensitivity 3, §9); means estimated on dev
  reference-language windows.
- **B2 as a language-aware affinity offset on the clustering output:** in the style of Thienpondt et al. (2022),
  the affinity between two clusters whose LID-predicted languages differ is raised by δ. pyannote's
  centroid-linkage AHC works on embeddings, not on a pairwise affinity matrix, so the offset is applied to its
  output (`csdiar/pipelines/m2b_hooks.py`, `b2_merge`). After AHC at threshold θ, each output speaker has a
  duration-weighted unit centroid of its segment embeddings and a majority language (most LID-predicted speech
  time; ties to the lexicographically smaller label). The closest pair of speakers with different majority
  languages is merged while its cosine distance is below θ + δ (ties by label). The larger speaker keeps its
  label, the centroid and language durations are recomputed after each merge, and same-language pairs are never
  merged, since AHC already compared them at θ. δ = 0 reproduces B0. This raises the affinity of every
  cross-language pair, so a language switch is less likely to split one speaker, and a cross-language turn
  between two speakers is more likely to be merged; the second effect is what H5 measures. δ is the dev
  verification-score shift (same-speaker same-language minus same-speaker cross-language segment cosines),
  clamped to [0, 0.5], as the D1 B2 (H4 supplement §4). The clamp carries over unchanged because the units are
  the same: δ is a cosine difference, AHC works on cosine distance 1 − cos, and D1's d(t) is also 1 − cos, so
  raising an affinity by δ equals lowering a distance by δ. On P0 (secondary) B2 keeps its D1 score-offset form.
- **M1 SubspaceProjector:** the training-free projector `csdiar/mitigations/m1_subspace.py`, v̂ the unit mean
  over (speaker, video/recording) of within-speaker cross-language embedding differences, estimated on dev;
  e′ = unit(e − (e·v̂)v̂), applied before AHC-cosine.
- **M2 LanguageGatedSCD, two deployments:**
  - **as the P0 gate** (secondary back end): τ(t) = τ0 + α·λ(t), λ the renormalised-posterior JSD (§2); used also
    for the D1 stage-2 test (§6).
  - **as the P1-AHC post-processor** (primary; `m2b_hooks.m2_relabel`): after clustering, for two consecutive
    segments of the exclusive diarization with different labels and a gap of at most 2 s, at the gap's middle
    t, **relabel the shorter of the two to its neighbour's speaker** when λ(t) is at least 0.5 bit **and** the
    cosine distance between the two segments' embeddings (the system's own embedding of each output segment) is
    below τ(t) = τ0 + α·λ(t) with τ0 = 0 (§3), so α = 0 never relabels. Every pair is decided on
    the original labels, so the order of processing does not matter. A segment claimed by both neighbours takes
    the nearer one's label (smaller distance; ties to the earlier neighbour), and of two equally long segments
    the later one is relabelled. This is a **local relabel of the shorter segment only, never a global cluster
    merge** (PLAN.md §5). Variants: oracle λ and time-shuffled λ (§2).
- **M3:** the five all-data fine-tuned models (one per seed 0–4, trained on all 15 D1 train channels plus D5 and
  early-stopped on the calibration channels; H4 supplement §7), used as the E3 replacement embedding inside
  P1-AHC; the per-seed results are averaged for the point estimate and their spread is reported (PLAN.md §5, §8).
- **What each system writes** (fixed after round 1 of the `run_m2b` code review, before any DISPLACE data was
  read): one overlapping and one exclusive RTTM per recording, threshold and parameter value
  (`csdiar/pipelines/run_m2b.py`). B0, B1, M1 and M3 change the clustering input, so both annotations come from
  the re-clustering. B2 merges whole output speakers, so its merge map is applied to both annotations. M2 relabels
  exclusive segments, and the overlapping annotation takes the same relabel over the same spans: inside a
  relabelled span, the source speaker's overlapping track gets the target label. FAR and MR are always scored on
  the exclusive RTTM (§2) and DER on the overlapping one (§3), so each system's DER describes the same decisions
  as its FAR and MR.
- **Native back ends:** native P1 (Community-1 with its own VBx) and P2 (DiariZen) are the pipeline systems of the
  conversation H2 test (§7); they are not part of the H4/H5 family, whose back end is P1-AHC+E3.

## 5. Operating point and the trade-off DET
- **DET (PLAN.md §2, §7):** sweep the named threshold (the AHC-cosine threshold for P1-AHC; τ0 for P0) with α
  fixed, and plot P(miss | DS-DL) against P(false alarm | SS-DL) on probit axes. **Primary summary:** MR_DS-DL at
  **FAR_SS-DL = 10%**, interpolated on the probit scale. **Secondary:** MR_DS-DL at FAR_SS-DL = 5%, and FAR_SS-DL
  at MR_DS-DL = 5% and 10%. DETs are also stratified by turn gap (PLAN.md §2).
- **Reading a DET at a target** (fixed after round 2 of the `score_m2b` code review, before any DISPLACE data was
  read): the DET is the polyline through the sweep's operating points **in threshold order**. Every segment
  between two adjacent thresholds that crosses the target, and every sweep point that hits it, gives a reading by
  probit interpolation; a run of consecutive sweep points at the target counts as one crossing. A threshold sweep
  need not give a monotone curve, so the target can be crossed more than once. The summary is then the **best**
  reading (the lowest MR at a target FAR, the lowest FAR at a target MR), which is an operating point the sweep
  actually reaches, and the number of crossings is reported beside it. A target the sweep never crosses is **not
  reached**: nothing is extrapolated, and the system or session is reported as not reaching it.
- **Operating point for H4/H5:** both are evaluated on eval at each system's frozen dev threshold (§3): B0 at its
  lowest-dev-DER threshold, each mitigation at the threshold matching B0's dev MR_DS-DL (PLAN.md §7). The DET
  summary at FAR_SS-DL = 10% is the H6 statistic and is reported beside H4. Each system's frozen threshold, τ0, α
  and clustering threshold are written into the record (PLAN.md §7 "the VBx parameter grid is fixed at setup and
  written into the frozen document").

## 6. M2's D1 stage-2 test
The α frozen **here** (the P0-with-E3 gate value, §3) is recorded into `planning/FREEZE_H4.json` by
`freeze_h4 add-m2` (H4 supplement §4, §10), and only then does `confirm_h4.py --stage 2` run **once**, scoring
M2 on D1 with M1 and M3's stage-1 verdicts carried unchanged (their Holm verdicts are final; `confirm_h4.py`
`test_family` copies them with `carried_from_stage1`). If DISPLACE never arrives, M2's α is never frozen, and M2's
D1 H4 is **reported as not tested** (H4 supplement §4). This supplement does not recompute M1 or M3 on D1.

## 7. Hypotheses and analysis plan
Primary conversation back end: **P1-AHC with E3**, DISPLACE eval, clustered by recording session. Primary collar
c = 0.5 s. SS-DL run-length strata pooled (strata reported).

| H | Prediction | Primary test | Supported if (smallest effect) |
|---|---|---|---|
| H2 (pipelines, conv.) | More hypothesized changes at SS-DL points than at controls for native P1 and P2 | Per-session log RR (ratio of sums of fires over SS-DL points and over their cell controls); sign-flip over sessions; Holm over the M2a family (m = 5) with P0's observed D1 p-values included (PREREGISTRATION.md §4) | Holm p < .05, and RR ≥ 1.25 with ΔFAR ≥ 1 pp |
| H4 (conv.) | Each mitigation lowers the SS-DL excess ΔFAR vs its retuned B0 (M1–M3; m = 3) | Per-session change in cell-standardised ΔFAR at the matched operating point (§3); MR_DS-DL at FAR_SS-DL = 10% reported beside it (PLAN.md's "also"); cluster sign-flip over sessions | Holm p < .05 and ≥ 25% relative reduction in ΔFAR |
| H5 | It does so without raising misses on real DS-DL turns (M1–M3; m = 3) | Non-inferiority on MR_DS-DL, clustered by session; **intersection–union with H4 per mitigation** | One-sided 95% upper bound below **both +2 pp and +25% relative** |
| H6 | M3 has the best trade-off and M1 the smallest gain (secondary) | Pairwise differences in MR_DS-DL at FAR_SS-DL = 10%, Holm (m = 3); two-stage bootstrap over sessions and seeds | Ordering significant after Holm |

- **Intersection–union (H4 × H5), per mitigation** (PLAN.md §7 "intersection–union with H4 per mitigation";
  PREREGISTRATION.md §6's grading is the same logic): a mitigation is declared a **net improvement** only if, for
  that mitigation, **both** H4 is supported (significant reduction, ≥ 25% relative) **and** H5's non-inferiority
  holds (upper bound below both +2 pp and +25% relative). Each component is tested at α = .05 within its Holm
  family; because intersection–union takes the **less** favourable of the two, no further multiplicity adjustment
  is made across the two components (as in PREREGISTRATION.md §6). A mitigation that passes H4 but fails H5 is
  reported as "reduces false splits at the cost of missed turns"; one that passes H5 but fails H4 as "no reduction
  shown".
- **Families and Holm m:** H2 (pipelines) completes the M2a family (m = 5: P0 with E1–E3 at their observed D1
  one-sided p-values, each 2^−17 ≈ 7.6 × 10^−6 by the exact sign-flip over 17 channels (decision 23), with their
  M2a verdicts final; P1 and P2 from here); H4 over M1–M3 (m = 3, Holm); H5 over M1–M3 (m = 3, Holm); H6 pairwise
  over the three mitigations (m = 3, Holm). The families are kept separate, as in PREREGISTRATION.md §4 and the
  H4 supplement.
- **Smallest effects:** H4, ≥ 25% relative reduction in ΔFAR (PLAN.md §7, H4 supplement §6), relative to S_B0 =
  B0's mean cell-standardised ΔFAR over the included eval sessions at B0's frozen threshold, computed once and held
  fixed in the bootstrap. H5, the one-sided
  95% upper bound of ΔMR_DS-DL below **both +2 pp and +25% relative** (PLAN.md §7), i.e. below
  min(0.02, 0.25 · S_MR). S_MR is B0's mean MR_DS-DL over the sessions included in H5, computed once and held
  fixed in the bootstrap, as S_B0 is in the H4 supplement §6; S_B0 uses the sessions included in H4. The two
  inclusion rules count different units, so the two session sets can differ. H6, the ordering (M3 best, M1
  smallest) significant after Holm.
- **Test statistics and unit structure:** the cluster is the **recording session** (PLAN.md §7). With few sessions
  (at most 32 eval recordings; the number passing the inclusion rule is reported at scoring), the primary test is a **cluster sign-flip** over sessions on the per-session statistic,
  computed by `stats.signflip` as frozen: exact over all 2^k patterns when 2^k ≤ 2^17 (k ≤ 17 sessions), and
  otherwise Monte Carlo with 2^17 random patterns from the seeded generator (as H7, decision 24). H5's non-inferiority uses the one-sided upper confidence bound of the
  per-session mean ΔMR_DS-DL, a BCa bound (10,000 whole-session resamples); H5 does not use the sign-flip, which
  gives a p-value but no non-inferiority bound, so the two intersection–union components use different machinery
  by design, as in the H4 supplement §6. The per-session statistics are ΔFAR_std (H4, `pilot.h2_cell`-style cell
  standardisation adapted to sessions) and MR_DS-DL (H5), with B0 subtracted per session as in the H4 supplement
  (δ_M(c) = metric_M(c) − metric_B0(c)).
- **BCa levels:** BCa bootstrap over **sessions**, 10,000 whole-session resamples, jackknife acceleration
  (PREREGISTRATION.md §4; H4 supplement §6). Two-sided intervals at 1 − 0.05/m = **98.3%** (m = 3); TOST intervals
  at 1 − 2(0.05/m) = **96.7%**. H5's non-inferiority bound is the **one-sided 95%** upper bound (PLAN.md §7),
  reported alongside the two-sided interval.
- **Outcome categories** (PREREGISTRATION.md §4, `csdiar/analysis/stats.py`): for H4 and for each component,
  significant (Holm one-sided p < .05); meets the smallest effect; reversed (Holm two-sided p < .05, wrong
  direction); equivalent (TOST interval inside the equivalence bounds). Precedence: supported, nonzero but
  trivial, reversed, equivalent, inconclusive. H5 adds **non-inferior** (upper bound below the margin) vs
  **inferior / not shown**.
- **"Not assessable" rules:** following PREREGISTRATION.md §6 and the H4 supplement §5 — a session enters a test
  only with at least 10 units (H4: SS-DL points with cell controls; H5: DS-DL turns), the D1 minimum of
  decision 20, fixed now. The dev rule of §3 (5 sessions with ≥ 10 DS-DL turns) decides only whether α can be
  tuned; it does not set eval inclusion. If fewer than the minimum number of
  **sessions** have enough units (minimum 5, since an exact one-sided sign-flip with k ≤ 4 cannot reach .05,
  decision 21), the test is **not assessable** and said to be so. If a system's dev tuning returns a degenerate
  operating point (no finite threshold, as `tau = -inf` in `confirm_h4.py`), that system is reported not
  assessable. If H5's dev sample-size simulation shows the eval set cannot reach the needed size, **H5 is reported
  descriptively** (PLAN.md §7, §10; decision 6) and the intersection–union net-improvement claim is withheld.
- **Power / size check from dev counts** (PLAN.md §7 "H5 is sized by simulation (design effect included) on
  DISPLACE dev"): the H5 sample-size simulation runs on DISPLACE **dev** with the design effect from
  session-level clustering. Its input for eval is the **projected** eval DS-DL count: the dev rate of DS-DL turns
  per hour of speech times the eval hours from the file listing, never a count read from the eval references. It
  asks whether that projection gives power ≥ 0.8 for the non-inferiority test at the +2 pp / +25% margin, with
  dev's B0 miss rate standing in for S_MR. PLAN.md's own simulation target (roughly 600–1,000 DS-DL turns on D2)
  is the reference order of magnitude. The verdict (confirmatory or descriptive) is written into
  `FREEZE_M2B.json` before any eval scoring and is not revisited once the eval turns are counted. A matching power
  check for the conversation H2/H4 SS-DL excess is run from dev SS-DL counts and reported.

## 8. Secondaries
Reported, not tested for support:
- **M2's λ against DS-SL** (decision 15's deferred half): λ's AUC separating SS-DL points from DS-SL turns and
  from DS-DL turns on dev, before any use of α — does λ track language rather than speaker change (PLAN.md §5)?
  With the oracle and time-shuffled λ variants and the full-posterior variant (§2).
- **P0 on DISPLACE** (E1–E3): the p0L1-style AUC and ΔFAR at conversation SS-DL points, as support for the D1
  H2-P0 result. Pipeline H2 (native P1, P2) is confirmatory and sits in §7.
- **D2-P test shows** (PLAN.md §6 E-B (iv), decision 6): the same trade-off on Nepali conversation, split by show,
  clustered by show, as support for H5 on Nepali — **confirmatory only if D2-P reaches the simulated size**
  (PLAN.md §7, §10); otherwise descriptive and labelled so. D2-P counts are **not determined** (no episode was annotated;
  decisions 6 and 8).
- **DISPLACE-M** as a secondary set for P1-AHC tuning and descriptive trade-off (decisions 9, 29).
- Each mitigation against **B1 and B2**, and B1/B2 against B0 (same statistics, unadjusted p), as in the H4
  supplement §8; how B2's offset interacts with H5 is in §12, item 5.
- Per-system DET curves, FAR_SS-DL and MR_DS-DL at both operating points, collar curves (0.1–1.0 s), the DER and
  its components via `pyannote.metrics` (PLAN.md §2), and the AMI language-driven split index against a
  circular-shift null (PLAN.md §2) on the conversation data.
- M1's k-dimensional subspace and specificity controls; M3's distillation curve, ablations and per-seed spread
  (H4 supplement §8). The two UMAP companion numbers (linear-probe language accuracy, speaker-ID accuracy,
  PLAN.md §6).
- `PREREGISTRATION.md` §5 sensitivities as they apply to the conversation data (§9).

## 9. Sensitivity analyses (pre-registered; descriptive)
1. **Collar curve:** every primary result at c ∈ {0.1, …, 1.0} s (PLAN.md §2), since the DISPLACE language
   references are segment-level and the primary c = 0.5 s may be optimistic or pessimistic at the segment edges.
2. **Overlap re-included:** DS-DL turns and SS-DL points in overlapped regions re-included, scored against the
   overlap-aware hypothesis (§2 overlap handling).
3. **Full-posterior λ** in place of the renormalised-posterior gate for M2 and B1/B2's language prediction (§2),
   since DISPLACE's languages differ from (ne, en).
4. **Oracle λ and time-shuffled λ** (M2 upper bound and null control).
5. **Backchannel-only segments included** (PLAN.md §2 names this a sensitivity analysis), if the references mark
   them; if they do not, reported as not available.
6. **Minimum-run-length variants:** the ≥ 1 s both-runs rule relaxed to the achievable minimum and tightened,
   reported with the counts each leaves (the segment-granularity dependency of §2).
7. **Without DISPLACE-M** (if both sets are granted): the confirmatory result stands on DISPLACE 2024 alone;
   DISPLACE-M enters only secondary (§8).
8. **D2-P** as the Nepali support set (§8), descriptive unless it reaches the simulated size.

Code or inputs created after the freeze (the overlap re-inclusion run, the full-posterior variant if not already
built, DISPLACE-M ingest) are committed and their hashes logged in `planning/DEVIATIONS.md` before they run on an
eval recording; they never change the frozen primary code or features.

## 10. Fallback if DISPLACE 2024 is not granted
If only DISPLACE-M is available (decision 29): the confirmatory H5 runs on DISPLACE-M with **LID-derived DS-DL
turns** (automatic language labels, since DISPLACE-M has no language annotation), and is **labelled partly
automatic** and reported as **supporting, not confirmatory** — the confirmatory H5 claim then rests on D2-P if and
only if D2-P reaches the simulated size, and otherwise H5 is reported descriptively across both (PLAN.md §10
"H5 falls back to descriptive"). The limitation (reference language labels are automatic) is stated wherever the
DISPLACE-M H5 number appears. M2's α is then tuned on DISPLACE-M dev with the same automatic-label caveat.

## 11. Hygiene
- **One run:** confirmatory outcomes are computed only by `csdiar/analysis/confirm_m2b.py`, which refuses to run
  unless every code hash, feature hash, reference checksum and input path in `FREEZE_M2B.json` matches, and runs
  **once per frozen configuration** (one-shot, as `confirm.py` and `confirm_h4.py`). The M2 D1 stage-2 run is the
  separate one-shot `confirm_h4.py --stage 2` (§6).
- **What the freeze record covers:** the guarded code (`confirm_m2b.py`, `ingest_displace.py`, `p1_ahc.py` with
  the B2 affinity-offset and M2 relabel hooks, `m2_lambda.py`, `stats.py`, the mitigation and M3 modules, this
  document); the DISPLACE dev **and** eval feature files and both reference RTTMs with checksums; the dev/eval
  split as released; the frozen α (both gates), τ0, clustering thresholds and VBx grid; the H5 sample-size
  simulation result; the seed (20261018).
- **Freeze order:** α is tuned on dev and frozen here; it is then added to `FREEZE_H4.json` (`freeze_h4 add-m2`)
  before the D1 stage-2 run (§6). M3's all-data checkpoints and their DISPLACE eval stage-1 caches are hashed
  and appended before `confirm_m2b.py` runs on the conversation data (the D1 dense grids are covered by
  `FREEZE_H4.json`).
- **Logging:** every run logs its configuration, seed (20261018), git SHA and data revision (PLAN.md §7).
- **Deviations:** changes after the freeze go to `planning/DEVIATIONS.md`, with dates and the hash of any
  after-the-freeze code before it touches an eval recording.

## 12. Departures from PLAN.md, and what was known
1. **DISPLACE 2024 is the primary H5 set, not DISPLACE-M** (decisions 6, 29); decision 29 supersedes the interim
   position of decision 9 (which made DISPLACE-M primary): DISPLACE-M has no language annotation, so its DS-DL turns would be
   automatic (§2, §10).
2. **The ±0.25 s cell is not used on DISPLACE**; c = 0.5 s is the finest primary collar, because the language
   references are segment-level (PLAN.md §3, §11). On D1 the primary analyses used word-level alignment; they are
   not reproduced here.
3. **The LID gate renormalises over the recording's reference languages**, not over (ne, en) as on D1 (§2),
   because DISPLACE is not Nepali–English; the full-posterior variant is the deployable fallback.
4. **The minimum-run-length rule matching D1's `p0L1` (both runs ≥ 1 s) is applied only where the segment
   references allow it** (§2); the achievable minimum is reported rather than assumed.
5. **B2 is a language-aware AHC affinity offset on the conversation back end**, not the P0 score offset of D1
   (§4). On D1, B2 was biased in its favour because D1 has no cross-speaker turns, so its offset could lower
   switch-point hits at no cost (H4 supplement §8). On DISPLACE the same offset also raises the affinity across
   real cross-language turns, so its cost appears in MR_DS-DL and H5 measures it. Comparisons with B2 stay
   **secondary and are read together with H5**, and the confirmatory family stays M1–M3 (m = 3).
6. **M2's α is tuned separately for the two gates** (P1-AHC post-processor and P0-with-E3), because their score
   scales differ (§3; H4 supplement §4); the P0-with-E3 α is the one recorded into `FREEZE_H4.json` for the D1
   stage-2 test.
7. **H5 may be reported descriptively** if the dev simulation shows the eval set is too small (PLAN.md §7, §10),
   in which case the intersection–union net-improvement claim is withheld and D2-P carries the Nepali support.
8. **Sessions are the cluster**, so the sign-flip is exact up to 17 sessions and Monte Carlo (2^17 patterns)
   beyond, as `stats.signflip` does and as in H7 (decision 24; §7).
9. **M2's D1 α comes from another corpus and another language pair.** The P0-gate α used in D1's stage 2 is
   tuned on DISPLACE dev, on the recordings with exactly two reference languages, with λ the two-language JSD of
   §3: renormalised over (ne, en) on D1 and over the recording's pair on DISPLACE. The functional form and its
   scale in bits are the same; the label pair is not. The transfer assumes the coefficient depends only on how
   much the two-language posterior moves, which D1 cannot check because it has no cross-speaker turns. The H4
   supplement §4 already chose a DISPLACE-tuned α, because a D1-tuned α would let M2 win by construction; this
   document only fixes how the transfer is done, and the D1 M2 result is reported with this caveat.
10. **Fixed tuning grids and a not-assessable rule for M2** (§3) replace PLAN.md's unspecified "tuned on dev".
   B0's conversation threshold is the lowest-dev-DER grid value, and each mitigation is matched to B0's dev
   MR_DS-DL (PLAN.md §7).
11. **B2 acts on P1-AHC's output, not inside every AHC step.** PLAN.md §5 calls B2 an offset "in the clustering
   affinities". pyannote's centroid-linkage AHC has no pairwise affinity matrix to offset, and replacing the
   linkage would change B0 as well. So B2 is a final cross-language merge of output speakers at θ + δ (§4). It
   can change which speakers end up together, but not the order of AHC's earlier merges.

**What was known when this was written.** The **D1 H4 result will be known when this freezes**: the H4 supplement
is frozen (tag `h4-freeze`) and its confirmatory run (`confirm_h4.py --stage 1`, M1 and M3) runs before DISPLACE
arrives in the build order (decision 26, steps 6–8). The M2a confirmatory results (decision 23) and the H7 result
(decision 25) are known. This supplement's conversation-specific choices — the back-end mapping of B2 and M2 onto
P1-AHC, the renormalised-posterior gate, the session clustering, the intersection–union wording and the DET
operating points — were written from PLAN.md and the decisions, not from any DISPLACE eval statistic, which does
not exist at the freeze. M2's α, τ0 and the clustering thresholds are set on DISPLACE **dev** only; no eval
recording informs any choice in this document.
