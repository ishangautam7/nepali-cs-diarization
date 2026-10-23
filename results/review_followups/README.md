# External-review follow-ups (EXPLORATORY, post hoc)

Work done after all pre-registered results, in answer to the external review handled in PR #61 (branch `paper-review-fixes`, DEVIATIONS.md entry 14). **Everything here is exploratory.**

- **Nothing frozen changed:** no frozen code, frozen result file or pre-registered verdict was touched. PR #62 itself did not edit `paper/main.tex`; the paper reports these results since PR #63 (Section VIII) and the leave-one-channel-out check since the second external review (Section V).
- **Code and outputs:** the scripts are `scripts/review_*.py` and `scripts/score_human_subset.py`, with tests in `tests/test_review_followups.py`, `tests/test_review_channel_exclusion.py` and `tests/test_review_human_subset.py`; the outputs are in this folder.

**Inputs.** The frozen M2a inputs were lost with the first GPU box (DEVIATIONS.md entry 6). What exists now are the features rebuilt after that loss.

- **What still matches the freeze:** only E3's grid embeddings (`emb/e3_L1/{train,test}.npz`) are byte-identical to the hashes in FREEZE_M2A.json. The rebuilt point tables reproduce every frozen count exactly (Task 2).
- **Paths:** the rebuilt features sit in a backup folder beside the repository on the analysis host: `BOUNDARIES=../nepdiar-box-backup/data/d1/boundaries/lex_v1b`, `EMB=../nepdiar-box-backup/data/d1/emb` and `ALIGN=../nepdiar-box-backup/data/d1/align`.
- **D1 audio:** `D1=../nepdiar-host-data/d1`, the public corpus at revision dd00ca41 ingested with `csdiar.data.ingest_nepali_cs`. Its manifest is byte-identical to the GPU box's.
- **Where it ran:** Tasks 1, 2 and 4 and the Task 3 file set ran on the host's CPU. Task 3's pipelines ran on a GPU box (RTX PRO 6000 Blackwell).

## 1. Fig. 1 against Table II (issue 1): `fig1_reconciliation.json`

**The check.** Step (d) must reproduce the reference ΔFAR to within 1e-6. All three models pass with difference 0, but against different references (Ishan's decision of 2026-10-22, option B):

- **E3** (frozen inputs) is checked against Table II, `results/confirm/confirm_m2a.json`: +2.7287 pp.
- **E1 and E2** (rebuilt inputs) are checked against the re-run on rebuilt inputs, `results/sensitivity/base/confirm_sens_base.json`: +2.5124 and +3.0977 pp. Their Table II values are +2.4617 and +2.9546 pp. The file labels both models as rebuilt.

| Step (pp, p0L1 cell, frozen τ) | E1 (rebuilt) | E2 (rebuilt) | E3 (frozen) |
|---|---|---|---|
| (a) old figure: raw rates, all 20 channels, points with no NMS peak in ±c left out | −1.02 | −0.16 | +0.22 |
| points with no peak counted as no fire | +0.94 | +0.88 | +0.43 |
| 3 pilot channels dropped (17 confirmatory) | −0.28 | −0.39 | −0.46 |
| switch points without cell controls dropped | −0.11 | −0.01 | −0.04 |
| (c) cell standardisation, pooled over points | +1.28 | +1.69 | +1.02 |
| (d) equal channel weighting = Table II's statistic | +1.70 | +1.09 | +1.56 |
| **(d) value** | **+2.51** | **+3.10** | **+2.73** |

The intermediate values, as cumulative levels in pp for E1/E2/E3:

| Level | E1 | E2 | E3 |
|---|---|---|---|
| (a), points without a peak counted as no fire | −0.09 | +0.73 | +0.65 |
| (b), 17 confirmatory channels | −0.36 | +0.34 | +0.18 |
| (b2), switch points with cell controls only | −0.47 | +0.33 | +0.15 |
| (c), cell-standardised and point-weighted | +0.81 | +2.01 | +1.17 |

E3's (c) equals the point-weighted mean in PR #61's new figure, which is +1.17 pp.

**What explains the gap.** Two steps account for it:

- **Cell standardisation:** +1.0 to +1.7 pp. Each switch point is compared with the controls of its own video × pause × position cell.
- **Equal channel weighting:** +1.1 to +1.7 pp.

The old figure's other two differences roughly cancel. It dropped points without a peak from both rates, which moved it by +0.4 to +0.9 pp. It also included the pilot channels, which moved it by −0.3 to −0.5 pp. The reviewer's reading of the old figure was about −1.1/−0.2/+0.2 pp; step (a) gives −1.02/−0.16/+0.22 pp.

```
PYTHONPATH=. python -m scripts.review_fig1_reconciliation --boundaries $BOUNDARIES --emb $EMB \
    --pilot results/pilot/pilot_v2.json --confirm results/confirm/confirm_m2a.json \
    --rebuilt-models e1 e2 --rebuilt-reference results/sensitivity/base/confirm_sens_base.json \
    --out results/review_followups/fig1_reconciliation.json          # numpy 2.5.4, as .venv-pyannote4
```

## 2. Exact sample counts (issue 3): `counts.json`

These are the 17 confirmatory channels under the frozen P0 rules. The counts are the same for E1, E2 and E3, and they match the frozen totals: 4,069 scored and 107 without cell controls, so 3,962 with cell controls, and 120,630 for pooled det.

| Cell, split (channels) | Switch points eligible | Scored | Without cell controls | With cell controls | Controls scored | Controls in the cells used (cells) |
|---|---|---|---|---|---|---|
| p0L1, train (15) | 3,954 | 3,954 | 103 | 3,851 | 33,556 | 29,528 (894) |
| p0L1, test (2) | 115 | 115 | 4 | 111 | 6,735 | 2,316 (70) |
| **p0L1, total (17)** | **4,069** | **4,069** | **107** | **3,962** | **40,291** | **31,844 (964)** |
| det, train (15) | 129,792 | 116,911 | 503 | 116,408 | 87,607 | 86,456 (2,041) |
| det, test (2) | 4,722 | 4,267 | 45 | 4,222 | 11,345 | 10,699 (301) |
| **det, total (17)** | **134,514** | **121,178** | **548** | **120,630** | **98,952** | **97,155 (2,342)** |

How to read the columns:

- **Controls:** p0L1 uses det_w1 controls (40,297 eligible, 40,291 scored); the det cell uses det controls (106,041 eligible).
- **Unscored switch points:** in the det cell, 13,336 have no d(t), because they lie within 1 s of a clip start.

```
PYTHONPATH=. python -m scripts.review_counts --boundaries $BOUNDARIES --emb $EMB \
    --confirm results/confirm/confirm_m2a.json --out results/review_followups/counts.json
```

## 3. Full diarization pipelines (issue 2): `pipeline_exploratory.json`, `pipeline_exploratory.md`

**Set** (`pipeline/set.json`, seed 20261023): 1,479 files, 6.848 h, with 300 boundaries of each type on the 17 confirmatory channels. Each type is drawn per channel in proportion to that channel's p0L1 count, with no shortfall. The types are:

- **switch:** a natural p0L1 switch point with cell controls, inside its own clip.
- **control:** a natural det_w1 control from the same cell.
- **join_ss_same:** two consecutive clips of one video, same language at the join.
- **join_xs_same:** clips of two channels, same language at the join.
- **join_xs_cross:** clips of two channels, a language change at the join.

Every join side has an edge run of at least 1 s. **Clip joins are artificial boundaries:** two separately recorded segments are concatenated with a 10 ms fade, so each join also carries a recording cut.

**Reference** (`pipeline/reference.rttm`): speech regions from the aligned words, with the channel as the speaker. D1 has no speaker labels; issue 4 asks whether a channel really is one speaker.

**Systems**, as released, with default hyperparameters and no oracle speaker count:

- **P1:** stock pyannote Community-1 (pyannote.audio 4.0.7) at revision 3533c8cf. This is not the paper's P1-AHC: that variant's threshold was to be tuned on DISPLACE dev data that never arrived (DEVIATIONS.md entry 7).
- **P2:** DiariZen `BUT-FIT/diarizen-wavlm-large-s80-md` at 6285693d, with the WeSpeaker ResNet34-LM embedding at 837717dd and code at 844f5555. It ran with torch 2.8.0+cu128 instead of DiariZen's pinned 2.1.1, which has no Blackwell kernels.
- **Hardware:** both ran on CUDA. Real-time factors were 0.008 (P1) and 0.015 (P2).
- **CPU runs not used:** an earlier CPU run on the host was stopped at 78 (P1) and 129 (P2) files once the GPU results were in. It is not used here.

**Measures.** A change is "detected" when the hypothesis switches to a different non-empty speaker set within ±0.5 s; silence in between does not count. The false-change rate applies to within-speaker boundaries, the miss rate to cross-speaker joins. DER uses pyannote.metrics with a 0.5 s collar (±0.25 s), overlap scored. Contrasts are per-channel differences: an exact sign-flip test over 17 channels and a BCa 95% interval (`csdiar.analysis.stats`), with no multiplicity correction.

| Boundary | P1 change rate, pooled (channel mean) | P2 change rate, pooled (channel mean) |
|---|---|---|
| switch (natural, within speaker) | 1.3% (1.8%) | 0.3% (0.4%) |
| control (natural, within speaker) | 0.3% (0.6%) | 0.3% (0.8%) |
| join_ss_same (artificial, within speaker) | 1.7% (4.1%) | 3.3% (6.8%) |
| join_xs_same (artificial, two speakers): miss rate | 39.7% | 6.0% |
| join_xs_cross (artificial, two speakers): miss rate | 34.7% | 4.0% |

| Contrast (per channel, A − B) | P1: mean, 95% BCa, sign-flip p one-/two-sided | P2: mean, 95% BCa, p |
|---|---|---|
| false change: switch − control | +1.2 pp, −0.8 to +3.6, p = .156 / .313 | −0.4 pp, −3.4 to +0.8, p = .750 / 1.000 |
| false change: switch − join_ss_same | −2.3 pp, −9.3 to +0.6, p = .797 / .422 | −6.4 pp, −14.7 to −2.0, p = 1.000 / .031 |
| false change: join_ss_same − control | +3.5 pp, −0.2 to +10.5, p = .156 / .313 | +6.0 pp, +2.3 to +12.5, p = .016 / .031 |
| miss: join_xs_cross − join_xs_same | −9.0 pp, −16.9 to −1.6, p = .980 / .041 | −1.5 pp, −4.2 to +1.1, p = .858 / .285 |

**DER**, mean (median):

| Files | P1 | P2 |
|---|---|---|
| natural clips (579) | 7.1% (1.9%) | 6.2% (1.6%) |
| join_ss_same | 10.5% (3.5%) | 10.8% (3.5%) |
| join_xs_same | 23.9% (15.8%) | 13.8% (6.5%) |
| join_xs_cross | 19.7% (13.2%) | 11.4% (5.6%) |

Two further rates, for P1 and P2: natural clips with more than one speaker found, 5.5% and 2.2%; cross-speaker joins merged into one label, 37.7% and 29.3% (P1) against 3.7% and 3.0% (P2), for same-language and cross-language joins. The per-boundary, per-file and per-channel values are in `pipeline_exploratory.json`.

**Reading.** In neither system does a natural code-switch within one speaker produce a false speaker change more often than a matched same-language boundary. Both rates are near zero, and both contrasts are far from significant. An artificial same-speaker join does raise false changes for DiariZen, as a recording cut would.

P1 misses many real speaker changes at joins, and fewer when the language also changes: −9.0 pp, two-sided p = .041, uncorrected and exploratory. DiariZen misses few either way.

```
PYTHONPATH=. python -m scripts.review_pipeline_set --boundaries $BOUNDARIES --emb $EMB --align $ALIGN --d1 $D1 \
    --out results/review_followups/pipeline                                          # host CPU, numpy 2.5.4
# GPU box: D1 = the 579 natural clips' WAVs copied from the host, MODELS = the folder holding the pinned snapshots above
PYTHONPATH=. python -m scripts.review_pipeline_run --set results/review_followups/pipeline/set.json --d1 $D1 \
    --system p1 --model-dir $MODELS/pyannote-community-1 --out results/review_followups/pipeline --device cuda
PYTHONPATH=. TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m scripts.review_pipeline_run \
    --set results/review_followups/pipeline/set.json --d1 $D1 --system p2 --model-dir $MODELS/diarizen-wavlm-large-s80-md \
    --embedding $MODELS/pyannote-wespeaker-voxceleb-resnet34-LM/pytorch_model.bin --out results/review_followups/pipeline --device cuda
PYTHONPATH=. python -m scripts.review_pipeline_score --set results/review_followups/pipeline/set.json \
    --systems p1 p2 --out results/review_followups/pipeline_exploratory               # pyannote.metrics 4.2.0
```

The hypotheses are `pipeline/hyp_p1.rttm` and `pipeline/hyp_p2.rttm`, with per-file timings in `pipeline/timing_p*.jsonl` and run records in `pipeline/run_p*.json`. The joined audio, `pipeline/audio/`, is git-ignored, and the first command rebuilds it. `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` lets DiariZen's pyannote 3.1.1 load its pinned checkpoints under torch ≥ 2.6.

## 4. Human-verified subset (issue 4): `human_subset/`

The subset is prepared but **not labelled**.

- **Sample:** 150 p0L1 switch points with cell controls and 150 matched controls, one per switch point from the same cell. They were drawn once with seed 20261022 from the population of 3,962, in proportion to channel size (1 to 22 per channel, all 17 channels).
- **Snippets:** each is 4.000 s with the boundary at 2.000 s; 56 are padded with silence (26 switch, 30 control). Names and order are blinded.
- **Language sides:** the switch items are 81 ne→en and 69 en→ne; the controls are 107 ne-ne and 43 en-en.
- **Files:** the rater sheet is `human_subset/template.csv`, the hidden key `human_subset/key.csv` and the record `human_subset/subset.json`. The instructions and commands are in `human_subset/README.md`. The audio is git-ignored (CC BY-NC), and `--audio-from-key` recreates it byte-identically.
- **Scoring:** `scripts/score_human_subset.py` computes Cohen's kappa per question, plus speaker, language and boundary contamination rates with Wilson intervals, once two bilingual raters have filled their sheets.
- **Second external review (2026-10-22, before any rating):**
  - **Sheet:** the rater sheet gained `switch_time_s` (the heard switch time, for timing agreement) and `unusable` (overlap or noise). The sample is unchanged: `key.csv` and `subset.json` are byte-identical.
  - **Protocol:** `human_subset/PROTOCOL.md` fixes the analysis in advance:
    - per-question agreement on the independent sheets;
    - contamination rates;
    - adjudication of disagreements in `adjudication.csv`, keeping both original answers;
    - the verified subset;
    - a minimum of 100 verified switch items and 100 verified controls from at least 10 channels.
  - **Central comparison:** run only when that minimum is met, and descriptive. It covers an H1-type similarity difference from the grid embeddings and the cell-standardised ΔFAR, with E3 as the hash-verified reference.
  - **Scorer:** it refuses any sheet that carries key columns. Tests: `tests/test_review_human_subset.py`.
  - **Rater package:** the 300 snippets, the blank sheet and the instructions, kept outside git.
- **Earlier single-rater checks**, unchanged:
  - pair check: 117 of 230 done, `results/listening/listening_purity.json`;
  - audibility check: 352 of 450 done, `results/listening/listening_d1.json`.

  Both pages hold every planned item in a fixed order, so a rater can resume where they stopped; `scripts/score_listening.py --first N` scores the finished part.

## 5. Leave-one-channel-out sensitivity (second external review, item 2): `channel_exclusion.json`, `channel_exclusion.md`

Added on 2026-10-22, post hoc and exploratory. Each main effect is re-estimated 17 times, each time without one confirmatory channel, channel-weighted (as frozen) and switch-weighted, with the exact one-sided sign-flip p over the remaining 16 channels (unadjusted). Every fit is reported; no other exclusion rule was tried.

- **Part A, exact:** H1, H2-P0, ΔFAR and H3 come from the frozen per-channel statistics in `results/confirm/confirm_m2a.json`. Recomputed full-sample values and sign-flip p equal the frozen ones (max |diff| 5.6e-17, floating-point rounding). The switch-weighted AUC is the n_ssdl-weighted mean of channel AUCs, which equals the cell-standardised AUC pooled over switch points. H3 has no switch-weighted form here (per-channel trial counts are not saved).
- **Part B, rebuilt inputs:** the pooled det and det_w1 AUCs need per-channel values the frozen run did not save, so the repository's `score_points`/`h2_cell` run at the frozen τ on the rebuilt inputs. They reproduce the frozen p0L1, det and det_w1 AUCs to three decimals; ΔFAR differs for E1/E2 (+2.51 and +3.10 pp against +2.46 and +2.95 pp), as in section 1.
- **Results:** no estimate changes direction, and every primary test stays significant (p ≤ 3.1e-5). Channel-weighted ranges: H1 d_z 0.254–0.290 / 0.181–0.204 / 0.245–0.266 (E1/E2/E3); H2-P0 AUC 0.567–0.589, always ≥ 0.55; H3 −0.224 to −0.167 SD. E2's H1 crosses its smallest effect in 5 of 17 fits and E3's H3 in 4. The pooled det AUC stays at 0.482–0.498 in every fit and both weightings.
- **Least stable:** E1's channel-averaged ΔFAR, +2.46 pp, falls to +1.17 pp (p = .092) without the smallest channel, `UChx9-bynjfm6Js3k54Sf1Mg` (13 p0L1 switch points, ΔFAR +23.2 pp). E2 and E3 stay at +2.0 to +3.6 pp.
- **Possible shared speaker:** pilot video `zJYS-XNDP2Q` (channel `UCeq9rTo9W93NfJNP9n1eRGw`) is the closest cross-channel match of all four videos of confirmatory channel `UCOC6C8_U9PsAs3tQodDDmGw` (cos 0.82–0.84 TitaNet, 0.84–0.88 CAM++; its best cosine to its own channel is 0.54/0.60). Without that channel no primary estimate moves by more than 0.006. Because those four videos are the channel's only videos, the video-level exclusion equals the channel-level one.

```
PYTHONPATH=. python scripts/review_channel_exclusion.py --confirm results/confirm/confirm_m2a.json \
    --purity results/purity/purity_audit.json --boundaries $BOUNDARIES --emb $EMB \
    --out results/review_followups/channel_exclusion.json --md results/review_followups/channel_exclusion.md
                                                                     # numpy 2.2.6 (.venv-pyannote4 class env)
```

