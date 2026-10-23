# Pre-registration supplement: H7, the D1 diagnostic replicated on D4

Status: **draft for freeze**. It is frozen as `planning/FREEZE_H7.json` (tag `h7-freeze`) before any D4 outcome statistic is computed. It supplements `planning/PREREGISTRATION.md` (stage M2a), as decision 19 provides. Changes after the freeze go to `planning/DEVIATIONS.md`. Design decisions are in decision 24.

## 1. Hypothesis
H7 (PLAN.md §7): the SS-DL effect replicates on D4. H1 (cosine drift) and H2-P0 (change-detector AUC) are tested as in D1, with the same predicted direction and Holm within H7. PLAN.md does not include attribution or H3 in H7, so neither is tested here.

**Verdict.** For each hypothesis (H1, H2-P0) and each language pair, the verdict depends on how many of the three embeddings replicate (§5):
- *replicated:* all three;
- *partly replicated:* one or two;
- *not replicated:* none.

D1's smallest effects are reported next to every verdict, as the number of embeddings whose D1 category is "supported", but they do not enter it. With several hundred speakers, a significant but trivial effect is possible, so a verdict is always read together with that count and the estimates.

## 2. Data (D4)
- **Source:** MUCS 2021 sub-task 2, OpenSLR 104, CC BY-SA 4.0. These are real spoken-tutorial recordings.

  | Pair | Split | Hours | Segments | Speaker IDs |
  |---|---|---|---|---|
  | Hindi–English | train | 89.55 | 52,825 | 520 |
  | Hindi–English | test | 5.18 | 3,136 | 30 |
  | Bengali–English | train | 45.94 | 26,606 | 267 |
  | Bengali–English | test | 7.00 | 4,275 | 40 |

  - The archives' SHA-256 values are in `~/data/d4/raw/SHA256SUMS`, and the freeze record carries them.
  - The blind test set is not used.
- **Clips:** each published segment is cut from its continuous recording (`csdiar/data/ingest_d4.py`). Nothing is joined or modified, and no segment was skipped.
- **Clusters:** MUCS speaker IDs.
  - Every speaker ID has exactly one recording except one ID, which has two. A speaker ID is therefore in effect a recording, and one narrator may appear under several IDs. §6 tests this.
  - No speaker ID appears in both splits.
- **No D4 pilot:** both published splits are analysed together. Every definition and threshold comes from the frozen D1 code, except the two pair-specific inputs in §3 and §4: the low-alignment threshold, under D1's rule, and the lexicons.

## 3. Language tags (decision 24)
- **Tagger** (`csdiar/data/tags_d4.py`): D1's convention, with the pair's native script as a parameter.
  - Wholly Devanagari (hi-en) or Bengali (bn-en) words → native.
  - Wholly Latin words → English.
  - Digits, mixed-script and other-script words → neutral.
  - The single-letter rule of decision 12 applies.
  - Loan-lexicon words → `loan`, which breaks runs and is excluded from windows. Names → neutral.
- **Why a D4 lexicon is needed:** MUCS transcripts often write English words in the native script, e.g. क्लिक "click", ফাংশন "function". A script rule alone would call them native.
- **How it was built (text only):**
  - **Review lists.** Each list has the 1,500 most frequent native-script word types, plus every type seen at least 5 times whose uroman romanisation has the consonant skeleton of an English word. That is 2,698 Hindi types, covering 97.0% of native-script tokens, and 2,131 Bengali types, covering 94.4%. Latin-script types that are rare in English (Zipf < 3, seen at least 5 times) were added: 643 Hindi and 422 Bengali.
  - **Labels.** Each type was labelled LOAN, NATIVE, NAME or UNSURE from up to three corpus contexts, by LLM-based labellers in 8 parallel runs.
  - **Checks.** Every file was checked row by row against its list. A gloss-similarity test found that one file was shifted by one row; it was rewritten and re-checked. A 60-item seeded random spot-check by the analyst agreed on 59 items; the exception was Bengali কলাম "column", labelled native.
  - **UNSURE:** no type received this label. An UNSURE type would have kept its script tag.
- **Result** (`planning/lexicon/d4/`; SHA-256 in the freeze record):

  | Pair | Loan types | Loan tokens | Share of native-script tokens | Names | Native words in Latin script |
  |---|---|---|---|---|---|
  | hi-en | 763 | 35,041 | 7.1% | 67 | 1 |
  | bn-en | 393 | 9,100 | 4.7% | 21 | 1 |

- **Use of the labels:**
  - Native-script LOAN words, and native words written in Latin script (1 type per pair), go into the loan list, so they are tagged `loan`. NAME words go into the name list and are tagged neutral.
  - Labels were used as produced, except the shifted file, which was re-labelled. The one disagreement found by the spot-check, কলাম, was not edited, so the list stays the output of a fixed procedure.
- **Residual contamination:** English words left native-tagged, from unreviewed types or label errors, make switch points and controls less distinct, which biases H7 towards no effect.
  - It is estimated after the freeze on 200 random native-tagged tokens per pair (seed 20261018), shown with their clip contexts.
  - An LLM-based labeller with the §3 instructions labels them, and the analyst checks 50 of them.
  - The share labelled English is reported descriptively.

## 4. Features (checksums in the freeze record)
- **Alignment:** uroman (`hin` / `ben`) + MMS_FA (`csdiar/data/align.py --lang`).
- **VAD:** Silero, as in D1.
- **Embeddings:** dense L = 1 s grids and point-anchored windows (L = 0.5, 1.0 and 1.5 s) for E1–E3.
- **Boundaries:** version `lex_d4v1`, produced by `csdiar/data/boundaries_d4.py`. This is D1's frozen `boundaries.py` with the D4 tagger swapped in, and every other rule unchanged.
- **Low-alignment thresholds** (5th percentile of the pair's train clips): hi-en 0.2995, bn-en 0.3662.
- **Counts** (these are not outcomes):

  | Pair | `emb1` switch points (train + test) | `p0L1` switch points (train + test) |
  |---|---|---|
  | hi-en | 5,491 + 237 | 5,428 + 235 |
  | bn-en | 3,269 + 580 | 3,253 + 584 |

## 5. Tests (`csdiar/analysis/confirm_h7.py`)
- **Per pair and embedding** (E1–E3), H1 and H2-P0 are computed by D1's frozen functions (`pilot.h1`, `pilot.h2`):
  - **H1:** the per-speaker mean paired Δcos over the L = 1.0 s matched sets, with dz = unweighted mean of speaker means / pooled within-speaker SD. The denominator is recomputed in each bootstrap resample. Predicted > 0.
  - **H2-P0:** the per-speaker cell-standardised AUC of d(t) at the point: `p0L1` switch points against `det_w1` controls, with cells video × pause bin × position bin, weighted by switch points. Predicted > 0.5.
- **Inclusion:** a speaker enters a test with at least 5 units.
  - For H1, the units are matched sets with at least one control.
  - For H2-P0, they are `p0L1` switch points with cell controls. This is the count `pilot.h2_cell` reports as `n_ssdl`, which includes only switch points whose cell has a control.
  - Excluded speakers are listed with their counts.
- **Test:** a one-sided sign-flip over speakers.
  - With 17 or fewer speakers, every sign pattern is enumerated, as in D1.
  - With more, m = 2^17 random sign patterns are drawn (numpy `default_rng(20261018)`, via `csdiar.analysis.stats.signflip`). The p-value is (b + 1)/(m + 1), where b counts the sampled patterns whose mean is at least the observed mean (one-sided) or whose |mean| is at least the observed |mean| (two-sided).
- **Family and intervals:**
  - Holm over the 12 tests (H1 and H2-P0 × E1–E3 × hi-en and bn-en).
  - BCa intervals from 10,000 speaker resamples at 1 − 0.05/12 (99.58%); TOST intervals at 1 − 2(0.05/12).
- **Replication criterion** (PLAN.md §7, "same direction"): a test replicates if its Holm one-sided p < .05.
  - D1's outcome categories are also reported for each test, with the same smallest effects (dz 0.2, AUC 0.55) and the same precedence (PREREGISTRATION.md §4).
- **Descriptive:**
  - plain dz;
  - H1 at L = 0.5 and 1.5 s;
  - cell-standardised ΔFAR at D1's frozen τ (calibrated on D1, so FAR_CTRL is not 10% on D4);
  - the pooled `det` and `det_w1` AUCs with their strata, reported next to the primary result. These use every speaker with a finite AUC, with no minimum;
  - per-direction counts.
- **No D4 power analysis:** there is no D4 pilot, so no formal power analysis is possible. With 250–550 speakers per pair, the intervals will be narrow.
  - An inconclusive result would therefore indicate an effect clearly smaller than D1's, or speakers that disagree.
  - An equivalent result would indicate an effect within the smallest effect. Both would be reported as such.

## 6. Sensitivity (pre-registered; descriptive)
- **Voice clusters:** the H7 tests are repeated with recordings grouped by voice, and Holm is applied over the same 12 tests.
  - **Centroids:** each recording's centroid is the unit-length mean of its clip means of E2 dense L = 1 s windows.
  - **Clustering:** average linkage, cut at cosine θ. θ is the 99th percentile of D1 cosines between video centroids of different channels, computed from D1 features at run time.
  - **Values at the freeze:** θ = 0.514. On D1, 91.6% of same-channel video pairs exceed it, and on the 3 pilot channels it groups their 14 videos into exactly 3 clusters.
  - **Checks at run time:** before the primary analysis starts, the code computes θ and the clusters, and refuses to run if θ does not reproduce 0.5137 (4 decimals). A recording without any E2 window (none expected) stays a cluster of its own and is counted.
  - **How a divergence is read:** a test that replicates under speaker IDs but not under voice clusters is reported as "not robust to speaker dependence". Some dependence is likely to remain even then: on D1, 8.4% of same-channel video pairs fall below θ.
- **Contamination audit:** see §3.

## 7. Hygiene
- **Before this freeze, no D4 outcome statistic was computed.** The only things computed were:
  - counts and features;
  - the label-quality checks;
  - a smoke test (`scripts/smoke_confirm_h7.py`). It runs `confirm_h7.py`'s code on the D1 pilot channels laid out as D4, and reproduces D1 pilot v2's numbers.
- **One run:** `confirm_h7.py` is run once. It refuses to run in any of these cases:
  - any hash or input path in `FREEZE_H7.json` differs, or the output folder is not the frozen one;
  - the seed is not 20261018;
  - its output already exists.
- **What the record covers:**
  - the guarded code, the lexicons and this document;
  - every D4 feature file read;
  - the D4 archive checksums;
  - the D1 manifest and the E2 embeddings behind θ;
  - the pilot JSON holding τ.
