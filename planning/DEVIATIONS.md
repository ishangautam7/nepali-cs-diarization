# Deviations from the frozen pre-registration

Each entry records a change made after a freeze: the date, the freeze it departs from (`m2a-freeze`, or a later supplement), what changed and why, and whether any confirmatory outcome had been seen when the change was made. It also records the commit and SHA-256 of any code written after the freeze that runs on confirmatory channels (PREREGISTRATION.md §5).

## 1. Attribution: the empty-video_id refusal is removed (2026-10-18, after confirm.py had run)

- **Freeze:** `m2a-freeze` (11eb77c).
- **What happened:** the frozen `attribution.py` refused to run because 16 points have no `video_id`. They are in the 3 test clips without provenance (`noprov_test_001`–`003`, channel `UChx9-bynjfm6Js3k54Sf1Mg`). The refusal had been added in decision 22 on a reviewer's minor note.
- **The change:** the refusal is replaced by a count, reported as `points_without_video_id`.
- **Effect: none on any number.**
  - **The 16 points** (all in channel `UChx9-bynjfm6Js3k54Sf1Mg`, with eligible families in brackets):
    - `noprov_test_001:ssdl:17` (none);
    - `noprov_test_003:ssdl:5` (det);
    - controls in `noprov_test_002`:
      - `ctrl:1`, `10`, `11`, `12`, `13`, `17`, `22`, `27`, `29` (det, emb0.5);
      - `ctrl:18`, `21`, `28` (det, det_w1, p0L1, emb0.5, emb1);
      - `ctrl:19`, `20` (the same plus emb1.5).
  - **Why none can enter an attribution row:**
    - **H1 rows are matched sets.** Matching is within the same clip or video, and no `emb1` switch point exists in these clips, so no set contains them.
    - **P0 rows** are `p0L1` switch points and the `det_w1` controls in their cells. Neither switch point here is `p0L1`-eligible, so no cell with video "" has a switch point.
    - P0 cells keep `pilot.h2_cell`'s key (video_id, pause bin, position bin), as before.
- **What had been seen:**
  - The primary analysis (`confirm.py`) had finished when the change was made, and its outcome table was visible in the run log.
  - No attribution outcome had been computed.
  - The change removes a guard and makes no analytic choice.
- **Code and record:**
  - `attribution.py` SHA-256 changed from `ec729327c7cd8dfe1933d8e8da86887233183b432747209f98f8bf01e1f9d28e` to `181bc02aecfb7ab5c3b23aa03b0f0e88a3ea7a21d453ce6102a2a83716a60ab0`.
  - `planning/FREEZE_M2A.json` lists this under `amendments` and holds the new hash, so the guard accepts the fixed file and only it.
  - `confirm.py` is not re-run.

## 2. H7: `pilot.h2_cell` no longer crashes on a speaker with no cell-matched controls (2026-10-18, before any H7 outcome was written)

- **Freeze:** `h7-freeze` (a68e5b1).
- **What happened:** the one-shot H7 run crashed in `pilot.h2_cell` (imported by `confirm_h7`) with `TypeError: unsupported operand type(s) for +: 'float' and 'NoneType'`. A MUCS speaker can have SS-DL switch points but none with a matched control in the same (video, pause-bin, position-bin) cell; that speaker's per-channel `delta_far_std`/`auc`/`far_ssdl` are then `None`/`nan`, and the three cell-level aggregates (`auc_mean_of_channels`, `auc_between_channel_sd`, `delta_far_std_mean_of_channels`) summed over `None`. D1 has no such channel (17 populous channels), so this path never ran on D1 and is specific to D4's long tail of small MUCS speakers.
- **The change:** the three aggregates now skip channels whose estimate is `None`/`nan`, and a new `n_channels_with_estimate` records how many remain. These are **descriptive** cell-level summaries; the H7 **primary** outcome is the channel-level AUC via `confirm_h7.h2_test`'s `included()` (min 5 units) and the sign-flip, which were never reached because the crash was in the descriptive aggregation.
- **Effect: none on any D1 number, none on any H7 primary outcome.**
  - The D1 pilot reproduces byte-for-byte after the fix (dz 0.345/0.304/0.242; p0L1 AUC 0.615/0.586/0.592; τ 0.917/0.863/0.929; ΔFAR +0.029/+0.025/+0.030), because every D1 channel has cell controls so no element is dropped and `len(aucs) > 1` holds.
  - No H7 outcome had been computed or written: the process raised before `write()`, and `~/data/d4/results/h7/` was empty.
- **What had been seen:** nothing — the run produced no result table, no JSON, no stdout rows. The change is a crash fix in descriptive aggregation, not an analytic choice, and was made before any H7 number existed.
- **Code and record:**
  - `csdiar/analysis/pilot.py` SHA-256 changed from `d15644be035fd58948ca21414a8c3519af494cd168fd3895de26c98cdd585525` to `34665bde6f2df6b8e59fda173a25a9d1165825aed9867ba641ad9f9e7e2c1816`.
  - `planning/FREEZE_H7.json` lists this under `amendments` and holds the new hash, so the guard accepts the fixed file and only it.
  - H7 is then run once, for the first time producing a result.

## 3. Speaker-purity audit code, recorded before it runs (2026-10-19, before any audit result exists)

- **Freeze:** `m2a-freeze` (11eb77c), PREREGISTRATION.md §5 item 4, whose rules this code implements. The primary H1–H3 and H7 outcomes were already known when it was written, as §5 anticipates ("the audit runs after the freeze"). No audit result has been computed.
- **Code (SHA-256), committed with this entry:**
  - `csdiar/data/audit_embed.py` `bad408e433517441c3cd53375cf44bd96234c42e6e5aa007d3e1c9c9c1727ba0` (clip embeddings);
  - `csdiar/analysis/purity_audit.py` `3670db52ab0ddc5d2f0529432ffeec8c8b42ad70284b83d97e4cc5e6f323e04e` (rules, subset, listening sample);
  - `ops/env_audit.sh` `c384dbbfcd3f08fa221427dbc8c035c58288f4665f314d0454004d430a97e025`; tests in `tests/test_purity_audit.py`.
- **Embeddings:** TitaNet-L through NeMo 2.7.3 on the GPU (`nvidia/speakerverification_en_titanet_large` at `0dc382f4`, `.nemo` SHA-256 `e838520693f269e7984f55bc8eb3c2d60ccf246bf4b896d4be9bcabe3e4b0fe3`). CAM++ through its official ONNX export with onnxruntime 1.31.0 on the CPU (`Wespeaker/wespeaker-voxceleb-campplus` at `acf623ad`, `.onnx` SHA-256 `b50810498b5bcf5773d086f6993d344476bd0c88b566a41e8d801aaf8461efad`). TitaNet-L runs, so the ERes2NetV2 fallback is not used. Environment `.venv-audit`: torch 2.8.0+cu128, lightning 2.4.0, numpy 2.5.4; full list in `.venv-audit/freeze.txt` on the box.
- **Operational readings of §5 item 4, fixed before running (no outcome informs them):**
  - "Mean over speech": Silero VAD frames with p ≥ 0.5 are speech, gaps of at most 0.25 s are bridged, and every speech run of at least 0.5 s is embedded on its own. Runs are never joined, so no audio is concatenated. The clip embedding is the duration-weighted mean of the unit run embeddings, re-normalised; a clip with no such run is embedded whole and counted.
  - "That stratum's median" and the MAD are taken over the clips of the same video and stratum, the set the comparison is made in. A cell whose MAD is 0 flags nothing.
  - CMI tertiles are cut at the 1/3 and 2/3 quantiles of the channel's clip CMI values (tertile 0 if CMI ≤ q1, 1 if ≤ q2, else 2).
  - Video centroids are the re-normalised mean of all the video's unit clip embeddings. The split-half null uses every video with at least 4 clips, 20 random halvings each (seed 0), separately per embedding. A channel with one video has no split-off test.
  - The 3 test clips without provenance (no `video_id`) are left out of the audit.
  - Same teacher on two channels (PLAN §4 step 6c) is reported descriptively and changes nothing in the subset.

## 4. Purity-subset sensitivity run of the M2a analysis, recorded before it runs (2026-10-19)

- **Freeze:** `m2a-freeze` (11eb77c), PREREGISTRATION.md §5 item 4. The audit result is decision 27; no sensitivity number exists yet.
- **Code:** `csdiar/analysis/purity_sensitivity.py`, SHA-256 `f23b0809d94dc8710cc6c4dc7cdbbc492d2087ba1704d0d2e45fe4228ef203a3`, committed with this entry. It calls the frozen `confirm.run_h1`, `run_h2` and `run_h3` unchanged, with the frozen τ, on the confirmatory splits (train and test).
- **What it changes, as §5 item 4 says:** the audit's 688 flagged clips are dropped, and the 34 split-off videos become clusters of their own. H1 matched sets survive only if their switch point and control both do. H3 reads a filtered copy of the frozen clean windows.
- **Known property, stated before the run:** the frozen split-off rule also splits recording sessions of one teacher (decision 27), so the subset has more clusters than speakers, and its sign-flip treats sessions of one teacher as independent. The result is read with that caveat; it is descriptive and changes no primary verdict.

## 5. H4 secondary-analysis code, recorded before it runs on any scored channel (2026-10-19)

- **Freeze:** `h4-freeze` (cd22caf) with `planning/FREEZE_H4.json` (c1f3fa6, tag `h4-freeze-record`). The code
  implements the section 8 secondaries of the H4 supplement that `confirm_h4.py` does not compute. It was written
  after the freeze and before stage 1 has run, so no H4 outcome exists.
- **Code (SHA-256), committed at 7d7c3be:** `csdiar/analysis/secondary_h4.py`
  `e6a5b1666b9bf8b90f06d57767cd1dc77ed1dbb0486de7657f62f52badc5c60c`; tests in `tests/test_secondary_h4.py`.
  The same hash is on the box and in the trial-run records below.
- **Reviews:** two code reviews (round 1: 2 major, 4 minor, fixed in 94dd3ae; round 2: no major, 2 of 4 minor
  applied in 7d7c3be).
- **Trial runs, calibration channels only (structure, not effects):** every part, the M3 parts with the first
  finished primary M3 grid (`m3_f0_s0_w0.3_none`), exit 0, no NaN in any output
  (`~/data/d1/results/h4_dryrun/` on the box). No scored channel was read.
- **How it runs:** once, after `confirm_h4.py` stage 1, with `--scope cv` and then `--scope test`. It refuses to
  continue unless its own recomputation of stage 1's per-channel numbers matches stage 1 exactly.

## 6. GPU box lost; D1 features, M3 inputs and M3 models regenerated with the frozen code (2026-10-19, before any H4 outcome exists)

- **What happened:** the g7e.2xlarge box stopped answering at about 05:45 UTC on 2026-10-19 (a Spot interruption,
  per Ishan) and was replaced by a new instance with a fresh disk. Everything under `~/data` on the old box was
  lost: the D1 ingest and features, the M3 inputs, the 4 finished primary M3 models (of 25), and the trial-run
  records of entry 5. Code, plans, reviews and every reported result were on GitHub and on a backup host and
  are unaffected.
- **No H4 outcome existed:** neither `confirm_h4.py` stage 1 nor `secondary_h4.py` had read a scored channel. The
  M2a and H7 confirmatory results were computed and committed before the loss and are not re-run.
- **Same code:** all 18 files hashed in `FREEZE_H4.json` match their frozen SHA-256, and every unguarded producer
  (`ingest_nepali_cs`, `align`, `vad`, `windows`, `audit_embed`, `lid`, `point_embed`, `scripts/download_d1.py`,
  `ops/run_embeddings.sh`, the env scripts and the lexicon files) is unchanged since the freeze commit `cd22caf`.
- **Regeneration:** `ops/rebuild_features.sh`, committed with this entry, re-runs the original chain in the
  rebuilt environments (`ops/rebuild_envs.sh`), from the same public D1 parquet and from IndicVoices Nepali and
  pyannote Community-1 at their pinned revisions. Two departures from the original runs, neither touching a frozen
  rule: only the L = 1 s grids are rebuilt (H4 and its secondaries read nothing else), and parquet and audio I/O
  (`ingest_nepali_cs`, `m3_data`) runs in `.venv-audit`, the one rebuilt environment with pyarrow, soundfile and
  numpy together.
- **Hashes will differ:** GPU-computed features (alignment, VAD, embeddings, LID) are expected to differ from the
  frozen files in low-order bits, and files derived from them (boundaries, windows, M3 window labels, the dev
  trials) can differ more. Before stage 1 runs, the new SHA-256 of every pinned feature file is written to
  `FREEZE_H4.json` as a dated amendment that keeps each old value next to the new one, and files that reproduce
  exactly are listed as such.
- **M3:** the 65 jobs of PREREGISTRATION_H4.md §7 (25 primary fold models, 5 all-data models, 35 seed-0 variants)
  are trained from scratch with the frozen commands and the frozen recipe (decision 28), from `ops/queues/m3_h4.txt`.
- **Outcome (recorded 2026-10-19, amendment `rebuild-2026-10-19` in `FREEZE_H4.json`, made with
  `scripts/amend_freeze_h4_rebuild.py` and verified on the box before this commit):** 19 of the 27 pinned files
  reproduced byte-for-byte, including the E3 L = 1 s grids, the LID posteriors, both audit embeddings, the purity
  subset, the boundaries summary (same counts), the M3 window labels, the D5 pool and the M3 audio memmaps. The 8
  that differ all derive from the word alignment and VAD frames, which are not pinned: the four lex_v1b point and
  match tables and the two lex_v1 window tables (sizes within 0.8%), the E3 point embeddings at those points, and
  the M3 dev trials drawn from the windows. The first rebuilt primary M3 model reached a dev EER of 0.135, against
  0.136 for its lost counterpart.
- **Environments (recorded 2026-10-19, before any H4 outcome):** the package lists of the five rebuilt Python
  environments are committed as `ops/locks/venv-*.freeze.txt` (`uv pip freeze`, Python 3.12.14 in each).
  `confirm_h4` and `secondary_h4` run in `.venv-pyannote4` (numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1); M3 trains
  and embeds in `.venv-e3` (torch 2.8.0+cu128, numpy 2.2.6, s3prl 0.4.18). `.venv-audit` has every version entry 3
  lists (torch 2.8.0+cu128, lightning 2.4.0, numpy 2.5.4, NeMo 2.7.3, onnxruntime 1.31.0).

## 7. DISPLACE 2024 access not granted; the conversation stage (M2b) is not run (2026-10-19, before any H4 outcome exists)

- **What happened:** the DISPLACE 2024 password never arrived, and Ishan decided on 2026-10-19 to finish the study
  without it. Both encrypted archives had been fetched and checksum-verified (`ops/get_displace2024.sh`); they were
  never opened, listed beyond their file names, or scored.
- **No fallback data:** the M2b supplement's fallback (its §10) is DISPLACE-M, which is encrypted as well and has no
  language references (decision 29), and no D2-P hour has been annotated. No available dataset has reference
  speaker and language turns.
- **What follows, as the plans specify where they can:**
  - M2's gate weight α is never frozen, so M2's D1 H4 test is reported as not tested (H4 supplement §4, M2b
    supplement §6). `confirm_h4.py` stage 1 runs as frozen, with M2 in the H4 Holm family at p = 1, so its M1 and M3
    verdicts are final; stage 2 never runs.
  - H5, H6, the joint H4 × H5 test and the conversation parts of H2 and H4 are not tested. PLAN.md's fallback ("H5
    falls back to descriptive") needs DS-DL turns, which no available dataset supplies, so no H5 number is reported,
    descriptive or otherwise.
  - The M2b supplement stays unfrozen (there is no `FREEZE_M2B.json`). Its code (`ingest_displace`, `m2b_hooks`,
    `score_m2b`, `run_m2b`, `tune_m2b`, `tune_p0_m2b`, `power_m2b`, `confirm_m2b`, `freeze_m2b`) stays in the
    repository, reviewed but never run on data.
- **Unchanged:** H4 on D1 for M1 and M3, that is `confirm_h4.py` stage 1 and the secondaries of `secondary_h4.py`,
  and all 65 M3 jobs of PREREGISTRATION_H4.md §7, including the five all-data models that `secondary_h4 --scope
  test` reads.

## 8. The two listening checks are done by the author (2026-10-19, before any listening answer exists)

- **The plan:** hired bilingual annotators do all annotation, including the D1 listening checks, and a second
  annotator redoes 20% for agreement (decision 2).
- **Instead:** no annotator was hired. Ishan Gautam, the author, a Nepali–English bilingual, does both listening
  tasks himself, at his own request:
  - the 450-item blind D1 check (PLAN.md §4 step 5; `scripts/make_listening_check.py`, seed 20261018, unchanged),
    drawn from the rebuilt `lex_v1b` boundary tables of entry 6, whose sizes are within 0.8% of the frozen ones;
  - the purity audit's 200 flagged clips (PLAN.md §4 step 6d, `listening_sample.csv` of `purity_audit.py`). Each is
    paired with an unflagged clip of the same video, the two played in random order as A and B under the question
    "same speaker?", and 30 pairs of two unflagged clips from the same videos are mixed in unannounced, so the
    listener's own rate of hearing a different speaker can be read beside the flagged pairs' rate.
- **Consequences:** there is one rater, so no agreement statistic exists, and the rater knows the hypotheses. The D1
  page hides the condition (switch points and controls mixed, no transcript), and the purity page does not say which
  clip of a pair is flagged or which pairs are controls. Both results are reported as one rater's judgements, and
  the contamination-corrected ΔFAR (PREREGISTRATION.md §5 item 5) is computed from them.

## 9. Code for the variant boundary runs (sensitivity items 2, 3 and 8), recorded before it runs (2026-10-19)

- **Why:** PREREGISTRATION.md §5 says the code for items 2, 3 and 8 is committed, with its hash recorded here,
  before it runs on a confirmatory channel.
- **Code (SHA-256):** `scripts/sensitivity_variants.py`
  `3b75c330f21c5247be9b19d98c8bc6d950fba3ccb2bd8304227c53eac335e8ff`; tests in `tests/test_sensitivity_variants.py`;
  code review (no major; its minors applied before this hash).
- **How:** each variant swaps one function of the frozen `csdiar/data/boundaries.py` for the run, the way
  `boundaries_d4.py` swaps its tagger, and runs the frozen boundary command with only `--out` changed; no frozen file
  is edited. Item 2 tags "I" and capital "A" en where the single-letter rule made them neutral; item 3 keeps the
  low-alignment clips (every clip with an aligned word scores +inf, so none falls below the threshold); item 8 tags
  loan-lexicon words in the language they are pronounced in (Devanagari en, Latin ne; `loans.txt` is all
  Devanagari). The frozen M2a analysis (`confirm.primary_block`, frozen tau, seed 20261018, train and test) then
  runs on each variant's points and their point embeddings from `csdiar.models.point_embed`, descriptively.
- **Reference run:** the inputs are the ones rebuilt after the box loss (entry 6), whose point tables differ
  slightly from those of the M2a confirmatory run. So the same analysis also runs once on the rebuilt, unvaried
  `lex_v1b` (`base`), each variant is read against base, and base against the M2a numbers shows what the rebuild
  alone changed. None of these runs changes an M2a verdict, which stays as computed at the freeze.

## 10. How the two listening checks are scored, fixed before any answer is read (2026-10-19)

- **Why:** PLAN.md §4 step 5 and PREREGISTRATION.md §5 item 5 say the listening check's contamination rate "feeds a
  contamination-corrected ΔFAR" but give no formula, and the purity listening (PLAN.md §4 step 6d) names no summary.
  Both are fixed here, with their code, before any answer exists.
- **Code (SHA-256):** `scripts/score_listening.py`
  `0e5f222ec2c20f4b174d1bb2229f8629ae32b912b1c6939bf711f64cab7f6ce3`; tests in `tests/test_score_listening.py`;
  code reviews (round 1: 3 major, fixed; round 2: no major).
- **D1 check:** items are weighted by the inverse of their stratum's sampling fraction (the page's four pools), and
  the rates use the confirmatory splits only (train, test), the 17 channels of M2a's ΔFAR. Over the switch points of
  the primary cell (`p0L1`): π, the weighted share answered "no language change"; δ, the share answered "different
  speaker" or "overlap or noise" (both pooled into δ); and the share whose marker is within about 0.1 s of a word
  boundary. Over its `det_w1` controls: κ, the share answered "language change". An "unsure" leaves that item out of
  that question's rate. The corrected excess is ΔFAR / (1 − π − κ) for each embedding's primary-cell ΔFAR, assuming
  that a mislabelled switch point fires like a control and a mislabelled control like a switch point, and that
  contamination does not depend on channel or cell; it is descriptive. δ and the boundary share are reported beside
  it, not folded in. Intervals: 10,000 stratified bootstraps of the items (seed 20261018), 2.5 and 97.5 percentiles.
  Coverage (points of the cell that neither pool could sample) is counted over the confirmatory splits; the
  per-stratum answer tables describe the whole sample, pilot items included.
- **Purity check:** for flagged and for control pairs, the share answered "different speaker" among the usable
  pairs judged same or different (the headline), the same share over all judged pairs, and the share marked
  unusable, with Wilson 95% intervals and the differences flagged minus control.

## 11. The D1 listening check stopped after 352 of its 450 items; it is scored on those 352 (2026-10-20, before any answer value was read)

- **What happened:** the author answered the page's first 352 items, in page order, and then stopped for lack of time.
  All three questions are answered for 350 of them. Items L158 and L312 lack only the third (boundary) answer, and
  items L353–L450 are unanswered. Before this entry and its code were committed, only that structure was inspected
  (which cells are blank); no answer value was read.
- **Why the 352 still follow the plan's design:** the page shows its 450 items in a seeded random order
  (`scripts/make_listening_check.py`: the four strata are drawn first, then one `rng.shuffle` permutes the combined
  list of 450, and item numbers are assigned after it). The first 352 are therefore a simple random sample of the
  450, and within each of the four strata a simple random sample of that stratum's sampled items. Stopping depended
  on time, not on any answer.
- **Rule** (`score_listening.py d1 --first 352`): only L001–L352 are scored, and any answer after them stops the run.
  Inside them, q1 and q2 must be answered; a blank q3 is left out of the q3 shares, like "unsure", and counted, in
  total and per stratum. Each stratum's weight is its pool size over its count among the scored items. Everything
  else is as in entry 10, and without `--first` entry 10's rules are unchanged.
- **Consequences:** 22% fewer items than planned, so intervals about 13% wider. The paper reports a 352-item check.
- **Code (SHA-256):** `scripts/score_listening.py`
  `069bbcdf031c72e1d89942d4260f8396ef00c2256cbd0353d8805a8a593c2f65` (entry 10's version was `0e5f222e…`); tests in
  `tests/test_score_listening.py` (12 pass); code review (no major).
  Recorded and committed before the run.

## 12. The purity listening check stopped after 117 of its 230 pairs; it is scored on those pairs (2026-10-20, before any answer value was read)

- **What happened:** the author answered the page's first 117 pairs (P001–P117), in page order, with both questions
  answered for every one of them, and P118–P230 unanswered. Before this entry and its code were committed, only that
  structure was inspected; no answer value was read. If the author sends a longer export before the run, its answered
  prefix replaces this one under the same rule, and the count here is updated before any value is read.
- **Why the prefix still follows the page's design:** `scripts/make_purity_listening.py` builds the 200 flagged and
  30 control pairs, permutes all of them with one `rng.shuffle`, and numbers them after it. A uniform permutation's
  first N pairs are a simple random sample of the 230 and, restricted to each kind, a simple random sample of that
  kind's pairs. Stopping depended on time, not on any answer.
- **Rule** (`score_listening.py purity --first N`): only P001–PN are scored, and any answer after them stops the run.
  Both questions must be answered inside N. The summaries are those of entry 10, computed over the scored pairs. The
  d1 path is unchanged.
- **Consequences:** about half the planned pairs, so the flagged pairs' interval is roughly 1.4 times as wide, and
  the control pairs, about 15 of them, bound only a large difference. Item 4's listening part is reported as a
  117-pair check.
- **Code (SHA-256):** `scripts/score_listening.py`
  `05f61a4b464ba9f10eda94f380af98d47a12d41a2d358dc75482a9717abdb1a3` (entry 11's version was `069bbcdf…`; its d1
  code is unchanged); tests in `tests/test_score_listening.py` (13 pass); code
  review: no major. Recorded and committed before the run.

## 13. H4 stage 1 was interrupted by the loss of the GPU box and is restarted once (2026-10-20, before any H4 outcome existed)

- **What happened:** after all 25 primary M3 models finished and their hashes were recorded and merged (PR #46),
  `confirm_h4 --stage 1` started at 11:06Z on the box at 54.175.89.133. It passed its freeze check at 11:08Z, writing
  `confirm_h4_stage1.started`, and was computing when the box stopped answering at about 11:15Z. The box never came
  back, and Ishan replaced it at about 12:36Z with a new On-Demand g7e.2xlarge at 52.90.239.193. No stage-1 result was
  written: at the last contact, at 11:12Z, `results/h4` held only the start marker and the process had printed
  nothing. Nobody saw any H4 statistic. The cause is unknown, because the old box's logs went with it.
- **Recovery:** the new box was rebuilt with the same environments. All five match `ops/locks/` after six packages
  that had drifted were pinned back to their locked versions. The 30-minute backup was restored, and the inputs it
  leaves out were regenerated: the D1 WAVs (the manifest again matches its frozen hash), the D5 pool and the M3 audio
  memmaps. Before the restart, every file `FREEZE_H4.json` pins was checked against its recorded hash: the 27 pinned
  inputs and 18 guarded code files by `scripts/amend_freeze_h4_rebuild.py verify`, and the 75 M3 artifacts by a direct
  SHA-256 comparison. All matched (2026-10-20, 15:3xZ), and `confirm_h4`'s own freeze check repeats both at the start.
- **Restart:** stage 1 runs once on the new box, with the frozen code, inputs, folds and seed, and with the box to
  itself (the M3 queue restarts after it), in case memory pressure caused the loss. Its one-shot marker existed only
  on the lost disk, so nothing on the new box would stop a second start. This entry records that the restart is the
  second start of the same pre-registered run, and the first and only one to produce a result.
- **Unchanged:** the frozen code, inputs, decision rules and the M3 recipe. The secondaries of `secondary_h4` and the
  M3 jobs not yet finished (the queue resumes, skipping finished jobs) are unaffected.
- **All-data M3 models (added 2026-10-20, about 19:20Z, before `secondary_h4 --scope test` ran):** these five models
  feed only the descriptive test-scope secondaries. On the new box the queue skipped `all_s0`, finished on the old box
  and restored from the backup, and trained and embedded `all_s1` to `all_s4` once each, from scratch, with the
  pre-registered configuration and seeds (15:59–19:15Z). `all_s2` had been embedding on the old box when it was
  lost; no file of that interrupted run was read. Before the test-scope run, all five folders were checked: each
  `meta.json` names the SHA-256 of its run's `best.pt`, and every split (validation, train, test) is complete and
  finite, with E3's clips and window counts (2,338, 16,702 and 1,765 clips).

## 14. Reporting changes after an external manuscript review (2026-10-22, after all results; no statistic recomputed)

- **Fig. 1 replaced.** `det_ssdl_vs_dsdl.pdf` (`csdiar/figures/det_fig.py`) plotted raw fire rates pooled over all switch points and controls of all 20 channels, pilot included. It is point-weighted and not cell-standardised, so its operating points (about −1.1/−0.2/+0.2 pp) did not show Table II's ΔFAR (+2.46/+2.95/+2.73 pp, an equal-weight mean of per-channel, cell-standardised values on the 17 confirmatory channels), although its caption said they matched. The new Fig. 1, `delta_far_channels.pdf` (`csdiar/figures/delta_far_fig.py`), is drawn from `results/confirm/confirm_m2a.json` alone: per-channel ΔFAR, the equal-weight mean (asserted equal to the reported statistic) and the point-weighted mean (+0.78/+2.00/+1.17 pp).
- **"Language-attributable" reported as "covariate-robust".** The grading rule is unchanged; only its name in the paper changes, because the design is observational.
- **D4 voice-cluster sensitivity.** Every voice-cluster test failed to replicate (Holm p 0.1875 Hindi, 0.875 Bengali), which PREREGISTRATION_H7.md would label "not robust to speaker dependence". With 6 and 3 clusters no Holm-adjusted p can fall below .05, so the paper reports the computed verdict and the rule, and reads the analysis as uninformative rather than applying the label (as DECISIONS.md decision 25 already did).
- **Scope.** The paper is retitled and reframed as a study of embedding and change-score sensitivity, since no full pipeline (P1/P2) and no DS-DL test was run.

## 15. Post hoc additions and reporting changes after the second external review (2026-10-22, after all results; no frozen file or verdict changed)

- **Exploratory full-pipeline check** (PR #62, in answer to the first external review; reported in the paper's Section VIII since PR #63): stock pyannote Community-1 and DiariZen on D1 clips and artificial clip joins (`results/review_followups/pipeline_exploratory.*`). It is not the pre-registered P1/P2 test, which still was not run; the H2 family keeps its two places at p = 1.
- **Leave-one-channel-out check** (`scripts/review_channel_exclusion.py`, `results/review_followups/channel_exclusion.*`), as the reviewer asked:
  - Every main effect is refitted 17 times, each time without one confirmatory channel, channel- and switch-weighted, and every fit is reported. No other exclusion rule was tried.
  - H1, H2-P0, ΔFAR and H3 are exact from the frozen per-channel statistics. The pooled det and det_w1 AUCs come from the rebuilt inputs (entry 6) and are labelled so.
  - No estimate changes direction and every primary test stays significant. E1's channel-averaged ΔFAR falls from +2.46 to +1.17 pp (p = .09) without its smallest channel (13 switch points); the paper says so in Section V and in the Discussion.
  - The confirmatory channel that may share a teacher with a pilot-channel video is examined separately: without it no primary estimate moves by more than 0.006.
- **Listening audit for two independent raters:** the protocol, sheets and analysis are fixed in `results/review_followups/human_subset/PROTOCOL.md` before any rating, on the predefined sample of PR #62 (seed 20261022; `key.csv` and `subset.json` unchanged). No rating exists yet, so the paper calls its evidence preliminary.
- **Wording, no statistic changed:**
  - The grade *nonzero but trivial* is reported as "significant, below the smallest effect", because E2's intervals include the smallest effect of interest. The pre-registered label is named in the paper's statistics section and in the Table I caption.
  - "We powered the study" is replaced by the pre-registered power table (PREREGISTRATION.md, Power): the sample was fixed by the data.
  - "Purity-verified" becomes "purity-audited".
  - The paper states that Holm adjusts the tests' p-values, not the estimates.
  - A short Conclusion is added.
- **Bibliography:** seven entries corrected against their authoritative records.

## 16. Corrections and an omission found in a review of the paper (2026-10-23, after all results; no frozen file or verdict changed)

- **Voice-cluster Holm p (correcting entry 14).** Entry 14 gives "Holm p 0.1875 Hindi, 0.875 Bengali". In `results/h7/confirm_h7.json` five Hindi tests have 0.1875 and the sixth (H2-P0 with E1) has 0.875; all six Bengali tests have 0.875. The paper now says so (Section VI and the Table III caption).
- **Tag noise on the 777 human-verified D1 clips was not measured.** `PREREGISTRATION_H4.md` says that tag noise on the M3 training windows "is measured on the 777 human-verified clips (window labels from the silver transcript vs the verified one) and reported". It was not run, and no result exists. The paper's Limitations now say so.
- **Reporting fixes** (audit against the result files):
  - The D4 counts now separate eligible switch points from those with matched or cell controls.
  - Three roundings are corrected: Table III's Bengali H2-P0 AUC for E1 is 0.53, E3's covariate-adjusted p is .008, and E2's H3 without the test channels is −0.17.
  - The switch-point rule is described as `csdiar/data/boundaries.py` implements it, and the phrase-boundary contrast as pre-registered.
- **Pre-registered secondary outcomes now reported** (frozen values; nothing recomputed): H1 at 0.5 s and 1.5 s windows, the cross- and mono-language EER on H3's trials, and P0's fire rates by boundary type.
