# Analysis protocol — blinded two-rater listening audit of the D1 human subset

**Dated 2026-10-22. POST HOC and EXPLORATORY** (external review, issue 4). Written before any
label exists for this subset. It fixes every analysis choice in advance; nothing here is changed
after a sheet is read. It does not alter any pre-registered confirmatory analysis.

The subset is the predefined, frozen 300-item sample in this folder: 150 primary-cell (p0L1)
switch points and 150 matched same-language controls, one per switch point from the same channel
and cell (video × pause bin × position bin), drawn once with seed 20261022. `key.csv` and
`subset.json` are frozen; their sha256 are recorded in `subset.json` and must not change.

## 1. Raters and blinding

Two independent bilingual (Nepali–English) raters. Each fills a private copy of `template.csv`
(`rater1.csv`, `rater2.csv`). Raters work alone; they do not discuss any item before both sheets
are frozen; they never see `key.csv`, `subset.json`, the model scores, or each other's sheets.

The key is committed to the repository (its SHA-256 is in `subset.json`), and the repository
becomes public when the paper is posted. Raters therefore get only the rater package (snippets,
blank sheet, instructions) and agree in writing not to open the repository, or anything under
`results/` in it, until both sheets are frozen. The adjudication record notes each rater's
agreement (added 2026-10-23, before any rating).

Blinding already built into the package (verified, see §6): the 300 items carry opaque ids
(`i001`…`i300`) in a single seeded-shuffled order that mixes switch points and controls, so
neither the id, the row order, nor the file name reveals switch-vs-control, channel or position.
Every snippet is the same length (4.000 s, boundary at 2.000 s) and the same 16 kHz mono format,
so WAV duration and metadata reveal nothing either.

## 2. Rater questions (fields in `template.csv`)

Each rater answers, per item:

- `same_speaker` — `y` same person just before and after the 2.00 s mark, `n` different person,
  `unsure`. Blank only if unjudgeable.
- `audible_language_switch` — `y` the speech changes between Nepali and English at/near 2.00 s,
  `n` it does not, `unsure`. Blank only if unjudgeable.
- `switch_time_s` — the time in seconds **from the start of the snippet** (0.000–4.000) at which
  the rater hears the switch. Recorded whenever `audible_language_switch = y`; left blank
  otherwise. Makes the timing check quantitative so inter-rater timing agreement can be computed.
- `boundary_within_0.1s` — `y` the switch/word boundary falls within ±0.1 s of 2.00 s (i.e.
  `switch_time_s` in [1.90, 2.10]), `n` otherwise, `unsure`. Pre-defined tolerance unchanged.
- `unusable` — `y` if overlap, music or noise makes the item unusable, `n` otherwise. (New field
  for the reviewer's check 4.)
- `notes` — free text.

Allowed values are exactly those above; a value outside them stops the scorer. A blank cell means
"could not judge at all".

## 3. Disagreement resolution (adjudication)

After **both** independent sheets are frozen (their sha256 recorded), the two raters re-listen
**together** to every item on which they disagree on any of the five fields, and agree a resolved
answer; or, failing agreement, a third bilingual listener decides. No model scores, no `key.csv`
are consulted during adjudication.

Adjudication is recorded in an adjudication sheet (`adjudication.csv`), one row per adjudicated
item, keeping both original answers, the resolved answer and a one-line reason:

```
item_id, field, rater1, rater2, resolved, reason
```

`field` is one of `same_speaker, audible_language_switch, switch_time_s, boundary_within_0.1s,
unusable`. Items not listed are treated as agreed; the agreed value is used as the resolved value.
The resolved label of an item is its two raters' agreed/ adjudicated answer on each field.

## 4. Pre-specified analyses

All reported descriptively with 95 % intervals; no inferential claim is made.

### 4.1 Per-question inter-rater agreement (independent sheets, before adjudication)
Cohen's κ and raw agreement for `same_speaker`, `audible_language_switch`, `boundary_within_0.1s`,
`unusable`, each over the items both raters answered (non-blank). For `same_speaker`, κ is also
reported over `y`/`n` answers only (dropping `unsure`). Timing agreement: on items where both
raters gave a `switch_time_s`, the mean and 95th-percentile absolute difference |t1 − t2|, and the
share of pairs agreeing within 0.1 s.

### 4.2 Contamination rates (Wilson 95 % intervals)
Per kind (switch / control) and per reading (rater 1, rater 2, both-agree, either-flags), as the
existing scorer already reports:
- switch items with no audible switch (`audible_language_switch = n`) — a false switch label;
- switch items with a different speaker (`same_speaker = n`);
- control items with an audible switch (`audible_language_switch = y`);
- items off the boundary (`boundary_within_0.1s = n`);
- unusable items (`unusable = y`).
The sample is proportional to channel size, so unweighted rates estimate point-level rates over the
population defined in `subset.json`.

### 4.3 Verified subset (after adjudication)
Using the resolved labels:
- a **verified switch item** = resolved `same_speaker = y` AND `audible_language_switch = y` AND
  `boundary_within_0.1s = y` (timing within tolerance) AND `unusable = n`;
- a **verified control** = resolved `same_speaker = y` AND `audible_language_switch = n` AND
  `unusable = n`.
Report the verified counts per kind and per channel.

### 4.4 Minimum size for re-running the central comparison
Re-run the central comparison only if the verified subset holds **≥ 100 verified switch items and
≥ 100 verified controls, from ≥ 10 distinct channels**. Below any of these, report the verified
counts and contamination descriptively only, and do **not** report a re-run comparison.

### 4.5 Central comparison on the verified items (if the minimum is met)
Two estimators close to the paper's, computed on the verified items only, for E1, E2 and E3:

1. **Same-speaker similarity difference (H1-type)** — per channel, the mean cosine between the two
   adjacent 1 s windows at the verified controls minus that at the verified switch points; then the
   equal-weight mean over channels, and the same in control-score SD units. The cosine is
   1 − d(t) from the grid embeddings (`pilot.p0_curves`). This is not the paper's H1 estimator: H1
   uses point-anchored embeddings with matched control sets, and those embeddings were rebuilt after
   the first host loss and are not hash-verified, so the grid-based version is used here
   (clarified on 2026-10-22, before any rating).
2. **Change-score comparison** — the H2/P0 cell-standardised ΔFAR
   (`csdiar.analysis.pilot.h2_cell`, cell = video × pause bin × position bin): per channel the
   fire rate at verified switch points minus the cell-standardised fire rate at verified controls,
   using the **frozen τ** from `results/confirm/confirm_m2a.json` (`tau.{e1,e2,e3}`); then the
   equal-weight mean over channels. Reported beside the paper's full-subset ΔFAR
   (`h2_p0.primary.{e}.delta_far_std` in `confirm_m2a.json`).

Both are restricted to the verified item set and are DESCRIPTIVE: the subset is small, post hoc and
not powered, so no p-value or confidence claim is attached to the re-run.

## 5. Per-item model scores — data source

The subset's switch/control `point_id`s are drawn from the frozen P0 boundary tables (`lex_v1b`)
and the E3 grid embeddings, exactly as `scripts/review_human_subset.py` drew them. Per-item model
scores for the central comparison are therefore **computed exactly from the frozen inputs**, not
read from a precomputed per-item file (none exists for these 300 ids):

- **E3** — the only family whose grid embeddings in the backup match the frozen hashes recorded in
  `subset.json` (`inputs_sha256`). The comparison **uses E3** as its reference family, computed
  from:
  - boundaries `…/nepdiar-box-backup/data/d1/boundaries/lex_v1b/points_{train,test}.csv.gz`
    (sha256 in `subset.json`);
  - grid embeddings `…/nepdiar-box-backup/data/d1/emb/e3_L1/` (and `e1_L1`, `e2_L1` for E1/E2);
  - D1 manifest `…/nepdiar-host-data/d1/manifest.csv` (sha256 in `subset.json`).
- **E1, E2** — their grid embeddings exist in the backup (`e1_L1`, `e2_L1`) but are **not**
  hash-verified against the frozen set, so E1/E2 are reported only as a secondary cross-check and
  the headline re-run uses E3.
- The frozen τ for all three families is read from `results/confirm/confirm_m2a.json`.

The scorer recomputes `d(t)` and the NMS peak per item with `pilot.p0_curves` / `pilot.score_points`
(unchanged frozen code), so the per-item scores are byte-reproducible from the frozen inputs. It
never reads `key.csv` into any rater-facing path; the central comparison runs only on explicitly
supplied `--emb`/`--boundaries` inputs.

## 6. Blinding and integrity checks (automated)
`scripts/score_human_subset.py` refuses to run on any sheet that carries a key column
(`kind`, `point_id`, `t`, `clip_id`, `split`, `lang_left`, `lang_right`, …), so a sheet
accidentally built from `key.csv` cannot be scored. The 300 ids, the single shuffled order and the
uniform 4.000 s / 16 kHz format are what keep switch-vs-control, channel and position hidden.
