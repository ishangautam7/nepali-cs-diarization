# Human-verified D1 subset (EXPLORATORY; prepared, not labelled)

Prepared for the external review's issue 4. Nobody has labelled anything here yet. The analysis is pre-specified in `PROTOCOL.md` (post hoc, exploratory).

- **300 items:** 150 primary-cell (p0L1) switch points and 150 matched controls, one per switch point, from the same channel and cell (video × pause bin × position bin).
- **Sample:** drawn once with seed 20261022 from the 3,962 p0L1 switch points that have cell controls on the 17 confirmatory channels. Each channel gets a share in proportion to its count, 1 to 22 items (`subset.json`).
- **Snippets:** each item is a 4.000 s clip with the boundary at exactly 2.000 s. Where the recording clip is shorter, the file is padded with silence: 56 items (26 switch, 30 control).
- **Blinding:** names and order are shuffled, so neither shows which items are switch points. Every clip is 4.000 s, 16 kHz mono, so duration and metadata reveal nothing either.

| File | What it is |
|---|---|
| `template.csv` | The answer sheet. Each rater fills in a copy. |
| `key.csv` | Which item is which: kind, point, clip, times, padding. **Raters must not open it, nor `subset.json`.** |
| `PROTOCOL.md` | The pre-specified analysis (agreement, contamination, verified subset, central comparison). |
| `audio/` | The 300 snippets. Not in git: D1 is CC BY-NC 4.0 and the paper redistributes no audio. The folder is only for the two raters and is not shared publicly. |

## For the raters

Two **independent** bilingual (Nepali–English) listeners. **Work alone.** Do not discuss any item, and do not compare answers, until both sheets are finished and handed back. Never open `key.csv` or `subset.json`, never look at any model score, and never look at the other rater's sheet.

1. Copy `template.csv` to `rater1.csv` (the other rater: `rater2.csv`).
2. Listen to each item with headphones. An audio editor with a timeline, such as Audacity, helps with the timing fields. Then fill in, for each item:
   - **same_speaker** — `y` if the same person speaks just before and just after the 2.00 s mark, `n` if it is a different person, `unsure` if you cannot tell.
   - **audible_language_switch** — `y` if the speech changes between Nepali and English at or near 2.00 s, `n` if it does not, `unsure` if you cannot tell.
   - **switch_time_s** — only when `audible_language_switch = y`: the time **in seconds from the start of the clip** (a number between 0.000 and 4.000) at which you hear the switch. Leave blank if there is no switch. (The mark is at 2.00 s, so a switch right on the mark is about `2.0`.)
   - **boundary_within_0.1s** — `y` if that switch / word boundary falls within a tenth of a second of the 2.00 s mark (i.e. between 1.90 and 2.10 s), `n` otherwise, `unsure` if you cannot tell. For a control with no switch, judge the nearest word boundary to the mark.
   - **unusable** — `y` if overlapping voices, music or noise make the item impossible to judge, `n` otherwise.
   - **notes** — anything unusual (noise, music, overlap, unclear words).
3. Leave a cell blank only if you truly could not judge it at all. Silence at the very start or end of a clip is padding; ignore it.

Allowed values: `same_speaker` ∈ {y, n, unsure}; `audible_language_switch` ∈ {y, n, unsure}; `boundary_within_0.1s` ∈ {y, n, unsure}; `unusable` ∈ {y, n}; `switch_time_s` a number in [0, 4] or blank.

## After both sheets are in: adjudication

1. Freeze both sheets and record their sha256 **before** anyone re-listens.
2. The two raters re-listen **together** to every item they disagree on (any field), and agree a resolved answer; if they cannot agree, a third bilingual listener decides. No model scores and no `key.csv` during this step.
3. Record the outcome in `adjudication.csv`, one row per adjudicated (item, field):

   ```
   item_id,field,rater1,rater2,resolved,reason
   ```

   `field` ∈ {same_speaker, audible_language_switch, switch_time_s, boundary_within_0.1s, unusable}. Items/fields not listed are taken as already agreed.

## Commands (repo root)

On the analysis host the inputs were `BOUNDARIES=../nepdiar-box-backup/data/d1/boundaries/lex_v1b`, `EMB=../nepdiar-box-backup/data/d1/emb` and `D1=../nepdiar-host-data/d1`, folders beside the repository (D1 is the public corpus ingested with `csdiar.data.ingest_nepali_cs`; its manifest is byte-identical to the GPU box's). Only E3's grid embeddings match the frozen hashes in `subset.json`, so the central comparison uses E3.

```
# draw once (done; refuses if key.csv exists)
python -m scripts.review_human_subset --boundaries $BOUNDARIES --emb $EMB --d1 $D1 --out results/review_followups/human_subset
# recreate audio/ in a fresh clone from the committed key (byte-identical; no new draw)
python -m scripts.review_human_subset --boundaries $BOUNDARIES --emb $EMB --d1 $D1 --out results/review_followups/human_subset --audio-from-key
# score the two independent sheets (+ optional adjudication, + optional per-item model scores)
python -m scripts.score_human_subset --subset results/review_followups/human_subset \
    --rater1 results/review_followups/human_subset/rater1.csv --rater2 results/review_followups/human_subset/rater2.csv \
    --adjudication results/review_followups/human_subset/adjudication.csv \
    --boundaries $BOUNDARIES --emb $EMB \
    --out results/review_followups/human_subset/scores
```

The scorer writes `scores.json` and `scores.md`. They hold Cohen's κ and raw agreement for each question (for `same_speaker` also over y/n only), the timing-agreement summary, the disagreement list, the contamination rates with Wilson 95 % intervals (per rater, both-agree, either-flags, for switch items and controls), the verified-subset counts against the minimum (≥100 switch, ≥100 control, ≥10 channels), and — when `--boundaries`/`--emb` are given and the minimum is met — the central comparison (same-speaker similarity difference and cell-standardised ΔFAR) on the verified items for E1–E3, beside the paper's full-subset values. The scorer refuses to run on any sheet that contains key columns.
