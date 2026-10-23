# Speaker-embedding and change-score sensitivity at Nepali–English code-switch points

Code, pre-registration, frozen hashes and aggregate results for the paper *Speaker-Embedding and Change-Score Sensitivity at Nepali–English Code-Switch Points* (Ishan Gautam, Pulchowk Campus, Institute of Engineering, Tribhuvan University). The LaTeX source is in `paper/`.

The study asks whether, within one speaker, speaker-embedding similarity drops and a two-window speaker-change score rises across a natural Nepali–English switch. Real recordings are used throughout. The only artificial audio is the clip joins of the post hoc pipeline check, which the paper labels as such.

- **Reproducing the results:** `REPRODUCE.md`.
- **Pre-registration and freezes:** `planning/PREREGISTRATION.md` (with its H4 and H7 supplements) and the freeze records `planning/FREEZE_*.json`.
- **Decisions and deviations:** `planning/DECISIONS.md` and `planning/DEVIATIONS.md`.
- **Frozen confirmatory results:** `results/confirm/`, `results/h7/` and `results/h4/`.
- **Post hoc, exploratory analyses:** `results/review_followups/` (see its README).

**Data.** No audio is stored in this repository.

- **D1** is the public Hugging Face dataset `saileshbro/nepali-cs-asr` at revision `dd00ca41`, licensed CC BY-NC 4.0. `scripts/fetch_d1_metadata.py` records the pinned revision.
- **D4** is MUCS 2021; see `ops/get_d4.sh`.

**Tests:** `python -m pytest`.

**This is a release copy.** It was made from the project's development repository when the paper was posted. The development repository holds the full history, including the freeze tags `m2a-freeze` (`11eb77c`), `h7-freeze` (`a68e5b1`) and `h4-freeze` (`cd22caf`) that the paper cites, and some planning notes and review records that are not included here. It is kept unchanged and is available to reviewers and editors on request.
