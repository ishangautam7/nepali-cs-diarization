# Reproducing the paper

How to reproduce the results in *Speaker-Embedding and Change-Score Sensitivity at Nepali–English Code-Switch Points*. Audio is **not** redistributed: you obtain the corpora yourself at the pinned revisions. The confirmatory analyses are frozen (git tags). The post hoc work done after the external reviews lives in `results/review_followups/` and is listed separately below.

## 0. Layout

- **Code:** `csdiar/` (data, analysis, mitigations, pipelines, figures), `scripts/` and `ops/`.
- **Plans and freezes:**
  - `planning/PREREGISTRATION*.md` and `planning/PLAN.md`;
  - `planning/FREEZE_*.json`;
  - `planning/DECISIONS.md` and `planning/DEVIATIONS.md`.
- **Results:** `results/{confirm,h7,h4,sensitivity,purity,listening,descriptive,review_followups}/`.
- **Tests:** `python -m pytest`.

## 1. Environment

The core (tagging, boundaries, metrics) is dependency-light (`pyproject.toml`). The GPU toolkits live in separate virtualenvs, each captured as a `pip freeze` lockfile in `ops/locks/`:

| Stage | venv | Lockfile | Toolkit |
|---|---|---|---|
| Embeddings E3, M3 training | `.venv-e3` | `ops/locks/venv-e3.freeze.txt` | torch 2.8.0+cu128, numpy 2.2.6, s3prl 0.4.18 |
| Confirmatory, attribution, H4, H7 | `.venv-pyannote4` | `ops/locks/venv-pyannote4.freeze.txt` | pyannote.audio 4.0.7 |
| Purity audit (TitaNet-L, CAM++) | `.venv-audit` | `ops/locks/venv-audit.freeze.txt` | — |
| Pipeline P1 (Community-1) | `.venv-p1` | `ops/locks/venv-p1.freeze.txt` | pyannote.audio 4.0.7 |
| Data ingest and metadata | `.venv-data` | `ops/locks/venv-data.freeze.txt` | — |
| GLMM sensitivity model (R) | — | `ops/locks/r-glmm.explicit.txt` | R + lme4 |

Recreate each from its lockfile, e.g. `python -m venv .venv-e3 && .venv-e3/bin/pip install -r ops/locks/venv-e3.freeze.txt`. The bootstrap helpers are `ops/env_*.sh` and `ops/bootstrap_al2023.sh`.

The post hoc pipeline check ran DiariZen under torch 2.8.0+cu128 instead of its pinned 2.1.1, with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`. Its package lists are `results/review_followups/pipeline/env_p*gpu.freeze.txt`.

## 2. Data and models at the pinned revisions

- **D1 (Nepali–English, diagnostic):** Hugging Face `saileshbro/nepali-cs-asr`, revision `dd00ca41` (CC BY-NC 4.0).
  - Record the revision with `python -m scripts.fetch_d1_metadata`, then ingest with `python -m csdiar.data.ingest_nepali_cs`.
  - The pinned revision is also in `planning/FREEZE_M2A.json`.
- **D4 (MUCS 2021 Hindi/Bengali–English):** run `ops/get_d4.sh`, then `python -m csdiar.data.ingest_d4`. The manifests are in `data/d4_meta/`.
- **Models:** download each at its pinned revision.
  - **Embeddings:** E1 is ECAPA-TDNN; E2 is the WeSpeaker ResNet34-LM in pyannote Community-1 (revision `3533c8cf`); E3 is WavLM-Large+ECAPA.
  - **Diarization pipelines:** P1 is pyannote Community-1 `3533c8cf`. P2 is DiariZen `BUT-FIT/diarizen-wavlm-large-s80-md` `6285693d`, with WeSpeaker ResNet34-LM `837717dd` and code `844f5555`.
  - **Other tools:** torchaudio MMS_FA / MMS-300M for alignment, Silero VAD, VoxLingua107 for language ID, and TitaNet-L and CAM++ for the purity audit. `ops/env_audit.sh` fetches the audit checkpoints at their pinned revisions and prints their SHA-256.

The feature files are not in the repository. Regenerate them from the audio with the frozen code, and compare them with the `feature_sha256` blocks in the freeze records.

## 3. Verify the freezes

```
python - <<'PY'
import json, hashlib, pathlib
for fz in ["planning/FREEZE_M2A.json", "planning/FREEZE_H7.json", "planning/FREEZE_H4.json"]:
    for f, h in json.load(open(fz))["code_sha256"].items():
        ok = hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest() == h
        print("OK " if ok else "MISMATCH", fz, f)
PY
```

The freeze commits and tags (`m2a-freeze` `11eb77c`, `h7-freeze` `a68e5b1`, `h4-freeze` `cd22caf`) are in the development repository, which is available on request (see `README.md`). Its tags are unsigned, and commit and tag times come from the author's machine, so check the commit hashes quoted in the paper rather than tag times.

Changes made after a freeze are logged with old and new SHA-256 in each freeze record's `amendments` and in `planning/DEVIATIONS.md`. That includes the rebuild after the GPU-box loss of 2026-10-19 (entry 6) and the post hoc additions (entries 14 and 15).

## 4. Regenerate each table and figure

The commands assume the inputs are under `~/data/d1` and `~/data/d4`; adjust the paths to yours. They are the frozen commands from `planning/FREEZE_*.json` and `results/confirm/commands.txt`.

### Confirmatory (frozen)

| Element | Command | Writes |
|---|---|---|
| **Tables I and II** | `.venv-pyannote4/bin/python -m csdiar.analysis.confirm --boundaries ~/data/d1/boundaries/lex_v1b --point-emb ~/data/d1/point_emb/lex_v1b --emb ~/data/d1/emb --windows ~/data/d1/windows/lex_v1 --align ~/data/d1/align --pilot ~/data/d1/results/pilot/pilot_v2.json --out ~/data/d1/results/confirm --splits train test --tag m2a` | `results/confirm/confirm_m2a.json` |
| Covariate attribution | `.venv-pyannote4/bin/python -m csdiar.analysis.attribution … --confirm …/confirm_m2a.json --out ~/data/d1/results/confirm --splits train test --tag m2a` | `results/confirm/attribution_m2a.json` |
| **Table III** (H7, D4) | `.venv-pyannote4/bin/python -m csdiar.analysis.confirm_h7 --d4-root ~/data/d4 --d1-root ~/data/d1 --pilot ~/data/d1/results/pilot/pilot_v2.json --out ~/data/d4/results/h7` | `results/h7/confirm_h7.json` |
| **Table IV** (H4) | train M3 (`.venv-e3/bin/python -m csdiar.mitigations.m3_train --fold F --seed S --w-dist 0.3`) and embed it (`csdiar.mitigations.m3_embed`), then `.venv-pyannote4/bin/python -m csdiar.analysis.confirm_h4 --stage 1` and `--stage 2` | `results/h4/confirm_h4_stage1.json` |
| **Fig. 1** | `python -m csdiar.figures.delta_far_fig --confirm results/confirm/confirm_m2a.json --out paper/figures/delta_far_channels.pdf` | reads `confirm_m2a.json` only |
| **Fig. 2** | `python -m csdiar.figures.umap_fig --boundaries ~/data/d1/boundaries/lex_v1b --point-emb ~/data/d1/point_emb/lex_v1b --model e3 --out paper/figures/umap_speaker_language_invariance.pdf` | boundaries and E3 point embeddings |

`delta_far_fig.py` checks that its plotted means equal Table II's statistic. `det_fig.py` and `det_ssdl_vs_dsdl.pdf` are the superseded old Fig. 1; the paper no longer uses them.

### Post hoc, exploratory (`results/review_followups/`)

These are kept apart from the frozen results; none changed a frozen file or a verdict. `results/review_followups/README.md` gives every number with its file and the full command.

| Element | Script | Writes |
|---|---|---|
| **Table V**, pipeline check | `scripts/review_pipeline_set.py`, then `scripts/review_pipeline_run.py --system p1` and `--system p2`, then `scripts/review_pipeline_score.py` | `pipeline_exploratory.json` and `.md` |
| Exact sample counts (Method) | `scripts/review_counts.py` | `counts.json` |
| Fig. 1 reconciliation (Ethics and Reproducibility) | `scripts/review_fig1_reconciliation.py` | `fig1_reconciliation.json` |
| Leave-one-channel-out check (Results) | `scripts/review_channel_exclusion.py` | `channel_exclusion.json` and `.md` |
| Listening audit (protocol fixed, not yet rated) | `scripts/review_human_subset.py` prepares the sheets; `scripts/score_human_subset.py` scores them | `human_subset/` |

The earlier single-rater listening checks are scored by `scripts/score_listening.py` into `results/listening/listening_{d1,purity}.json` (deviation log, entries 10–12). The sensitivity variants come from `ops/run_sensitivity_variants.sh` and `scripts/sensitivity_variants.py`, and are written to `results/sensitivity/`.

## 5. Build the paper

```
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The arXiv source package is built by `paper/arxiv/make_arxiv.sh`, which compiles the package from a clean folder with pdflatex, as arXiv does.

## 6. Tests

Run `python -m pytest` for the whole repository. The post hoc scripts are covered by `tests/test_review_followups.py`, `tests/test_review_channel_exclusion.py` and `tests/test_review_human_subset.py`.
