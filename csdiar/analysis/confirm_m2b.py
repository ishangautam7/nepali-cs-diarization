#!/usr/bin/env python3
"""The one-shot DISPLACE 2024 confirmatory analysis for the M2b conversation study (PREREGISTRATION_M2B.md §7,
§8, §11). Every confirmatory conversation outcome -- H4 (SS-DL excess reduction), H5 (DS-DL non-inferiority),
their per-mitigation intersection-union, H6 (trade-off ordering) and the conversation H2 (native P1/P2) -- is
computed HERE and nowhere else.

Order and hygiene (§11, as `confirm.py` and `confirm_h4.py`). The script REFUSES to run unless
`planning/FREEZE_M2B.json` matches every guarded code hash, every dev/eval feature and reference hash, every
tuning value, the power verdict, the M3 artifacts, every eval hypothesis hash and every input path, with the
seed 20261018, and unless neither `confirm_m2b.json` nor `confirm_m2b.started` exists in the output dir. It then
writes `confirm_m2b.started` (with the commit) BEFORE reading any eval reference -- the first eval read in the
whole study -- ingests the eval references in-process with `ingest_displace.process_recording` at the frozen eval
VAD, scores every system's frozen eval hypotheses with `score_m2b`, runs the statistics, and writes one
`confirm_m2b.json`.

`--dry-run` runs the SAME analysis on the DEV split (dev references, dev hypotheses), with NO freeze check, NO
markers, and NO eval path ever touched, writing `confirm_m2b_dryrun.json`. It is how the pipeline is exercised on
real data before the single eval shot.

Reused verbatim, never re-implemented (frozen / reviewed): `csdiar.analysis.stats` (`signflip`, `bca_ci`,
`holm`, `outcome`); `csdiar.analysis.score_m2b` (`score_system`, `score_session`, `hypothesis_changes`,
`read_det`, `det_summary`, `parse_hypothesis_rttm`); `csdiar.data.ingest_displace` (`process_recording`);
`csdiar.analysis.confirm_h4` (`secondary_vs` for the unadjusted §8 comparisons). The H4 family machinery differs
from `confirm_h4.test_family` ONLY in that its cluster is the recording SESSION, not the D1 channel, and the
per-session statistic is `score_m2b`'s cell-standardised delta-FAR rather than `pilot.h2_cell`'s channel one;
`confirm_h4.test_family` is NOT reused because it hard-codes the M1/M2/M3 stage split and the D1 per-channel
dictionaries (see `_h4_family` below, and the test that pins the two agree on a shared one-family input).

Open choices (the supplement leaves these to the implementation; each is a NAMED CONSTANT here and is listed in
the final report):
  * `H5_HOLM`           -- how Holm applies to H5's bound-based non-inferiority test. "p_from_bound": a
                          non-inferiority p-value is formed by inverting the one-sided BCa bound on the SAME
                          resamples (`_ni_pvalue`), then Holm over M1-M3 (m = 3); the raw bound and whether it
                          lies below the margin are always reported beside it.
  * `H5_DECISION`       -- how the per-mitigation non-inferiority VERDICT is reached (settled for this round,
                          kept a named open choice for the design round). "holm_p_from_inverted_bound": from ONE
                          set of BCA_N_BOOT whole-session resamples drawn exactly as `stats.bca_ci` draws them
                          from the module generator in a fixed order, (a) the one-sided 95% BCa upper bound is
                          reported, equal bit-for-bit to `bca_ci(x, mean_stat, 0.90, rng)["hi"]` at that
                          generator state, and (b) the non-inferiority p is that bound inverted at the margin.
                          Holm over M1-M3 (a not-assessable member enters p = 1); NON-INFERIOR IFF Holm p < .05,
                          and the intersection-union uses this decision.
  * `H5_TWO_SIDED_LEVEL`-- the level of H5's reported two-sided interval. 98.3% (= 1 - 0.05/3, the same level as
                          H4's two-sided BCa; §7 "BCa levels"), reported ALONGSIDE the one-sided 95% bound.
  * `H6_PVALUE`         -- the H6 pairwise bootstrap p-value. "directional_sign": a genuine two-stage bootstrap
                          resamples sessions then, within each drawn session, M3's seeds; the ONE-SIDED p in the
                          predicted direction (the share of replicates whose mean pairwise MR difference is at
                          or below 0) feeds Holm and "significant", and the TWO-SIDED p drives "reversed", as
                          `stats.outcome`'s convention does elsewhere. Holm m = 3; not-reached replicates are
                          dropped and counted.
  * `H6_ORDERING`       -- what "ordering significant" means. "all_pairs": the predicted order (M3 best, M1
                          smallest gain) holds only when ALL THREE predicted pairwise directions are significant
                          after Holm (the strict reading); the per-pair verdicts are reported either way.
  * `H6_DET_POOLING`    -- the DET for H6 is read per session then averaged, "by_session" (H6 is a per-session
                          clustered statistic like H4/H5, so the MR-at-FAR=10% that enters it is the session's
                          own DET reading; the pooled DET is reported beside it).
  * `H2_CONTROL_NEVER_FIRES` -- a session whose numerator OR denominator sum is zero has an undefined log RR.
                          "either_sum_zero_drops_session": the session is dropped from the H2 log-RR sign-flip
                          and COUNTED whenever EITHER sum is zero -- a zero denominator (its cell controls never
                          fire, a 0/0 or x/0 ratio) OR a zero numerator (no SS-DL-with-cell-control point fired,
                          a 0/x ratio). Neither forms a finite log; §7's unit is the per-session log RR, so a
                          session that cannot form one is not assessable for H2, mirroring the >= MIN_UNITS
                          inclusion rule. (The name's "control never fires" is the denominator case; the value
                          records that a zero numerator drops the session too.)
  * `H2_RR_AGGREGATE`   -- whether the RR >= 1.25 support check uses exp(mean session log RR) or a pooled ratio.
                          "exp_mean_log": exp of the mean per-session log RR -- the same per-session quantity the
                          sign-flip tests, so the point estimate and the test agree (the pooled fires/controls
                          ratio is reported beside it).
  * `NATIVE_TAG`        -- the fixed output tag of the native P1/P2 eval RTTM directories, whose runners do not
                          exist yet. "P1_native" / "P2_native" under <hyp_root>/eval/<tag>/ (layout analogous to
                          `run_m2b.rttm_paths`, one <rid>.exclusive.rttm per recording, no threshold sweep -- a
                          native back end has its own fixed VBx operating point). FLAGGED in the output as an
                          assumption about files that do not yet exist.

All frozen numbers (the collar, the +2pp/+25% H5 margin, the 25% H4 SESOI, the three Holm families, the BCa
levels and resample counts, the >= MIN_UNITS / >= 5-session inclusion rules, the RR >= 1.25 / delta-FAR >= 1pp
H2 thresholds, the seed) come straight from §7.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from csdiar.analysis import score_m2b
from csdiar.analysis.confirm_h4 import secondary_vs
from csdiar.analysis.stats import bca_ci, holm, outcome, signflip

REPO = Path(__file__).resolve().parents[2]
RECORD = REPO / "planning" / "FREEZE_M2B.json"
EVAL_SPLIT = "eval"

# ----------------------------------------------------------------------------------------------------- #
# frozen constants (PREREGISTRATION_M2B.md §7, §11)
# ----------------------------------------------------------------------------------------------------- #
SEED = 20261018
ALPHA = 0.05
MIN_UNITS = score_m2b.MIN_UNITS                     # >= 10 units per session (§7; same as score_m2b)
MIN_SESSIONS = 5                                    # a test needs >= 5 included sessions (§7, decision 21)
C = score_m2b.C                                     # primary collar 0.5 s
THRESHOLD_GRID = tuple(round(0.30 + 0.05 * k, 2) for k in range(15))   # 0.30 .. 1.00 step 0.05 (§3, §5)

H4_FAMILY = ("M1", "M2", "M3")
M_H4 = 3
M_H5 = 3
M_H6 = 3
M_H2 = 5                                             # the M2a family completed by native P1/P2 (§7)
SESOI_REL_H4 = 0.25                                 # >= 25% relative reduction in delta-FAR (§7)
H5_MARGIN_PP = 0.02                                 # +2 pp (§7)
H5_MARGIN_REL = 0.25                                # +25% relative (§7)
H4_CI_LEVEL = 1 - ALPHA / M_H4                      # 98.3% two-sided BCa (§7)
H4_TOST_LEVEL = 1 - 2 * ALPHA / M_H4                # 96.7% TOST (§7)
H5_ONE_SIDED_LEVEL = 0.95                           # one-sided 95% upper bound (§7)
BCA_N_BOOT = 10000                                  # 10,000 whole-session resamples (§7)
H2_RR_MIN = 1.25                                    # RR >= 1.25 (§7)
H2_DFAR_MIN_PP = 0.01                               # delta-FAR >= 1 pp (§7)
FAR_PRIMARY = 0.10                                  # DET primary operating point FAR_SS-DL = 10% (§5)

# ----------------------------------------------------------------------------------------------------- #
# open choices, fixed as named constants (see the module docstring and the final report)
# ----------------------------------------------------------------------------------------------------- #
H5_HOLM = "p_from_bound"
# brief fix 4: the two-sided interval reported beside H5's one-sided bound is at 98.3% (1 - 0.05/3, §7 "BCa
# levels"), the same level as H4's two-sided BCa -- NOT a plain 95% (that was the review M4 defect).
H5_TWO_SIDED_LEVEL = H4_CI_LEVEL                     # 98.3% (1 - ALPHA/M_H5), §7 "BCa levels"
# H5_DECISION (open choice, settled for this round, kept a NAMED choice for the design round, brief fix 4):
# how the per-mitigation non-inferiority verdict is reached. "holm_p_from_inverted_bound": from ONE set of
# BCA_N_BOOT whole-session resamples drawn exactly as stats.bca_ci draws them from the module generator in a
# fixed order, (a) the one-sided 95% BCa upper bound is reported and is bit-for-bit stats.bca_ci(x, mean_stat,
# 0.90, rng)["hi"] at that generator state, and (b) the non-inferiority p-value is that bound inverted at the
# margin (the smallest one-sided alpha at which the 1 - alpha BCa upper bound is below the margin). Holm over
# M1-M3 (a not-assessable member enters p = 1); NON-INFERIOR IFF the Holm-adjusted p < .05, and the
# intersection-union uses this decision.
H5_DECISION = "holm_p_from_inverted_bound"
H6_PVALUE = "directional_sign"
H6_ORDERING = "all_pairs"
H6_DET_POOLING = "by_session"
H2_CONTROL_NEVER_FIRES = "either_sum_zero_drops_session"
H2_RR_AGGREGATE = "exp_mean_log"
NATIVE_TAG = {"P1": "P1_native", "P2": "P2_native"}

# ----------------------------------------------------------------------------------------------------- #
# guarded code + frozen-artifact lists (PREREGISTRATION_M2B.md §11)
# ----------------------------------------------------------------------------------------------------- #
# Every file whose edit must invalidate the freeze. "the mitigation modules and the M3 code" (§11) and the real
# paths of each are listed; any the freeze author is unsure of is flagged in the final report, not silently
# dropped. These all exist in the tree at the M2b branch (verified against the repo listing).
GUARDED = (
    "csdiar/analysis/confirm_m2b.py",
    "csdiar/analysis/freeze_m2b.py",
    "csdiar/analysis/tune_m2b.py",
    "csdiar/analysis/tune_p0_m2b.py",
    "csdiar/analysis/score_m2b.py",
    "csdiar/analysis/stats.py",
    "csdiar/analysis/m2_lambda.py",
    "csdiar/analysis/confirm_h4.py",
    "csdiar/analysis/pilot.py",
    "csdiar/data/ingest_displace.py",
    "csdiar/pipelines/p1_ahc.py",
    "csdiar/pipelines/m2b_hooks.py",
    "csdiar/pipelines/run_m2b.py",
    "csdiar/mitigations/m1_subspace.py",
    "csdiar/mitigations/m3_train.py",
    "csdiar/mitigations/m3_embed.py",
    "csdiar/mitigations/m3_data.py",
    "csdiar/mitigations/folds.py",
    "csdiar/models/e3.py",
    "csdiar/models/embed.py",
    "third_party/unispeech/ecapa_tdnn.py",
    "planning/PREREGISTRATION_M2B.md",
)
# The guarded files whose code hash a tuning record (tune_m2b / tune_p0_m2b) also carries, checked by add-tuning.
TUNING_GUARDED = ("csdiar/analysis/tune_m2b.py", "csdiar/analysis/tune_p0_m2b.py")

# Every path `check_freeze` pins to its frozen value (keys into FREEZE_M2B.json["paths"]).
PATH_ARGS = ("root", "cache_dir", "vad_dir", "wav_dir", "units_dev", "units_eval", "out",
             "label_map", "confirm_m2a")

# M3 all-data checkpoints (5 seeds; §4 "five all-data fine-tuned models"). The run-folder name is exactly what
# m3_train names it (m3_train.py: "{'all' if fold is None else f'f{fold}'}_s{seed}_w{w_dist:g}_{ablation}"),
# i.e. `all_s<seed>_w0.3_none`, and run_m3_queue.sh writes every checkpoint to `~/data/m3/runs/<name>/best.pt`.
# On DISPLACE each all-data model is the E3-replacement embedding inside P1-AHC, run by `run_m2b` with
# `--model m3:<run>/best.pt`; its filesystem tag is `model_tag` = `m3-<run folder>` (run_m2b.model_tag /
# m3_ckpt_name), so its stage-1 cache lands at `<cache_dir>/stage1/m3-<name>/<rid>.npz`.
M3_SEEDS = (0, 1, 2, 3, 4)
M3_PRIMARY = "w0.3_none"
M3_NAMES = tuple(f"all_s{s}_{M3_PRIMARY}" for s in M3_SEEDS)
# `M3_GRIDS` are the D1 (H4) dense-grid stems m3_embed writes (`m3_<name>_L1` under ~/data/m3/emb), hashed by
# `freeze_h4 add-m3` for the D1 P0 path. They are NOT a DISPLACE confirm_m2b input (confirm_m2b never reads a
# dense grid -- see m3_artifacts), and are kept here only so the naming test can pin the `m3_<name>` stem.
M3_GRIDS = tuple(f"m3_{name}" for name in M3_NAMES)

# The M3 model tags `run_m2b` uses on DISPLACE (`m3-<run folder>`), the folder each seed's stage-1 cache lands
# in. Kept beside M3_NAMES so a reader sees the stage-1 layout m3_artifacts hashes.
M3_MODEL_TAGS = tuple(f"m3-{name}" for name in M3_NAMES)


def m3_artifacts(rec: dict | None = None) -> list[str]:
    """The M3-dependent DISPLACE artifacts the freeze must pin before `confirm_m2b` runs, in the layout their
    producers write (§11 "M3's all-data checkpoints and grids are hashed and appended before confirm_m2b runs").

    What `confirm_m2b` actually depends on for M3, and where each is pinned:
      * the five all-data CHECKPOINTS `~/data/m3/runs/<name>/best.pt` (m3_train / run_m3_queue) -- the model
        identity; nothing else in the M2b freeze hashes them, so they are listed here;
      * each seed's DISPLACE STAGE-1 cache `<cache_dir>/stage1/m3-<name>/<rid>.npz` (run_m2b.stage1_cache_path,
        model tag `m3-<name>`) for every EVAL recording -- the per-seed embeddings `run_m2b` clustered to
        produce the frozen M3 eval hypotheses. `freeze_m2b init` hashes only the `e3` stage-1 caches, so the M3
        ones are pinned here (the DISPLACE analogue of the `e3` feature caches `init` pins for the other
        systems). Eval recording IDs come from `rec["split"]["eval"]` -- the SAME source `freeze_m2b` uses for
        them (`eval_hypothesis_files`, the split taken from the reference file names) -- and the cache root from
        the frozen `cache_dir` path.

    Deliberately NOT listed (so nothing is hashed twice or against a non-existent layout):
      * the M3 EVAL HYPOTHESIS RTTMs (`<hyp_root>/eval/M3_<name>/th<tt>/<rid>.exclusive.rttm`): these ARE what
        confirm_m2b reads for M3, but `freeze_m2b add-eval-hyps` already hashes every one of them into
        `eval_hyps_sha256` over the whole clustering grid -- listing them here would double-hash;
      * the dense D1 grids `~/data/m3/emb/m3_<name>_L1/{validation,train,test}.npz`: these are read by the D1 P0
        path (`confirm_h4` / `pilot.p0_curves`) and hashed by `freeze_h4 add-m3`; `confirm_m2b` never opens a
        dense grid on the DISPLACE path. (The previous `~/data/m3/emb_displace/..._L1/{dev,eval}.npz` listing
        was a layout NOTHING writes -- neither m3_embed nor run_m3_queue produces an `emb_displace` tree or
        `dev`/`eval` grid splits -- so `add-m3` would have raised on a missing file; review m3-naming MAJOR.)

    Called with `rec` (the freeze record) by `check_freeze` and `freeze_m2b add-m3`; called with no record only
    by the naming test, where it returns just the checkpoint paths (the record-free part)."""
    out: list[str] = [f"~/data/m3/runs/{name}/best.pt" for name in M3_NAMES]
    if rec is None:
        return out
    from csdiar.pipelines import run_m2b as R
    cache_dir = rec["paths"]["cache_dir"]
    eval_ids = rec["split"][EVAL_SPLIT]
    for name in M3_NAMES:
        model = f"m3:~/data/m3/runs/{name}/best.pt"      # model_tag -> m3-<name>
        for rid in eval_ids:
            out.append(str(R.stage1_cache_path(cache_dir, model, rid)))
    return out


# ----------------------------------------------------------------------------------------------------- #
# hashing / markers
# ----------------------------------------------------------------------------------------------------- #
def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def confirm_markers_exist(out_dir: Path) -> tuple[bool, bool]:
    """(started, done): whether `confirm_m2b.started` and `confirm_m2b.json` exist in `out_dir` (§11 one-shot)."""
    out_dir = Path(out_dir)
    return (out_dir / "confirm_m2b.started").exists(), (out_dir / "confirm_m2b.json").exists()


# ----------------------------------------------------------------------------------------------------- #
# frozen eval hypothesis files (names only; the freeze hashes them, confirm reads them)
# ----------------------------------------------------------------------------------------------------- #
def _frozen_systems(rec: dict) -> dict:
    """The P1-AHC systems and their frozen operating points from the tuning section of the record. Each maps a
    system name -> {"threshold", "delta"?, "alpha"?, "tau0"?, "m3_ckpt"?}. A system recorded not assessable in
    tuning is omitted (it has no frozen hypothesis)."""
    tuning = rec.get("tuning", {})
    thr = tuning.get("thresholds", {})
    na = tuning.get("not_assessable", {})
    out: dict = {}
    for name, th in thr.items():
        if name in na or th is None:
            continue
        spec = {"threshold": th}
        if name == "B2":
            spec["delta"] = tuning.get("b2_delta")
        if name == "M2":
            spec["alpha"] = tuning.get("m2_post_alpha")
            spec["tau0"] = tuning.get("m2_post_tau0")
            spec["threshold"] = tuning.get("m2_post_threshold", th)
        if name.startswith("M3:"):
            spec["m3_ckpt"] = name.split(":", 1)[1]
        out[name] = spec
    return out


def eval_hypothesis_files(hyp_root, rec: dict) -> list[Path]:
    """Every expected eval hypothesis RTTM over the WHOLE clustering grid (the DET needs the sweep) for every
    P1-AHC system at its frozen parameters, plus the single operating-point file (same path at the frozen
    threshold). Layout mirrors `run_m2b.rttm_paths`: <hyp_root>/eval/<system_tag>/th<tt>/<rid>.exclusive.rttm.
    The grid sweep uses `system_tag` with the frozen delta/alpha/tau0/ckpt so distinct operating points never
    collide. Returns the EXCLUSIVE RTTM paths (FAR/MR are scored on the exclusive RTTM, §4)."""
    from csdiar.pipelines import run_m2b as R
    hyp_root = Path(hyp_root)
    rids = rec["split"][EVAL_SPLIT]
    systems = _frozen_systems(rec)
    files: list[Path] = []
    for name, spec in systems.items():
        base_system = "M3" if name.startswith("M3:") else name
        tag = R.system_tag(base_system, delta=spec.get("delta", 0.0) or 0.0,
                           alpha=spec.get("alpha", 0.0) or 0.0, tau0=spec.get("tau0", 0.0) or 0.0,
                           m3_ckpt=spec.get("m3_ckpt"))
        for th in THRESHOLD_GRID:
            base = hyp_root / EVAL_SPLIT / tag / R.threshold_tag(th)
            for rid in rids:
                files.append(base / f"{rid}.exclusive.rttm")
    return files


def expected_native_dirs(hyp_root, rec: dict) -> list[Path]:
    """The native P1 and P2 eval RTTM directories (their runners do not exist yet; NATIVE_TAG, flagged). One
    directory per native back end holding <rid>.exclusive.rttm per recording, no threshold sweep."""
    hyp_root = Path(hyp_root)
    return [hyp_root / EVAL_SPLIT / NATIVE_TAG[p] for p in ("P1", "P2")]


def expected_native_files(hyp_root, rec: dict) -> dict[str, list[Path]]:
    """M2: every per-recording native eval hypothesis <native dir>/<rid>.exclusive.rttm, keyed by 'P1'/'P2'.
    `add-eval-hyps` hashes these and `check_freeze` verifies them (unless the freeze was taken --native-absent).
    `confirm_m2b`'s native read uses the SAME paths, so what is scored is exactly what was hashed."""
    hyp_root = Path(hyp_root)
    rids = rec["split"][EVAL_SPLIT]
    return {p: [hyp_root / EVAL_SPLIT / NATIVE_TAG[p] / f"{rid}.exclusive.rttm" for rid in rids]
            for p in ("P1", "P2")}


# The real run restricts every hypothesis read to the exact files hashed in the freeze record (brief M1): a
# file whose resolved path is not in this set is refused rather than silently scored. None (the default, and
# --dry-run) means no restriction. `main` sets it from the record before the real run.
_FROZEN_HYP_PATHS: set | None = None


def _check_frozen_read(path: Path) -> None:
    """Refuse a hypothesis read whose resolved path is outside the frozen allow-set (brief M1). No-op when the
    set is None (dry-run / unit tests that patch the read functions)."""
    if _FROZEN_HYP_PATHS is None:
        return
    if str(Path(path).resolve()) not in _FROZEN_HYP_PATHS:
        raise SystemExit(f"refusing: hypothesis file {path} is not among the frozen eval hypotheses "
                         "(confirm_m2b reads only the files hashed in FREEZE_M2B.json)")


# ----------------------------------------------------------------------------------------------------- #
# freeze check (PREREGISTRATION_M2B.md §11)
# ----------------------------------------------------------------------------------------------------- #
def check_freeze(args) -> dict:
    """Refuse unless the record matches every input, the power verdict and tuning values are present, and the
    one-shot markers are absent. Mirrors `confirm_h4.check_freeze`, extended to M2b's sections."""
    if not RECORD.exists():
        raise SystemExit(f"refusing: {RECORD} does not exist")
    rec = json.loads(RECORD.read_text())
    # code
    for f in GUARDED:
        if f not in rec["code_sha256"]:
            raise SystemExit(f"refusing: {f} is not covered by the freeze record")
    for f, h in rec["code_sha256"].items():
        if sha256(REPO / f) != h:
            raise SystemExit(f"refusing: {f} changed since the freeze")
    # features + references + M3 artifacts (bytes; never parsed)
    if "m3_artifacts_sha256" not in rec:
        raise SystemExit("refusing: the record does not hash the M3 artifacts (freeze_m2b add-m3)")
    if sorted(rec["m3_artifacts_sha256"]) != sorted(m3_artifacts(rec)):
        raise SystemExit("refusing: the record does not hash every expected M3 artifact (freeze_m2b add-m3)")
    hashed = {**rec["feature_sha256"], **rec["reference_sha256"], **rec["m3_artifacts_sha256"]}
    for f, h in hashed.items():
        if sha256(Path(f).expanduser()) != h:
            raise SystemExit(f"refusing: {f} changed since the freeze")
    if rec.get("confirm_m2a_sha256") != sha256(Path(rec["paths"]["confirm_m2a"]).expanduser()):
        raise SystemExit("refusing: the M2a result changed since the freeze")
    if rec.get("label_map_sha256") != sha256(Path(rec["paths"]["label_map"]).expanduser()):
        raise SystemExit("refusing: the label map changed since the freeze")
    # eval hypotheses. review-v2 m1: before hashing, refuse any hash key whose path is not under the frozen
    # hyp_root subtree -- so a hand-edited record that moved a hash key out of the pinned root is caught here
    # directly, not only transitively via the --hyp-root equality check below.
    if "hyp_root" not in rec.get("paths", {}):
        raise SystemExit("refusing: the record does not pin the hypothesis root (freeze_m2b add-eval-hyps)")
    frozen_hr = Path(rec["paths"]["hyp_root"]).expanduser().resolve()

    def _under_frozen_root(f) -> bool:
        return Path(f).expanduser().resolve() == frozen_hr or \
            frozen_hr in Path(f).expanduser().resolve().parents

    if "eval_hyps_sha256" not in rec:
        raise SystemExit("refusing: the record does not hash the eval hypotheses (freeze_m2b add-eval-hyps)")
    for f, h in rec["eval_hyps_sha256"].items():
        if not _under_frozen_root(f):
            raise SystemExit(f"refusing: eval hypothesis {f} is not under the frozen hypothesis root "
                             f"{frozen_hr} (record tampered)")
        if not Path(f).exists():
            raise SystemExit(f"refusing: frozen eval hypothesis {f} is missing")
        if sha256(Path(f)) != h:
            raise SystemExit(f"refusing: eval hypothesis {f} changed since the freeze")
    # M2: the native P1/P2 eval hypotheses are content-hashed too, unless they were frozen absent. A record
    # written before this fix (no native_hyps_sha256 and no native_absent flag) is refused: re-run add-eval-hyps.
    if not rec.get("native_absent", False):
        if "native_hyps_sha256" not in rec:
            raise SystemExit("refusing: the record does not hash the native P1/P2 eval hypotheses "
                             "(freeze_m2b add-eval-hyps); re-run add-eval-hyps, or use --native-absent")
        for f, h in rec["native_hyps_sha256"].items():
            if not _under_frozen_root(f):
                raise SystemExit(f"refusing: native eval hypothesis {f} is not under the frozen hypothesis "
                                 f"root {frozen_hr} (record tampered)")
            if not Path(f).exists():
                raise SystemExit(f"refusing: frozen native eval hypothesis {f} is missing")
            if sha256(Path(f)) != h:
                raise SystemExit(f"refusing: native eval hypothesis {f} changed since the freeze")
    # M1: the hypothesis root the confirm run reads from is pinned to the frozen one, and confirm reads only the
    # exact files hashed above (built from the record), never re-globbing an arbitrary root (see run_analysis).
    # `frozen_hr` was resolved above (where the hash keys are checked to live under it).
    given_hr = Path(getattr(args, "hyp_root")).expanduser().resolve()
    if given_hr != frozen_hr:
        raise SystemExit(f"refusing: --hyp-root {given_hr} is not the frozen hypothesis root {frozen_hr}")
    # input paths, seed, power verdict, tuning values
    for k in PATH_ARGS:
        given = Path(getattr(args, k)).expanduser().resolve()
        frozen = Path(rec["paths"][k]).expanduser().resolve()
        if given != frozen:
            raise SystemExit(f"refusing: --{k.replace('_', '-')} {given} is not the frozen input {frozen}")
    if rec.get("seed") != SEED:
        raise SystemExit("refusing: the freeze record's seed differs from the code's")
    if "power" not in rec or rec["power"].get("verdict") not in ("confirmatory", "descriptive"):
        raise SystemExit("refusing: the record has no H5 power verdict (freeze_m2b add-power)")
    if "tuning" not in rec or not rec["tuning"].get("thresholds"):
        raise SystemExit("refusing: the record has no frozen tuning operating points (freeze_m2b add-tuning)")
    out_dir = Path(rec["paths"]["out"]).expanduser()
    started, done = confirm_markers_exist(out_dir)
    if started or done:
        raise SystemExit("refusing: confirm_m2b has already started or finished (one-shot)")
    return rec


# ----------------------------------------------------------------------------------------------------- #
# per-session statistics helpers
# ----------------------------------------------------------------------------------------------------- #
def mean_stat(xs) -> float:
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else float("nan")


def _per_session_delta(metric_m: dict[str, float], metric_b0: dict[str, float],
                       sessions) -> dict[str, float]:
    """delta_M(s) = metric_M(s) - metric_B0(s) over the sessions present in BOTH (§7 'B0 subtracted per
    session')."""
    return {s: metric_m[s] - metric_b0[s] for s in sessions
            if s in metric_m and s in metric_b0
            and metric_m[s] == metric_m[s] and metric_b0[s] == metric_b0[s]}


def _ni_pvalue(deltas: list[float], margin: float, rng, n_boot: int = BCA_N_BOOT) -> float:
    """A non-inferiority p-value for H0: mean(delta) >= margin vs H1: mean(delta) < margin, by inverting the BCa
    bound on the SAME whole-session resamples `bca_ci` uses (H5_DECISION='holm_p_from_inverted_bound'). It is the
    smallest one-sided alpha at which the 1 - alpha BCa upper bound still lies at or below `margin`: we find the
    largest one-sided level whose upper bound <= margin by a monotone search over the bootstrap distribution's
    BCa-adjusted quantiles, which is exactly `bca_ci`'s construction. The caller REWINDS the module generator to
    the state `bca_ci` consumed for the reported bound before calling this, so `boots` here is the identical
    index draw and the inverted bound matches the reported one bit-for-bit on the live path (not only for a hand-
    shared seed). Returns 1.0 when even the median BCa upper bound exceeds the margin. Pure given `rng`."""
    k = len(deltas)
    if k < 2:
        return float("nan")
    clusters = [[v] for v in deltas]
    stat = lambda cs: float(np.mean([c[0] for c in cs]))
    theta = stat(clusters)
    # Reproduce bca_ci's resampling to get the bootstrap distribution ONCE, then read its upper bound at any
    # level without re-bootstrapping (the test asserts this matches bca_ci bit-for-bit for a shared seed).
    boots = np.array([stat([clusters[i] for i in rng.integers(0, k, k)]) for _ in range(n_boot)],
                     dtype=np.float64)
    boots = boots[np.isfinite(boots)]
    if len(boots) == 0 or not np.isfinite(theta):
        return float("nan")
    from statistics import NormalDist
    _N = NormalDist()
    frac = ((boots < theta).sum() + 0.5 * (boots == theta).sum()) / len(boots)
    frac = min(max(frac, 1 / (len(boots) + 1)), len(boots) / (len(boots) + 1))
    z0 = _N.inv_cdf(frac)
    jack = np.array([stat(clusters[:i] + clusters[i + 1:]) for i in range(k)], dtype=np.float64)
    d = jack.mean() - jack
    denom = 6 * (d ** 2).sum() ** 1.5
    a = float((d ** 3).sum() / denom) if denom > 0 else 0.0

    def upper_bound(level: float) -> float:
        # one-sided upper (1-alpha) BCa bound, alpha = 1 - level: the BCa-adjusted (level) quantile.
        z = _N.inv_cdf(level)
        q = _N.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        return float(np.quantile(boots, q))

    # p = 1 - (largest one-sided level whose upper bound <= margin). Bisect on level in (0.5, 1).
    if upper_bound(0.5) > margin:          # even the median bound exceeds the margin: not non-inferior at all
        return 1.0
    lo, hi = 0.5, 1.0 - 1e-6
    for _ in range(60):
        mid = (lo + hi) / 2
        if upper_bound(mid) <= margin:
            lo = mid
        else:
            hi = mid
    return float(max(0.0, min(1.0, 1.0 - lo)))


# ----------------------------------------------------------------------------------------------------- #
# H4 family (session-clustered; the confirm_h4.test_family analogue, §7)
# ----------------------------------------------------------------------------------------------------- #
def _h4_family(deltas_by_m: dict[str, dict[str, float]], s_b0: float, rng,
               not_assessable: set[str]) -> dict:
    """H4 over M1-M3 (m = 3), clustered by SESSION. `deltas_by_m` maps a mitigation -> {session: delta-FAR vs
    B0}. Exact one-sided sign-flip over sessions on the predicted NEGATIVE delta, Holm over M1-M3, BCa 98.3% /
    TOST 96.7%, SESOI = 25% of S_B0 (held fixed), outcome via `stats.outcome`. A member listed not assessable
    enters Holm with p = 1 and is reported so, as `confirm_h4.test_family` does for a staged member."""
    res: dict = {}
    for n in H4_FAMILY:
        if n in not_assessable:
            res[n] = {"not_assessable": True}
            continue
        x = [deltas_by_m[n][s] for s in sorted(deltas_by_m[n])] if n in deltas_by_m else []
        if len(x) < MIN_SESSIONS:
            res[n] = {"not_assessable": True, "reason": f"only {len(x)} included sessions (< {MIN_SESSIONS})"}
            continue
        sf = signflip([-v for v in x], rng)            # predicted negative (a reduction in the SS-DL excess)
        res[n] = {"n_sessions": len(x), "mean_delta": float(np.mean(x)),
                  "relative_reduction": float(-np.mean(x) / s_b0) if s_b0 > 0 else None,
                  "signflip": sf,
                  "ci": bca_ci(x, mean_stat, H4_CI_LEVEL, rng),
                  "tost": bca_ci(x, mean_stat, H4_TOST_LEVEL, rng),
                  "per_session": deltas_by_m[n]}
    p1 = [res[n]["signflip"]["p_one"] if "signflip" in res[n] else 1.0 for n in H4_FAMILY]
    p2 = [res[n]["signflip"]["p_two"] if "signflip" in res[n] else 1.0 for n in H4_FAMILY]
    a1, a2 = holm(p1), holm(p2)
    for i, n in enumerate(H4_FAMILY):
        r = res[n]
        if "signflip" not in r:
            continue
        r["p_one_holm"], r["p_two_holm"] = a1[i], a2[i]
        if s_b0 > 0:
            r["outcome"] = outcome(a1[i], a2[i], r["mean_delta"], r["tost"]["lo"], r["tost"]["hi"],
                                   SESOI_REL_H4 * s_b0, direction=-1)
        else:
            r["outcome"] = {"category": "inconclusive", "reason": "S_B0 <= 0: no excess to reduce"}
    return res


# ----------------------------------------------------------------------------------------------------- #
# H5 family (non-inferiority on MR_DS-DL; §7)
# ----------------------------------------------------------------------------------------------------- #
def _h5_family(dmr_by_m: dict[str, dict[str, float]], s_mr: float, rng, not_assessable: set[str],
               descriptive: bool) -> dict:
    """H5 over M1-M3, clustered by SESSION. `dmr_by_m` maps a mitigation -> {session: delta MR_DS-DL vs B0}.
    margin = min(+2pp, +25% * S_MR), S_MR held fixed. For each member, from ONE set of BCA_N_BOOT whole-session
    resamples drawn exactly as `stats.bca_ci` draws them from the module generator in a fixed order
    (H5_DECISION='holm_p_from_inverted_bound', brief fix 4): (a) the one-sided 95% BCa upper bound, reported and
    equal bit-for-bit to `bca_ci(x, mean_stat, 2*H5_ONE_SIDED_LEVEL-1, rng)["hi"]` at the same generator state;
    (b) the non-inferiority p by inverting that bound at the margin. The two-sided interval beside it is at
    H5_TWO_SIDED_LEVEL (98.3%). Holm over M1-M3 (a not-assessable member enters p = 1). **Non-inferior iff the
    Holm-adjusted p < .05**; the intersection-union uses this decision. When the power verdict is descriptive,
    the claim is withheld (`claim_withheld`) and `non_inferior` is None."""
    margin = min(H5_MARGIN_PP, H5_MARGIN_REL * s_mr) if s_mr == s_mr else float("nan")
    res: dict = {"margin": margin, "s_mr": s_mr, "descriptive": descriptive,
                 "decision": H5_DECISION, "two_sided_level": H5_TWO_SIDED_LEVEL}
    members: dict = {}
    for n in H4_FAMILY:
        if n in not_assessable:
            members[n] = {"not_assessable": True}
            continue
        x = [dmr_by_m[n][s] for s in sorted(dmr_by_m[n])] if n in dmr_by_m else []
        if len(x) < MIN_SESSIONS:
            members[n] = {"not_assessable": True,
                          "reason": f"only {len(x)} included sessions (< {MIN_SESSIONS})"}
            continue
        # The bound (a) and the inverted-bound p (b) must come from the SAME resamples. bca_ci advances the
        # module rng; capture its state, read the bound, then REWIND to that same state so _ni_pvalue draws the
        # identical BCA_N_BOOT index sets in the same order -- so the p inverts exactly the bound reported.
        state = rng.bit_generator.state
        one = bca_ci(x, mean_stat, 2 * H5_ONE_SIDED_LEVEL - 1.0, rng)   # upper endpoint = one-sided 95% bound
        bound = one["hi"]
        if margin == margin:
            rng.bit_generator.state = state
            p_ni = _ni_pvalue(x, margin, rng)                          # inverts `bound` on the same resamples
        else:
            p_ni = float("nan")
        two = bca_ci(x, mean_stat, H5_TWO_SIDED_LEVEL, rng)            # 98.3% two-sided, reported beside it
        members[n] = {"n_sessions": len(x), "mean_delta_mr": float(np.mean(x)),
                      "one_sided_95_upper": bound, "two_sided_ci": {"lo": two["lo"], "hi": two["hi"],
                                                                    "level": H5_TWO_SIDED_LEVEL},
                      "bound_below_margin": bool(bound == bound and margin == margin and bound < margin),
                      "ni_pvalue": p_ni, "claim_withheld": descriptive,
                      "per_session": dmr_by_m[n]}
    # Holm over the non-inferiority p-values (members without one enter as p = 1). The CONFIRMATORY decision is
    # the Holm-adjusted p < .05 (brief fix 4), not the raw bound < margin; the raw bound and whether it lies
    # below the margin are reported beside it so Holm never hides the primary bound.
    pv = [members[n]["ni_pvalue"] if "ni_pvalue" in members[n]
          and members[n]["ni_pvalue"] == members[n]["ni_pvalue"] else 1.0 for n in H4_FAMILY]
    adj = holm(pv)
    for i, n in enumerate(H4_FAMILY):
        if "ni_pvalue" in members[n]:
            members[n]["ni_pvalue_holm"] = adj[i]
            members[n]["non_inferior"] = None if descriptive else bool(adj[i] < ALPHA)
    res["members"] = members
    return res


# ----------------------------------------------------------------------------------------------------- #
# intersection-union (H4 x H5 per mitigation; §7)
# ----------------------------------------------------------------------------------------------------- #
def _intersection_union(h4: dict, h5: dict, descriptive: bool) -> dict:
    """Per mitigation: net improvement iff H4 supported AND H5 non-inferior. §7 names only the two one-sided
    cases -- "reduces false splits at the cost of missed turns" (H4 ok, H5 fails) and "no reduction shown"
    (H5 ok, H4 fails). Review m4 / brief fix 8: the FOURTH cell (fails both) gets its OWN label,
    "no reduction shown and misses raised", so the report distinguishes it from the H5-ok/H4-fail case rather
    than collapsing the two into one "no reduction shown" branch (the former dead branch). When H5 is
    descriptive the net-improvement claim is withheld."""
    out: dict = {}
    for n in H4_FAMILY:
        r4, r5 = h4.get(n, {}), h5.get("members", {}).get(n, {})
        if r4.get("not_assessable") or r5.get("not_assessable"):
            out[n] = {"label": "not assessable",
                      "h4_not_assessable": bool(r4.get("not_assessable")),
                      "h5_not_assessable": bool(r5.get("not_assessable"))}
            continue
        h4_supported = r4.get("outcome", {}).get("category") == "supported"
        h5_non_inferior = bool(r5.get("non_inferior"))
        if descriptive:
            out[n] = {"label": "descriptive (power verdict)", "h4_supported": h4_supported,
                      "h5_non_inferior": None, "net_improvement": None, "claim_withheld": True}
            continue
        if h4_supported and h5_non_inferior:
            label = "net improvement"
        elif h4_supported and not h5_non_inferior:
            label = "reduces false splits at the cost of missed turns"
        elif h5_non_inferior and not h4_supported:
            label = "no reduction shown"
        else:                                           # fails BOTH H4 and H5 (brief fix 8: its own label)
            label = "no reduction shown and misses raised"
        out[n] = {"label": label, "h4_supported": h4_supported, "h5_non_inferior": h5_non_inferior,
                  "net_improvement": bool(h4_supported and h5_non_inferior)}
    return out


# ----------------------------------------------------------------------------------------------------- #
# H6 (trade-off ordering; §7 secondary)
# ----------------------------------------------------------------------------------------------------- #
# Predicted pairwise directions of MR_DS-DL at FAR_SS-DL = 10% (lower MR = better trade-off): M3 best, M1 the
# smallest gain. So predicted MR(M3) < MR(M2) < MR(M1): the three ordered pairs and the sign we expect for
# MR_a - MR_b (predict > 0 means a has the HIGHER miss rate, i.e. the worse trade-off).
H6_PAIRS = (("M1", "M2"), ("M1", "M3"), ("M2", "M3"))   # predict MR_a - MR_b > 0 for each


def _h6(mr_at_far_by_m: dict[str, dict[str, float]], m3_seed_mr: dict[str, dict[str, float]], rng) -> dict:
    """H6 pairwise ordering of MR_DS-DL at FAR = 10%, Holm (m = 3), GENUINE two-stage bootstrap over sessions
    then M3's seeds (H6_PVALUE='directional_sign', H6_ORDERING='all_pairs'; brief fix 3).

    `mr_at_far_by_m` maps M1/M2/M3 -> {session: MR-at-FAR reading} (H6_DET_POOLING='by_session'); for M3 the
    value is the seed-mean, used for the OBSERVED statistic. `m3_seed_mr` maps M3 seed -> {session: reading},
    the second bootstrap stage. In each replicate we resample SESSIONS with replacement, and for each drawn
    session each system contributes one reading: M1 and M2 their single per-session reading, M3 the average of a
    WITH-REPLACEMENT resample of that session's seed readings (one reading per seed). The three pairwise
    differences are formed from that replicate and averaged over the drawn sessions.

    The p-value follows `stats.outcome`'s convention (brief fix 3): "significant" uses the ONE-SIDED bootstrap p
    in the predicted direction (predict mean diff > 0, i.e. `a` misses more), "reversed" the TWO-SIDED p. A
    replicate whose averaged diff is not finite (a drawn session's target not reached) is DROPPED and counted in
    `n_not_reached_replicates`; both p-values are shares over the finite replicates."""
    res: dict = {"pairs": {}, "det_pooling": H6_DET_POOLING, "pvalue_rule": H6_PVALUE,
                 "ordering_rule": H6_ORDERING}
    present = {m for m in H4_FAMILY if m in mr_at_far_by_m}
    pvals = []            # one-sided p per pair (feeds Holm; "significant" uses this)
    pair_order = []

    def session_reading(system: str, sess: str, draw_seeds):
        """One system's MR reading for a drawn session in a replicate. M1/M2: the per-session reading. M3: the
        mean of a with-replacement seed resample of that session's per-seed readings (second stage). When M3 has
        no per-seed readings for this session (a single collapsed model, or the degenerate test fixture), the
        seed stage is a no-op and the per-session seed-mean reading is used."""
        if system != "M3" or not m3_seed_mr:
            return mr_at_far_by_m[system].get(sess, float("nan"))
        seed_vals = [m3_seed_mr[sd][sess] for sd in m3_seed_mr
                     if sess in m3_seed_mr[sd] and m3_seed_mr[sd][sess] == m3_seed_mr[sd][sess]]
        if not seed_vals:
            return mr_at_far_by_m["M3"].get(sess, float("nan"))
        js = draw_seeds(len(seed_vals))
        return float(np.mean([seed_vals[j] for j in js]))

    for a, b in H6_PAIRS:
        pair_order.append((a, b))
        key = f"{a}_vs_{b}"
        if a not in present or b not in present:
            res["pairs"][key] = {"not_assessable": True}
            pvals.append(1.0)
            continue
        sessions = sorted(set(mr_at_far_by_m[a]) & set(mr_at_far_by_m[b]))
        diffs = [mr_at_far_by_m[a][s] - mr_at_far_by_m[b][s] for s in sessions
                 if mr_at_far_by_m[a][s] == mr_at_far_by_m[a][s]
                 and mr_at_far_by_m[b][s] == mr_at_far_by_m[b][s]]
        if len(diffs) < MIN_SESSIONS:
            res["pairs"][key] = {"not_assessable": True,
                                 "reason": f"only {len(diffs)} sessions with both readings"}
            pvals.append(1.0)
            continue
        usable = [s for s in sessions
                  if mr_at_far_by_m[a][s] == mr_at_far_by_m[a][s]
                  and mr_at_far_by_m[b][s] == mr_at_far_by_m[b][s]]
        obs = float(np.mean(diffs))
        k = len(usable)
        # two-stage bootstrap: resample sessions, then resample M3's seeds within each drawn session.
        le = 0            # replicates with statistic <= 0 (contradict the prediction a > b)
        ge = 0            # replicates with statistic >= 0
        not_reached = 0
        total = 0
        for _ in range(BCA_N_BOOT):
            sess_idx = rng.integers(0, k, k)
            per = []
            for si in sess_idx:
                s = usable[si]
                va = session_reading(a, s, lambda m: rng.integers(0, m, m))
                vb = session_reading(b, s, lambda m: rng.integers(0, m, m))
                per.append(va - vb)
            rep = float(np.mean(per)) if per else float("nan")
            if not np.isfinite(rep):
                not_reached += 1
                continue
            total += 1
            if rep <= 0:
                le += 1
            if rep >= 0:
                ge += 1
        # one-sided p in the predicted direction (predict > 0): share of replicates at or below 0.
        p_one = (le / total) if total else float("nan")
        # two-sided p (for "reversed"): 2 * the smaller tail share, capped at 1.
        p_two = min(1.0, 2.0 * min(le, ge) / total) if total else float("nan")
        pvals.append(p_one if p_one == p_one else 1.0)
        res["pairs"][key] = {"n_sessions": k, "mean_mr_diff": obs,
                             "predicted_direction": "a_misses_more", "p_one_sided": p_one,
                             "p_two_sided": p_two, "n_not_reached_replicates": not_reached,
                             "n_seeds_m3": len(m3_seed_mr)}
    adj = holm(pvals)
    sig = []
    for i, (a, b) in enumerate(pair_order):
        key = f"{a}_vs_{b}"
        if "p_one_sided" in res["pairs"][key]:
            res["pairs"][key]["p_one_holm"] = adj[i]
            # significant in the PREDICTED direction: one-sided Holm-adjusted p < alpha and observed sign > 0.
            res["pairs"][key]["significant"] = bool(adj[i] < ALPHA
                                                    and res["pairs"][key]["mean_mr_diff"] > 0)
            # reversed: two-sided p < alpha with the WRONG observed sign (reported, not part of ordering).
            res["pairs"][key]["reversed"] = bool(res["pairs"][key]["p_two_sided"] == res["pairs"][key]["p_two_sided"]
                                                 and res["pairs"][key]["p_two_sided"] < ALPHA
                                                 and res["pairs"][key]["mean_mr_diff"] < 0)
            sig.append(res["pairs"][key]["significant"])
        else:
            sig.append(False)
    res["ordering_significant"] = bool(sig and all(sig)) if H6_ORDERING == "all_pairs" else bool(sig and any(sig))
    return res


# ----------------------------------------------------------------------------------------------------- #
# H2 conversation (native P1/P2; §7)
# ----------------------------------------------------------------------------------------------------- #
def _h2_log_rr_per_session(score: dict) -> dict[str, float]:
    """Per-session log RR over the SS-DL points that HAVE a cell control (§7 'ratio of sums of fires over SS-DL
    points and over their cell controls'; brief fix 5). RR = (fires over the SS-DL points that have cell
    controls) / (sum of those points' own cell-control fire rates), the two quantities `score_m2b.score_session`
    accumulates as `ssdl_fires_with_cell_controls` and `cell_ctrl_rate_sum` in the same loop that builds
    `delta_far_std`. This is NOT FAR_SS-DL / FAR_CTRL: the numerator counts only the cell-matched points and the
    denominator is the sum of each matched point's OWN cell control rate, so a point in a never-firing cell
    contributes 0 to both. A session whose numerator or denominator is zero (either side <= RR_ZERO_SUM_EPS) has
    no finite log RR and is dropped (H2_CONTROL_NEVER_FIRES='either_sum_zero_drops_session': a zero numerator,
    a zero denominator, or both). Returns {session: log RR}."""
    out: dict[str, float] = {}
    for sess, s in score["sessions"].items():
        if sess not in score["inclusion"]["ssdl"]["included"]:
            continue
        num, den = s.ssdl_fires_with_cell_controls, s.cell_ctrl_rate_sum
        if not (num == num and den == den):            # nan either side: not assessable for H2
            continue
        if num <= score_m2b.RR_ZERO_SUM_EPS or den <= score_m2b.RR_ZERO_SUM_EPS:
            continue                                    # zero-sum drop: 0/0, x/0 or a 0 numerator
        out[sess] = math.log(num) - math.log(den)
    return out


def _h2_native(native_scores: dict, p0_pvalues: list[float], rng, native_absent: bool = False) -> dict:
    """Conversation H2 for native P1 and P2 (§7). Per-session log RR, sign-flip over sessions (predict log RR >
    0), Holm over the M2a family of m = 5 with P0's observed D1 one-sided p-values (read from the frozen M2a
    result, never typed). Supported iff Holm p < .05 AND RR >= 1.25 with delta-FAR >= 1 pp.

    `native_scores` maps 'P1'/'P2' -> that system's `score_m2b.score_system` output. `p0_pvalues` are the three
    frozen M2a P0 one-sided p-values (e1, e2, e3) that complete the m = 5 family. When `native_absent` (the
    freeze was taken --native-absent, brief M2), the native hypotheses were never frozen, so each pipeline
    member is reported "native hypotheses not frozen" and enters Holm with p = 1 -- the M2a P0 members still get
    their adjusted p from the family of m = 5."""
    members: dict = {}
    raw_p_one = list(p0_pvalues)          # the three P0 D1 p-values enter Holm first (M2a verdicts final)
    order = []
    for sys_name in ("P1", "P2"):
        sc = None if native_absent else native_scores.get(sys_name)
        if native_absent:
            members[sys_name] = {"not_done": True, "reason": "native hypotheses not frozen"}
            raw_p_one.append(1.0)
            order.append(sys_name)
            continue
        if sc is None:
            members[sys_name] = {"not_assessable": True, "reason": "native hypothesis not available"}
            raw_p_one.append(1.0)
            order.append(sys_name)
            continue
        log_rr = _h2_log_rr_per_session(sc)
        n_sessions_in = len(sc["inclusion"]["ssdl"]["included"])
        dropped = n_sessions_in - len(log_rr)
        if len(log_rr) < MIN_SESSIONS:
            members[sys_name] = {"not_assessable": True,
                                 "reason": f"only {len(log_rr)} sessions with a finite log RR "
                                           f"(< {MIN_SESSIONS}); {dropped} dropped (controls/points never fire)"}
            raw_p_one.append(1.0)
            order.append(sys_name)
            continue
        x = [log_rr[s] for s in sorted(log_rr)]
        sf = signflip(x, rng)                          # predict mean log RR > 0
        mean_log = float(np.mean(x))
        rr = math.exp(mean_log)                        # H2_RR_AGGREGATE='exp_mean_log'
        # brief fix 5: the >= 1 pp support check uses the cell-standardised delta-FAR, not far_ssdl - far_ctrl,
        # over the same included sessions that formed a finite log RR.
        dfar = mean_stat([sc["sessions"][s].delta_far_std for s in sorted(log_rr)])
        members[sys_name] = {"n_sessions": len(x), "mean_log_rr": mean_log, "rr": rr,
                             "delta_far": dfar, "signflip": sf, "n_sessions_dropped": dropped}
        raw_p_one.append(sf["p_one"])
        order.append(sys_name)
    adj = holm(raw_p_one)                              # m = 5: [e1, e2, e3, P1, P2]
    labels = ["e1_p0", "e2_p0", "e3_p0"] + order
    adj_map = dict(zip(labels, adj))
    for sys_name in ("P1", "P2"):
        m = members[sys_name]
        if "signflip" not in m:
            continue
        m["p_one_holm"] = adj_map[sys_name]
        m["supported"] = bool(m["p_one_holm"] < ALPHA and m["rr"] >= H2_RR_MIN
                              and m["delta_far"] >= H2_DFAR_MIN_PP)
    return {"members": members, "holm_family_m": M_H2,
            "p0_d1_one_sided_pvalues": {"e1": p0_pvalues[0], "e2": p0_pvalues[1], "e3": p0_pvalues[2]},
            "holm_adjusted": adj_map}


# ----------------------------------------------------------------------------------------------------- #
# scoring every system (ingest, hypotheses, per-session scores)
# ----------------------------------------------------------------------------------------------------- #
def _ingest_split(units_dir: Path) -> tuple[dict, dict]:
    """Load `ingest_displace`'s reader rows (ssdl/turns/ctrl csv.gz) grouped by recording, and the durations
    from summary.json, for scoring. This reads the ALREADY-INGESTED unit tables; the in-process
    `process_recording` ingest of the eval references (the first eval read) happens in `main` AFTER the marker.
    """
    import csv
    import gzip
    rows_by_rec: dict = defaultdict(list)
    for kind in ("ssdl", "turns", "ctrl"):
        for p in units_dir.glob(f"{kind}_*.csv.gz"):
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    rows_by_rec[row["recording"]].append(row)
    summary = json.loads((units_dir / "summary.json").read_text())
    # review m1 / brief fix 6: the duration comes from the summary's duration_s (written from the frozen WAV
    # header by _ingest_eval_in_process for eval, and by ingest_displace for dev) -- a frozen, non-reference
    # source. It is NEVER re-derived from the reference row times here; a missing/zero duration stays nan so
    # changes/min is reported nan rather than silently wrong.
    durations = {}
    for r in summary.get("recordings", []):
        d = float(r.get("duration_s", 0.0) or 0.0)
        durations[r["recording"]] = d if d > 0.0 else float("nan")
    return dict(rows_by_rec), durations


def _changes_from_rttms(hyp_root: Path, split: str, tag: str, threshold, rids) -> dict:
    """Hypothesized change times per recording, read from the frozen exclusive RTTMs at one (system, threshold):
    <hyp_root>/<split>/<tag>/th<tt>/<rid>.exclusive.rttm -> `score_m2b.hypothesis_changes`."""
    from csdiar.pipelines import run_m2b as R
    base = hyp_root / split / tag / R.threshold_tag(threshold)
    out: dict = {}
    for rid in rids:
        p = base / f"{rid}.exclusive.rttm"
        if not p.exists():
            out[rid] = []
            continue
        _check_frozen_read(p)
        segs = score_m2b.parse_hypothesis_rttm(p.read_text(encoding="utf-8"))
        out[rid] = score_m2b.hypothesis_changes(segs)
    return out


def _native_changes(hyp_root: Path, split: str, tag: str, rids) -> dict:
    base = hyp_root / split / tag
    out: dict = {}
    for rid in rids:
        p = base / f"{rid}.exclusive.rttm"
        if not p.exists():
            out[rid] = []
            continue
        _check_frozen_read(p)
        segs = score_m2b.parse_hypothesis_rttm(p.read_text(encoding="utf-8"))
        out[rid] = score_m2b.hypothesis_changes(segs)
    return out


def _score_system_sweep(rows_by_session, durations, hyp_root, split, tag, rids) -> dict:
    """Score one system over the whole threshold grid. Returns {"by_threshold": {th: score_system output},
    "det_points": [(th, far, mr)], "det_by_session": {s: [(th,far,mr)]}}. FAR/MR on the exclusive RTTM (§4)."""
    by_threshold: dict = {}
    det_points: list = []
    det_by_session: dict = defaultdict(list)
    for th in THRESHOLD_GRID:
        changes = _changes_from_rttms(hyp_root, split, tag, th, rids)
        res = score_m2b.score_system(rows_by_session, changes, durations)
        by_threshold[th] = res
        inc_far = set(res["inclusion"]["ssdl"]["included"])
        inc_mr = set(res["inclusion"]["dsdl"]["included"])
        fars = [res["sessions"][s].far_ssdl for s in inc_far]
        mrs = [res["sessions"][s].mr_dsdl for s in inc_mr]
        far = mean_stat(fars)
        mr = mean_stat(mrs)
        if far == far and mr == mr:
            det_points.append((float(th), far, mr))
        for s in inc_far & inc_mr:
            fs, ms = res["sessions"][s].far_ssdl, res["sessions"][s].mr_dsdl
            if fs == fs and ms == ms:
                det_by_session[s].append((float(th), fs, ms))
    return {"by_threshold": by_threshold, "det_points": det_points,
            "det_by_session": dict(det_by_session)}


def _op_score(rows_by_session, durations, hyp_root, split, tag, threshold, rids) -> dict:
    """Score one system at its FROZEN operating-point threshold (nothing tuned on eval, §4). Returns the
    `score_m2b.score_system` output at that single threshold."""
    changes = _changes_from_rttms(hyp_root, split, tag, threshold, rids)
    return score_m2b.score_system(rows_by_session, changes, durations)


def _mr_at_far_by_session(sweep: dict, far_target: float = FAR_PRIMARY) -> dict[str, float]:
    """Per-session MR_DS-DL read at FAR_SS-DL = far_target from each session's own DET sweep (H6_DET_POOLING=
    'by_session'; §5 crossing rule via `score_m2b.read_det`). Only sessions whose sweep reaches the target."""
    out: dict[str, float] = {}
    for s, triples in sweep["det_by_session"].items():
        r = score_m2b.read_det(triples, far_target, along="far")
        if r.value == r.value:
            out[s] = r.value
    return out


# ----------------------------------------------------------------------------------------------------- #
# main
# ----------------------------------------------------------------------------------------------------- #
def _frozen_tag(rec: dict, name: str, spec: dict):
    from csdiar.pipelines import run_m2b as R
    base_system = "M3" if name.startswith("M3:") else name
    return R.system_tag(base_system, delta=spec.get("delta", 0.0) or 0.0,
                        alpha=spec.get("alpha", 0.0) or 0.0, tau0=spec.get("tau0", 0.0) or 0.0,
                        m3_ckpt=spec.get("m3_ckpt"))


def run_analysis(rows_by_session, durations, hyp_root, split, rec, rng, descriptive: bool,
                 p0_pvalues, hyp_root_native=None) -> dict:
    """The whole confirmatory analysis on one split's rows + hypotheses (shared by the real run and --dry-run).

    `rec` carries the frozen operating points (tuning section). For a dry run on dev the same section names the
    thresholds; the dev hypotheses live under the same layout. `descriptive` is the frozen power verdict's
    effect on H5."""
    from csdiar.pipelines import run_m2b as R
    systems = _frozen_systems(rec)
    hyp_root = Path(hyp_root)
    hyp_root_native = Path(hyp_root_native) if hyp_root_native is not None else hyp_root

    # ---- score every P1-AHC system: sweep (DET) + operating point ----
    sweeps: dict = {}
    op_scores: dict = {}
    for name, spec in systems.items():
        tag = _frozen_tag(rec, name, spec)
        sweeps[name] = _score_system_sweep(rows_by_session, durations, hyp_root, split, tag,
                                           rec["split"][split] if split in rec["split"] else sorted(rows_by_session))
        op_scores[name] = _op_score(rows_by_session, durations, hyp_root, split, tag, spec["threshold"],
                                    rec["split"][split] if split in rec["split"] else sorted(rows_by_session))

    not_assessable: dict[str, str] = {}
    # review m6: compute the present mitigation base-names once (M3's seed systems are "M3:<ckpt>", so their
    # base name is "M3"), then a family member is not assessable iff its base name is absent. §7's degenerate
    # operating point (tau = -inf) shows up here as no frozen operating point for that system at all.
    present = {s.split(":", 1)[0] for s in op_scores}
    for name in H4_FAMILY:
        if name not in present:
            not_assessable[name] = "no frozen operating point (tuning degenerate or missing)"

    if "B0" not in op_scores:
        return {"not_done": "B0 has no frozen operating point", "systems": sorted(op_scores)}

    def per_session(score, attr, included_key):
        inc = set(score["inclusion"][included_key]["included"])
        return {s: getattr(score["sessions"][s], attr) for s in inc}

    b0 = op_scores["B0"]
    b0_dfar = per_session(b0, "delta_far_std", "ssdl")
    b0_mr = per_session(b0, "mr_dsdl", "dsdl")

    # merge M3's seeds: if several M3:<ckpt> systems exist, M3's per-session value is the mean over seeds (§7)
    m3_names = [s for s in op_scores if s.startswith("M3")]

    def mitigation_dfar(name):
        if name == "M3" and m3_names:
            per_seed = defaultdict(list)
            for mn in m3_names:
                for s, v in per_session(op_scores[mn], "delta_far_std", "ssdl").items():
                    per_seed[s].append(v)
            return {s: float(np.mean(v)) for s, v in per_seed.items()}
        return per_session(op_scores[name], "delta_far_std", "ssdl") if name in op_scores else {}

    def mitigation_mr(name):
        if name == "M3" and m3_names:
            per_seed = defaultdict(list)
            for mn in m3_names:
                for s, v in per_session(op_scores[mn], "mr_dsdl", "dsdl").items():
                    per_seed[s].append(v)
            return {s: float(np.mean(v)) for s, v in per_seed.items()}
        return per_session(op_scores[name], "mr_dsdl", "dsdl") if name in op_scores else {}

    # ---- H4 ----
    h4_included = sorted(s for s in b0_dfar if b0_dfar[s] == b0_dfar[s])
    s_b0 = float(np.mean([b0_dfar[s] for s in h4_included])) if h4_included else float("nan")
    deltas_h4 = {m: _per_session_delta(mitigation_dfar(m), b0_dfar, h4_included)
                 for m in H4_FAMILY if (m in op_scores or (m == "M3" and m3_names))}
    h4 = _h4_family(deltas_h4, s_b0, rng, set(not_assessable))

    # ---- H5 ----
    h5_included = sorted(s for s in b0_mr if b0_mr[s] == b0_mr[s])
    s_mr = float(np.mean([b0_mr[s] for s in h5_included])) if h5_included else float("nan")
    dmr_h5 = {m: _per_session_delta(mitigation_mr(m), b0_mr, h5_included)
              for m in H4_FAMILY if (m in op_scores or (m == "M3" and m3_names))}
    h5 = _h5_family(dmr_h5, s_mr, rng, set(not_assessable), descriptive)

    # ---- intersection-union ----
    iu = _intersection_union(h4, h5, descriptive)

    # ---- H6 (MR at FAR=10%, per session) ----
    mr_at_far = {}
    for m in H4_FAMILY:
        if m == "M3" and m3_names:
            per_seed = defaultdict(list)
            for mn in m3_names:
                for s, v in _mr_at_far_by_session(sweeps[mn]).items():
                    per_seed[s].append(v)
            mr_at_far["M3"] = {s: float(np.mean(v)) for s, v in per_seed.items()}
        elif m in sweeps:
            mr_at_far[m] = _mr_at_far_by_session(sweeps[m])
    m3_seed_mr = {mn: _mr_at_far_by_session(sweeps[mn]) for mn in m3_names}
    h6 = _h6(mr_at_far, m3_seed_mr, rng)

    # ---- H2 conversation (native P1/P2) ----
    # brief M2 fix: when the freeze was taken --native-absent, the native hypotheses were never frozen, so H2's
    # pipeline arm is NOT done and nothing is read from the native dirs.
    native_absent = bool(rec.get("native_absent", False))
    native_scores: dict = {}
    if not native_absent:
        for sys_name in ("P1", "P2"):
            tag = NATIVE_TAG[sys_name]
            rids = rec["split"][split] if split in rec["split"] else sorted(rows_by_session)
            changes = _native_changes(hyp_root_native, split, tag, rids)
            if all(len(v) == 0 for v in changes.values()) and not (hyp_root_native / split / tag).is_dir():
                native_scores[sys_name] = None
            else:
                native_scores[sys_name] = score_m2b.score_system(rows_by_session, changes, durations)
    h2 = _h2_native(native_scores, list(p0_pvalues), rng, native_absent=native_absent)

    # ---- secondaries (§8), each mitigation vs B1/B2 and B1/B2 vs B0, unadjusted; DER; DET; collar; etc. ----
    secondary: dict = {}
    base_d = {b: _per_session_delta(per_session(op_scores[b], "delta_far_std", "ssdl"), b0_dfar, h4_included)
              for b in ("B1", "B2") if b in op_scores}
    for b in base_d:
        secondary[f"{b}_vs_B0"] = secondary_vs(base_d[b], {s: 0.0 for s in base_d[b]}, rng)
    for m in H4_FAMILY:
        dm = deltas_h4.get(m, {})
        for b in base_d:
            secondary[f"{m}_vs_{b}"] = secondary_vs(dm, base_d[b], rng)
    det_summaries = {name: sweeps[name]["det_points"] and
                     score_m2b.det_summary(sweeps[name]["det_points"],
                                           list(sweeps[name]["det_by_session"].values()) or None)
                     for name in systems}
    # MR_DS-SL (secondary) and changes/min per system at operating point
    mr_dssl = {name: mean_stat([op_scores[name]["sessions"][s].mr_dssl
                                for s in op_scores[name]["inclusion"]["dssl"]["included"]])
               for name in systems}
    cpm = {name: mean_stat([op_scores[name]["sessions"][s].changes_per_min
                            for s in op_scores[name]["sessions"]]) for name in systems}

    # what §8/§9 asks for that this script does NOT compute
    not_computed_here = [
        "P0 on DISPLACE (E1-E3 p0L1-style AUC/delta-FAR; §8)",
        "M2's lambda AUC vs DS-SL / DS-DL (§8, decision 15 half)",
        "DISPLACE-M (secondary set; §8, §9.7)",
        "D2-P Nepali support set (§8, §9.8)",
        "sensitivities 2-8 (overlap re-inclusion, full-posterior/oracle/time-shuffled lambda, backchannel,"
        " min-run-length variants; §9) -- post-freeze, hashed in DEVIATIONS.md",
        "AMI language-driven split index vs circular-shift null (§8)",
        "UMAP companion probe accuracies (§8)",
    ]

    inclusion_report = {
        "H4": {"included": h4_included, "s_b0": s_b0,
               "excluded": {s: op_scores["B0"]["sessions"][s].n_ssdl_with_cell_controls
                            for s in op_scores["B0"]["sessions"] if s not in h4_included}},
        "H5": {"included": h5_included, "s_mr": s_mr,
               "excluded": {s: op_scores["B0"]["sessions"][s].n_dsdl
                            for s in op_scores["B0"]["sessions"] if s not in h5_included}},
    }

    return {
        "split": split,
        "collar": C, "seed": SEED,
        "power_verdict": "descriptive" if descriptive else "confirmatory",
        "operating_points": {name: {"threshold": spec["threshold"], **{k: spec[k] for k in
                              ("delta", "alpha", "tau0", "m3_ckpt") if k in spec}}
                             for name, spec in systems.items()},
        "inclusion": inclusion_report,
        "not_assessable": not_assessable,
        "H4": h4, "H5": h5, "intersection_union": iu, "H6": h6, "H2_conversation": h2,
        "secondary": {"vs": secondary, "det_summary": det_summaries, "mr_dssl": mr_dssl,
                      "changes_per_min": cpm},
        "not_computed_here": not_computed_here,
        "open_choices": _open_choices(),
    }


def _open_choices() -> dict:
    return {"h5_holm": H5_HOLM, "h5_decision": H5_DECISION, "h5_two_sided_level": H5_TWO_SIDED_LEVEL,
            "h6_pvalue": H6_PVALUE,
            "h6_ordering": H6_ORDERING, "h6_det_pooling": H6_DET_POOLING,
            "h2_control_never_fires": H2_CONTROL_NEVER_FIRES, "h2_rr_aggregate": H2_RR_AGGREGATE,
            "native_tag": NATIVE_TAG}


def _p0_pvalues_from_m2a(confirm_m2a_path: Path) -> list[float]:
    """The three P0 D1 observed one-sided p-values (e1, e2, e3) that complete the H2 Holm family (m = 5), read
    from the frozen M2a result -- NEVER typed in (§7). They live at h2_p0.primary.<m>.signflip.p_one."""
    d = json.loads(Path(confirm_m2a_path).read_text())
    prim = d["h2_p0"]["primary"]
    return [float(prim[m]["signflip"]["p_one"]) for m in ("e1", "e2", "e3")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, help="DISPLACE archive root (eval references; real run)")
    ap.add_argument("--cache-dir", dest="cache_dir", type=Path)
    ap.add_argument("--vad-dir", dest="vad_dir", type=Path)
    ap.add_argument("--wav-dir", dest="wav_dir", type=Path)
    ap.add_argument("--units-dev", dest="units_dev", type=Path, required=True)
    ap.add_argument("--units-eval", dest="units_eval", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label-map", dest="label_map", type=Path)
    ap.add_argument("--confirm-m2a", dest="confirm_m2a", type=Path, required=True,
                    help="frozen M2a result (H2 reads its P0 D1 p-values)")
    ap.add_argument("--hyp-root", dest="hyp_root", type=Path, required=True, help="run_m2b hypothesis output root")
    ap.add_argument("--dry-run", action="store_true",
                    help="run the whole analysis on DEV (dev refs + dev hyps); no freeze check, no markers, "
                         "never touch eval")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(SEED)

    if args.dry_run:
        # DEV only: no freeze check, no markers, eval never touched. The tuning operating points come from the
        # tuning record if present, else a minimal record assembled from FREEZE_M2B.json's tuning section.
        rec = json.loads(RECORD.read_text()) if RECORD.exists() else {"split": {}, "tuning": {}}
        rows, durations = _ingest_split(args.units_dev)
        p0 = _p0_pvalues_from_m2a(args.confirm_m2a)
        descriptive = rec.get("power", {}).get("verdict") == "descriptive"
        res = run_analysis(rows, durations, args.hyp_root, "dev", rec, rng, descriptive, p0)
        res["dry_run"] = True
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "confirm_m2b_dryrun.json").write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps({"written": str(args.out / "confirm_m2b_dryrun.json"),
                          "systems": sorted(res.get("operating_points", {}))}))
        return 0

    # ---- real run: one shot ----
    rec = check_freeze(args)
    out_dir = Path(rec["paths"]["out"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    # write the started marker BEFORE reading any eval reference (the first eval read of the study)
    (out_dir / "confirm_m2b.started").write_text(json.dumps({"commit": rec["commit"], "date":
                                                 __import__("datetime").datetime.now().isoformat()}))

    # eval ingest IN-PROCESS: process_recording on each eval recording's SD + LD references at the frozen eval
    # VAD, exactly as dev was ingested. This is the FIRST read of any eval reference.
    _ingest_eval_in_process(rec)

    rows, durations = _ingest_split(Path(rec["paths"]["units_eval"]))
    p0 = _p0_pvalues_from_m2a(Path(rec["paths"]["confirm_m2a"]))
    descriptive = rec["power"]["verdict"] == "descriptive"
    # brief M1: restrict every hypothesis read to the exact files hashed in the record (built from the record,
    # not re-globbed). The native files are included only when they were frozen (not --native-absent).
    global _FROZEN_HYP_PATHS
    frozen = {str(Path(f).resolve()) for f in rec["eval_hyps_sha256"]}
    if not rec.get("native_absent", False):
        frozen |= {str(Path(f).resolve()) for f in rec.get("native_hyps_sha256", {})}
    _FROZEN_HYP_PATHS = frozen
    try:
        res = run_analysis(rows, durations, args.hyp_root, EVAL_SPLIT, rec, rng, descriptive, p0)
    finally:
        _FROZEN_HYP_PATHS = None
    res.update({"dry_run": False, "freeze": {k: rec.get(k) for k in ("tag", "commit", "date")},
                "input_sha256": {"code": rec["code_sha256"], "features": rec["feature_sha256"],
                                 "references": rec["reference_sha256"], "eval_hyps": rec["eval_hyps_sha256"],
                                 "native_hyps": rec.get("native_hyps_sha256", {}),
                                 "native_absent": rec.get("native_absent", False),
                                 "m3_artifacts": rec["m3_artifacts_sha256"]},
                "tuning_values": rec["tuning"]})
    (out_dir / "confirm_m2b.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"written": str(out_dir / "confirm_m2b.json"),
                      "intersection_union": {m: iu.get("label") for m, iu in res["intersection_union"].items()}},
                     default=float))
    return 0


def _wav_duration_s(wav_path: Path) -> float:
    """Recording duration in seconds from the WAV HEADER (stdlib `wave`, no audio decoded), the frozen
    non-reference source the freeze path-pins as `wav_dir` (review m1 / brief fix 6). Never derived from the
    eval references. nan when the header cannot be read (the caller records nan, so changes/min is nan, not a
    silently-wrong rate)."""
    import wave
    try:
        with wave.open(str(wav_path), "rb") as w:
            frames, rate = w.getnframes(), w.getframerate()
        return (frames / rate) if rate else float("nan")
    except Exception:
        return float("nan")


def _ingest_eval_in_process(rec: dict) -> None:
    """Ingest the eval references with `ingest_displace.process_recording` at the frozen eval VAD, writing the
    eval unit tables confirm then scores. Mirrors `ingest_displace.main` for the eval split, but called in
    process AFTER the started marker (the first eval read, §11). Each recording's duration is read from the
    frozen WAV header (review m1 / brief fix 6), looked up by the WAV extension `init` recorded (review-v2 m5),
    and written into the summary as `duration_s`, so changes/min is never derived from the eval reference row
    times."""
    import numpy as _np
    from csdiar.data import ingest_displace as ING
    root = Path(rec["paths"]["root"]).expanduser()
    wav_dir = Path(rec["paths"]["wav_dir"]).expanduser()
    lay = ING.LAYOUT[EVAL_SPLIT]
    sd_dir, ld_dir = root / lay["sd"], root / lay["ld"]
    # review m2 / brief fix 7: use the SD/LD suffixes `freeze_m2b init` recorded, so a dev ingest done with a
    # custom suffix is matched here instead of silently reading the default-suffix files.
    sd_suffix = rec.get("sd_suffix", ING.SD_SUFFIX)
    ld_suffix = rec.get("ld_suffix", ING.LD_SUFFIX)
    # review-v2 m5: the WAV extension `freeze_m2b init` recorded (one for all recordings). The duration lookup
    # uses it instead of a hard-coded '.wav', so a recording stored as e.g. '.flac' is found rather than
    # yielding a nan duration from a missing '<rid>.wav'. Defaults to '.wav' for a legacy record with no field.
    wav_ext = rec.get("wav_ext", ".wav")
    ids = sorted(p.name[: -len(sd_suffix)] for p in sd_dir.glob(f"*{sd_suffix}"))
    vad_arrays: dict = {}
    vad_path = Path(rec["paths"]["vad_dir"]).expanduser() / f"{EVAL_SPLIT}.npz"
    if vad_path.exists():
        with _np.load(vad_path) as z:
            vad_arrays = {k: z[k].astype(float).tolist() for k in z.files}
    out = Path(rec["paths"]["units_eval"]).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    all_rows: list = []
    summaries: list = []
    for rid in ids:
        sd_text = (sd_dir / f"{rid}{sd_suffix}").read_text(encoding="utf-8")
        ld_text = (ld_dir / f"{rid}{ld_suffix}").read_text(encoding="utf-8")
        rows, summary = ING.process_recording(rid, EVAL_SPLIT, sd_text, ld_text, vad_arrays.get(rid))
        # record the duration from the frozen WAV header (non-reference source), so _ingest_split reads a real
        # duration_s and never falls back to the reference row times (review m1 / confirm m1 / brief fix 6).
        summary["duration_s"] = _wav_duration_s(wav_dir / f"{rid}{wav_ext}")
        all_rows.extend(rows)
        summaries.append(summary)
    all_rows = ING.order_rows(all_rows)
    for kind in ("ssdl", "turns", "ctrl"):
        subset = [r for r in all_rows if (r["kind"] in ("dsdl", "dssl") if kind == "turns"
                                          else r["kind"] == kind)]
        ING.write_csv_gz(out / f"{kind}_{EVAL_SPLIT}.csv.gz", subset)
    (out / "summary.json").write_text(json.dumps({"split": EVAL_SPLIT, "n_recordings": len(ids),
                                                  "recordings": summaries}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
