#!/usr/bin/env python3
"""DISPLACE 2024 dev tuning for the M2b conversation study: fits, sweep and operating points
(PREREGISTRATION_M2B.md §3, §4, §5; the §7 inclusion rule for "not assessable").

This is the DEV-ONLY tuner for **P1-AHC with E3** (and the M3 all-data checkpoints): it fits B1/M1/B2 on the
DISPLACE dev references, sweeps every system over the clustering-threshold grid (and M2 over its alpha grid),
scores each dev recording with `csdiar.analysis.score_m2b`, and freezes one operating point per system by the
§3/§5 rules. Nothing here reads an eval split (a guard refuses it), and no eval statistic is produced -- the
one-shot confirmatory run is `confirm_m2b.py` (NOT this file), which is a later step.

Scope (brief, §4): **P1-AHC + E3 and the M3 checkpoints only.** The P0-with-E3 gate alpha (§3, §6) is tuned by
`csdiar.analysis.tune_p0_m2b`, and the native P1/P2 back ends are later work; `p0_gate_alpha` here always raises
NotImplementedYet, so a caller cannot mistake this module for the source of the P0-gate value.

Layout
------
Pure, model-free pieces (unit-tested in `tests/test_tune_m2b.py`):
  * Fits on dev embeddings/segment cosines: `fit_b1_means`, `fit_m1_v_hat`, `fit_b2_delta`. Each returns the
    value AND the counts it used, and RAISES `InsufficientData` when it is under-defined.
  * Selection rules: `select_b0_threshold`, `select_matched_threshold`, `select_m2_alpha`,
    `assess_m2_tunable`, `degenerate_operating_point`. Each is a small deterministic function over numbers.
  * B0's DER is a POOLED (corpus-level) DER, `pyannote.metrics` native: summed error time over summed reference
    time across dev recordings (§3). `der_components` (pyannote `detailed=True` per recording) and
    `pooled_der` are the helpers; the curve's `der` field carries this pooled DER regardless of `AGGREGATION`,
    which governs only FAR/MR aggregation (an open design-round choice).
  * `SystemCurve` / `curve_from_scores`: turn a system's per-threshold `score_m2b.score_system` results plus the
    per-recording DER components into the per-threshold (pooled DER, FAR_SS-DL, MR_DS-DL, DET summary) record the
    selection rules and the output JSON read.
Orchestration (needs the runner + a Community-1 cache; exercised end-to-end through fakes):
  * `run_dev_tuning` reads the dev manifest, builds the fit inputs from the stage-1 / LID caches, the wavs and
    the dev references, runs the three fits, writes the B1 npz and M1 npy the runner reads (hashing them),
    sweeps B0/B1/M1/B2/M3 over the threshold grid, selects B0, sweeps M2's alpha at B0's threshold, scores every
    output (`score_m2b` from the exclusive annotation; pooled DER components from the overlapping annotation
    against the reference speaker RTTM), builds the curves, runs every selection and assessability rule, and
    writes `curves`, `operating_points`, `fits`, the alpha table, counts and flags.
  * `--dry-run` checks inputs and prints the plan without running or scoring anything.

Open choices the plan leaves to the implementation (all are named constants / parameters here; every one is
listed in the task's final report):
  * `AGGREGATION` -- how dev FAR / MR enter the selection rules: POOLED over units vs SESSION-AVERAGED. Default
    `"session_mean"` (the confirmatory tests cluster by session, §7, so the operating point is chosen on the
    same session-level quantity the tests use). **B0's DER is NOT governed by this** (fixed after review M1): §3
    pins B0's threshold to the lowest `pyannote.metrics` POOLED dev DER, so `curve_from_scores` always computes
    the pooled DER for selection, independently of `AGGREGATION`. `score_m2b.det_summary` already reports the DET
    both pooled and by session.
  * `B0_TIE` -- B0's lowest-dev-DER tie rule: `"lower_threshold"` (the plan does not pin it; the lower threshold
    merges less, so it is the conservative, reproducible choice).
  * `M2_AT_B0_THRESHOLD` -- whether M2 runs at B0's frozen clustering threshold (True) and is NOT re-matched
    afterwards. True: M2 is a post-processor on B0's own clustering (§4), and §3's alpha rule is stated relative
    to B0, so re-matching M2's threshold would double-count the operating-point choice. M2's threshold is thus
    B0's; only alpha is swept.
  * `M1_SIGN` -- the M1 direction sign convention: `"english_minus_other"` (v points from the non-English
    language mean to the English mean, per reference language pair; units with no English side are skipped).
  * `B2_PAIRING` -- the B2 delta segment pairing: `"same_speaker_within_recording"` same-speaker same-language
    minus same-speaker cross-language SEGMENT cosines (the system's own `SegmentEmbedder` embeddings), pooled
    over dev recordings; clamped to [0, B2_DELTA_MAX]. The segments are the reference resolver's clean
    single-speaker single-language spans, so BOTH the speaker and the language of a segment are the references'
    (never LID).
  * `B2_MAX_PAIRS_PER_UNIT` -- a deterministic cap on same-speaker segment pairs per (speaker, recording): 0
    (default) takes every pair; a positive cap truncates a shuffle of the sorted pair list seeded per
    (recording, speaker) from a stable key, so a unit's sample is reproducible AND independent of the rest of
    the corpus (§4, review M1/m5).
  * `ALPHA_ZERO_ALWAYS_FEASIBLE` -- alpha = 0 reproduces B0 (deltaMR = 0) so it always meets the +1 pp bound; the
    feasible set is never empty (§3).

All frozen numeric constants (the grids, the +1 pp bound, the 80% one-sided BCa level, the 10,000 resamples,
the 5-session / 10-turn assessability rule, the seed) come straight from §3 and §7.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

# numpy is used by the fits and the BCa resampling, but imported lazily where possible so the SELECTION rules
# and the dataclasses import on a host without numpy (the box analysis env has numpy; a dev host may not).
# score_m2b / stats are stdlib-friendly at import (their heavy deps are lazy).
from csdiar.analysis import score_m2b
from csdiar.analysis.stats import bca_ci

# ----------------------------------------------------------------------------------------------------- #
# frozen constants (PREREGISTRATION_M2B.md §3, §5, §7)
# ----------------------------------------------------------------------------------------------------- #
SEED = 20261018                               # §11 logging / resampling seed
THRESHOLD_GRID = tuple(round(0.30 + 0.05 * k, 2) for k in range(15))   # 0.30 .. 1.00 step 0.05 (§3)
ALPHA_GRID = tuple(round(0.05 * k, 2) for k in range(21))              # 0.00 .. 1.00 step 0.05 (§3)
M2_TAU0 = 0.0                                 # deployed P1-AHC post-processor form (§3)
B2_DELTA_MAX = 0.5                            # B2 delta clamp upper bound (§4; D1 B2_MAX)
MR_BOUND_PP = 0.01                            # M2's upper bound on deltaMR_DS-DL vs B0: +1 pp (§3)
BCA_ONE_SIDED_LEVEL = 0.80                    # one-sided 80% BCa bound for the M2 alpha rule (§3)
BCA_N_BOOT = 10000                            # 10,000 whole-session resamples (§3)
ASSESS_MIN_SESSIONS = 5                       # >= 5 dev sessions ... (§3 M2 assessability)
ASSESS_MIN_DSDL_PER_SESSION = 10              # ... each with >= 10 DS-DL turns (§3)
FAR_PRIMARY = 0.10                            # DET primary operating point FAR_SS-DL = 10% (§5)
FAR_SECONDARY = 0.05                          # DET secondary FAR_SS-DL = 5% (§5)

# ----------------------------------------------------------------------------------------------------- #
# open choices, fixed as named constants (see the module docstring and the final report)
# ----------------------------------------------------------------------------------------------------- #
AGGREGATION = "session_mean"                  # "session_mean" | "pooled" -- how FAR/MR enter selection (NOT DER)
B0_TIE = "lower_threshold"                    # B0 lowest-DER tie rule
M2_AT_B0_THRESHOLD = True                     # M2 runs at B0's frozen threshold; its threshold is not re-matched
M1_SIGN = "english_minus_other"              # M1 direction sign convention
B2_PAIRING = "same_speaker_within_recording"  # B2 delta segment pairing
B2_MAX_PAIRS_PER_UNIT = 0                      # deterministic cap on same-speaker segment pairs per (speaker,
#                                               recording); 0 = no cap (take every pair). When > 0 the pairs are
#                                               taken in SORTED (start, end) segment order with a shuffle seeded
#                                               per (recording, speaker) from a stable key, so a unit's sample is
#                                               reproducible AND independent of the rest of the corpus (§4,
#                                               review M1/m5).
ALPHA_ZERO_ALWAYS_FEASIBLE = True             # alpha=0 == B0 always meets the bound (§3)

ENGLISH_CODE = "en"                           # VoxLingua107 code for English, for the M1 sign convention

GUARDED_SELF = "csdiar/analysis/tune_m2b.py"  # this file hashes itself into the output record


class InsufficientData(ValueError):
    """A fit (B1 means / M1 v / B2 delta) does not have enough data to be defined (§4 'refuses when too little
    data')."""


class NotImplementedYet(NotImplementedError):
    """A piece the brief defers to a later step (the P0 gate alpha, native P1/P2). Raised so a caller never
    mistakes a missing value for a frozen one."""


# ===================================================================================================== #
# 0. REFERENCE RESOLVER on the dev speaker + language RTTMs (PREREGISTRATION_M2B.md §2, §4; review M1/M3)
# ===================================================================================================== #
# M1 units and B2 segment pairs read the DISPLACE references EXACTLY as the conversation reader does, by
# building the same elementary timeline from the two reference tracks (`ingest_displace.parse_rttm` +
# `elementary_timeline`) and asking the same "clean single-speaker single-language run" question
# `ingest_displace._window_in_clean_run` answers. Nothing here reads LID: on a real run the slot's (speaker,
# language) both come from the references (review M1's requirement, and review m3's exclusion-zone contract).
class DisplaceRefResolver:
    """Answer, from one dev recording's speaker + language reference RTTMs, two questions the fits need (§2, §4):

      * ``resolve(lo, hi)`` -> ``(speaker, language)`` iff the whole span ``[lo, hi]`` lies inside CLEAN timeline
        intervals of exactly one speaker and exactly one language, with no gap, no overlap interval and no
        ambiguous interval touching the span -- the same rule as `ingest_displace._window_in_clean_run`, so a
        span that straddles a reference speaker change, a language change, an overlap or an ambiguous interval
        gets ``None`` (review m3: attribution excluded at exactly those boundaries). ``resolve(t)`` with one
        argument answers for the instant ``[t, t]`` (an M1 slot is attributed by its active span, see below).
      * ``clean_spans()`` -> the list of ``(speaker, start, end, language)`` maximal clean single-speaker
        single-language spans, for B2's same-speaker segment pairs.

    Built on `ingest_displace`'s own parser and timeline so the fits and the reader never diverge. Pure and
    model-free (plain-text RTTMs); numpy is not needed here.
    """

    def __init__(self, sd_text: str, ld_text: str, rid: str = ""):
        from csdiar.data import ingest_displace as ING
        self._ING = ING
        self.rid = rid
        self._sd = ING.parse_rttm(sd_text)
        self._ld = ING.parse_rttm(ld_text)
        # reuse the reader's label-variant guard so 'ne'/'NE' cannot fabricate a run split here either.
        ING.check_label_variants(rid or "?", "SD", {s.label for s in self._sd})
        ING.check_label_variants(rid or "?", "LD", {s.label for s in self._ld})
        self._timeline = ING.elementary_timeline(self._sd, self._ld)

    @classmethod
    def from_paths(cls, sd_path, ld_path, rid: str = "") -> "DisplaceRefResolver":
        sd_text = Path(sd_path).read_text(encoding="utf-8")
        ld_text = Path(ld_path).read_text(encoding="utf-8")
        return cls(sd_text, ld_text, rid=rid or Path(sd_path).stem)

    def resolve(self, lo: float, hi: Optional[float] = None):
        """``(speaker, language)`` if a single clean (speaker, language) run fully, cleanly covers ``[lo, hi]``
        (``[lo, lo]`` when ``hi`` is None), else ``None``. Uses `ingest_displace._window_in_clean_run` for the
        coverage test, so the contract matches the reader's exactly."""
        ING = self._ING
        if hi is None:
            hi = lo
        if hi < lo:
            lo, hi = hi, lo
        # candidate (speaker, lang): the clean intervals overlapping the span must all agree on one speaker and
        # one language, and must fully cover it with no gap. _window_in_clean_run enforces coverage + cleanliness
        # for a GIVEN (speaker, lang); we first read the candidate off the clean intervals touching the span.
        # The ±EPS widening here is ONLY to PICK candidate atoms (so an atom touching the span to within EPS is
        # considered); it is NOT the coverage tolerance -- that is _window_in_clean_run's own EPS, applied below
        # to the RAW lo/hi (m1: a single source of tolerance, bit-identical to the reader's call site).
        touching = [iv for iv in self._timeline
                    if iv.is_clean and iv.end > lo - ING.EPS and iv.start < hi + ING.EPS]
        if not touching:
            return None
        speakers = {iv.speakers[0] for iv in touching}
        langs = {iv.languages[0] for iv in touching}
        if len(speakers) != 1 or len(langs) != 1:
            return None                              # a speaker or language change inside the span: no answer
        spk = next(iter(speakers))
        lang = next(iter(langs))
        # coverage test on the RAW bounds: _window_in_clean_run's own EPS is the only tolerance, exactly as the
        # reader calls it (m1). A span ending 1·EPS past a clean run's end therefore fails here just as it does
        # for the reader, with no extra ±EPS widening double-counting the slack.
        if not ING._window_in_clean_run(self._timeline, lo, hi, spk, lang):
            return None
        # reject if any overlap or ambiguous interval touches the span (coverage alone cannot see an overlap that
        # shares the span with the clean run; the reader's exclusion zone forbids it, review m3). The ±EPS here
        # only decides "touching"; it is not the coverage tolerance.
        for iv in self._timeline:
            if iv.end <= lo - ING.EPS or iv.start >= hi + ING.EPS:
                continue
            if iv.is_overlap or iv.is_ambiguous:
                return None
        return (spk, lang)

    def clean_spans(self) -> list:
        """Maximal clean single-speaker single-language spans ``(speaker, start, end, language)`` for B2 (§4).

        A span is a run of adjacent clean timeline intervals sharing one speaker and one language with no gap;
        it breaks at a gap, a speaker change, a language change, or a non-clean (overlap/ambiguous) interval.
        """
        ING = self._ING
        spans: list = []
        cur = None  # [speaker, start, end, language]
        for iv in self._timeline:
            if not iv.is_clean:
                cur = None
                continue
            spk, lang = iv.speakers[0], iv.languages[0]
            if (cur is not None and cur[0] == spk and cur[3] == lang
                    and iv.start <= cur[2] + ING.EPS):
                cur[2] = iv.end
            else:
                cur = [spk, iv.start, iv.end, lang]
                spans.append(cur)
        return [(s[0], s[1], s[2], s[3]) for s in spans]


def default_ref_resolver_factory(inputs: "TuningInputs", rid: str) -> DisplaceRefResolver:
    """Build a `DisplaceRefResolver` for one dev recording from `inputs.ref_rttm_dir` (SD) and
    `inputs.lang_rttm_dir` (LD): ``<ref_rttm_dir>/<rid>_SPEAKER.rttm`` and
    ``<lang_rttm_dir>/<rid>_LANGUAGE.rttm`` (§2 layout; `ingest_displace.SD_SUFFIX` / `LD_SUFFIX`). Raises when
    either directory is missing or either file is absent -- a real run with references present must actually
    read them (review M1), never silently fall back to the unfittable branch."""
    from csdiar.data.ingest_displace import SD_SUFFIX, LD_SUFFIX
    if inputs.ref_rttm_dir is None:
        raise ValueError("default_ref_resolver_factory needs ref_rttm_dir (speaker RTTMs)")
    if inputs.lang_rttm_dir is None:
        raise ValueError("default_ref_resolver_factory needs lang_rttm_dir (language RTTMs); pass --lang-rttm-dir")
    sd_path = Path(inputs.ref_rttm_dir) / f"{rid}{SD_SUFFIX}"
    ld_path = Path(inputs.lang_rttm_dir) / f"{rid}{LD_SUFFIX}"
    for kind, p in (("speaker", sd_path), ("language", ld_path)):
        if not Path(p).exists():
            raise FileNotFoundError(f"reference {kind} RTTM for dev recording {rid!r} not found: {p}")
    return DisplaceRefResolver.from_paths(sd_path, ld_path, rid=rid)


# ===================================================================================================== #
# 1. FITS on dev (PREREGISTRATION_M2B.md §4)
# ===================================================================================================== #
def fit_b1_means(slot_embeddings: dict) -> dict:
    """B1 language means on dev (§4): per reference language, the mean of the stage-1 slot embeddings whose
    LID-argmax language (over the recording's two reference languages, `slot_languages`) is that language,
    pooled over all dev recordings.

    `slot_embeddings` maps a language KEY -> an iterable of (dim,) slot-embedding vectors already assigned to
    that language (the caller does the slot->language assignment with `run_m2b.slot_languages`, so this function
    is pure and model-free). The key may be a VoxLingua107 code (for the pure tests) or a `pair` index (the
    orchestration passes the pair indices the runner's `b1_transform` reads); the function is agnostic and
    returns the means keyed exactly as given. Returns ``{"means": {key: (dim,) float array}, "counts": {key: n},
    "dim": dim}``.

    Raises `InsufficientData` when a language has no slots (its mean is undefined), or when the two languages'
    slots disagree on dimension.
    """
    import numpy as np

    means: dict = {}
    counts: dict = {}
    dim = None
    for code, vecs in slot_embeddings.items():
        arr = np.asarray([np.asarray(v, dtype=np.float64).reshape(-1) for v in vecs], dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise InsufficientData(f"B1: language {code!r} has no slot embeddings; its mean is undefined")
        if dim is None:
            dim = arr.shape[1]
        elif arr.shape[1] != dim:
            raise InsufficientData(f"B1: language {code!r} embeddings have dim {arr.shape[1]} != {dim}")
        means[code] = arr.mean(axis=0)
        counts[code] = int(arr.shape[0])
    if not means:
        raise InsufficientData("B1: no languages with slot embeddings")
    return {"means": means, "counts": counts, "dim": int(dim)}


def fit_m1_v_hat(units: Sequence[dict], english_code: str = ENGLISH_CODE, sign: str = M1_SIGN) -> dict:
    """M1 direction v on dev (§4): the UNIT mean, over (reference speaker, recording) units, of the within-speaker
    cross-language difference of embedding MEANS, in the SAME embedding space the M1 transform acts on (the
    stage-1 slot embeddings; the orchestration builds the units from those).

    Each element of `units` is one (speaker, recording) unit:
      ``{"speaker": str, "recording": str, "by_lang": {code: (dim,) mean embedding of that speaker's windows in
      that language in that recording}}``.
    A unit contributes only when it has BOTH of its recording's two reference languages. The per-unit difference
    is oriented by `sign`:
      * ``"english_minus_other"`` (default): English-language mean minus the other language's mean. A unit with
        no English side is SKIPPED (and counted), so the sign is consistent across language pairs even when the
        non-English language differs (Hindi, Bengali, ...). This is the sign convention the brief asks to fix and
        list.
    The per-unit differences are averaged (an unweighted mean over units), and the result is unit-normalised to
    the direction v_hat `run_m2b.load_v_hat_npy` reads.

    Returns ``{"v_hat": (dim,) unit array, "n_units": n, "n_skipped_no_english": s, "dim": dim}``.
    Raises `InsufficientData` when no unit contributes, or when the averaged difference is a zero / non-finite
    vector (no direction defined).
    """
    import numpy as np

    if sign != "english_minus_other":
        raise ValueError(f"unknown M1 sign convention {sign!r}")
    diffs = []
    n_skipped = 0
    dim = None
    for u in units:
        by = u["by_lang"]
        langs = list(by.keys())
        if len(langs) < 2:
            continue                                   # unit saw only one language: no cross-language contrast
        if english_code not in by:
            n_skipped += 1                             # no English side -> cannot orient; skip (counted)
            continue
        others = [c for c in langs if c != english_code]
        if not others:
            continue
        other = others[0]                              # the recording's non-English reference language
        e_en = np.asarray(by[english_code], dtype=np.float64).reshape(-1)
        e_ot = np.asarray(by[other], dtype=np.float64).reshape(-1)
        if dim is None:
            dim = e_en.shape[0]
        if e_en.shape[0] != dim or e_ot.shape[0] != dim:
            raise InsufficientData("M1: unit embedding means disagree on dimension")
        diffs.append(e_en - e_ot)                      # English minus the other language
    if not diffs:
        raise InsufficientData("M1: no (speaker, recording) unit had both languages with an English side")
    v = np.mean(np.asarray(diffs, dtype=np.float64), axis=0)
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm <= 0.0:
        raise InsufficientData("M1: the averaged within-speaker cross-language difference is a zero vector")
    return {"v_hat": (v / norm), "n_units": len(diffs), "n_skipped_no_english": n_skipped, "dim": int(dim)}


def fit_b2_delta(same_lang_cosines: Sequence[float], cross_lang_cosines: Sequence[float],
                 delta_max: float = B2_DELTA_MAX) -> dict:
    """B2 delta on dev (§4): the verification-score shift = mean same-speaker SAME-language segment cosine minus
    mean same-speaker CROSS-language segment cosine, clamped to [0, delta_max].

    `same_lang_cosines` / `cross_lang_cosines` are the pooled same-speaker segment-pair cosines (the system's own
    `SegmentEmbedder` embeddings; the caller forms the pairs, so this is pure). The raw shift can be negative;
    the clamp to [0, delta_max] is §4's, carried over from the D1 B2 (delta is a cosine difference and AHC works
    on 1 - cos, so raising an affinity by delta equals lowering a distance by delta).

    Returns ``{"delta": clamped, "raw": raw, "n_same": n, "n_cross": n, "clamped_low": bool, "clamped_high":
    bool}``. Raises `InsufficientData` when either pool is empty (the mean is undefined).
    """
    import numpy as np

    same = np.asarray(list(same_lang_cosines), dtype=np.float64)
    cross = np.asarray(list(cross_lang_cosines), dtype=np.float64)
    if same.size == 0 or cross.size == 0:
        raise InsufficientData(f"B2: need at least one same- and one cross-language cosine "
                               f"(got {same.size} same, {cross.size} cross)")
    raw = float(same.mean() - cross.mean())
    clamped = min(max(raw, 0.0), float(delta_max))
    return {"delta": clamped, "raw": raw, "n_same": int(same.size), "n_cross": int(cross.size),
            "clamped_low": raw < 0.0, "clamped_high": raw > float(delta_max)}


# ===================================================================================================== #
# 1b. POOLED (corpus-level) DER, pyannote-native (PREREGISTRATION_M2B.md §3, review M1 + m3)
# ===================================================================================================== #
# The pyannote.metrics DiarizationErrorRate detailed-component keys. Accumulated over recordings pyannote reports
# a corpus-level POOLED DER = sum of (false alarm + missed detection + confusion) over sum of total reference
# time (§3's "pyannote.metrics, overlap scored, no collar"). We read those three error components and the total
# per recording and pool them ourselves, so B0's threshold does not inherit the FAR/MR session-mean path.
DER_ERROR_KEYS = ("false alarm", "missed detection", "confusion")
DER_TOTAL_KEY = "total"


def der_components(reference_rttm_path, hypothesis_rttm_path) -> dict:
    """Per-recording pyannote DER COMPONENTS (collar 0.0, overlap scored), for pooling into a corpus DER.

    A thin wrapper around ``DiarizationErrorRate(collar=0.0, skip_overlap=False)`` called with ``detailed=True``
    -- the same metric `score_m2b.der` wraps, but returning the raw error/total TIMES instead of the ratio, so
    the caller can accumulate Σerror / Σtotal (a pooled DER) rather than a mean of ratios (review M1). pyannote
    is imported lazily. Reuses `score_m2b`'s single-recording guard semantics. Returns
    ``{"error": fa+miss+conf seconds, "total": reference seconds, "fa", "miss", "conf"}``.
    """
    from pyannote.metrics.diarization import DiarizationErrorRate
    from pyannote.database.util import load_rttm

    def _one(path):
        ann = load_rttm(str(path))
        if not ann:
            raise ValueError(f"no annotation parsed from {path}")
        if len(ann) != 1:
            raise ValueError(f"der_components scores exactly one recording, but {path} holds {len(ann)} "
                             f"({sorted(ann.keys())}); pass a single-recording RTTM")
        return next(iter(ann.values()))

    reference = _one(reference_rttm_path)
    hypothesis = _one(hypothesis_rttm_path)
    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    detail = metric(reference, hypothesis, detailed=True)
    # Strict keys (checked on the box against pyannote.metrics 4.x: 'false alarm', 'missed detection',
    # 'confusion', 'total'): a renamed component must raise, never count as zero error.
    fa = float(detail["false alarm"])
    miss = float(detail["missed detection"])
    conf = float(detail["confusion"])
    total = float(detail[DER_TOTAL_KEY])
    return {"error": fa + miss + conf, "total": total, "fa": fa, "miss": miss, "conf": conf}


def pooled_der(components_by_recording: dict) -> float:
    """Corpus-level pooled DER = Σ error time / Σ reference time over the given recordings (§3, review M1/m3).

    `components_by_recording` maps a recording id -> a component dict with ``error`` and ``total`` seconds (as
    `der_components` returns). This is pyannote's native accumulated DER, NOT a mean of per-recording ratios and
    NOT a duration-weighted mean of ratios (review m3): it sums error and total separately and divides once.
    A recording with a missing (None/empty) or non-finite component RAISES ValueError: B0's threshold rides on
    this number, and silently dropping a recording would change the corpus it is computed over. Returns nan
    only when there are no recordings or no positive total reference time.
    """
    from math import isfinite
    tot_err = 0.0
    tot_tot = 0.0
    for rid, comp in components_by_recording.items():
        err = comp.get("error") if comp else None
        tot = comp.get("total") if comp else None
        if err is None or tot is None or not (isfinite(err) and isfinite(tot)):
            raise ValueError(f"pooled_der: recording {rid!r} has no finite DER components ({comp!r}); "
                             "every dev recording must be scored against its reference before B0 is chosen")
        tot_err += float(err)
        tot_tot += float(tot)
    return (tot_err / tot_tot) if tot_tot > 0.0 else float("nan")


# ===================================================================================================== #
# 2. CURVES: per-threshold dev metrics from score_m2b (PREREGISTRATION_M2B.md §3, §5)
# ===================================================================================================== #
@dataclass
class ThresholdPoint:
    """One system at one threshold, aggregated to the dev-level numbers the selection rules read."""
    threshold: float
    der: float                      # dev DER -- always the POOLED (corpus) DER (§3, review M1)
    far_ssdl: float                 # dev FAR_SS-DL (aggregation per AGGREGATION)
    mr_dsdl: float                  # dev MR_DS-DL (aggregation per AGGREGATION)
    per_session_mr_dsdl: dict = field(default_factory=dict)   # session -> MR_DS-DL (included sessions only)
    n_sessions_far: int = 0         # sessions included for FAR/H4
    n_sessions_mr: int = 0          # sessions included for MR/H5
    der_components: dict = field(default_factory=dict)        # recording -> {error, total, ...} that pooled it


@dataclass
class SystemCurve:
    system: str
    points: list                    # list[ThresholdPoint] in threshold order
    det: dict = field(default_factory=dict)        # score_m2b.det_summary output over the sweep
    param_tag: str = ""


def _aggregate(values_by_session: dict, included: Sequence[str], weights: Optional[dict] = None,
               aggregation: str = AGGREGATION) -> float:
    """Aggregate a per-session metric over the INCLUDED sessions (§7), by AGGREGATION.

    ``session_mean``: unweighted mean over included sessions (the confirmatory tests cluster by session, so the
    operating point is chosen on the same session-level quantity). ``pooled``: weighted by `weights` (unit
    counts) when given, else equal weight. Returns nan when no session is included. This governs FAR / MR only;
    B0's DER uses the corpus pooled DER (`pooled_der`), never this path.
    """
    xs = [values_by_session[s] for s in included if s in values_by_session
          and values_by_session[s] == values_by_session[s]]          # drop nan
    if not xs:
        return float("nan")
    if aggregation == "pooled" and weights is not None:
        ws = [float(weights.get(s, 1.0)) for s in included if s in values_by_session
              and values_by_session[s] == values_by_session[s]]
        tot = sum(ws)
        return sum(x * w for x, w in zip(xs, ws)) / tot if tot > 0 else float("nan")
    return sum(xs) / len(xs)


def curve_from_scores(system: str, threshold_scores: dict, der_components_by_threshold: dict,
                      aggregation: str = AGGREGATION, param_tag: str = "") -> SystemCurve:
    """Build a `SystemCurve` from a system's per-threshold `score_m2b.score_system` output plus per-recording DER
    COMPONENTS.

    `threshold_scores` maps threshold -> the dict returned by `score_m2b.score_system` for that threshold (it
    carries ``sessions`` = {sess: SessionScore} and ``inclusion`` = {'ssdl'|'dsdl'|...: {included, excluded}}).
    `der_components_by_threshold` maps threshold -> {recording: {error, total, ...}} (from `der_components`).

    DER per threshold is the corpus POOLED DER over those components (§3, review M1): Σerror / Σtotal, computed
    by `pooled_der`, independently of `aggregation`. FAR / MR over sessions follow `aggregation`. The DET summary
    is read over the (threshold, FAR, MR) sweep with `score_m2b.det_summary`, both pooled and per session
    (review m4): the per-session DET passes one (threshold, FAR, MR) sweep per session so `det_summary`'s
    `by_session` reading is populated.

    The §7 inclusion rule is applied by `score_system` already: FAR/H4 use sessions with >= MIN_UNITS SS-DL
    points that have a cell control; MR/H5 use sessions with >= MIN_UNITS DS-DL turns. Only included sessions
    enter the aggregate (an excluded session's nan MR is never averaged).
    """
    points: list = []
    det_triples: list = []
    # per-session DET sweeps: session -> list of (threshold, far, mr) over the thresholds that session is in.
    det_by_session: dict = defaultdict(list)
    for th in sorted(threshold_scores):
        res = threshold_scores[th]
        sessions = res["sessions"]
        inc_far = res["inclusion"]["ssdl"]["included"]
        inc_mr = res["inclusion"]["dsdl"]["included"]

        far_by_sess = {s: sessions[s].far_ssdl for s in sessions}
        mr_by_sess = {s: sessions[s].mr_dsdl for s in sessions}
        far_w = {s: sessions[s].n_ssdl_with_cell_controls for s in sessions}
        mr_w = {s: sessions[s].n_dsdl for s in sessions}

        far = _aggregate(far_by_sess, inc_far, far_w, aggregation)
        mr = _aggregate(mr_by_sess, inc_mr, mr_w, aggregation)

        comp = der_components_by_threshold.get(th, {})
        der = pooled_der(comp)                             # POOLED (corpus) DER, §3, review M1

        points.append(ThresholdPoint(
            threshold=float(th), der=der, far_ssdl=far, mr_dsdl=mr,
            per_session_mr_dsdl={s: mr_by_sess[s] for s in inc_mr},
            n_sessions_far=len(inc_far), n_sessions_mr=len(inc_mr),
            der_components=dict(comp)))
        # pooled DET sweep uses the aggregated FAR/MR per threshold
        if far == far and mr == mr:
            det_triples.append((float(th), far, mr))
        # per-session DET sweep: each session that is included for BOTH FAR and MR at this threshold contributes
        # a (threshold, far, mr) node to its own sweep, so det_summary's by_session reading is populated (m4).
        for s in set(inc_far) & set(inc_mr):
            fs, ms = far_by_sess[s], mr_by_sess[s]
            if fs == fs and ms == ms:
                det_by_session[s].append((float(th), fs, ms))

    by_session_sweeps = [v for v in det_by_session.values() if v]
    det = score_m2b.det_summary(det_triples, by_session_sweeps or None) if det_triples else {}
    return SystemCurve(system=system, points=points, det=det, param_tag=param_tag)


# ===================================================================================================== #
# 3. SELECTION rules (PREREGISTRATION_M2B.md §3, §5; small pure functions, each unit-tested)
# ===================================================================================================== #
def degenerate_operating_point(points: Sequence[ThresholdPoint]) -> bool:
    """True when a system's dev sweep has NO finite operating point (§7 'degenerate operating point ... not
    assessable'): every threshold's DER is nan/inf, i.e. no threshold produced a scorable system. Mirrors
    `confirm_h4`'s `tau = -inf` degeneracy for the P1-AHC threshold sweep.
    """
    from math import isfinite
    return not any(isfinite(p.der) for p in points)


def select_b0_threshold(points: Sequence[ThresholdPoint], tie: str = B0_TIE) -> dict:
    """B0's operating point (§3): the grid threshold with the LOWEST dev DER (the corpus POOLED DER, review M1).

    `tie = "lower_threshold"`: among thresholds tied on the minimum DER (exact float equality), the smaller
    threshold wins (it merges less -- the conservative, reproducible choice the plan leaves open). Returns
    ``{"threshold": th, "der": der, "ties": [tied thresholds]}``. Raises `InsufficientData` when no threshold has
    a finite DER (a degenerate B0, reported not assessable by the caller).
    """
    from math import isfinite
    finite = [p for p in points if isfinite(p.der)]
    if not finite:
        raise InsufficientData("B0: no threshold has a finite dev DER (degenerate)")
    best = min(p.der for p in finite)
    tied = sorted(p.threshold for p in finite if p.der == best)
    if tie == "lower_threshold":
        chosen = tied[0]
    elif tie == "higher_threshold":
        chosen = tied[-1]
    else:
        raise ValueError(f"unknown B0 tie rule {tie!r}")
    return {"threshold": chosen, "der": best, "ties": tied}


def select_matched_threshold(points: Sequence[ThresholdPoint], b0_mr_dsdl: float) -> dict:
    """A mitigation's operating point (§3): the grid threshold whose dev MR_DS-DL is CLOSEST to B0's dev
    MR_DS-DL; ties (equal |MR - B0_MR|) go to the LOWER dev FAR_SS-DL.

    Only thresholds with a finite MR_DS-DL and FAR_SS-DL are candidates. A further tie on FAR goes to the lower
    threshold (a last deterministic fallback, same spirit as B0_TIE). Returns ``{"threshold", "mr_dsdl",
    "far_ssdl", "distance"}``. Raises `InsufficientData` when no threshold has a finite MR (degenerate).
    """
    from math import isfinite
    cand = [p for p in points if isfinite(p.mr_dsdl) and isfinite(p.far_ssdl)]
    if not cand:
        raise InsufficientData("mitigation: no threshold has a finite dev MR_DS-DL and FAR_SS-DL (degenerate)")
    if not isfinite(b0_mr_dsdl):
        raise InsufficientData("mitigation: B0 dev MR_DS-DL is not finite; cannot match")
    # closest MR, then lower FAR, then lower threshold
    best = min(cand, key=lambda p: (abs(p.mr_dsdl - b0_mr_dsdl), p.far_ssdl, p.threshold))
    return {"threshold": best.threshold, "mr_dsdl": best.mr_dsdl, "far_ssdl": best.far_ssdl,
            "distance": abs(best.mr_dsdl - b0_mr_dsdl)}


def assess_m2_tunable(dsdl_turns_per_session: dict,
                      min_sessions: int = ASSESS_MIN_SESSIONS,
                      min_turns: int = ASSESS_MIN_DSDL_PER_SESSION) -> dict:
    """§3 M2 assessability: alpha can be frozen only when at least `min_sessions` dev sessions have at least
    `min_turns` DS-DL turns each. (For the P0 gate this is restricted to two-language recordings; for the P1-AHC
    post-processor the brief scopes us to all dev sessions.)

    `dsdl_turns_per_session` maps session -> its DS-DL turn count. Returns ``{"tunable": bool, "n_qualifying":
    k, "qualifying": [sessions], "min_sessions": .., "min_turns": ..}``. Pure and model-free.
    """
    qualifying = sorted(s for s, n in dsdl_turns_per_session.items() if n >= min_turns)
    return {"tunable": len(qualifying) >= min_sessions, "n_qualifying": len(qualifying),
            "qualifying": qualifying, "min_sessions": min_sessions, "min_turns": min_turns}


def _per_session_delta_mr(alpha_curve: SystemCurve, b0_point: ThresholdPoint) -> dict:
    """Per-session deltaMR_DS-DL (alpha system minus B0) at the SAME threshold (B0's frozen threshold, §3/§4), one
    value per session present in BOTH the alpha point's and B0's included-MR session sets. The intersection is
    exercised in `tests/test_tune_m2b.py` with asymmetric session sets (review m2).
    """
    # M2 runs at B0's threshold (M2_AT_B0_THRESHOLD), so the alpha curve has a single ThresholdPoint at that
    # threshold; align session-wise with B0's per-session MR.
    assert len(alpha_curve.points) == 1, "M2 alpha curve must hold exactly one threshold point (B0's)"
    ap = alpha_curve.points[0]
    out = {}
    for s, mr in ap.per_session_mr_dsdl.items():
        if s in b0_point.per_session_mr_dsdl:
            out[s] = mr - b0_point.per_session_mr_dsdl[s]
    return out


def select_m2_alpha(alpha_curves: dict, b0_point: ThresholdPoint, rng,
                    bound_pp: float = MR_BOUND_PP, level: float = BCA_ONE_SIDED_LEVEL,
                    n_boot: int = BCA_N_BOOT) -> dict:
    """M2's alpha (§3, the P1-AHC post-processor): minimise dev FAR_SS-DL subject to the UPPER one-sided
    `level` (80%) BCa bound of per-session deltaMR_DS-DL (vs B0) being at most `bound_pp` (+1 pp). Ties on FAR go
    to the SMALLER alpha, then the lower dev MR_DS-DL. alpha = 0 always qualifies (deltaMR = 0).

    `alpha_curves` maps alpha -> a `SystemCurve` with exactly one `ThresholdPoint` (at B0's frozen threshold).
    `b0_point` is B0's `ThresholdPoint` at that same threshold. `rng` is a numpy Generator (seed 20261018). The
    one-sided 80% UPPER bound is read off `stats.bca_ci` at a two-sided level 2*level - 1 = 0.60 (its upper
    endpoint is the one-sided 80% bound), resampling whole sessions -- exactly §3's "one-sided 80% BCa bound
    over dev sessions (10,000 whole-session resamples)".

    Returns ``{"alpha", "far_ssdl", "mr_dsdl", "feasible": [...], "per_alpha": {alpha: {...}}}``. Never empty:
    alpha 0 is feasible by construction.
    """
    from math import isfinite

    def mean_stat(xs):
        xs = [x for x in xs if x == x]
        return sum(xs) / len(xs) if xs else float("nan")

    two_sided = 2.0 * level - 1.0                      # 0.80 one-sided upper == 0.60 two-sided upper endpoint
    per_alpha = {}
    feasible = []
    for alpha in sorted(alpha_curves):
        curve = alpha_curves[alpha]
        pt = curve.points[0]
        deltas = _per_session_delta_mr(curve, b0_point)
        clusters = [[v] for v in deltas.values()]      # bca_ci resamples whole clusters; one session = one cluster
        if alpha == 0.0 and ALPHA_ZERO_ALWAYS_FEASIBLE:
            # alpha 0 == B0: deltaMR identically 0, bound = 0 <= bound_pp. Always feasible (§3), no resampling.
            # The stub carries the SAME keys bca_ci returns (review m1), so downstream code reading bca["a"] /
            # bca["z0"] on the alpha-0 row never KeyErrors.
            bound = 0.0
            ci = {"estimate": 0.0, "lo": 0.0, "hi": 0.0, "level": two_sided, "z0": float("nan"),
                  "a": float("nan"), "n_boot": 0, "n_dropped": 0}
        elif len(clusters) >= 2:
            ci = bca_ci(clusters, lambda cs: mean_stat([c[0] for c in cs]), two_sided, rng, n_boot=n_boot)
            bound = ci["hi"]
        else:
            ci = {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan"), "level": two_sided,
                  "z0": float("nan"), "a": float("nan"), "n_boot": 0, "n_dropped": 0}
            bound = float("nan")
        # alpha 0 is feasible by construction (§3); every other alpha must clear the bound (review m1: single,
        # non-dead feasibility assignment).
        ok = (alpha == 0.0 and ALPHA_ZERO_ALWAYS_FEASIBLE) or (isfinite(bound) and bound <= bound_pp + 1e-12)
        per_alpha[alpha] = {"far_ssdl": pt.far_ssdl, "mr_dsdl": pt.mr_dsdl,
                            "delta_mr_upper_bound": bound, "n_sessions": len(clusters),
                            "bca": ci, "feasible": ok}
        if ok:
            feasible.append(alpha)

    # minimise FAR; ties -> smaller alpha; then lower MR
    def key(a):
        p = per_alpha[a]
        far = p["far_ssdl"] if isfinite(p["far_ssdl"]) else float("inf")
        mr = p["mr_dsdl"] if isfinite(p["mr_dsdl"]) else float("inf")
        return (far, a, mr)

    chosen = min(feasible, key=key)
    return {"alpha": chosen, "far_ssdl": per_alpha[chosen]["far_ssdl"],
            "mr_dsdl": per_alpha[chosen]["mr_dsdl"], "feasible": sorted(feasible), "per_alpha": per_alpha}


def p0_gate_alpha(*_args, **_kwargs):
    """The P0-with-E3 gate alpha (§3, §6) is tuned by `csdiar.analysis.tune_p0_m2b`, not here. The P0 gate acts on
    a different score scale with its own tau0 grid and is recorded into FREEZE_H4.json for the D1 stage-2 test; it
    is not part of this P1-AHC dev tuner, so calling this is always an error."""
    raise NotImplementedYet(
        "The P0-with-E3 gate alpha (PREREGISTRATION_M2B.md §3, §6) is tuned by csdiar.analysis.tune_p0_m2b; "
        "this tuner freezes the P1-AHC post-processor alpha only.")


# ===================================================================================================== #
# 4. counts / flags for the record (PREREGISTRATION_M2B.md §3, §7)
# ===================================================================================================== #
def dev_counts(summary: dict) -> dict:
    """Pull the dev counts the record and the assessability rules need from `ingest_displace`'s summary.json
    (§3 'report the counts either way, including how many dev recordings have exactly two languages').

    `summary` is the parsed summary.json. Returns per-recording DS-DL turn counts, language counts, the number of
    two-language recordings, and the session-level DS-DL>=10 tally. Pure.
    """
    recs = summary.get("recordings", [])
    dsdl_per_session = {r["recording"]: int(r.get("n_dsdl", 0)) for r in recs}
    n_languages = {r["recording"]: len(r.get("languages", [])) for r in recs}
    two_lang = sorted(r["recording"] for r in recs if len(r.get("languages", [])) == 2)
    n_sessions_ge10_dsdl = sum(1 for n in dsdl_per_session.values() if n >= ASSESS_MIN_DSDL_PER_SESSION)
    return {"dsdl_per_session": dsdl_per_session, "n_languages_per_session": n_languages,
            "two_language_recordings": two_lang, "n_two_language_recordings": len(two_lang),
            "n_recordings": len(recs), "n_sessions_ge10_dsdl": n_sessions_ge10_dsdl}


# ===================================================================================================== #
# 4b. per-session record at an operating point, and per-mitigation paired-turn counts vs B0
#     (the dev inputs power_m2b's H5 simulation reads; review m1 and the old interface gap)
# ===================================================================================================== #
# `curve_from_scores` keeps only the per-session MR on each ThresholdPoint. The H5 sizing simulation
# (`power_m2b`) needs richer per-session dev values at the frozen operating points -- the session's MR_DS-DL,
# cell-standardised delta-FAR, FAR_SS-DL, its H4/H5 inclusion unit counts -- plus, per mitigation vs B0, the
# per-session DISCORDANT paired-turn counts that give the within-session variance of the per-session difference
# (review M1/M2). These helpers read the SessionScore objects and the hypothesis changes the sweep already
# produced; they add fields to the record and NEVER touch a selection result.
def per_session_from_scores(score_result: dict, inclusion_ssdl: Sequence[str],
                            inclusion_mr: Sequence[str]) -> dict:
    """Per-session dev values at one operating point, from a `score_m2b.score_system` result.

    Returns ``{recording: {mr_dsdl, delta_far_std, far_ssdl, n_dsdl, n_ssdl_with_cell_controls}}`` over EVERY
    scored session (the H4/H5 inclusion lists are reported separately, so a session below the floor is still
    visible), plus ``_inclusion`` = {"h4_ssdl": [...], "h5_dsdl": [...]} (the §7 inclusion lists this operating
    point produced). Pure; reads the `SessionScore` dataclass fields only.
    """
    sessions = score_result["sessions"]
    per_session = {
        rid: {"mr_dsdl": s.mr_dsdl, "delta_far_std": s.delta_far_std, "far_ssdl": s.far_ssdl,
              "n_dsdl": int(s.n_dsdl), "n_ssdl_with_cell_controls": int(s.n_ssdl_with_cell_controls)}
        for rid, s in sessions.items()
    }
    return {"per_session": per_session,
            "inclusion": {"h4_ssdl": sorted(inclusion_ssdl), "h5_dsdl": sorted(inclusion_mr)}}


def paired_turn_counts_vs_b0(ds_pool_by_rec: dict, b0_changes_by_rec: dict,
                             mit_changes_by_rec: dict, c: float = score_m2b.C) -> dict:
    """Per-session DS-DL paired-turn counts for one mitigation vs B0, matching the FULL DS pool as
    `score_m2b.score_session` does, then tallying only the DS-DL turns (review M1).

    ``ds_pool_by_rec`` maps a recording to ``{"times": [...], "n_dsdl": k}`` where ``times`` is the FULL DS pool
    (DS-DL turns FIRST, then DS-SL turns) and ``n_dsdl`` is how many of them are DS-DL -- exactly the ordering
    `score_session` builds (``ds_times = dsdl_rows + dssl_rows``) and the index convention it uses (a DS-DL turn
    is index ``i < n_dsdl``). For each recording the full pool is matched ONCE against B0's hypothesis changes
    and once against the mitigation's, under the same one-to-one collar matching `score_m2b.match_turns` uses, so
    a DS-DL turn counts as MATCHED only if it WON the collar competition that the DS-SL turns also entered --
    the same quantity `score_session`'s ``mr_dsdl`` is computed from. The missed/matched tally is then restricted
    to the DS-DL indices (``i < n_dsdl``), so the per-system missed count over ``n_dsdl`` equals that system's
    MR_DS-DL. A DS-DL turn can be "missed" because a nearby DS-SL turn took the change -- which the old DS-DL-only
    matching never saw, biasing the within-session noise (review M1's trace).

    Returns ``{recording: {n_dsdl, missed_by_mit_only, missed_by_b0_only, missed_by_both, matched_by_both}}``
    over the recordings present in `ds_pool_by_rec`. `score_m2b.match_turns` is REUSED verbatim -- `score_m2b.py`
    is never edited.
    """
    out: dict = {}
    for rid, pool in ds_pool_by_rec.items():
        ds_times = pool["times"]
        n_dsdl = int(pool["n_dsdl"])
        b0_changes = b0_changes_by_rec.get(rid, [])
        mit_changes = mit_changes_by_rec.get(rid, [])
        # match the FULL DS pool one-to-one (as score_session does); the DS-SL turns compete for the same changes
        b0_matched = {p[0] for p in score_m2b.match_turns(ds_times, b0_changes, c)["pairs"]}
        mit_matched = {p[0] for p in score_m2b.match_turns(ds_times, mit_changes, c)["pairs"]}
        # restrict the tally to the DS-DL turns (indices i < n_dsdl), so it equals MR_DS-DL
        dsdl_idx = set(range(n_dsdl))
        b0_missed = dsdl_idx - b0_matched
        mit_missed = dsdl_idx - mit_matched
        out[rid] = {"n_dsdl": n_dsdl,
                    "missed_by_mit_only": len(mit_missed - b0_missed),
                    "missed_by_b0_only": len(b0_missed - mit_missed),
                    "missed_by_both": len(b0_missed & mit_missed),
                    "matched_by_both": len((dsdl_idx & b0_matched) & mit_matched)}
    return out


def _ds_turn_times_by_recording(rows_by_rec: dict) -> dict:
    """The FULL DS pool per recording for `paired_turn_counts_vs_b0`, in `score_session`'s order (review M1).

    MR_DS-DL is tallied over DS-DL turns, but `score_session` forms the match by competing the FULL DS pool
    (``dsdl_rows + dssl_rows``) one-to-one against the changes, so a DS-DL turn is matched only if it beat the
    nearby DS-SL turns for a change. To reproduce that exactly here we return the WHOLE pool with the DS-DL
    turns first (matching ``ds_times = dsdl_rows + dssl_rows``), plus ``n_dsdl`` so the caller knows the DS-DL
    turns are indices ``i < n_dsdl``. Returns ``{recording: {"times": [...], "n_dsdl": k}}``.
    """
    out: dict = {}
    for rid, rows in rows_by_rec.items():
        dsdl = [float(r["t"]) for r in rows if r.get("kind") == "dsdl" and r.get("t") not in (None, "")]
        dssl = [float(r["t"]) for r in rows if r.get("kind") == "dssl" and r.get("t") not in (None, "")]
        out[rid] = {"times": dsdl + dssl, "n_dsdl": len(dsdl)}
    return out


# ===================================================================================================== #
# 5. hashing / IO helpers
# ===================================================================================================== #
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def self_sha256() -> str:
    """sha256 of this module's own source (written into the record, §11 logging)."""
    return sha256_file(_repo_root() / GUARDED_SELF)


def read_reader_rows(units_dir, split: str) -> dict:
    """Load `ingest_displace`'s per-split rows (ssdl / turns / ctrl CSVs) grouped by recording, for `score_m2b`.

    GUARD (§3): refuses any split other than 'dev' -- tuning reads dev only, nothing may read eval. Returns
    ``{recording: [row dict, ...]}`` with every reader column preserved.
    """
    if split != "dev":
        raise ValueError(f"tune_m2b reads the DEV split only (PREREGISTRATION_M2B.md §3); refusing split "
                         f"{split!r}. No eval recording is read before the freeze.")
    units_dir = Path(units_dir)
    rows_by_rec: dict = defaultdict(list)
    for kind in ("ssdl", "turns", "ctrl"):
        path = units_dir / f"{kind}_{split}.csv.gz"
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows_by_rec[row["recording"]].append(row)
    return dict(rows_by_rec)


# ===================================================================================================== #
# 6. ORCHESTRATION (needs the runner + a Community-1 cache)
# ===================================================================================================== #
@dataclass
class TuningInputs:
    """Everything `run_dev_tuning` needs, resolved by the CLI (or a test). Paths are dev-only."""
    units_dir: Path                 # ingest_displace output for dev (ssdl/turns/ctrl csv.gz, summary.json)
    cache_dir: Path                 # run_m2b stage-1 + LID caches
    wav_dir: Path                   # dev wavs (B2/M2 embed their output segments)
    out_path: Path                  # output JSON
    ref_rttm_dir: Optional[Path] = None   # dev reference speaker RTTMs (for DER); <ref_rttm_dir>/<rid>_SPEAKER.rttm
    lang_rttm_dir: Optional[Path] = None  # dev reference language RTTMs; <lang_rttm_dir>/<rid>_LANGUAGE.rttm (§2)
    fits_dir: Optional[Path] = None       # where the fitted B1 npz / M1 npy are written (default: out's dir)
    model: str = "e3"
    m3_checkpoints: tuple = ()      # m3:<ckpt> ids (the 5 all-data seeds)
    means_npz: Optional[Path] = None   # pre-fit B1 means (else fit on dev here)
    v_hat_npy: Optional[Path] = None   # pre-fit M1 v (else fit on dev here)
    recordings: tuple = ()          # dev recording ids (default: discovered from the units summary)
    pair_by_recording: dict = field(default_factory=dict)   # rec -> (code_a, code_b) reference pair


def plan(inputs: TuningInputs) -> dict:
    """The dry-run plan: what would be fit, swept and selected, and on which inputs -- without running anything.

    Checks the dev inputs exist and reads the dev summary for the recording list, language pairs and counts.
    Raises if asked to touch eval (via `read_reader_rows`'s guard, which the caller uses on the real run).
    """
    units = Path(inputs.units_dir)
    summary_path = units / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"plan: dev summary.json not found at {summary_path}")
    summary = json.loads(summary_path.read_text())
    if summary.get("split") != "dev":
        raise SystemExit(f"plan: summary.json is split {summary.get('split')!r}, expected 'dev' (dev-only tuning)")
    counts = dev_counts(summary)
    assess = assess_m2_tunable(counts["dsdl_per_session"])
    systems = ["B0", "B1", "M1", "B2"] + [f"M3:{c}" for c in inputs.m3_checkpoints] + ["M2"]
    return {
        "split": "dev",
        "model": inputs.model,
        "n_recordings": counts["n_recordings"],
        "n_two_language_recordings": counts["n_two_language_recordings"],
        "systems": systems,
        "threshold_grid": list(THRESHOLD_GRID),
        "alpha_grid": list(ALPHA_GRID),
        "m2_assessable": assess,
        "fits": {
            "b1_means": "fit on dev" if inputs.means_npz is None else f"load {inputs.means_npz}",
            "m1_v_hat": "fit on dev" if inputs.v_hat_npy is None else f"load {inputs.v_hat_npy}",
            "b2_delta": "fit on dev from SegmentEmbedder cosines",
        },
        "choices": _open_choices(),
        "outputs": str(inputs.out_path),
        "p0_gate": "tuned separately by csdiar.analysis.tune_p0_m2b",
        "counts": counts,
    }


def _open_choices() -> dict:
    return {"aggregation_far_mr": AGGREGATION, "b0_der": "pooled_corpus_pyannote", "b0_tie": B0_TIE,
            "m2_at_b0_threshold": M2_AT_B0_THRESHOLD, "m1_sign": M1_SIGN, "b2_pairing": B2_PAIRING,
            "b2_max_pairs_per_unit": B2_MAX_PAIRS_PER_UNIT,
            "alpha_zero_always_feasible": ALPHA_ZERO_ALWAYS_FEASIBLE}


def _ref_rttm_path(inputs: TuningInputs, rid: str) -> Optional[Path]:
    """Dev reference speaker RTTM for DER scoring: <ref_rttm_dir>/<rid>_SPEAKER.rttm (§2 layout). None if the dir
    was not supplied (the DER columns are then left out of the record and flagged)."""
    if inputs.ref_rttm_dir is None:
        return None
    return Path(inputs.ref_rttm_dir) / f"{rid}_SPEAKER.rttm"


def _lang_rttm_path(inputs: TuningInputs, rid: str) -> Optional[Path]:
    """Dev reference language RTTM for the M1/B2 resolver: <lang_rttm_dir>/<rid>_LANGUAGE.rttm (§2 layout). None
    when the dir was not supplied (a real run then has no resolver and M1/B2 are unfittable)."""
    if inputs.lang_rttm_dir is None:
        return None
    return Path(inputs.lang_rttm_dir) / f"{rid}_LANGUAGE.rttm"


def _hash_inputs(inputs: TuningInputs, recordings: Sequence[str], fitted_paths: dict) -> dict:
    """sha256 EVERY file that feeds the fits and the sweep (§3 'input file hashes'; §11; review M3).

    Covers: the reader outputs (ssdl/turns/ctrl csv.gz, summary.json), the reference speaker RTTMs, the stage-1
    caches per model (E3 + each M3 checkpoint), the LID caches, the dev wavs, the M3 checkpoints, and the fitted
    B1 npz / M1 npy. Two runs that differ only in one cache's embeddings (a re-extracted stage-1 cache) therefore
    produce DIFFERENT records (the test pins this), closing review M3.
    """
    from csdiar.pipelines import run_m2b as R

    hashes: dict = {}
    units = Path(inputs.units_dir)
    # reader outputs
    for name in ("ssdl_dev.csv.gz", "turns_dev.csv.gz", "ctrl_dev.csv.gz", "summary.json"):
        p = units / name
        if p.exists():
            hashes[str(p)] = sha256_file(p)
    # reference speaker RTTMs (DER) and reference language RTTMs (M1/B2 resolver): both feed the fits (review M1)
    for rid in recordings:
        rp = _ref_rttm_path(inputs, rid)
        if rp is not None and Path(rp).exists():
            hashes[str(rp)] = sha256_file(rp)
        lp = _lang_rttm_path(inputs, rid)
        if lp is not None and Path(lp).exists():
            hashes[str(lp)] = sha256_file(lp)
    # stage-1 caches per model (E3 + each M3 checkpoint) and the LID caches, and the wavs
    models = [inputs.model] + [c for c in inputs.m3_checkpoints]
    for rid in recordings:
        for model in models:
            try:
                sp = R.stage1_cache_path(inputs.cache_dir, model, rid)
            except Exception:
                sp = None
            if sp is not None and Path(sp).exists():
                hashes[str(sp)] = sha256_file(sp)
        try:
            lp = R.lid_cache_path(inputs.cache_dir, rid)
        except Exception:
            lp = None
        if lp is not None and Path(lp).exists():
            hashes[str(lp)] = sha256_file(lp)
        for ext in (".wav", ".flac"):
            wp = Path(inputs.wav_dir) / f"{rid}{ext}"
            if wp.exists():
                hashes[str(wp)] = sha256_file(wp)
    # M3 checkpoints themselves
    for c in inputs.m3_checkpoints:
        cp = Path(c[3:] if str(c).startswith("m3:") else c).expanduser()
        if cp.exists() and cp.is_file():
            hashes[str(cp)] = sha256_file(cp)
    # the fitted artefacts the runner reads (B1 npz, M1 npy)
    for p in fitted_paths.values():
        if p is not None and Path(p).exists():
            hashes[str(p)] = sha256_file(p)
    return hashes


# --- default fit-input builders (injectable so the orchestration is testable through fakes) --------------
def _default_build_b1_slot_embeddings(inputs, recordings, caches, lids, pairs):
    """B1 fit inputs: per `pair`-index language, the stage-1 slot embeddings `slot_languages` assigns to it,
    pooled over dev recordings. Returns {pair_index: [ (dim,) vectors ]} for `fit_b1_means`.

    Keyed by the pair INDEX (not the code) so the fitted means line up with what the runner's `b1_transform`
    reads (`slot_languages` labels slots by pair index). Uses the real `run_m2b.slot_languages`.
    """
    import numpy as np
    from csdiar.pipelines import run_m2b as R
    by_index: dict = defaultdict(list)
    for rid in recordings:
        cache = caches[rid]
        lid = lids[rid]
        pair = pairs[rid]
        slot_langs = R.slot_languages(cache, lid["post"], pair)
        emb = np.asarray(cache["embeddings"], dtype=np.float64)   # (chunks, speakers, dim)
        C, S = slot_langs.shape
        for c in range(C):
            for p in range(S):
                lang = int(slot_langs[c, p])
                if lang < 0:
                    continue
                row = emb[c, p]
                if np.isfinite(row).all():
                    by_index[lang].append(row)
    return dict(by_index)


def _default_build_m1_units(inputs, recordings, caches, lids, pairs, refs):
    """M1 fit inputs: one (reference speaker, recording) unit per speaker-recording, with `by_lang` = the mean
    STAGE-1 SLOT embedding of that speaker's slots in each reference language (so M1's v is fit in the very space
    the row-wise projector acts on). Returns a list of units for `fit_m1_v_hat`.

    A stage-1 slot counts ONLY when the reference resolver gives an answer over the slot's active span (review
    M1): both the slot's SPEAKER and its LANGUAGE come from the references (never from LID), via
    ``resolver.resolve(lo, hi)`` on the slot's active frame span. A slot whose active span straddles a reference
    speaker change, a language change, an overlap or an ambiguous interval gets no answer and is dropped (review
    m3's exclusion-zone attribution). `refs` maps rid -> a `DisplaceRefResolver` (or any object with a matching
    ``resolve`` method). When `refs` is None the units cannot be built and an empty list is returned (the caller
    flags M1 as unfittable).
    """
    import numpy as np
    from csdiar.pipelines import run_m2b as R
    if refs is None:
        return []
    units = []
    for rid in recordings:
        cache = caches[rid]
        lid = lids[rid]
        pair = pairs[rid]
        slot_langs = R.slot_languages(cache, lid["post"], pair)
        times = R.slot_frame_times(cache)                       # (C, S, F) abs frame MID times, NaN inactive
        # m2: slot_frame_times returns frame MIDPOINTS, so the min/max mid span omits rf_duration/2 of real
        # audio before the first mid and after the last mid. Attribute each slot over its TRUE acoustic extent
        # -- first active frame's start to last active frame's end -- by widening the midpoint span by half a
        # frame's duration on each side. Read rf_duration from the stage-1 cache exactly as slot_frame_times
        # does (never hard-coded), so a slot whose widened extent crosses a reference language/speaker change is
        # correctly dropped by resolve().
        half_frame = float(cache["rf_duration"]) / 2.0
        emb = np.asarray(cache["embeddings"], dtype=np.float64)
        resolver = refs[rid]
        C, S = slot_langs.shape
        # accumulate per (reference speaker, reference language) the slot embeddings for this recording
        acc: dict = defaultdict(list)
        for c in range(C):
            for p in range(S):
                if int(slot_langs[c, p]) < 0:
                    continue                                    # inactive / unlabelled slot: no active span
                ft = times[c, p]
                ft = ft[np.isfinite(ft)]
                if ft.size == 0:
                    continue
                lo = float(ft.min()) - half_frame               # first active frame's START (mid - rf_dur/2)
                hi = float(ft.max()) + half_frame               # last active frame's END   (mid + rf_dur/2)
                ans = resolver.resolve(lo, hi)                  # (speaker, language) from the REFERENCES, or None
                if ans is None:
                    continue                                    # span not inside one clean (speaker, lang) run
                spk, code = ans
                row = emb[c, p]
                if np.isfinite(row).all():
                    acc[(spk, code)].append(row)
        # group by speaker into a unit with by_lang means
        by_spk: dict = defaultdict(dict)
        for (spk, code), vecs in acc.items():
            by_spk[spk][code] = np.mean(np.asarray(vecs, dtype=np.float64), axis=0)
        for spk, by_lang in by_spk.items():
            units.append({"speaker": spk, "recording": rid, "by_lang": by_lang})
    return units


def _default_build_b2_cosines(inputs, recordings, caches, lids, pairs, refs, embedders,
                              max_pairs_per_unit: int = B2_MAX_PAIRS_PER_UNIT):
    """B2 fit inputs: same-speaker SAME-language vs same-speaker CROSS-language SEGMENT cosines, pooled over dev
    recordings (B2_PAIRING). Returns (same_lang_cosines, cross_lang_cosines) for `fit_b2_delta`.

    The clean single-speaker single-language spans come straight from the reference resolver
    (`refs[rid].clean_spans()` -> ``(speaker, start, end, language)``), so BOTH the speaker and the language of
    every segment are the REFERENCES' (review m3: the language is the reference language of the span, never LID).
    Each span is embedded once with the system's own `SegmentEmbedder` (`embedders[rid]`); all within-speaker
    segment pairs contribute a cosine to the same- or cross-language pool.

    `max_pairs_per_unit` (`B2_MAX_PAIRS_PER_UNIT`, a listed choice) deterministically caps the pairs per
    (speaker, recording): the spans are embedded in SORTED (start, end) order, pairs enumerated in that order,
    and when the cap is exceeded a shuffle of the sorted pair list is truncated. The shuffle's generator is
    seeded per (recording, speaker) from a STABLE key -- ``np.random.default_rng([SEED, crc32(rid), crc32(spk)])``
    -- so a unit's sampled pairs depend only on its own id, NOT on how many pairs earlier recordings drew (m5).
    Python's salted `hash()` is deliberately avoided. 0 means take every pair. When `refs` or `embedders` is None
    the pools cannot be built and ([], []) is returned (the caller flags B2 as unfittable).
    """
    import numpy as np
    if refs is None or embedders is None:
        return [], []
    same: list = []
    cross: list = []
    for rid in recordings:
        resolver = refs[rid]
        embedder = embedders[rid]
        spans = resolver.clean_spans() if hasattr(resolver, "clean_spans") else None
        if not spans:
            continue
        # embed each clean (speaker, start, end, lang) span once, in sorted (start, end) order for determinism
        by_spk: dict = defaultdict(list)
        for (spk, start, end, code) in sorted(spans, key=lambda s: (s[1], s[2])):
            vec = embedder.embed(start, end)
            if vec is None:
                continue
            by_spk[spk].append((code, np.asarray(vec, dtype=np.float64)))
        for spk, items in by_spk.items():
            pair_idx = [(a, b) for a in range(len(items)) for b in range(a + 1, len(items))]
            if max_pairs_per_unit and len(pair_idx) > max_pairs_per_unit:
                # m5: a STABLE per-(recording, speaker) seed, so this unit's sample is independent of the rest of
                # the corpus (adding/removing/reordering other recordings never shifts it). zlib.crc32 is a
                # stable, process-independent hash; Python's salted hash() must NOT be used here.
                rng = np.random.default_rng([SEED, zlib.crc32(str(rid).encode("utf-8")),
                                             zlib.crc32(str(spk).encode("utf-8"))])
                order = rng.permutation(len(pair_idx))[:max_pairs_per_unit]
                pair_idx = [pair_idx[k] for k in sorted(order)]
            for a, b in pair_idx:
                ca, va = items[a]
                cb, vb = items[b]
                na = float(np.linalg.norm(va)); nb = float(np.linalg.norm(vb))
                if na <= 0 or nb <= 0:
                    continue
                cos = float(va @ vb) / (na * nb)
                (same if ca == cb else cross).append(cos)
    return same, cross


def run_dev_tuning(inputs: TuningInputs, *, runner=None, scorer=None, der_components_fn=None,
                   lid_loader=None, cache_loader=None, pipe_loader=None,
                   embedder_factory=None, ref_resolver_factory=None,
                   build_b1=None, build_m1=None, build_b2=None,
                   fit_hooks: Optional[dict] = None, write: bool = True) -> dict:
    """Drive the whole dev tuning and write one JSON (PREREGISTRATION_M2B.md §3, §4, §5).

    The heavy collaborators are injected so the orchestration is TESTABLE with `run_system_from_cache` and the
    scorers monkeypatched (the tests pass fakes for `runner`, `scorer`, `der_components_fn`, caches, embedders
    and the fit-input builders):
      * `runner(pipe, cache, threshold, system, **kw)` -> (overlapping, exclusive, info) -- defaults to
        `run_m2b.run_system_from_cache`;
      * `scorer(rows_by_session, changes_by_recording, durations, ...)` -> `score_m2b.score_system` output --
        defaults to `score_m2b.score_system`;
      * `der_components_fn(ref_rttm, hyp_rttm)` -> {error, total, ...} -- defaults to `der_components`;
      * `lid_loader(cache_dir, rid)` / `cache_loader(cache_dir, model, rid)` / `pipe_loader(model, th)` /
        `embedder_factory(model, rid, wav)` -- default to the `run_m2b` loaders;
      * `build_b1` / `build_m1` / `build_b2` -- the fit-input builders above (default to the `_default_build_*`).

    A real run: reads the manifest; builds the fits from the caches/references and runs them (writing the B1 npz
    and M1 npy the runner reads, with their sha256 in the record); sweeps B0/B1/M1/B2/M3 over the threshold grid;
    selects B0; sweeps M2's alpha at B0's threshold; scores every output (FAR/MR from the exclusive annotation,
    pooled DER components from the overlapping annotation against the reference speaker RTTM); builds the curves,
    runs every selection/assessability rule, and writes `curves`, `operating_points`, `fits`, the alpha table,
    counts and flags. On a degenerate system it records "not assessable" rather than raising, so one bad system
    does not sink the whole record (§7).
    """
    import numpy as np
    from csdiar.pipelines import run_m2b as R

    runner = runner or R.run_system_from_cache
    scorer = scorer or score_m2b.score_system
    der_components_fn = der_components_fn or der_components
    lid_loader = lid_loader or R.load_lid
    cache_loader = cache_loader or R.load_stage1
    pipe_loader = pipe_loader or R.load_stage2_pipeline
    embedder_factory = embedder_factory or (lambda model, rid, wav, device="cpu": R.SegmentEmbedder(
        model, rid, wav, device=device))
    build_b1 = build_b1 or _default_build_b1_slot_embeddings
    build_m1 = build_m1 or _default_build_m1_units
    build_b2 = build_b2 or _default_build_b2_cosines
    fit_hooks = fit_hooks or {}

    rng = np.random.default_rng(SEED)
    units = Path(inputs.units_dir)
    summary = json.loads((units / "summary.json").read_text())
    if summary.get("split") != "dev":
        raise ValueError("run_dev_tuning: summary.json is not the dev split (dev-only tuning, §3)")
    if inputs.ref_rttm_dir is None:
        raise ValueError("run_dev_tuning needs ref_rttm_dir: B0's threshold is the lowest pooled dev DER (§3)")
    counts = dev_counts(summary)
    rows_by_rec = read_reader_rows(units, "dev")        # guard: refuses eval
    recordings = list(inputs.recordings) or sorted(rows_by_rec)

    # m5: an included recording MUST have a finite duration; otherwise changes/min is nan and would silently
    # degrade. m4: the summary must carry `duration_s` in SECONDS -- an `hours` fallback would silently read an
    # hours value as seconds (a ~60x unit error with no flag), so a summary that carries `hours` but no
    # `duration_s` is REFUSED rather than mis-read. With no summary duration at all, fall back to the reader rows.
    hours_only = [r["recording"] for r in summary.get("recordings", [])
                  if "duration_s" not in r and "hours" in r]
    if hours_only:
        raise ValueError(f"run_dev_tuning: {len(hours_only)} dev recording(s) carry only `hours`, not "
                         f"`duration_s`: {sorted(hours_only)}. Durations must be given in seconds as "
                         f"`duration_s` (an hours value read as seconds is a silent ~60x unit error, review m4).")
    durations = {r["recording"]: float(r.get("duration_s", 0.0) or 0.0)
                 for r in summary.get("recordings", [])}
    flags: dict = {}
    nonfinite_durations = []
    for rec in recordings:
        if durations.get(rec, 0.0) <= 0.0:
            ts = [float(x["t"]) for x in rows_by_rec.get(rec, []) if x.get("t")]
            durations[rec] = (max(ts) + 1.0) if ts else float("nan")
        if not (durations[rec] == durations[rec] and durations[rec] not in (float("inf"), float("-inf"))
                and durations[rec] > 0.0):
            nonfinite_durations.append(rec)
    if nonfinite_durations:
        # m5: flag and COUNT; refuse rather than scoring with a nan duration for an included recording.
        raise ValueError(f"run_dev_tuning: {len(nonfinite_durations)} included recording(s) have a non-finite "
                         f"or non-positive duration and cannot be scored (changes/min undefined): "
                         f"{sorted(nonfinite_durations)}. Fix the dev summary.json durations.")

    models = {inputs.model: inputs.model}

    # ----- load caches / LID grids / pairs / wavs / embedders / reference resolvers once per recording -----
    def _load_wav(rid):
        for ext in (".wav", ".flac"):
            p = Path(inputs.wav_dir) / f"{rid}{ext}"
            if p.exists():
                return R._read_wav_mono(p)
        return None

    pairs: dict = {}
    lids: dict = {}
    caches: dict = {}             # E3 (primary) stage-1 cache per recording, for the fits
    m3_caches: dict = defaultdict(dict)   # m3 ckpt -> rid -> cache
    wavs: dict = {}
    embedders: dict = {}
    pair_index: dict = {}
    for rid in recordings:
        lid = lid_loader(inputs.cache_dir, rid)
        lids[rid] = lid
        codes = inputs.pair_by_recording.get(rid)
        if codes is None:
            raise ValueError(f"no reference-language pair for {rid}; pass pair_by_recording")
        i, j = R.pair_indices(lid["labels"], codes[0], codes[1])
        pair_index[rid] = (i, j)
        pairs[rid] = (i, j)
        caches[rid] = cache_loader(inputs.cache_dir, inputs.model, rid)
        for ckpt in inputs.m3_checkpoints:
            m3_caches[ckpt][rid] = cache_loader(inputs.cache_dir, ckpt, rid)
        wav = _load_wav(rid)
        wavs[rid] = wav
        embedders[rid] = embedder_factory(inputs.model, rid, wav if wav is not None else np.zeros(1,
                                          dtype=np.float32))

    # Reference resolver (review M1): a real run builds M1 units and B2 pairs from the references. When the
    # caller does not inject a factory (the CLI never does), use the default one built on ingest_displace's
    # parser -- so a real run with both reference dirs present actually reads them instead of silently landing
    # in the unfittable branch. A run missing lang_rttm_dir has no resolver (M1/B2 recorded unfittable).
    if ref_resolver_factory is None and inputs.lang_rttm_dir is not None:
        ref_resolver_factory = default_ref_resolver_factory
    refs = None
    if ref_resolver_factory is not None:
        refs = {rid: ref_resolver_factory(inputs, rid) for rid in recordings}

    # ----- FITS -----
    fits_dir = Path(inputs.fits_dir) if inputs.fits_dir is not None else Path(inputs.out_path).parent
    fits_dir.mkdir(parents=True, exist_ok=True)
    fitted_paths: dict = {"b1_npz": None, "v_hat_npy": None}
    fits_record: dict = {}

    # B1 means (fit on dev unless a pre-fit file was supplied)
    means_by_index_by_rec: dict = {}            # rid -> {pair_index: mean vector} the runner reads for B1
    if inputs.means_npz is not None:
        # pre-fit means keyed by code; resolve to each recording's pair indices via load_means_npz
        dim = int(np.asarray(next(iter(caches.values()))["embeddings"]).shape[-1]) if caches else None
        for rid in recordings:
            codes = inputs.pair_by_recording[rid]
            code_means = R.load_means_npz(inputs.means_npz, codes[0], codes[1], dim)
            i, j = pairs[rid]
            means_by_index_by_rec[rid] = {i: code_means[codes[0]], j: code_means[codes[1]]}
        fitted_paths["b1_npz"] = Path(inputs.means_npz)
        fits_record["b1_means"] = {"source": str(inputs.means_npz)}
    elif "b1_means" in fit_hooks:
        fits_record["b1_means"] = fit_hooks["b1_means"]
    else:
        slot_emb = build_b1(inputs, recordings, caches, lids, pairs)
        try:
            b1 = fit_b1_means(slot_emb)
            fits_record["b1_means"] = {"counts": b1["counts"], "dim": b1["dim"]}
            # write the npz the runner reads: it is keyed by VoxLingua107 CODE (load_means_npz contract), so map
            # each fitted pair-index mean back to its code. A pair index not present across every recording is
            # fine -- each recording resolves its own two codes.
            code_means: dict = {}
            for rid in recordings:
                i, j = pairs[rid]
                codes = inputs.pair_by_recording[rid]
                if i in b1["means"]:
                    code_means[codes[0]] = np.asarray(b1["means"][i], dtype=np.float64)
                if j in b1["means"]:
                    code_means[codes[1]] = np.asarray(b1["means"][j], dtype=np.float64)
                means_by_index_by_rec[rid] = {k: b1["means"][k] for k in (i, j) if k in b1["means"]}
            b1_path = fits_dir / "b1_means_dev.npz"
            np.savez_compressed(b1_path, **{c: v for c, v in code_means.items()})
            fitted_paths["b1_npz"] = b1_path
        except InsufficientData as e:
            fits_record["b1_means"] = {"insufficient_data": str(e)}
            flags["b1_unfittable"] = str(e)

    # m1: the runner's B1 branch indexes BOTH pair-language means unconditionally (`means[pair[0]]`,
    # `means[pair[1]]`), so a recording whose pair language has no dev B1 mean (a language pooled corpus-wide but
    # with no slots in THIS recording) would KeyError inside the sweep. Guard it here: if any recording is
    # missing a pair-language mean, record B1 unfittable naming the language, rather than sinking the run.
    if "b1_unfittable" not in flags and inputs.means_npz is None and "b1_means" not in fit_hooks:
        for rid in recordings:
            i, j = pairs[rid]
            codes = inputs.pair_by_recording[rid]
            have = means_by_index_by_rec.get(rid, {})
            missing = [code for idx, code in ((i, codes[0]), (j, codes[1])) if idx not in have]
            if missing:
                msg = (f"B1: recording {rid!r} has no dev B1 mean for language(s) {sorted(set(missing))} "
                       f"(pair {codes}); the runner would KeyError on it")
                fits_record["b1_means"] = {"insufficient_data": msg}
                flags["b1_unfittable"] = msg
                break

    # M1 v_hat (fit on dev unless a pre-fit file was supplied)
    v_hat_by_rec: dict = {}
    if inputs.v_hat_npy is not None:
        dim = int(np.asarray(next(iter(caches.values()))["embeddings"]).shape[-1]) if caches else None
        vh = R.load_v_hat_npy(inputs.v_hat_npy, dim)
        for rid in recordings:
            v_hat_by_rec[rid] = vh
        fitted_paths["v_hat_npy"] = Path(inputs.v_hat_npy)
        fits_record["m1_v_hat"] = {"source": str(inputs.v_hat_npy)}
    elif "m1_v_hat" in fit_hooks:
        fits_record["m1_v_hat"] = fit_hooks["m1_v_hat"]
    else:
        m1_units = build_m1(inputs, recordings, caches, lids, pairs, refs)
        try:
            m1 = fit_m1_v_hat(m1_units)
            fits_record["m1_v_hat"] = {"n_units": m1["n_units"],
                                       "n_skipped_no_english": m1["n_skipped_no_english"], "dim": m1["dim"]}
            v_path = fits_dir / "m1_v_hat_dev.npy"
            np.save(v_path, np.asarray(m1["v_hat"], dtype=np.float64))
            for rid in recordings:
                v_hat_by_rec[rid] = np.asarray(m1["v_hat"], dtype=np.float64)
            fitted_paths["v_hat_npy"] = v_path
        except InsufficientData as e:
            fits_record["m1_v_hat"] = {"insufficient_data": str(e)}
            flags["m1_unfittable"] = str(e)

    # B2 delta (always fit on dev)
    b2_delta = 0.0
    if "b2_delta" in fit_hooks:
        fits_record["b2_delta"] = fit_hooks["b2_delta"]
        b2_delta = float(fit_hooks["b2_delta"].get("delta", 0.0))
    else:
        same, cross = build_b2(inputs, recordings, caches, lids, pairs, refs, embedders)
        try:
            b2 = fit_b2_delta(same, cross)
            fits_record["b2_delta"] = b2
            b2_delta = float(b2["delta"])
        except InsufficientData as e:
            fits_record["b2_delta"] = {"insufficient_data": str(e)}
            flags["b2_unfittable"] = str(e)

    record: dict = {
        "split": "dev", "model": inputs.model, "seed": SEED,
        "code_sha256": {GUARDED_SELF: self_sha256()},
        "choices": _open_choices(),
        "constants": {"threshold_grid": list(THRESHOLD_GRID), "alpha_grid": list(ALPHA_GRID),
                      "m2_tau0": M2_TAU0, "b2_delta_max": B2_DELTA_MAX, "mr_bound_pp": MR_BOUND_PP,
                      "bca_one_sided_level": BCA_ONE_SIDED_LEVEL, "bca_n_boot": BCA_N_BOOT,
                      "assess_min_sessions": ASSESS_MIN_SESSIONS,
                      "assess_min_dsdl_per_session": ASSESS_MIN_DSDL_PER_SESSION,
                      "far_primary": FAR_PRIMARY, "far_secondary": FAR_SECONDARY},
        "counts": counts,
        "fits": fits_record,
        "curves": {}, "operating_points": {}, "alpha_table": {}, "flags": flags,
        "p0_gate_alpha": "see tune_p0_m2b",
    }
    record["input_sha256"] = _hash_inputs(inputs, recordings, fitted_paths)

    # ----- helper: sweep one system over the thresholds and score every recording -----
    def sweep_system(system: str, thresholds, *, model=None, cache_map=None, delta=0.0, alpha=0.0, tau0=0.0):
        """Return (threshold_scores, der_components_by_threshold, changes_by_threshold) for one system.

        For each threshold and recording: run the system to (overlapping, exclusive), turn the exclusive
        annotation into hypothesis change times for `score_m2b`, write the two RTTMs needed for DER (overlapping
        hyp vs the reference speaker RTTM), and read the DER components. One `score_system` call per threshold
        aggregates the per-recording rows. `changes_by_threshold` maps threshold -> {recording: [change times]}
        so the per-session paired-turn counts vs B0 (review M1/M2) can be computed at the frozen operating
        points WITHOUT re-running or editing score_m2b.
        """
        model = model or inputs.model
        cache_map = cache_map or caches
        threshold_scores: dict = {}
        der_by_threshold: dict = {}
        changes_by_threshold: dict = {}
        for th in thresholds:
            changes_by_rec: dict = {}
            der_comp: dict = {}
            for rid in recordings:
                kw = dict(post=lids[rid]["post"], pair=pairs[rid])
                if system == "B1":
                    kw["means"] = means_by_index_by_rec.get(rid, {})
                if system == "M1":
                    kw["v_hat"] = v_hat_by_rec.get(rid)
                if system in ("B2", "M2"):
                    kw["embedder"] = embedders[rid]
                if system == "B2":
                    kw["delta"] = delta
                if system == "M2":
                    kw["alpha"] = alpha
                    kw["tau0"] = tau0
                diar, excl, _info = runner(pipe_loader(model, th), cache_map[rid], th, system, **kw)
                segs = R.segments_from_annotation(excl)
                changes_by_rec[rid] = score_m2b.hypothesis_changes(
                    [score_m2b.HypSegment(s, e, l) for s, e, l in segs])
                der_comp[rid] = _score_der(inputs, rid, diar, der_components_fn)
            rows_by_session = {rid: rows_by_rec.get(rid, []) for rid in recordings}
            threshold_scores[th] = scorer(rows_by_session, changes_by_rec, durations)
            der_by_threshold[th] = der_comp
            changes_by_threshold[th] = changes_by_rec
        return threshold_scores, der_by_threshold, changes_by_threshold

    # ----- PHASE 1: sweep B0, B1, M1, B2(fitted delta), each M3 over the grid; select B0 + each match -----
    curves: dict = {}
    op: dict = {}
    sweep_specs = [("B0", {}), ("B1", {}), ("M1", {}), ("B2", {"delta": b2_delta})]
    for ckpt in inputs.m3_checkpoints:
        sweep_specs.append((f"M3:{ckpt}", {"model": ckpt, "cache_map": m3_caches[ckpt]}))

    phase1_curves: dict = {}
    phase1_scores: dict = {}      # name -> threshold_scores (for the per-session record at the frozen point)
    phase1_changes: dict = {}     # name -> {threshold: {recording: [change times]}} (paired counts vs B0)
    for name, kw in sweep_specs:
        system = "M3" if name.startswith("M3:") else name
        # skip a mitigation whose fit is missing (recorded as unfittable above); B0 always runs.
        if system == "B1" and "b1_unfittable" in flags:
            op[name] = {"not_assessable": "B1 fit unavailable"}
            continue
        if system == "M1" and "m1_unfittable" in flags:
            op[name] = {"not_assessable": "M1 fit unavailable"}
            continue
        ts, dc, ch = sweep_system(system, THRESHOLD_GRID, **kw)
        curve = curve_from_scores(name, ts, dc, aggregation=AGGREGATION,
                                  param_tag=(f"delta={b2_delta:.3f}" if system == "B2" else ""))
        curves[name] = _curve_to_json(curve)
        phase1_curves[name] = curve
        phase1_scores[name] = ts
        phase1_changes[name] = ch

    # B0 selection
    b0_curve = phase1_curves.get("B0")
    b0_point = None
    if b0_curve is None or degenerate_operating_point(b0_curve.points):
        op["B0"] = {"not_assessable": "B0 has no finite dev DER (degenerate operating point)"}
        flags["b0_degenerate"] = True
    else:
        b0_sel = select_b0_threshold(b0_curve.points)
        b0_point = next(p for p in b0_curve.points if p.threshold == b0_sel["threshold"])
        op["B0"] = {"threshold": b0_sel["threshold"], "der": b0_sel["der"], "ties": b0_sel["ties"],
                    "mr_dsdl": b0_point.mr_dsdl, "far_ssdl": b0_point.far_ssdl}

    # match each mitigation / M3 to B0's dev MR
    if b0_point is not None:
        for name, curve in phase1_curves.items():
            if name == "B0":
                continue
            try:
                sel = select_matched_threshold(curve.points, b0_point.mr_dsdl)
                op[name] = {"threshold": sel["threshold"], "mr_dsdl": sel["mr_dsdl"],
                            "far_ssdl": sel["far_ssdl"], "match_distance": sel["distance"]}
            except InsufficientData as e:
                op[name] = {"not_assessable": str(e)}

    # ----- PHASE 2: sweep M2's alpha at B0's threshold; assess + select alpha -----
    assess = assess_m2_tunable(counts["dsdl_per_session"])
    record["m2_assessable"] = assess
    alpha_table: dict = {}
    if b0_point is None:
        op["M2"] = {"not_assessable": "B0 has no frozen threshold; M2 runs at it"}
    elif not assess["tunable"]:
        op["M2"] = {"not_assessable": f"only {assess['n_qualifying']} dev sessions with >= "
                                      f"{assess['min_turns']} DS-DL turns (< {assess['min_sessions']})"}
    else:
        b0_th = b0_point.threshold
        alpha_curves: dict = {}
        alpha_scores: dict = {}
        alpha_changes: dict = {}
        for alpha in ALPHA_GRID:
            ts, dc, ch = sweep_system("M2", [b0_th], alpha=alpha, tau0=M2_TAU0)
            ac = curve_from_scores(f"M2@a{alpha:.2f}", ts, dc, aggregation=AGGREGATION,
                                   param_tag=f"alpha={alpha:.2f}")
            alpha_curves[alpha] = ac
            alpha_scores[alpha] = ts
            alpha_changes[alpha] = ch
        m2_sel = select_m2_alpha(alpha_curves, b0_point, rng)
        op["M2"] = {"threshold": b0_th, "alpha": m2_sel["alpha"], "tau0": M2_TAU0,
                    "mr_dsdl": m2_sel["mr_dsdl"], "far_ssdl": m2_sel["far_ssdl"],
                    "feasible_alphas": m2_sel["feasible"]}
        alpha_table = {f"{a:.2f}": m2_sel["per_alpha"][a] for a in sorted(m2_sel["per_alpha"])}
        # the single M2 operating-point curve (at B0's threshold, chosen alpha) joins the curve record
        curves["M2"] = _curve_to_json(alpha_curves[m2_sel["alpha"]])
        # retain M2's chosen-alpha scores/changes for the per-session record (keyed by B0's threshold)
        phase1_scores["M2"] = alpha_scores[m2_sel["alpha"]]
        phase1_changes["M2"] = alpha_changes[m2_sel["alpha"]]

    record["curves"] = curves
    record["operating_points"] = op
    record["alpha_table"] = alpha_table

    # ----- per-session dev values + per-mitigation paired-turn counts at the FROZEN operating points -----
    # (review m1 / the old interface gap): write the per-session values power_m2b's H5 simulation reads, at
    # each system's chosen operating point, WITHOUT changing any selection result. These come straight from the
    # SessionScore objects and the hypothesis changes the sweeps already produced.
    ds_turn_times = _ds_turn_times_by_recording(rows_by_rec)
    per_session_record: dict = {}
    b0_changes_at_op = None
    if b0_point is not None and "B0" in phase1_scores:
        b0_th = b0_point.threshold
        b0_res = phase1_scores["B0"].get(b0_th)
        b0_changes_at_op = phase1_changes["B0"].get(b0_th, {})
        if b0_res is not None:
            per_session_record["B0"] = per_session_from_scores(
                b0_res, b0_res["inclusion"]["ssdl"]["included"], b0_res["inclusion"]["dsdl"]["included"])
    # each mitigation / M3 at its frozen operating point: per-session values + paired-turn counts vs B0
    for name in [n for n in op if n not in ("B0",)]:
        opn = op.get(name, {})
        th = opn.get("threshold")
        if th is None or name not in phase1_scores:
            continue
        res = phase1_scores[name].get(th)
        if res is None:
            # a matched threshold not in the retained sweep (shouldn't happen for grid thresholds); skip quietly
            continue
        entry = per_session_from_scores(
            res, res["inclusion"]["ssdl"]["included"], res["inclusion"]["dsdl"]["included"])
        if b0_changes_at_op is not None:
            mit_changes = phase1_changes[name].get(th, {})
            entry["paired_turns_vs_b0"] = paired_turn_counts_vs_b0(
                ds_turn_times, b0_changes_at_op, mit_changes)
        else:
            entry["paired_turns_vs_b0"] = None
            flags.setdefault("paired_turns_missing", []).append(name)
        per_session_record[name] = entry
    record["per_session_at_operating_points"] = per_session_record

    if write:
        Path(inputs.out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(inputs.out_path).write_text(json.dumps(record, indent=1, default=_json_default))
    return record


def _score_der(inputs: TuningInputs, rid: str, overlapping_annotation, der_components_fn) -> dict:
    """Write the overlapping hypothesis RTTM and score DER COMPONENTS against the reference speaker RTTM (§3/§4:
    DER on the overlapping annotation). A missing reference directory or file RAISES: B0 is chosen by the
    pooled dev DER over every dev recording, so a recording cannot be left out of it."""
    from csdiar.pipelines import run_m2b as R
    ref = _ref_rttm_path(inputs, rid)
    if ref is None:
        raise ValueError("a tuning run needs --ref-rttm-dir: B0's threshold is the lowest pooled dev DER (§3)")
    if not Path(ref).exists():
        raise FileNotFoundError(f"reference speaker RTTM for dev recording {rid!r} not found: {ref}")
    # write the overlapping hypothesis RTTM to the fits/scratch dir, one per (recording); DER is per recording.
    scratch = (Path(inputs.fits_dir) if inputs.fits_dir is not None else Path(inputs.out_path).parent) / "_der_hyp"
    scratch.mkdir(parents=True, exist_ok=True)
    hyp = scratch / f"{rid}.rttm"
    R.write_rttm(hyp, rid, R.segments_from_annotation(overlapping_annotation))
    return der_components_fn(ref, hyp)


def _curve_to_json(curve: SystemCurve) -> dict:
    """Serialise a SystemCurve into JSON-friendly primitives for the record."""
    return {
        "system": curve.system,
        "param_tag": curve.param_tag,
        "points": [{"threshold": p.threshold, "der": p.der, "far_ssdl": p.far_ssdl, "mr_dsdl": p.mr_dsdl,
                    "n_sessions_far": p.n_sessions_far, "n_sessions_mr": p.n_sessions_mr}
                   for p in curve.points],
        "det": curve.det,
    }


def _json_default(o):
    """json.dumps default: numpy scalars/arrays -> python; everything else -> float()."""
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return float(o)


# ===================================================================================================== #
# 7. CLI
# ===================================================================================================== #
def _build_inputs(args) -> TuningInputs:
    pair_by_rec = {}
    if args.pair:
        # --pair REC=a,b repeated, or a single a,b applied to every recording (resolved at run time)
        for item in args.pair:
            if "=" in item:
                rec, codes = item.split("=", 1)
                a, b = codes.split(",")
                pair_by_rec[rec] = (a.strip(), b.strip())
    return TuningInputs(
        units_dir=args.units, cache_dir=args.cache_dir, wav_dir=args.wav_dir, out_path=args.out,
        ref_rttm_dir=args.ref_rttm_dir, lang_rttm_dir=args.lang_rttm_dir, fits_dir=args.fits_dir,
        model=args.model, m3_checkpoints=tuple(args.m3 or ()),
        means_npz=args.means, v_hat_npy=args.v_hat, pair_by_recording=pair_by_rec)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--units", type=Path, required=True, help="ingest_displace DEV output dir")
    ap.add_argument("--cache-dir", type=Path, required=True, help="run_m2b stage-1 + LID caches")
    ap.add_argument("--wav-dir", type=Path, required=True, help="dev wavs (B2/M2 embed output segments)")
    ap.add_argument("--ref-rttm-dir", dest="ref_rttm_dir", type=Path, default=None,
                    help="dev reference speaker RTTMs for DER (<dir>/<rid>_SPEAKER.rttm)")
    ap.add_argument("--lang-rttm-dir", dest="lang_rttm_dir", type=Path, default=None,
                    help="dev reference LANGUAGE RTTMs for the M1/B2 resolver (<dir>/<rid>_LANGUAGE.rttm); "
                         "required for a real run -- without it M1 and B2 are recorded unfittable (§2, review M1)")
    ap.add_argument("--fits-dir", dest="fits_dir", type=Path, default=None,
                    help="where fitted B1 npz / M1 npy are written (default: --out's dir)")
    ap.add_argument("--out", type=Path, required=True, help="output JSON path")
    ap.add_argument("--model", default="e3", choices=("e1", "e2", "e3"))
    ap.add_argument("--m3", nargs="*", default=None, help="m3:<ckpt> ids (the all-data seeds)")
    ap.add_argument("--means", type=Path, default=None, help="pre-fit B1 means .npz (else fit on dev)")
    ap.add_argument("--v-hat", dest="v_hat", type=Path, default=None, help="pre-fit M1 v .npy (else fit on dev)")
    ap.add_argument("--pair", nargs="*", default=None, help="reference pair per recording: REC=a,b")
    ap.add_argument("--dry-run", action="store_true", help="check inputs + print the plan; run nothing")
    args = ap.parse_args(argv)

    inputs = _build_inputs(args)
    if args.dry_run:
        print(json.dumps(plan(inputs), indent=2, default=str))
        return 0
    rec = run_dev_tuning(inputs)
    print(json.dumps({"written": str(inputs.out_path), "systems": sorted(rec.get("curves", {})),
                      "operating_points": rec.get("operating_points", {}),
                      "p0_gate_alpha": rec["p0_gate_alpha"]}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
