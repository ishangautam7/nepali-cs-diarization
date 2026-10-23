#!/usr/bin/env python3
"""DISPLACE dev tuning of M2's P0-gate alpha and tau0, the value recorded into FREEZE_H4.json for the D1
stage-2 test (PREREGISTRATION_H4.md §2, §4, §5; PREREGISTRATION_M2B.md §2, §3, §6).

Scope (brief). This tunes **only the P0-with-E3 gate** -- the detector the D1 stage-2 test uses (H4 supplement
§4), NOT the P1-AHC post-processor that `csdiar/analysis/tune_m2b.py` freezes. The two gates act on different
score scales, so PLAN.md warns a single alpha cannot serve both (M2b §3, §6): this module freezes the P0-gate
alpha, and `tune_m2b.p0_gate_alpha` (a clearly-marked NotImplementedYet stub) is meant to point here.

The P0 detector, reused verbatim, NOT re-implemented:
  * `csdiar.analysis.pilot`: the frozen M2a P0 -- `p0_curves` (d(t) = 1 - cos(window k-10, window k) on the
    0.1 s grid), `peak_strength` (the strongest NMS peak within +-c = 0.5 s), `calibrate_tau` (tau at which 10%
    of dev controls fire, with the degenerate-case refusal PREREGISTRATION_H4.md §5 makes), and the hit rule
    "the strongest peak within +-0.5 s exceeds tau";
  * `csdiar.analysis.confirm_h4`: its stage-2 gated curve d - alpha*lambda for D1. The DISPLACE gate is LITERALLY
    `confirm_h4.system_curves(name="M2", ...)` -- the same function the D1 stage-2 test calls -- so the gate here
    is the same code as the D1 gate (brief: "reuse the same function or factor-free call path"). A
    `tests/test_tune_p0_m2b.py` test pins that the two agree on shared inputs.

lambda on the grid: `csdiar.pipelines.run_m2b.load_lid` / `pair_indices` give the full VoxLingua107 posterior
and the recording's two reference-language indices; lambda(t) = `confirm_h4.jsd` (base-2 JSD) of the
renormalised (lang_a, lang_b) posteriors of windows k-10 and k -- exactly the D1 lambda of `m2_lambda.lam`,
generalised to an explicit pair per M2B §2 (ne/en on D1, the recording's pair on DISPLACE).

The dense E3 grid of a whole DISPLACE recording is built by reusing `csdiar.models.embed` (`load`, `frames`,
`SR`): window k covers [0.1k, 0.1k + 1.0] s, the same grid P0 reads, so d(t) is P0's d(t). `device` is a
parameter (CPU-only on the acceptance box); the grid is cached per recording with the model and the code hash
of this module + embed.py in the cache, so a re-extracted grid never silently reuses a stale cache.

No DISPLACE archive is readable (encrypted): synthetic inputs only in the tests; nothing under
~/data/displace2024 is ever opened.

Tuning rule (PREREGISTRATION_M2B.md §3, the P0 gate; H4 supplement §4, §5):
  * tau0 = B0's dev tau (the tau at which 10% of dev controls fire, via `pilot.calibrate_tau` on the ungated
    curve d) plus the offset grid {-0.10, -0.09, ..., +0.10}; alpha in {0, 0.05, ..., 1.00}.
  * tuned JOINTLY to minimise dev FAR_SS-DL subject to the upper one-sided 80% BCa bound (10,000 whole-session
    resamples) of per-session delta MR_DS-DL vs B0 being <= +1 pp.
  * tuned on the two-language dev recordings only (so lambda is a two-language JSD in bits, as on D1). A
    recording is two-language only when its LD reference labels map -- through a required `--label-map` JSON
    (LD label string -> VoxLingua107 code, hashed into the record) -- to EXACTLY TWO distinct codes that are both
    in that recording's LID-cache label list. Each recording's pair is derived from the map, not the caller. A
    recording with any unmapped or unresolvable label is EXCLUDED from the P0-gate tuning, flagged and counted
    (never an abort; the full-posterior fallback of M2B §2 is for scoring, not tuning).
  * ties: smaller alpha, then lower dev MR_DS-DL, then smaller tau0.
  * not assessable with fewer than 5 RESOLVED two-language sessions having >= 10 DS-DL turns -- alpha is then not
    frozen and M2's D1 stage-2 test is reported as not tested (H4 supplement §4; M2B §3).

Output JSON (written by `run_p0_tuning`): the chosen alpha and tau0, the full (alpha, tau0) table (each cell with
a `far_defined` flag, the MR sessions the bound dropped, and both BCa levels), the dev tau and the count of edge
controls it dropped, the label map, the excluded recordings with reasons, the counts, the assessability flag, the
input sha256 hashes (including the label map) and this module's code hash. Only dense-grid / LID caches for
recording IDs in the dev summary.json are loaded; any other ID is refused (cache provenance beyond that
membership is pinned by the freeze record, not re-checked here).

Open choices, fixed as NAMED CONSTANTS and listed in the final report:
  * FAR_AGGREGATION / MR_AGGREGATION -- POOLED over units vs SESSION-MEAN. Default "session_mean" (the D1 H4
    stage-2 test clusters by channel/session, and the per-session delta MR the bound is taken over is a
    session-level quantity, so the objective is chosen on the same session-level scale).
  * TAU_CONTROLS -- which controls calibrate B0's dev tau and the gate's control fire rate. "all": every dev
    control (the DISPLACE reader has one control family, unlike D1's det/det_w1/p0L1; M2B §2). The 10% rule and
    `pilot.calibrate_tau` both read this set.
  * CELL -- the ΔFAR standardisation cell. "session_pause_bin": (session, pause bin), matching
    `score_m2b._cell` / M2B §2 (DISPLACE has no position bin; D1's was (video, pause bin, position bin)).
  * ALPHA_ZERO_TAU0_ZERO_IS_B0 -- alpha = 0 at tau0 offset 0 reproduces B0 exactly (deltaMR = 0), so it always
    meets the bound and the feasible set is never empty (M2B §3).
  * GATE_FEASIBLE_REQUIRES_FINITE_TAU -- an (alpha, tau0) whose tau0 is non-finite (it never is here: tau0 is a
    finite offset of a finite dev tau) or whose scoring is degenerate is infeasible; alpha 0 / tau0 offset 0 is
    exempt by construction.

Everything heavy (torch, soundfile, the embedding model) is imported lazily inside the functions that build the
dense grid, so this module imports -- and every pure function and the whole test file runs -- on a host with no
model present.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

# Reused D1 P0 + gate code (read, never edited): the detector, the gated curve, the JSD, the stats.
from csdiar.analysis import confirm_h4
from csdiar.analysis.pilot import HOP, C, calibrate_tau, peak_strength
from csdiar.analysis.stats import bca_ci

# ----------------------------------------------------------------------------------------------------- #
# frozen constants (PREREGISTRATION_M2B.md §3; H4 supplement §4, §5)
# ----------------------------------------------------------------------------------------------------- #
SEED = 20261018                                                        # §11 / confirm_h4.SEED
ALPHA_GRID = tuple(round(0.05 * k, 2) for k in range(21))             # 0.00 .. 1.00 step 0.05 (§3)
TAU0_OFFSET_GRID = tuple(round(-0.10 + 0.01 * k, 2) for k in range(21))  # -0.10 .. +0.10 step 0.01 (§3)
MR_BOUND_PP = 0.01                                                    # upper bound on deltaMR_DS-DL vs B0: +1 pp
BCA_ONE_SIDED_LEVEL = 0.80                                            # one-sided 80% BCa bound (§3)
BCA_N_BOOT = 10000                                                   # 10,000 whole-session resamples (§3)
ASSESS_MIN_SESSIONS = 5                                              # >= 5 two-language dev sessions ... (§3)
ASSESS_MIN_DSDL_PER_SESSION = 10                                     # ... each with >= 10 DS-DL turns (§3)
CTRL_FIRE_RATE = 0.10                                                # B0 dev tau: 10% of dev controls fire (§5)
LAG_K = confirm_h4.LAG_K                                             # d[k] compares windows k-10 and k (1 s)
WIN = 1.0                                                            # dense grid window length L (s)

# ----------------------------------------------------------------------------------------------------- #
# open choices, fixed as named constants (listed in the final report)
# ----------------------------------------------------------------------------------------------------- #
FAR_AGGREGATION = "session_mean"          # "session_mean" | "pooled" -- how dev FAR_SS-DL enters the objective
MR_AGGREGATION = "session_mean"           # "session_mean" | "pooled" -- how dev MR_DS-DL is reported (the bound
#                                           always uses PER-SESSION deltaMR, §3)
TAU_CONTROLS = "all"                       # which controls calibrate tau / give the control fire rate (one
#                                           DISPLACE control family, M2B §2; D1 used det_w1)
CELL = "session_pause_bin"                 # ΔFAR cell = (session, pause bin), score_m2b._cell / M2B §2
ALPHA_ZERO_TAU0_ZERO_IS_B0 = True          # alpha=0, tau0 offset 0 == B0 (deltaMR=0); feasible set never empty
GATE_FEASIBLE_REQUIRES_FINITE_TAU = True   # a non-finite / degenerate operating point is infeasible

GUARDED_SELF = "csdiar/analysis/tune_p0_m2b.py"
CACHE_VERSION = 1                          # bump when the dense-grid cache layout changes


class InsufficientData(ValueError):
    """The P0-gate alpha cannot be assessed (fewer than 5 two-language dev sessions with >= 10 DS-DL turns, §3),
    or a fit/scoring input is under-defined.

    `details` carries the structured numbers that decided the verdict (for the scored re-check: the per-session
    scorable DS-DL counts), so a not-assessable record is as machine-readable as a frozen one (review round 4,
    m2). It is empty when the message says everything (for example a degenerate dev tau)."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.details = dict(details or {})


# ===================================================================================================== #
# 0. the P0 detector on the dense grid -- reuse pilot, never re-implement
# ===================================================================================================== #
def d_curve(embeddings) -> "object":
    """P0's d(t) on a whole recording's dense E3 grid: 1 - cos(window k-10, window k), reusing `confirm_h4.curve`
    (which is `pilot.p0_curves`' per-clip arithmetic: d[k] = 1 - (e[k].e[k-LAG_K]), NaN for the first LAG_K
    windows). `embeddings` is a (n_windows, dim) array; it is unit-normalised first with `pilot.unit`, exactly
    as `p0_curves` does, so d(t) here is byte-identical to P0's d(t) on the same windows."""
    from csdiar.analysis.pilot import unit
    import numpy as np
    e = unit(np.asarray(embeddings, dtype=np.float32))
    return confirm_h4.curve(e)


def gated_curve(embeddings, posteriors, pair: tuple[int, int], alpha: float):
    """M2's gated curve d'(t) = d(t) - alpha*lambda(t) on the dense grid, produced by the SAME code the D1
    stage-2 test uses: `confirm_h4.system_curves(name="M2", ...)` (brief: reuse the same function so the DISPLACE
    gate is the same code as the D1 gate).

    `confirm_h4.load_lid` hands `system_curves` a per-clip array ALREADY renormalised over (ne, en) -- its M2
    branch computes d[k] -= alpha * jsd(p[k-10], p[k]) with no renormalisation of its own (`jsd` sums over the
    last axis). On DISPLACE the two reference languages are an arbitrary index pair in the 107-wide posterior, so
    we slice `posteriors[:, (i, j)]` and renormalise that 2-column pair to sum 1 -- reproducing exactly what
    `confirm_h4.load_lid` produces for (ne, en) on D1. That is the only adaptation the cross-corpus pair needs
    (M2B §2 departure 3): with it, `system_curves` subtracts `alpha * jsd(renorm(p[k-10]), renorm(p[k]))`, which
    is lambda(t) in bits, so the gate math is confirm_h4's own, unmodified.

    Returns the d' curve for the single pseudo-clip "rec". alpha = 0 returns d exactly (confirm_h4 subtracts
    0 * lambda), so it reproduces B0.
    """
    import numpy as np
    from csdiar.analysis.pilot import unit
    i, j = int(pair[0]), int(pair[1])
    if i == j:
        raise ValueError(f"pair must name two distinct indices, got {pair!r}")
    e = unit(np.asarray(embeddings, dtype=np.float32))
    post = np.asarray(posteriors, dtype=np.float64)
    if post.shape[0] != e.shape[0]:
        raise ValueError(f"LID grid has {post.shape[0]} windows, embeddings {e.shape[0]}")
    two = post[:, (i, j)]                              # (n, 2): the recording's reference-language pair
    two = two / np.maximum(two.sum(1, keepdims=True), 1e-12)   # renormalise as confirm_h4.load_lid does
    grid = {"rec": e}
    lid = {"rec": two}
    out = confirm_h4.system_curves("M2", {"rec"}, grid, lid, {"alpha": float(alpha)})
    return out["rec"]


def hit_on_curve(curve, t: float, tau: float) -> bool:
    """P0's hit rule, reused: the strongest NMS peak within +-C of grid index k = round(t/HOP) exceeds tau
    (`pilot.peak_strength` + the `> tau` comparison of `pilot.h2_cell`). NaN d(t) (within 1 s of the recording
    start, no left window) is not a hit."""
    import numpy as np
    k = int(round(float(t) / HOP))
    if k < 0 or k >= len(curve) or np.isnan(curve[k]):
        return False
    return peak_strength(curve, k) > tau


# ===================================================================================================== #
# 1. lambda on the grid -- confirm_h4.jsd over the renormalised reference-language pair (M2B §2)
# ===================================================================================================== #
def lambda_on_grid(posteriors, pair: tuple[int, int]):
    """lambda over the whole grid: base-2 JSD of the renormalised (lang_a, lang_b) posteriors of windows k-LAG_K
    and k, exactly as `m2_lambda.lam` / `confirm_h4.system_curves`' M2 branch do, generalised to the explicit
    reference-language `pair` (M2B §2). Returns a (n_windows,) array, NaN for the first LAG_K windows.

    This is the standalone lambda the tests check against `m2_lambda` on the same posteriors (pair (0, 1)); the
    gate itself uses `gated_curve` (confirm_h4's own subtraction), so the two never diverge.
    """
    import numpy as np
    i, j = int(pair[0]), int(pair[1])
    post = np.asarray(posteriors, dtype=np.float64)
    n = post.shape[0]
    out = np.full(n, np.nan, np.float64)
    if n <= LAG_K:
        return out
    two = post[:, (i, j)]
    s = two.sum(1, keepdims=True)
    renorm = two / np.maximum(s, 1e-12)
    for k in range(LAG_K, n):
        out[k] = confirm_h4.jsd(renorm[k - LAG_K], renorm[k])
    return out


# ===================================================================================================== #
# 2. the dense E3 grid of a whole recording -- reuse csdiar.models.embed; cached (device param, CPU on the box)
# ===================================================================================================== #
def _code_hash() -> str:
    """sha256 over this module's source AND csdiar/models/embed.py, so a change to either the gate/grid code or
    the embedder invalidates a cached grid (the cache carries this)."""
    repo = Path(__file__).resolve().parents[2]
    h = hashlib.sha256()
    for rel in (GUARDED_SELF, "csdiar/models/embed.py"):
        h.update((repo / rel).read_bytes())
    return h.hexdigest()


def dense_grid_cache_path(cache_dir, model: str, rid: str) -> Path:
    """<cache_dir>/dense/<model>_L1/<rid>.npz -- the P0 dense grid of one DISPLACE recording."""
    return Path(cache_dir) / "dense" / f"{model}_L{WIN:g}" / f"{rid}.npz"


def build_dense_grid(wav_path, cache_dir, rid: Optional[str] = None, model: str = "e3",
                     hop: float = HOP, win: float = WIN, batch: int = 256, device: str = "cpu",
                     overwrite: bool = False) -> dict:
    """E3 window embeddings at L = `win` s, hop `hop` s over a WHOLE DISPLACE recording, reusing
    `csdiar.models.embed.load` / `frames` (the frozen D1 dense-embedding code), cached per recording with the
    model id, this code's hash and the embed provenance in the npz.

    `device` is a parameter -- "cpu" on the acceptance box (the GPU is shared and off-limits). The embedding
    model is loaded lazily; a host with no model cannot build a grid but can still read a cache. Returns
    metadata (shapes, timing, whether the cache was reused).
    """
    import time
    import numpy as np
    import soundfile as sf
    import torch

    from csdiar.models.embed import SR, frames, load

    rid = rid or Path(wav_path).stem
    out_path = dense_grid_cache_path(cache_dir, model, rid)
    code_hash = _code_hash()
    if out_path.exists() and not overwrite:
        with np.load(out_path, allow_pickle=False) as z:
            if int(z.get("version", -1)) == CACHE_VERSION and str(z.get("code_hash", "")) == code_hash \
                    and str(z.get("model", "")) == model:
                return {"rid": rid, "model": model, "cache": str(out_path), "cached": True,
                        "n_windows": int(z["grid"].shape[0])}

    wav, sr = sf.read(str(wav_path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != SR:
        raise ValueError(f"{wav_path}: expected {SR} Hz, got {sr}")
    wav = np.ascontiguousarray(wav, dtype=np.float32)

    embed, prov = load(model, device)
    fr, padded = frames(wav, win, hop)
    t0 = time.time()
    embs = []
    with torch.inference_mode():
        for b in range(0, len(fr), batch):
            embs.append(embed(torch.from_numpy(fr[b:b + batch]).to(device)))
    grid = np.concatenate(embs).astype(np.float16)
    dt = time.time() - t0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, version=np.int64(CACHE_VERSION), rid=rid, model=model,
                        code_hash=code_hash, provenance=json.dumps(prov), grid=grid,
                        hop=np.float64(hop), win=np.float64(win), padded=np.int64(int(padded)))
    return {"rid": rid, "model": model, "cache": str(out_path), "cached": False,
            "n_windows": int(grid.shape[0]), "dim": int(grid.shape[1]), "embed_s": dt,
            "duration_s": len(wav) / SR}


def load_dense_grid(cache_dir, model: str, rid: str):
    """Load a dense-grid cache -> (grid (n,dim) float32, meta dict).

    m4 (review v1): validates the cache layout `version`, AND -- matching `build_dense_grid`'s reuse contract --
    the cached `code_hash` against the CURRENT `_code_hash()` and the cached `model` against the requested
    `model`, raising on a mismatch. This is the read path `run_p0_tuning` actually uses, so without these checks
    a grid built under a different embedder (e.g. after embed.py changed) would be silently accepted whenever the
    layout version still matched. The docstring's guarantee -- "a re-extracted grid never silently reuses a stale
    cache" -- now holds on load as well as on build.
    """
    import numpy as np
    path = dense_grid_cache_path(cache_dir, model, rid)
    with np.load(path, allow_pickle=False) as z:
        if int(z.get("version", -1)) != CACHE_VERSION:
            raise ValueError(f"{path}: cache version {int(z.get('version', -1))} != {CACHE_VERSION}; rebuild")
        cached_model = str(z["model"])
        if cached_model != model:
            raise ValueError(f"{path}: cache model {cached_model!r} != requested {model!r}; rebuild")
        cached_hash = str(z["code_hash"])
        cur_hash = _code_hash()
        if cached_hash != cur_hash:
            raise ValueError(f"{path}: cache code_hash {cached_hash[:12]}... != current {cur_hash[:12]}... "
                             f"(tune_p0_m2b.py / embed.py changed since this grid was built); rebuild")
        grid = z["grid"].astype(np.float32)
        meta = {"rid": str(z["rid"]), "model": cached_model, "code_hash": cached_hash,
                "hop": float(z["hop"]), "win": float(z["win"])}
    return grid, meta


# ===================================================================================================== #
# 3. scoring dev reader rows under the P0 rule (M2B §2; H4 supplement §4)
# ===================================================================================================== #
@dataclass
class SessionScore:
    """One recording session at one operating point (tau, and alpha on the gated curve)."""
    session: str
    far_ssdl: float                 # cell-standardised SS-DL ΔFAR (pilot.h2_cell style, DISPLACE cell)
    far_ssdl_raw: float             # raw SS-DL fire rate (before cell standardisation)
    ctrl_fire_rate: float           # control fire rate
    mr_dsdl: float                  # miss rate on DS-DL turns under the P0 rule
    n_ssdl_with_cell_controls: int  # H4 inclusion unit (M2B §7)
    n_ssdl_without_cell_controls: int
    n_dsdl: int                     # H5 inclusion unit (M2B §7)
    n_ctrl: int


def _cell_key(row: dict) -> tuple:
    """ΔFAR cell within a session: the pause bin (CELL = (session, pause bin), M2B §2; score_m2b._cell)."""
    return (row["pause_bin"],)


def _is_true(v) -> bool:
    return v is True or v == "True"


def _fire(curve, rows: Sequence[dict], tau: float) -> list[bool]:
    """P0 fire for each row's time t on `curve` at `tau` (the hit rule, reused)."""
    return [hit_on_curve(curve, float(r["t"]), tau) for r in rows]


def score_session_p0(session: str, rows: Sequence[dict], curve, tau: float) -> SessionScore:
    """Score one recording session under the P0 rule on a given curve (d for B0, d' for a gated alpha) at `tau`.

    SS-DL points, controls and DS-DL turns are read from the reader rows (`csdiar.data.ingest_displace`); a point
    fires when its strongest NMS peak within +-c exceeds tau. ΔFAR is `pilot.h2_cell`'s cell standardisation with
    the DISPLACE cell (session, pause bin): each scored SS-DL point's fire minus its own cell's control fire
    rate, controls reused, averaged over points that have a cell control. MR_DS-DL is the share of DS-DL turns
    with NO firing peak within +-c (the P0 detector's miss: a turn is matched when a peak fires at it). DS-SL
    turns and other kinds are ignored here.
    """
    import numpy as np

    def defined(r):
        # pilot.score_points drops a point with no defined d(t) (out of range, or NaN in the first second): it
        # is neither a fire nor a non-fire. The same rule applies here to SS-DL points, controls and DS-DL
        # turns (review round 2, m1), and is identical for B0 and M2 because gating never changes NaN positions.
        k = int(round(float(r["t"]) / HOP))
        return 0 <= k < len(curve) and not np.isnan(curve[k])

    ssdl = [r for r in rows if r["kind"] == "ssdl" and defined(r)]
    ctrl = [r for r in rows if r["kind"] == "ctrl" and defined(r)]
    dsdl = [r for r in rows if r["kind"] == "dsdl" and defined(r)]

    ssdl_fire = _fire(curve, ssdl, tau)
    ctrl_fire = _fire(curve, ctrl, tau)

    cell_fire: dict[tuple, list[bool]] = defaultdict(list)
    for r, f in zip(ctrl, ctrl_fire):
        cell_fire[_cell_key(r)].append(f)
    cell_rate = {c: (sum(v) / len(v) if v else float("nan")) for c, v in cell_fire.items()}

    deltas, without = [], 0
    for r, f in zip(ssdl, ssdl_fire):
        rate = cell_rate.get(_cell_key(r))
        if rate is None or rate != rate:
            without += 1
            continue
        deltas.append((1.0 if f else 0.0) - rate)
    far_std = float(np.mean(deltas)) if deltas else float("nan")
    far_raw = float(np.mean(ssdl_fire)) if ssdl_fire else float("nan")
    ctrl_rate = float(np.mean(ctrl_fire)) if ctrl_fire else float("nan")

    # MR_DS-DL under the P0 detector: a DS-DL turn is MISSED when no NMS peak fires within +-c of it.
    dsdl_fire = _fire(curve, dsdl, tau)
    mr = (1.0 - float(np.mean(dsdl_fire))) if dsdl_fire else float("nan")

    return SessionScore(session=session, far_ssdl=far_std, far_ssdl_raw=far_raw, ctrl_fire_rate=ctrl_rate,
                        mr_dsdl=mr, n_ssdl_with_cell_controls=len(deltas), n_ssdl_without_cell_controls=without,
                        n_dsdl=len(dsdl), n_ctrl=len(ctrl))


# ===================================================================================================== #
# 4. dev tau, aggregation and the joint (alpha, tau0) tuning (M2B §3; H4 supplement §4, §5)
# ===================================================================================================== #
def dev_tau(points_by_session: dict, curves_by_session: dict) -> tuple[float, int]:
    """B0's dev tau: the tau at which CTRL_FIRE_RATE (10%) of dev controls fire, via `pilot.calibrate_tau`
    (reused, including its degenerate-case refusal -> -inf when the finite-peak share is <= 10%,
    PREREGISTRATION_H4.md §5). Controls are pooled over all two-language dev sessions (TAU_CONTROLS = "all").

    `pilot.calibrate_tau(points, vals)` expects a `points` dict keyed by point id with a `kind` and the
    `elig_<TAU_CONTROLS>` flag `pilot.TAU_CONTROLS` names, and a `vals` dict of {"peak": strongest NMS peak}. We
    build both from the DISPLACE controls on the ungated curve d, tagging every control eligible so pilot's
    det_w1 eligibility test passes (DISPLACE has a single control family, M2B §2).

    M1 (review v1): a control whose grid point has no defined d(t) (k out of range, or d[k] NaN within the first
    LAG_K windows) is DROPPED exactly as `pilot.score_points` drops it on D1 -- it never enters the peaks array,
    rather than being passed to `calibrate_tau` as -inf. Returns (tau, n_dropped_controls); the dropped count is
    recorded.
    """
    import numpy as np
    pts: dict = {}
    vals: dict = {}
    dropped = 0
    for sess, rows in points_by_session.items():
        curve = curves_by_session[sess]
        for n, r in enumerate(rows):
            if r["kind"] != "ctrl":
                continue
            k = int(round(float(r["t"]) / HOP))
            # M1 (review v1): DROP an edge control exactly as pilot.score_points drops a point with no defined
            # d(t) -- k out of range, or d[k] NaN within the first LAG_K windows (no left window). pilot never
            # feeds such a control to calibrate_tau as -inf; it simply never enters the peaks array. Mirror that
            # here (skip, do NOT add a -inf peak) so the reused calibrate_tau sees the IDENTICAL control set it
            # sees on D1 -- otherwise -inf peaks lower `share` and shift the surviving quantile, giving a
            # different tau0 (and in the limit a spurious degenerate case) than the same P0 calibration on D1.
            if k < 0 or k >= len(curve) or np.isnan(curve[k]):
                dropped += 1
                continue
            pid = f"{sess}:{n}"
            pts[pid] = {"kind": "ctrl", f"elig_{_PILOT_TAU_CTRL}": True}
            vals[pid] = {"peak": peak_strength(curve, k)}
    tau, _share = calibrate_tau(pts, vals)
    return tau, dropped


# pilot.calibrate_tau reads elig_<pilot.TAU_CONTROLS>; mirror that name so the reused function sees our controls.
_PILOT_TAU_CTRL = __import__("csdiar.analysis.pilot", fromlist=["TAU_CONTROLS"]).TAU_CONTROLS


def _aggregate(values_by_session: dict, included: Sequence[str], weights: Optional[dict],
               aggregation: str) -> float:
    """Aggregate a per-session metric over INCLUDED sessions (M2B §7 inclusion) by `aggregation`
    ("session_mean" = unweighted mean; "pooled" = weighted by `weights`)."""
    xs, ws = [], []
    for s in included:
        if s in values_by_session and values_by_session[s] == values_by_session[s]:
            xs.append(values_by_session[s])
            ws.append(float(weights.get(s, 1.0)) if weights else 1.0)
    if not xs:
        return float("nan")
    if aggregation == "pooled":
        tot = sum(ws)
        return sum(x * w for x, w in zip(xs, ws)) / tot if tot > 0 else float("nan")
    return sum(xs) / len(xs)


def included_sessions(scores: dict, unit_of: Callable, min_units: int = ASSESS_MIN_DSDL_PER_SESSION) -> list:
    """Sessions with at least `min_units` of the given unit (M2B §7 inclusion rule). For FAR/H4 the unit is
    SS-DL points with a cell control; for MR/H5 it is DS-DL turns. The dev `min_units` is 10 (M2B §3)."""
    return sorted(s for s, sc in scores.items() if unit_of(sc) >= min_units)


def per_session_delta_mr(alpha_scores: dict, b0_scores: dict, included_mr: Sequence[str]) -> tuple[dict, list]:
    """Per-session deltaMR_DS-DL = MR(alpha) - MR(B0) over the sessions INCLUDED for MR and present in both,
    both at the SAME tau0 (the gate's tau0). The bound (§3) is taken over these per-session deltas.

    m1 (review v1): a session INCLUDED for MR but whose delta cannot be formed (absent from either score map, or
    a NaN MR on either side) is DROPPED from the bound. Returns (deltas_by_session, dropped_session_ids) so the
    cell can record how many included sessions the bound dropped and which -- a NaN / under-populated bound is
    then explainable rather than appearing as an unexplained infeasible cell.
    """
    out = {}
    dropped = []
    for s in included_mr:
        a, b = alpha_scores.get(s), b0_scores.get(s)
        if a is None or b is None:
            dropped.append(s)
            continue
        if a.mr_dsdl == a.mr_dsdl and b.mr_dsdl == b.mr_dsdl:
            out[s] = a.mr_dsdl - b.mr_dsdl
        else:
            dropped.append(s)
    return out, dropped


def mr_upper_bound(deltas_by_session: dict, rng, level: float = BCA_ONE_SIDED_LEVEL,
                   n_boot: int = BCA_N_BOOT) -> dict:
    """Upper one-sided `level` (80%) BCa bound of the per-session mean deltaMR, resampling whole sessions
    (`stats.bca_ci`, 10,000 resamples), exactly §3's bound. The one-sided 80% upper bound is the UPPER endpoint
    of the two-sided (2*level - 1) = 0.60 BCa interval. Returns {"bound", "estimate", "n_sessions", "bca"}.

    m5 (review v1): the stored `bca` records BOTH levels -- `one_sided_level` (0.80, the level the rule is stated
    in) and `bca_two_sided_level` (0.60, the level actually passed to `bca_ci`) -- so the persisted record is not
    read as a 60% bound. `level` is kept as the two-sided value for backward compatibility with `stats.bca_ci`'s
    own key.
    """
    clusters = [[v] for v in deltas_by_session.values()]   # one session = one cluster, as confirm_h4/stats do
    two_sided = 2.0 * level - 1.0
    levels = {"one_sided_level": level, "bca_two_sided_level": two_sided}
    if len(clusters) < 2:
        return {"bound": float("nan"), "estimate": float("nan"), "n_sessions": len(clusters),
                "bca": {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan"), "level": two_sided,
                        "z0": float("nan"), "a": float("nan"), "n_boot": 0, "n_dropped": 0, **levels}}
    ci = bca_ci(clusters, lambda cs: sum(c[0] for c in cs) / len(cs), two_sided, rng, n_boot=n_boot)
    ci = {**ci, **levels}
    return {"bound": ci["hi"], "estimate": ci["estimate"], "n_sessions": len(clusters), "bca": ci}


def assess_tunable(dsdl_turns_per_two_lang_session: dict,
                   min_sessions: int = ASSESS_MIN_SESSIONS,
                   min_turns: int = ASSESS_MIN_DSDL_PER_SESSION) -> dict:
    """§3 assessability for the P0 gate: alpha can be frozen only when at least `min_sessions` TWO-LANGUAGE dev
    sessions have at least `min_turns` DS-DL turns each. `dsdl_turns_per_two_lang_session` maps a two-language
    recording -> its DS-DL turn count (the caller restricts to two-language recordings)."""
    qualifying = sorted(s for s, n in dsdl_turns_per_two_lang_session.items() if n >= min_turns)
    return {"tunable": len(qualifying) >= min_sessions, "n_qualifying": len(qualifying),
            "qualifying": qualifying, "min_sessions": min_sessions, "min_turns": min_turns}


@dataclass
class GridCell:
    """One (alpha, tau0) operating point's dev summary."""
    alpha: float
    tau0: float
    far_ssdl: float
    mr_dsdl: float
    delta_mr_upper_bound: float
    feasible: bool
    n_sessions_far: int
    n_sessions_mr: int
    far_defined: bool = True            # m2: FAR could be computed on >= 1 included session (not NaN)
    n_mr_sessions_dropped: int = 0      # m1: included-for-MR sessions the bound could not use
    mr_sessions_dropped: list = field(default_factory=list)   # m1: which ones
    bca: dict = field(default_factory=dict)


def select_alpha_tau0(grid_cells: Sequence[GridCell], bound_pp: float = MR_BOUND_PP) -> dict:
    """The joint (alpha, tau0) choice (§3): among FEASIBLE cells (upper bound <= bound_pp), minimise dev
    FAR_SS-DL; ties -> smaller alpha, then lower dev MR_DS-DL, then smaller tau0. alpha=0 / tau0 offset 0 is
    feasible by construction (ALPHA_ZERO_TAU0_ZERO_IS_B0), so the feasible set is never empty.

    Returns {"alpha", "tau0", "far_ssdl", "mr_dsdl", "n_feasible", "far_defined",
    "n_feasible_far_undefined"}. Raises `InsufficientData` only if NO cell is feasible (which cannot happen while
    the B0 cell is present and feasible).

    m2 (review v1): a feasible cell with an undefined (NaN) FAR is ranked as +inf (worst) by the key, but the
    chosen cell's `far_defined` and the count of feasible cells with an undefined FAR are returned, so the
    selected alpha is never a silent artifact of a NaN->inf coercion.
    """
    from math import isfinite
    feasible = [c for c in grid_cells if c.feasible]
    if not feasible:
        raise InsufficientData("no feasible (alpha, tau0) cell; even alpha=0/tau0=0 was infeasible")

    def key(c: GridCell):
        far = c.far_ssdl if isfinite(c.far_ssdl) else float("inf")
        mr = c.mr_dsdl if isfinite(c.mr_dsdl) else float("inf")
        return (far, c.alpha, mr, c.tau0)

    best = min(feasible, key=key)
    return {"alpha": best.alpha, "tau0": best.tau0, "far_ssdl": best.far_ssdl, "mr_dsdl": best.mr_dsdl,
            "n_feasible": len(feasible), "far_defined": best.far_defined,
            "n_feasible_far_undefined": sum(1 for c in feasible if not c.far_defined)}


# ===================================================================================================== #
# 5. counts / hashing
# ===================================================================================================== #
def load_label_map(path) -> dict:
    """Load the LD-label -> VoxLingua107-code map (M2, review v1). A JSON object mapping each LD label STRING
    (as it appears in `ingest_displace`'s summary `languages`) to a VoxLingua107 code string (e.g.
    {"eng": "en", "HIN": "hi"}). Required for a real run and hashed into the record. Returns {str: str}."""
    obj = json.loads(Path(path).expanduser().read_text())
    if not isinstance(obj, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in obj.items()):
        raise ValueError(f"{path}: label map must be a JSON object of LD-label -> VoxLingua107-code strings")
    return {str(k): str(v) for k, v in obj.items()}


def resolve_recording_pair(ld_labels: Sequence[str], lid_labels: Sequence[str],
                           label_map: dict) -> dict:
    """Resolve one recording's LD reference labels to a VoxLingua107 reference-language PAIR (M2, review v1).

    "Two-language" is decided from the RESOLVED reference set, not the raw LD-label count (§2, §3): each LD label
    is mapped through `label_map` to a VoxLingua107 code, and the recording is two-language only when its LD
    labels map to EXACTLY TWO distinct codes that are BOTH present in this recording's LID-cache `lid_labels`.

    A recording with any unmapped LD label, any label mapping to a code absent from `lid_labels`, or whose labels
    collapse to a single distinct code (e.g. two labels mapping to the same code) is NOT two-language: it is
    excluded from the P0-gate tuning, flagged with a reason, and counted -- never an abort (§2's full-posterior
    fallback is for SCORING, not for tuning).

    Returns a dict:
      {"resolved": True,  "codes": (a, b), "indices": (i, j), "distinct_codes": [...]}            -> two-language
      {"resolved": False, "reason": "...", "unmapped": [...], "unresolved_codes": [...], "distinct_codes": [...]}
    """
    distinct = sorted({str(x) for x in ld_labels})
    unmapped = [lab for lab in distinct if lab not in label_map]
    if unmapped:
        return {"resolved": False, "reason": f"unmapped LD label(s) {unmapped}", "unmapped": unmapped,
                "unresolved_codes": [], "distinct_codes": []}
    codes = []
    for lab in distinct:
        c = label_map[lab]
        if c not in codes:
            codes.append(c)
    lidset = {str(x) for x in lid_labels}
    unresolved = [c for c in codes if c not in lidset]
    if unresolved:
        return {"resolved": False, "reason": f"code(s) {unresolved} not in the recording's LID label list",
                "unmapped": [], "unresolved_codes": unresolved, "distinct_codes": codes}
    if len(codes) != 2:
        return {"resolved": False,
                "reason": f"{len(codes)} distinct resolved code(s) {codes}, not exactly two",
                "unmapped": [], "unresolved_codes": [], "distinct_codes": codes}
    from csdiar.pipelines import run_m2b as R
    i, j = R.pair_indices(list(lid_labels), codes[0], codes[1])
    return {"resolved": True, "codes": (codes[0], codes[1]), "indices": (i, j), "distinct_codes": codes}


def two_language_sessions(summary: dict, label_map: dict, lid_labels_of: Callable[[str], Sequence[str]]) -> dict:
    """From `ingest_displace`'s summary.json, the label map and a per-recording LID-labels provider: the
    RESOLVED two-language recordings (M2, review v1).

    A recording is two-language only when `resolve_recording_pair` resolves its LD labels to exactly two distinct
    VoxLingua107 codes both present in that recording's LID label list. A recording with any unmapped or
    unresolvable label (or a same-code collapse) is EXCLUDED, flagged and counted -- never an abort. Assessability
    counts only resolved two-language recordings.

    Returns {dsdl_per_two_lang_session, pair_by_recording, excluded, n_*}.
    `lid_labels_of(rid)` may raise (e.g. a missing cache); such a recording is recorded as excluded with the
    error, not propagated.
    """
    recs = summary.get("recordings", [])
    two_lang: dict = {}
    pair_by_rec: dict = {}
    excluded: dict = {}
    for r in recs:
        rid = r["recording"]
        ld_labels = r.get("languages", [])
        try:
            lid_labels = lid_labels_of(rid)
        except Exception as e:                       # noqa: BLE001 -- a missing/unreadable LID cache excludes
            excluded[rid] = {"reason": f"LID labels unavailable: {e}", "ld_labels": sorted(ld_labels)}
            continue
        res = resolve_recording_pair(ld_labels, lid_labels, label_map)
        if res["resolved"]:
            two_lang[rid] = int(r.get("n_dsdl", 0))
            pair_by_rec[rid] = res["codes"]
        else:
            excluded[rid] = {"reason": res["reason"], "ld_labels": sorted(ld_labels),
                             "distinct_codes": res.get("distinct_codes", [])}
    return {"dsdl_per_two_lang_session": two_lang, "pair_by_recording": pair_by_rec,
            "excluded": excluded, "n_excluded": len(excluded),
            "n_two_language_recordings": len(two_lang), "n_recordings": len(recs),
            "n_two_lang_sessions_ge10_dsdl": sum(1 for n in two_lang.values()
                                                 if n >= ASSESS_MIN_DSDL_PER_SESSION)}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def self_sha256() -> str:
    return sha256_file(Path(__file__).resolve().parents[2] / GUARDED_SELF)


def read_reader_rows(units_dir, split: str) -> dict:
    """Load `ingest_displace`'s per-split rows (ssdl / turns / ctrl) grouped by recording, for scoring.

    GUARD (M2B §3): refuses any split other than 'dev' -- the tuner reads dev only; no eval recording is read
    before the freeze. Returns {recording: [row dict, ...]}.
    """
    import csv
    import gzip
    if split != "dev":
        raise ValueError(f"tune_p0_m2b reads the DEV split only (PREREGISTRATION_M2B.md §3); refusing "
                         f"{split!r}. No eval recording is read before the freeze.")
    units_dir = Path(units_dir)
    rows_by_rec: dict = defaultdict(list)
    for kind in ("ssdl", "turns", "ctrl"):
        path = units_dir / f"{kind}_{split}.csv.gz"
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows_by_rec[row["recording"]].append(row)
    return dict(rows_by_rec)


def open_choices() -> dict:
    return {"far_aggregation": FAR_AGGREGATION, "mr_aggregation": MR_AGGREGATION,
            "tau_controls": TAU_CONTROLS, "cell": CELL,
            "alpha_zero_tau0_zero_is_b0": ALPHA_ZERO_TAU0_ZERO_IS_B0,
            "gate_feasible_requires_finite_tau": GATE_FEASIBLE_REQUIRES_FINITE_TAU}


# ===================================================================================================== #
# 6. orchestration
# ===================================================================================================== #
@dataclass
class TuningInputs:
    units_dir: Path                 # ingest_displace DEV output (ssdl/turns/ctrl csv.gz, summary.json)
    cache_dir: Path                 # run_m2b LID caches + this module's dense-grid caches
    wav_dir: Path                   # dev wavs (the dense grid is built from these)
    out_path: Path
    model: str = "e3"
    device: str = "cpu"
    label_map: dict = field(default_factory=dict)          # M2: LD label string -> VoxLingua107 code
    label_map_path: Optional[Path] = None                  # hashed into the record when present
    pair_by_recording: dict = field(default_factory=dict)  # rec -> (code_a, code_b); derived from label_map
    recordings: tuple = ()


def _score_curve_over_sessions(curves_by_session: dict, rows_by_session: dict, tau: float) -> dict:
    """Score every session on its own curve at `tau` -> {session: SessionScore}."""
    return {s: score_session_p0(s, rows_by_session[s], curves_by_session[s], tau)
            for s in curves_by_session}


def tune(points_by_session: dict,
         d_curves_by_session: dict,
         gated_curves_by_session: Callable[[str, float], object],
         rng,
         far_aggregation: str = FAR_AGGREGATION,
         mr_aggregation: str = MR_AGGREGATION) -> dict:
    """The pure tuning core (no model, no IO): given the dev reader rows per session, the ungated d curve per
    session, and a callable giving the gated d' curve for (session, alpha), run the whole §3 rule.

    `gated_curves_by_session(session, alpha)` returns that session's d'(t) at `alpha` (the orchestration binds it
    to the recording's dense grid, posteriors and pair; the test binds a synthetic one). This is the single
    place the grids are turned into the chosen (alpha, tau0), so the CLI and the tests share it.

    Steps:
      1. B0's dev tau from the ungated curves (`dev_tau`, `pilot.calibrate_tau`).
      2. The B0 scores at tau0 = dev_tau (offset 0); the included session sets (§7, 10 units).
      3. For each (alpha, tau0 = dev_tau + offset): score every session on the alpha-gated curve at that tau0,
         aggregate dev FAR_SS-DL and MR_DS-DL over the included sets, and take the per-session deltaMR upper
         bound vs the B0 cell at the SAME tau0.
      4. Feasibility and the joint choice (`select_alpha_tau0`).
    Returns a dict with the chosen alpha/tau0, the full table and the dev tau. Raises `InsufficientData` when
    B0's dev tau is degenerate (-inf), mirroring confirm_h4's "tau = -inf -> not assessable".
    """
    from math import isfinite

    tau_base, n_tau_controls_dropped = dev_tau(points_by_session, d_curves_by_session)
    if not isfinite(tau_base):
        raise InsufficientData("B0 dev tau = -inf (<=10% of dev controls have a finite peak); "
                               "degenerate operating point, M2 not assessable on DISPLACE (H4 supplement §5)")

    # B0 scores per (tau0 offset): B0 is the ungated curve d, scored at tau0 = tau_base + offset.
    b0_scores_by_offset: dict = {}
    for off in TAU0_OFFSET_GRID:
        b0_scores_by_offset[off] = _score_curve_over_sessions(d_curves_by_session, points_by_session,
                                                              tau_base + off)
    # Inclusion sets are defined by unit counts, independent of the operating point, so take them off any B0 cell.
    ref_scores = b0_scores_by_offset[0.0]
    inc_far = included_sessions(ref_scores, lambda sc: sc.n_ssdl_with_cell_controls)
    inc_mr = included_sessions(ref_scores, lambda sc: sc.n_dsdl)
    # Review round 3, M1: the summary-based assessability count includes DS-DL turns P0 cannot score (no defined
    # d(t) in the first second or past the grid). An edge DS-DL turn is dropped, not counted as a miss, as on D1
    # (pilot.score_points), and identically for B0 and every alpha, since gating never moves NaN positions. So the
    # rule (>= 5 sessions with >= 10 DS-DL turns) is certified again on the SCORED counts the bound resamples.
    scored_dsdl = {s: ref_scores[s].n_dsdl for s in ref_scores}
    if len(inc_mr) < ASSESS_MIN_SESSIONS:
        raise InsufficientData(
            f"only {len(inc_mr)} two-language dev sessions keep >= {ASSESS_MIN_DSDL_PER_SESSION} scorable DS-DL "
            f"turns after dropping turns with no defined d(t) (< {ASSESS_MIN_SESSIONS}); M2 not assessable on "
            f"DISPLACE. Scored counts: {scored_dsdl}",
            details={"scored_dsdl_per_session": scored_dsdl,
                     "n_scored_two_lang_sessions_ge10_dsdl": len(inc_mr)})

    # Review round 4, m1: the re-check above, both inclusion sets and every bound rest on ONE scored population,
    # read off the offset-0 B0 cell. That holds because scoring drops only points with no defined d(t), never
    # by tau, and gating never moves NaN positions; it is checked on every cell, so a change that breaks it
    # fails loudly instead of silently mixing populations.
    def unit_counts(scores: dict) -> dict:
        return {s: (sc.n_ssdl_with_cell_controls, sc.n_ssdl_without_cell_controls, sc.n_dsdl, sc.n_ctrl)
                for s, sc in scores.items()}

    ref_units = unit_counts(ref_scores)

    def check_population(scores: dict, alpha: float, tau0: float) -> None:
        got = unit_counts(scores)
        if got != ref_units:
            changed = {s: {"b0_offset0": ref_units.get(s), "cell": got.get(s)}
                       for s in sorted(set(ref_units) | set(got)) if ref_units.get(s) != got.get(s)}
            raise RuntimeError(
                f"the scored population changed at alpha={alpha}, tau0={tau0}: {changed} (counts are SS-DL with "
                f"and without cell controls, DS-DL, controls). Scoring may drop only points with no defined d(t), "
                f"and gating must not move NaN positions; check the gated curve's inputs (e.g. NaN posteriors).")

    def aggregate_cell(scores: dict):
        far = _aggregate({s: scores[s].far_ssdl for s in scores}, inc_far,
                         {s: scores[s].n_ssdl_with_cell_controls for s in scores}, far_aggregation)
        mr = _aggregate({s: scores[s].mr_dsdl for s in scores}, inc_mr,
                        {s: scores[s].n_dsdl for s in scores}, mr_aggregation)
        return far, mr

    cells: list[GridCell] = []
    for alpha in ALPHA_GRID:
        # cache the gated curves for this alpha once; reuse across tau0 offsets (tau0 only shifts the threshold).
        gated = {s: (d_curves_by_session[s] if alpha == 0.0
                     else gated_curves_by_session(s, alpha)) for s in d_curves_by_session}
        for off in TAU0_OFFSET_GRID:
            tau0 = tau_base + off
            scores = _score_curve_over_sessions(gated, points_by_session, tau0)
            check_population(scores, alpha, tau0)
            far, mr = aggregate_cell(scores)
            from math import isnan
            far_defined = not (isinstance(far, float) and isnan(far))
            is_b0 = (alpha == 0.0 and off == 0.0)
            if is_b0 and ALPHA_ZERO_TAU0_ZERO_IS_B0:
                bound = 0.0
                bca = {"estimate": 0.0, "lo": 0.0, "hi": 0.0, "level": 2 * BCA_ONE_SIDED_LEVEL - 1,
                       "z0": float("nan"), "a": float("nan"), "n_boot": 0, "n_dropped": 0,
                       "one_sided_level": BCA_ONE_SIDED_LEVEL, "bca_two_sided_level": 2 * BCA_ONE_SIDED_LEVEL - 1}
                feasible = True
                mr_dropped: list = []
            else:
                b0_cell = b0_scores_by_offset[off]
                deltas, mr_dropped = per_session_delta_mr(scores, b0_cell, inc_mr)
                ub = mr_upper_bound(deltas, rng)
                bound, bca = ub["bound"], ub["bca"]
                feasible = isfinite(bound) and bound <= MR_BOUND_PP + 1e-12
                if GATE_FEASIBLE_REQUIRES_FINITE_TAU and not isfinite(tau0):
                    feasible = False
            cells.append(GridCell(alpha=alpha, tau0=round(tau0, 6), far_ssdl=far, mr_dsdl=mr,
                                  delta_mr_upper_bound=bound, feasible=feasible,
                                  n_sessions_far=len(inc_far), n_sessions_mr=len(inc_mr),
                                  far_defined=far_defined, n_mr_sessions_dropped=len(mr_dropped),
                                  mr_sessions_dropped=mr_dropped, bca=bca))

    chosen = select_alpha_tau0(cells)
    return {"dev_tau": tau_base, "n_tau_controls_dropped": n_tau_controls_dropped,
            "chosen": chosen, "cells": cells,
            "included_far": inc_far, "included_mr": inc_mr, "scored_dsdl_per_session": scored_dsdl,
            "tau0_offset_grid": list(TAU0_OFFSET_GRID), "alpha_grid": list(ALPHA_GRID)}


def _record_scored_counts(record: dict, assess: dict, scored: dict) -> None:
    """Write the scored (post-drop) DS-DL counts beside the reader-count verdict `m2_assessable`, on the frozen
    path AND the not-assessable path, and flag any difference in the number of qualifying sessions (review
    rounds 3 and 4, m2). `scored` is empty when no scoring happened (e.g. a degenerate dev tau): nothing is
    written then."""
    if not scored:
        return
    n_scored = scored["n_scored_two_lang_sessions_ge10_dsdl"]
    record["scored_dsdl_per_session"] = scored["scored_dsdl_per_session"]
    record["n_scored_two_lang_sessions_ge10_dsdl"] = n_scored
    if n_scored != assess.get("n_qualifying"):
        record["flags"]["scored_vs_summary_sessions"] = (
            f"{assess.get('n_qualifying')} sessions qualify on the reader's DS-DL counts, {n_scored} on the "
            f"scorable ones; the assessability verdict and the bound use the scorable set")


def run_p0_tuning(inputs: TuningInputs, *, grid_loader=None, lid_loader=None, write: bool = True) -> dict:
    """Drive the whole P0-gate tuning and write one JSON (M2B §3, §6; H4 supplement §4, §5).

    Reads the dev reader rows (eval guarded off), restricts to the TWO-LANGUAGE dev recordings, checks
    assessability, then -- when assessable -- builds each recording's d curve from its cached dense E3 grid and a
    gated-curve closure from its LID posteriors and reference pair, and runs the pure `tune` core. Heavy loaders
    are injectable so the orchestration is testable through fakes.

    Degenerate / not-assessable cases are recorded (alpha not frozen, M2 not assessable on DISPLACE, D1 stage-2
    reported not tested), not raised.
    """
    import numpy as np
    from csdiar.pipelines import run_m2b as R

    units = Path(inputs.units_dir)
    summary = json.loads((units / "summary.json").read_text())
    if summary.get("split") != "dev":
        raise ValueError("run_p0_tuning: summary.json is not the dev split (dev-only tuning, §3)")

    # M2: the label map is required for a real run (derives each recording's reference pair, hashed below).
    label_map = dict(inputs.label_map or {})
    if not label_map:
        raise ValueError("run_p0_tuning: a --label-map (LD label -> VoxLingua107 code) is required; the "
                         "two-language decision and each recording's reference pair are derived from it (M2)")

    # m3 (review v1): only dense-grid and LID caches for recording IDs listed in the dev summary.json may be
    # loaded; any other ID is refused. Cache provenance beyond this membership is pinned by the freeze record
    # (FREEZE_M2B.json: code hashes, feature checksums, the dev/eval split), not re-checked here.
    summary_ids = {r["recording"] for r in summary.get("recordings", [])}

    def _guard_id(rid: str):
        if rid not in summary_ids:
            raise ValueError(f"refusing cache for {rid!r}: not a recording ID in the dev summary.json (m3)")

    raw_grid_loader = grid_loader or (lambda rid: load_dense_grid(inputs.cache_dir, inputs.model, rid)[0])
    raw_lid_loader = lid_loader or (lambda rid: R.load_lid(inputs.cache_dir, rid))

    def grid_load(rid):
        _guard_id(rid)
        return raw_grid_loader(rid)

    def lid_load(rid):
        _guard_id(rid)
        return raw_lid_loader(rid)

    def lid_labels_of(rid):
        return lid_load(rid)["labels"]

    counts = two_language_sessions(summary, label_map, lid_labels_of)
    assess = assess_tunable(counts["dsdl_per_two_lang_session"])   # resolved two-language recordings only

    rng = np.random.default_rng(SEED)
    record: dict = {
        "split": "dev", "model": inputs.model, "seed": SEED, "gate": "p0_with_e3",
        "code_sha256": {GUARDED_SELF: self_sha256()},
        "choices": open_choices(),
        "constants": {"alpha_grid": list(ALPHA_GRID), "tau0_offset_grid": list(TAU0_OFFSET_GRID),
                      "mr_bound_pp": MR_BOUND_PP, "bca_one_sided_level": BCA_ONE_SIDED_LEVEL,
                      "bca_n_boot": BCA_N_BOOT, "assess_min_sessions": ASSESS_MIN_SESSIONS,
                      "assess_min_dsdl_per_session": ASSESS_MIN_DSDL_PER_SESSION,
                      "ctrl_fire_rate": CTRL_FIRE_RATE, "win_s": WIN, "hop_s": HOP, "lag_k": LAG_K},
        "label_map": label_map,
        "excluded_recordings": counts["excluded"],
        "counts": counts, "m2_assessable": assess, "flags": {},
        "input_sha256": {}, "alpha": None, "tau0": None,
        "d1_stage2": "not tested (alpha not frozen)",
    }
    if counts["excluded"]:
        record["flags"]["excluded_recordings"] = (
            f"{counts['n_excluded']} recording(s) excluded from P0-gate tuning (unmapped / unresolvable LD "
            f"label, or not exactly two distinct resolved codes); counted and flagged, not aborted (M2)")

    # tune on the RESOLVED two-language recordings only (§3); pairs come from the label map (M2), never the
    # caller (review round 2, M1: a caller pair would re-renormalise lambda over unchecked languages).
    if inputs.pair_by_recording:
        raise ValueError("pair_by_recording is not accepted: each recording's pair is derived from the label map")
    pair_by_rec = dict(counts["pair_by_recording"])
    recordings = list(inputs.recordings) or sorted(counts["dsdl_per_two_lang_session"])
    recordings = [r for r in recordings if r in counts["dsdl_per_two_lang_session"]]

    rows_by_rec_all = read_reader_rows(units, "dev")          # guard: refuses eval

    # hash the reader outputs + summary + label map always (they decide assessability and the counts)
    for name in ("ssdl_dev.csv.gz", "turns_dev.csv.gz", "ctrl_dev.csv.gz", "summary.json"):
        p = units / name
        if p.exists():
            record["input_sha256"][str(p)] = sha256_file(p)
    if inputs.label_map_path is not None and Path(inputs.label_map_path).exists():
        record["input_sha256"][str(inputs.label_map_path)] = sha256_file(inputs.label_map_path)
    # The two-language decision read every summary recording's LID labels, so hash every consulted LID cache,
    # excluded recordings included (review round 2, m2).
    for rr in summary.get("recordings", []):
        rid_c = rr.get("recording") if isinstance(rr, dict) else rr
        try:
            lp_c = R.lid_cache_path(inputs.cache_dir, rid_c)
        except Exception:                                    # noqa: BLE001 - an unbuildable path is just unhashed
            continue
        if Path(lp_c).exists():
            record["input_sha256"][str(lp_c)] = sha256_file(lp_c)

    if not assess["tunable"]:
        record["flags"]["not_assessable"] = (
            f"only {assess['n_qualifying']} two-language dev sessions with >= {assess['min_turns']} DS-DL turns "
            f"(< {assess['min_sessions']}); alpha not frozen, M2 not assessable on DISPLACE (H4 supplement §4)")
        if write:
            Path(inputs.out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(inputs.out_path).write_text(json.dumps(record, indent=1, default=_json_default))
        return record

    # Build per-recording d curves and a gated-curve closure from the caches.
    points_by_session: dict = {}
    d_curves: dict = {}
    posteriors: dict = {}
    pairs: dict = {}
    for rid in recordings:
        rows = rows_by_rec_all.get(rid, [])
        points_by_session[rid] = rows
        grid = grid_load(rid)
        lid = lid_load(rid)
        post = np.asarray(lid["post"], dtype=np.float64)
        codes = pair_by_rec.get(rid)
        if codes is None:
            raise ValueError(f"no reference-language pair for {rid}; expected it from the label map")
        i, j = R.pair_indices(lid["labels"], codes[0], codes[1])
        d_curves[rid] = d_curve(grid)
        posteriors[rid] = post
        pairs[rid] = (i, j)
        # hash the inputs that fed this recording's curves
        try:
            dp = dense_grid_cache_path(inputs.cache_dir, inputs.model, rid)
            if Path(dp).exists():
                record["input_sha256"][str(dp)] = sha256_file(dp)
            lp = R.lid_cache_path(inputs.cache_dir, rid)
            if Path(lp).exists():
                record["input_sha256"][str(lp)] = sha256_file(lp)
        except Exception:
            pass

    grids_for_gate = {rid: grid_load(rid) for rid in recordings}

    def gated_for(session: str, alpha: float):
        return gated_curve(grids_for_gate[session], posteriors[session], pairs[session], alpha)

    try:
        result = tune(points_by_session, d_curves, gated_for, rng)
    except InsufficientData as e:
        record["flags"]["not_assessable"] = str(e)
        _record_scored_counts(record, assess, e.details)
        if write:
            Path(inputs.out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(inputs.out_path).write_text(json.dumps(record, indent=1, default=_json_default))
        return record

    record["alpha"] = result["chosen"]["alpha"]
    record["tau0"] = result["chosen"]["tau0"]
    record["dev_tau"] = result["dev_tau"]
    record["n_tau_controls_dropped"] = result["n_tau_controls_dropped"]
    record["d1_stage2"] = "alpha frozen; recordable into FREEZE_H4.json via freeze_h4 add-m2 (H4 supplement §6)"
    record["chosen"] = result["chosen"]
    record["included_far"] = result["included_far"]
    record["included_mr"] = result["included_mr"]
    _record_scored_counts(record, assess, {"scored_dsdl_per_session": result["scored_dsdl_per_session"],
                                           "n_scored_two_lang_sessions_ge10_dsdl": len(result["included_mr"])})
    record["table"] = [{"alpha": c.alpha, "tau0": c.tau0, "far_ssdl": c.far_ssdl, "mr_dsdl": c.mr_dsdl,
                        "delta_mr_upper_bound": c.delta_mr_upper_bound, "feasible": c.feasible,
                        "far_defined": c.far_defined, "n_mr_sessions_dropped": c.n_mr_sessions_dropped,
                        "mr_sessions_dropped": c.mr_sessions_dropped,
                        "one_sided_level": c.bca.get("one_sided_level", BCA_ONE_SIDED_LEVEL),
                        "bca_two_sided_level": c.bca.get("bca_two_sided_level", 2 * BCA_ONE_SIDED_LEVEL - 1),
                        "n_sessions_far": c.n_sessions_far, "n_sessions_mr": c.n_sessions_mr}
                       for c in result["cells"]]

    if write:
        Path(inputs.out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(inputs.out_path).write_text(json.dumps(record, indent=1, default=_json_default))
    return record


def _json_default(o):
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
    label_map = load_label_map(args.label_map) if args.label_map else {}
    return TuningInputs(units_dir=args.units, cache_dir=args.cache_dir, wav_dir=args.wav_dir, out_path=args.out,
                        model=args.model, device=args.device, label_map=label_map,
                        label_map_path=(Path(args.label_map) if args.label_map else None))


def _cmd_cache(args) -> int:
    meta = build_dense_grid(args.wav, args.cache_dir, rid=args.rid, model=args.model, device=args.device,
                            overwrite=args.overwrite)
    print(meta, flush=True)
    return 0


def _cmd_tune(args) -> int:
    inputs = _build_inputs(args)
    rec = run_p0_tuning(inputs)
    print(json.dumps({"written": str(inputs.out_path), "alpha": rec.get("alpha"), "tau0": rec.get("tau0"),
                      "assessable": rec.get("m2_assessable", {}).get("tunable"),
                      "flags": rec.get("flags", {})}, default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", default="cpu", help="torch device for the dense grid (CPU-only on the box)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cache", help="build one recording's dense E3 grid -> cache")
    c.add_argument("--model", default="e3", choices=("e1", "e2", "e3"))
    c.add_argument("--wav", required=True, type=Path)
    c.add_argument("--cache-dir", required=True, type=Path)
    c.add_argument("--rid", default=None)
    c.add_argument("--overwrite", action="store_true")
    c.set_defaults(func=_cmd_cache)

    t = sub.add_parser("tune", help="dev-tune the P0-gate alpha and tau0 -> JSON")
    t.add_argument("--units", required=True, type=Path, help="ingest_displace DEV output dir")
    t.add_argument("--cache-dir", required=True, type=Path, help="LID + dense-grid caches")
    t.add_argument("--wav-dir", required=True, type=Path)
    t.add_argument("--out", required=True, type=Path)
    t.add_argument("--model", default="e3", choices=("e1", "e2", "e3"))
    t.add_argument("--label-map", default=None, type=Path,
                   help="JSON LD-label -> VoxLingua107-code map; required for a real run (M2), hashed into the "
                        "record. The two-language decision and each recording's reference pair are derived from it.")
    t.set_defaults(func=_cmd_tune)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
