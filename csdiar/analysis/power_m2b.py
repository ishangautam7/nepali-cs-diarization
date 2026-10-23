#!/usr/bin/env python3
"""H5 sample-size / power simulation and the SS-DL power check (PREREGISTRATION_M2B.md §7).

Before the DISPLACE 2024 freeze, this module decides -- from DISPLACE **dev** counts only, never an eval
reference -- whether H5 (non-inferiority of each mitigation's MR_DS-DL vs B0) is **confirmatory** or
**descriptive**. The verdict is written into ``FREEZE_M2B.json`` before any eval scoring and is not revisited
once the eval turns are counted (§3, §7, §11). No DISPLACE eval recording is read here; the eval DS-DL count is
PROJECTED from dev rates times the eval hours taken from the archive FILE LISTING (name, uncompressed size) via
a WAV byte rate, so nothing under ``~/data`` is opened and no count is read from an eval reference.

What §7 pins, and what this module does
---------------------------------------
* **Projection** (§7 "Power / size check from dev counts"): projected eval DS-DL count = dev rate of DS-DL turns
  per hour (of speech, §7) times the eval hours from the listing, plus projected per-session eval counts.
* **H5 simulation:** the test exactly as ``confirm_m2b`` will run it -- per-session MR difference vs B0 on
  sessions with >= ``score_m2b.MIN_UNITS`` (10) DS-DL turns, margin ``min(0.02, 0.25 * S_MR)`` with dev's B0
  miss rate standing in for ``S_MR``, a one-sided 95% BCa upper bound below the margin (non-inferior). Simulated
  under NO true inferiority, with the session-level clustering (design effect) and within-session turn noise
  taken from dev. Reports power with a Monte Carlo SE, the design effect, the projected counts and the
  ``verdict`` field ("confirmatory" if power >= 0.8 else "descriptive"); ``freeze_m2b`` copies ``verdict``.
* **SS-DL check** (§7 last sentence): the matching power for the conversation H2/H4 SS-DL excess from dev SS-DL
  counts, reported only (never sets the H5 verdict).
* **JSON output** with the inputs' sha256, the code hash, the seed (20261018), every simulation parameter and the
  results.

Reuse, never re-implement (and never edit): ``csdiar.analysis.stats.bca_ci`` (the frozen BCa; the vectorised
BCa upper bound here is tested to match its upper endpoint on shared inputs, ``_bca_upper_vec`` vs ``bca_ci``),
``csdiar.analysis.pilot`` (``power_h1`` / ``power_h2``, the M2a power simulations this follows),
``csdiar.analysis.score_m2b`` (``MIN_UNITS``), ``csdiar.analysis.tune_m2b`` (the dev tuning record: B0's operating
point and per-session dev scores) and ``csdiar.data.ingest_displace`` (the reader ``summary.json`` and rows).

Open choices (each is a named constant in the OPEN CHOICES block and listed in the final report):
  * ``DIFF_SPREAD_MODEL`` -- how the per-session MR DIFFERENCE is modelled. Default ``"paired_dev"``: model the
    WITHIN-session variance of the per-session difference from dev's discordant paired DS-DL turn counts (turns
    missed by the mitigation only + by B0 only, over n per session), per mitigation from its own discordance rate
    (review round 3, m3), and the BETWEEN-session variance by method of moments (observed variance of dev's
    per-session difference minus the mean within part, floored at 0; the largest across mitigations is used).
    A mitigation whose paired dev counts are missing falls back to ``"b0_mr_sd"`` (B0's own between-session MR
    SD, scaled by ``DIFF_SPREAD_FACTOR``) and is FLAGGED in the record.
  * ``DIFF_SPREAD_FACTOR`` -- the multiplier on the B0-spread FALLBACK proxy (default 1.0).
  * ``PROJECT_PER`` -- projection base. §7 says "per hour of SPEECH"; the listing gives AUDIO hours, so eval
    speech hours = eval audio hours * dev speech fraction (dev speech hours / dev audio hours). Default
    ``"speech"``; falls back to ``"audio"`` (flagged) when dev/eval speech hours are not supplied.
  * ``POWER_SCOPE`` -- ``"per_mitigation"`` (default: H5 is claimed per mitigation, so each claim needs its own
    power; the verdict uses the WEAKEST mitigation's power) vs ``"family"`` (the share of datasets where ALL three
    are declared non-inferior; never larger than the weakest mitigation's power). The simulation draws the three
    mitigations' session effects independently, so the family power is a conservative FLOOR for the real design,
    where the shared B0 correlates them (review round 3, m2). The verdict uses this scope; both are reported.
  * ``HOLM_IN_SIM`` -- whether Holm (m = 3) is applied inside the simulated test. Default ``True`` (the SETTLED
    H5 rule): in each simulated dataset the three mitigations are simulated TOGETHER, each one's non-inferiority
    p-value is obtained by inverting the one-sided 95% BCa upper bound at the margin (the smallest alpha whose
    1 - alpha BCa upper bound is below the margin), Holm is applied over the three, and a mitigation is declared
    NON-INFERIOR iff its Holm-adjusted p < ``HOLM_ALPHA`` (.05). Per-mitigation and family power are reported.
  * ``BETWEEN_SENSITIVITY`` -- a sensitivity table reports the power at 0.5x / 1x / 2x the between-session
    VARIANCE part, so the freeze author sees how the verdict moves with the one quantity dev can only estimate.
  * ``N_SIM`` / ``N_BOOT`` -- Monte Carlo datasets and bootstrap resamples per dataset. §7's confirmatory path
    uses 10,000 whole-session resamples, and so does this simulation by default (review round 2, m1): a
    VECTORISED BCa (bound AND inverted p-value), drawn in batches of datasets of about ``SIM_BATCH_ELEMENTS``
    resample indices so memory stays bounded, which tests pin to ``stats.bca_ci`` / ``confirm_m2b._ni_pvalue`` on
    SHARED resample indices (review m2/m3). Defaults N_SIM = 2000, N_BOOT = 10,000.

All arithmetic is pure and model-free; numpy is imported lazily so the module and its pure helpers import on a
host without numpy. CPU only; never the GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist
from typing import Optional, Sequence

from csdiar.analysis.score_m2b import MIN_UNITS

# ===================================================================================================== #
# frozen constants (PREREGISTRATION_M2B.md §3, §7, §11)
# ===================================================================================================== #
SEED = 20261018                       # §11 logging / resampling seed
MARGIN_ABS = 0.02                     # +2 pp absolute non-inferiority margin (§7)
MARGIN_REL = 0.25                     # +25% relative margin (§7); margin = min(MARGIN_ABS, MARGIN_REL * S_MR)
ONE_SIDED_LEVEL = 0.95                # H5's one-sided 95% upper bound (§7)
POWER_TARGET = 0.80                   # confirmatory iff power >= 0.80 (§7)
MIN_DSDL_PER_SESSION = MIN_UNITS      # §7 inclusion: a session enters H5 only with >= 10 DS-DL turns
PLAN_REFERENCE_TURNS = (600, 1000)    # PLAN.md §7/§10 order-of-magnitude reference (D2 ~600-1,000 DS-DL turns)

# WAV byte rate for the listing-to-seconds helper. DISPLACE 2024 far-field audio is 16 kHz mono 16-bit PCM
# (arXiv 2406.09494). A 44-byte canonical WAV header is subtracted before dividing by the byte rate; the header
# size is a parameter so a non-canonical header can be passed. These are the DEFAULTS of `listing_to_seconds`,
# never a frozen fact about a specific file -- the helper takes them as arguments and is tested synthetically.
WAV_SAMPLE_RATE_HZ = 16000
WAV_CHANNELS = 1
WAV_BYTES_PER_SAMPLE = 2              # 16-bit PCM
WAV_HEADER_BYTES = 44                # canonical PCM WAV header
WAV_BYTE_RATE = WAV_SAMPLE_RATE_HZ * WAV_CHANNELS * WAV_BYTES_PER_SAMPLE   # bytes per second of audio

# ===================================================================================================== #
# OPEN CHOICES -- fixed as named constants (see the module docstring and the final report)
# ===================================================================================================== #
DIFF_SPREAD_MODEL = "paired_dev"      # per-session MR DIFFERENCE model: "paired_dev" models the within- and
#                                       between-session variance of the per-session difference from dev's paired
#                                       DS-DL turn counts (missed by M only / by B0 only), falling back per
#                                       mitigation to "b0_mr_sd" (B0's own between-session MR SD) when those
#                                       counts are missing (flagged in the record).
DIFF_SPREAD_FACTOR = 1.0              # multiplier on the B0-spread FALLBACK proxy (only used when paired_dev data
#                                       is unavailable for a mitigation)
PROJECT_PER = "speech"                # projection base: "speech" (§7) with an audio->speech conversion, or "audio"
POWER_SCOPE = "per_mitigation"        # "per_mitigation" (weakest mitigation must reach 0.8) | "family" (all three; a conservative floor)
HOLM_IN_SIM = True                    # Holm(m=3) inside the simulated test: YES. The settled H5 rule inverts the
#                                       one-sided BCa upper bound at the margin to a non-inferiority p per
#                                       mitigation, Holm over M1-M3, and declares non-inferior iff Holm-adj p<.05.
N_SIM = 2000                          # Monte Carlo datasets
N_BOOT = 10000                        # bootstrap resamples per dataset: the confirmatory test's 10,000 (review r2 m1)
SIM_BATCH_ELEMENTS = 25_000_000       # resample-index entries per batch of datasets (about 200 MB of int64)
N_MITIGATIONS = 3                     # M1, M2, M3 -- the H5 confirmatory family (m = 3)
HOLM_ALPHA = 0.05                     # the Holm-adjusted non-inferiority level (§7: non-inferior iff Holm p<.05)
BETWEEN_SENSITIVITY = (0.5, 1.0, 2.0) # sensitivity table: power at these multiples of the between-session part

GUARDED_SELF = "csdiar/analysis/power_m2b.py"
_N = NormalDist()


# ===================================================================================================== #
# 1. listing -> seconds helper (tested synthetically; never opens audio)
# ===================================================================================================== #
def wav_seconds_from_size(size_bytes: int, byte_rate: int = WAV_BYTE_RATE,
                          header_bytes: int = WAV_HEADER_BYTES) -> float:
    """Seconds of audio in a PCM WAV of ``size_bytes`` uncompressed bytes: (size - header) / byte_rate.

    ``byte_rate`` is bytes per second of audio (sample_rate * channels * bytes_per_sample). A size at or below
    the header is 0.0 s (an empty or header-only file), never negative. Pure arithmetic; opens nothing.
    """
    data = max(0, int(size_bytes) - int(header_bytes))
    if byte_rate <= 0:
        raise ValueError(f"wav_seconds_from_size: byte_rate must be positive, got {byte_rate}")
    return data / float(byte_rate)


def listing_to_seconds(listing: Sequence[tuple], byte_rate: int = WAV_BYTE_RATE,
                       header_bytes: int = WAV_HEADER_BYTES) -> dict:
    """Map a file listing of ``(name, uncompressed_size_bytes)`` to ``{recording_id: seconds}``.

    ``name`` is a WAV file name (e.g. ``B001.wav`` or a path ending in it); the recording id is its stem (the
    base name without directory or ``.wav``/``.flac`` extension), matching ``ingest_displace`` recording ids.
    Only ``.wav`` / ``.flac`` names are taken; a non-audio listing entry is skipped. Pure; opens nothing (the
    size comes from the archive's central directory, §2). Raises on a duplicate recording id so two listing rows
    cannot silently collapse.
    """
    out: dict = {}
    for entry in listing:
        name, size = entry[0], entry[1]
        base = Path(str(name)).name
        low = base.lower()
        if not (low.endswith(".wav") or low.endswith(".flac")):
            continue
        rid = Path(base).stem
        secs = wav_seconds_from_size(size, byte_rate=byte_rate, header_bytes=header_bytes)
        if rid in out:
            raise ValueError(f"listing_to_seconds: duplicate recording id {rid!r} in the file listing")
        out[rid] = secs
    return out


def listing_hours(listing: Sequence[tuple], byte_rate: int = WAV_BYTE_RATE,
                  header_bytes: int = WAV_HEADER_BYTES) -> float:
    """Total audio HOURS across a file listing (sum of per-recording seconds / 3600). Pure."""
    secs = listing_to_seconds(listing, byte_rate=byte_rate, header_bytes=header_bytes)
    return sum(secs.values()) / 3600.0


# ===================================================================================================== #
# 2. projection (PREREGISTRATION_M2B.md §7)
# ===================================================================================================== #
def project_eval_dsdl(dev_dsdl_total: int, dev_hours: float, eval_hours: float) -> dict:
    """Projected eval DS-DL turn count = (dev DS-DL turns / dev hours) * eval hours (§7).

    ``dev_hours`` / ``eval_hours`` are the SAME base (both speech hours, or both audio hours -- the caller picks
    the base per ``PROJECT_PER`` and converts eval audio hours to speech hours before calling). Returns the rate
    per hour, the projected total, and the inputs. Raises on a non-positive dev_hours (the rate is undefined).
    """
    if dev_hours <= 0:
        raise ValueError(f"project_eval_dsdl: dev_hours must be positive, got {dev_hours}")
    rate = dev_dsdl_total / float(dev_hours)
    return {"dev_dsdl_total": int(dev_dsdl_total), "dev_hours": float(dev_hours),
            "eval_hours": float(eval_hours), "rate_per_hour": rate,
            "projected_eval_dsdl": rate * float(eval_hours)}


def project_per_session_counts(dev_per_session_dsdl: dict, dev_hours: float, eval_hours: float,
                               n_eval_sessions: int) -> dict:
    """Projected PER-SESSION eval DS-DL counts for the simulation (§7).

    The projected eval total (``project_eval_dsdl``) is distributed over ``n_eval_sessions`` eval sessions in the
    SHAPE of the dev per-session distribution, then rescaled so their sum equals the projected eval total. This
    keeps the dev clustering shape (a few turn-rich sessions, a tail of small ones) -- which drives the design
    effect and the inclusion rule -- while matching the projected volume, without reading any eval count.

    The shape is the sorted dev counts INTERPOLATED onto ``n_eval_sessions`` evenly spaced quantile positions
    (review round 3, m4). Tiling the sorted counts instead truncated the turn-poor tail whenever there are fewer
    eval than dev sessions (35 dev -> 32 eval dropped the three smallest dev sessions) and repeated the turn-rich
    head when there are more. With equal counts the dev shape is kept exactly. Returns the projected per-session
    counts (floats; the simulation rounds to integers), the eval total used, the number of sessions that clear
    ``MIN_DSDL_PER_SESSION`` under the projection, and how the shape was formed.
    """
    proj = project_eval_dsdl(sum(dev_per_session_dsdl.values()), dev_hours, eval_hours)
    total = proj["projected_eval_dsdl"]
    dev_counts = sorted((float(v) for v in dev_per_session_dsdl.values()), reverse=True)
    if not dev_counts or n_eval_sessions <= 0:
        return {"per_session": [], "eval_total": total, "n_sessions_ge_min": 0,
                "shape_method": "none", **proj}
    n_dev = len(dev_counts)
    if n_eval_sessions == n_dev:
        shaped = list(dev_counts)
        shape_method = "dev_counts"
    else:
        shaped = []
        for j in range(n_eval_sessions):
            x = (j + 0.5) * n_dev / n_eval_sessions - 0.5          # position among the sorted dev counts
            x = min(max(x, 0.0), n_dev - 1.0)
            lo = int(math.floor(x))
            hi = min(lo + 1, n_dev - 1)
            w = x - lo
            shaped.append(dev_counts[lo] * (1.0 - w) + dev_counts[hi] * w)
        shape_method = f"quantile_interpolation ({n_dev} dev -> {n_eval_sessions} eval sessions)"
    s = sum(shaped)
    scaled = [t * (total / s) if s > 0 else 0.0 for t in shaped]
    n_ge = sum(1 for c in scaled if c >= MIN_DSDL_PER_SESSION)
    return {"per_session": scaled, "eval_total": total, "n_sessions_ge_min": n_ge,
            "shape_method": shape_method, **proj}


def dev_speech_fraction(dev_speech_hours: Optional[float], dev_audio_hours: Optional[float]) -> Optional[float]:
    """Dev speech / audio hour ratio for the audio->speech projection conversion, or None when unavailable."""
    if dev_speech_hours is None or dev_audio_hours is None or dev_audio_hours <= 0:
        return None
    return float(dev_speech_hours) / float(dev_audio_hours)


# ===================================================================================================== #
# 3. vectorised BCa for the MEAN: one-sided UPPER bound AND the non-inferiority p-value, sharing draws
#    (tested to match stats.bca_ci on the SAME resample indices -- review m2, m3)
# ===================================================================================================== #
def _norm_cdf_vec(x):
    """Standard-normal CDF, numpy-vectorised (``0.5 * erfc(-x/sqrt2)`` via ``math.erfc``), so the BCa path does
    not build a ``statistics.NormalDist`` per element across every bisection step. Agrees with
    ``statistics.NormalDist().cdf`` to float precision, which the equivalence tests confirm. No scipy needed."""
    import numpy as np
    xa = np.asarray(x, dtype=np.float64)
    erfc = np.vectorize(math.erfc, otypes=[np.float64])
    return 0.5 * erfc(-xa / math.sqrt(2.0))


def _norm_ppf_vec(p):
    """Standard-normal inverse CDF (probit), numpy-vectorised over ``statistics.NormalDist().inv_cdf`` (stdlib,
    no scipy). Used only for the per-row z0 and the scalar level quantile, so the per-element loop is cheap."""
    import numpy as np
    ppf = np.vectorize(_N.inv_cdf, otypes=[np.float64])
    return ppf(np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1.0 - 1e-15))


def _bca_core(data, rng, n_boot: int, idx=None):
    """Shared BCa machinery for the row-mean statistic of ``data`` (shape (n_sim, k)), vectorised.

    Returns ``(boots_sorted, z0, a, theta)`` where ``boots_sorted`` is the per-row SORTED bootstrap means
    (n_sim, n_boot), ``z0`` the percentile-bias correction (share of bootstrap means below the observed mean,
    with the ``1/(B+1)`` edge clamp -- exactly ``stats.bca_ci``), ``a`` the jackknife acceleration from the
    leave-one-out means, and ``theta`` the per-row observed mean. Pass ``idx`` (shape (n_sim, n_boot, k)) to
    resample on CALLER-SUPPLIED indices -- this is how the equivalence tests drive ``stats.bca_ci`` and this
    vectorised path off the SAME draws (review m2/m3); omit it to draw from ``rng``. Rows with < 2 columns give
    all-nan.
    """
    import numpy as np
    x = np.asarray(data, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    n_sim, k = x.shape
    if k < 2:
        nan_boots = np.full((n_sim, n_boot), np.nan)
        return nan_boots, np.full(n_sim, np.nan), np.full(n_sim, np.nan), np.full(n_sim, np.nan)
    theta = x.mean(axis=1)                                                # (n_sim,)
    if idx is None:
        idx = rng.integers(0, k, size=(n_sim, n_boot, k))                # (n_sim, n_boot, k)
    boots = np.take_along_axis(x[:, None, :], idx, axis=2).mean(axis=2)   # (n_sim, n_boot)
    below = (boots < theta[:, None]).sum(axis=1) + 0.5 * (boots == theta[:, None]).sum(axis=1)
    frac = below / n_boot
    frac = np.clip(frac, 1.0 / (n_boot + 1), n_boot / (n_boot + 1))
    z0 = _norm_ppf_vec(frac)
    total = x.sum(axis=1, keepdims=True)
    jack = (total - x) / (k - 1)                                         # (n_sim, k) leave-one-out means
    jbar = jack.mean(axis=1, keepdims=True)
    d = jbar - jack
    num = (d ** 3).sum(axis=1)
    den = 6.0 * ((d ** 2).sum(axis=1)) ** 1.5
    a = np.where(den > 0, num / den, 0.0)
    return np.sort(boots, axis=1), z0, a, theta


def _bca_upper_from_core(boots_sorted, z0, a, level: float):
    """One-sided ``level`` UPPER BCa bound per row, from ``_bca_core``'s outputs. ``level`` a scalar or (n_sim,)
    array. Returns (n_sim,) upper bounds (nan where the core was nan)."""
    import numpy as np
    n_sim, n_boot = boots_sorted.shape
    z_u = _N.inv_cdf(level) if np.isscalar(level) else _norm_ppf_vec(level)
    adj = z0 + (z0 + z_u) / (1.0 - a * (z0 + z_u))
    q = np.clip(_norm_cdf_vec(np.atleast_1d(adj)), 0.0, 1.0)
    pos = q * (n_boot - 1)
    lo_i = np.floor(pos).astype(int)
    hi_i = np.ceil(pos).astype(int)
    frac_i = pos - lo_i
    rows = np.arange(n_sim)
    lo_v = boots_sorted[rows, np.clip(lo_i, 0, n_boot - 1)]
    hi_v = boots_sorted[rows, np.clip(hi_i, 0, n_boot - 1)]
    out = lo_v + (hi_v - lo_v) * frac_i
    bad = ~np.isfinite(z0)
    out[bad] = np.nan
    return out


def _bca_upper_vec(data, level: float, rng, n_boot: int, idx=None):
    """One-sided ``level`` UPPER BCa bounds for the row-mean of ``data`` (shape (n_sim, k)), vectorised.

    Thin wrapper over ``_bca_core`` + ``_bca_upper_from_core``. A two-sided ``stats.bca_ci`` at ``2*level - 1``
    has the SAME upper endpoint; the equivalence test shares ``idx`` with ``bca_ci`` and checks they agree to
    bootstrap rounding, not to a loose tolerance (review m2). Resampling uses ``rng`` unless ``idx`` is passed.
    """
    import numpy as np
    boots_sorted, z0, a, _ = _bca_core(data, rng, n_boot, idx=idx)
    return _bca_upper_from_core(boots_sorted, z0, a, level)


def _ni_pvalue_vec(data, margin: float, rng, n_boot: int, idx=None, n_bisect: int = 28):
    """Vectorised non-inferiority p-value per row of ``data`` (shape (n_sim, k)), mirroring
    ``confirm_m2b._ni_pvalue``: the smallest alpha at which the one-sided (1 - alpha) BCa UPPER bound is still
    at or below ``margin``, i.e. ``p = 1 - (largest one-sided level L with upper_bound(L) <= margin)``.

    The upper bound is monotone increasing in the level, so for every row we bisect the level in (0.5, 1) at
    once. A row whose MEDIAN (level 0.5) bound already exceeds the margin gets p = 1 (not non-inferior at all),
    exactly as the scalar routine returns. All rows share one ``_bca_core`` draw so the p-values and the raw
    upper bound agree bit-for-bit, and the equivalence test pins this against ``confirm_m2b._ni_pvalue`` on the
    same indices. Returns (n_sim,) p-values; nan rows (k < 2) give nan.
    """
    import numpy as np
    boots_sorted, z0, a, _ = _bca_core(data, rng, n_boot, idx=idx)
    n_sim = boots_sorted.shape[0]
    finite = np.isfinite(z0)
    p = np.full(n_sim, np.nan)
    if not finite.any():
        return p
    med = _bca_upper_from_core(boots_sorted, z0, a, 0.5)
    # rows whose median bound already exceeds the margin: p = 1 (even the median bound is above the margin)
    over = finite & ~(med <= margin)
    p[over] = 1.0
    live = finite & (med <= margin)
    if live.any():
        lo = np.full(n_sim, 0.5)
        hi = np.full(n_sim, 1.0 - 1e-6)
        for _ in range(n_bisect):
            mid = (lo + hi) / 2.0
            ub = _bca_upper_from_core(boots_sorted, z0, a, mid)
            ok = np.isfinite(ub) & (ub <= margin)                        # bound still below margin at this level
            lo = np.where(live & ok, mid, lo)
            hi = np.where(live & ~ok, mid, hi)
        p_live = np.clip(1.0 - lo, 0.0, 1.0)
        p[live] = p_live[live]
    return p


def _holm_reject_batch(pvals, alpha: float = HOLM_ALPHA):
    """Holm step-down over the m mitigations IN EACH simulated dataset, vectorised: ``pvals`` is (n_sim, m);
    returns a (n_sim, m) boolean of which hypotheses REJECT (declare non-inferior) at family level ``alpha``.

    Holm: sort each row ascending, the i-th smallest (0-based) is compared against ``alpha / (m - i)``, and once
    one fails every later (larger) p also fails. A nan p-value never rejects. This is the per-dataset analogue of
    ``stats.holm`` (which returns adjusted p); here we only need the reject/accept decision per mitigation.
    """
    import numpy as np
    p = np.asarray(pvals, dtype=np.float64)
    if p.ndim == 1:
        p = p[None, :]
    n_sim, m = p.shape
    reject = np.zeros((n_sim, m), dtype=bool)
    p_filled = np.where(np.isfinite(p), p, np.inf)
    order = np.argsort(p_filled, axis=1)                                 # ascending p per row
    rows = np.arange(n_sim)[:, None]
    thresh = alpha / (m - np.arange(m))[None, :]                         # alpha/m, alpha/(m-1), ..., alpha/1
    sorted_p = np.take_along_axis(p_filled, order, axis=1)
    passes = sorted_p <= thresh                                          # per sorted position
    # Holm stops at the FIRST failure: a position rejects iff it and all earlier positions pass
    cum_fail = np.cumsum(~passes, axis=1)
    sorted_reject = (cum_fail == 0) & passes
    # scatter the sorted decisions back to the original mitigation order
    np.put_along_axis(reject, order, sorted_reject, axis=1)
    return reject


# ===================================================================================================== #
# 4. H5 simulation (PREREGISTRATION_M2B.md §7)
# ===================================================================================================== #
def model_variance_parts(paired_counts_by_session, per_session_diff, rng=None):
    """Model the per-session MR DIFFERENCE's two variance parts from dev's PAIRED DS-DL turn counts (review M2).

    ``paired_counts_by_session`` maps a dev session -> ``{n_dsdl, missed_by_mit_only, missed_by_b0_only, ...}``
    (``tune_m2b.paired_turn_counts_vs_b0``'s output). Under NO true difference the per-session ΔMR
    (mitigation MR minus B0 MR) has, per session with ``n`` DS-DL turns:
      * a WITHIN-session (sampling) variance ``(missed_by_M_only + missed_by_B0_only) / n^2`` -- the discordant
        turns are the only ones that move ΔMR = (misses_M - misses_B0)/n, and under H0 each discordant turn is an
        independent +-1/n swing, so Var ≈ (#discordant)/n^2. The RATE of discordance is taken from dev.
      * a BETWEEN-session variance estimated by METHOD OF MOMENTS: the observed variance of dev's per-session
        ΔMR minus the mean within-session part, FLOORED at 0 (a negative moment estimate means no detectable
        true between-session spread).
    Returns ``{within_var_mean, between_var, per_session_within_var (dict), discordance_rate, n_sessions,
    observed_diff_var}``. Pure given the inputs (no RNG needed; ``rng`` accepted for signature symmetry).
    """
    import numpy as np
    sessions = sorted(paired_counts_by_session)
    within = {}
    rates = []
    for s in sessions:
        pc = paired_counts_by_session[s]
        n = float(pc.get("n_dsdl", 0) or 0)
        disc = float(pc.get("missed_by_mit_only", 0)) + float(pc.get("missed_by_b0_only", 0))
        if n > 0:
            within[s] = disc / (n * n)          # Var(ΔMR | session) ≈ (#discordant)/n^2 under H0
            rates.append(disc / n)
        else:
            within[s] = float("nan")
    finite_within = [v for v in within.values() if v == v]
    within_var_mean = float(np.mean(finite_within)) if finite_within else float("nan")
    # observed between-session variance of dev's per-session difference, MoM minus mean within, floored at 0
    diffs = [per_session_diff[s] for s in sessions if s in per_session_diff and per_session_diff[s] == per_session_diff[s]]
    observed_diff_var = float(np.var(diffs, ddof=1)) if len(diffs) >= 2 else float("nan")
    if observed_diff_var == observed_diff_var and within_var_mean == within_var_mean:
        between_var = max(0.0, observed_diff_var - within_var_mean)
    else:
        between_var = float("nan")
    return {"within_var_mean": within_var_mean, "between_var": between_var,
            "per_session_within_var": within,
            "discordance_rate": float(np.mean(rates)) if rates else float("nan"),
            "n_sessions": len(sessions), "observed_diff_var": observed_diff_var}


def _project_within_var(projected_counts, discordance_rate):
    """Per projected eval session, the within-session variance of ΔMR under H0: (rate * n)/n^2 = rate / n, with
    ``rate`` the dev discordance rate (#discordant turns per DS-DL turn) applied at the PROJECTED turn count. A
    session with ``n`` projected turns thus has within-var ``discordance_rate / n`` (fewer turns -> noisier)."""
    import numpy as np
    counts = np.asarray(projected_counts, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        wv = np.where(counts > 0, discordance_rate / counts, np.nan)
    return wv


def simulate_h5_power(per_session_counts: Sequence[float], s_mr: float,
                      within_var, between_var: float, rng, *, n_sim: int = N_SIM, n_boot: int = N_BOOT,
                      level: float = ONE_SIDED_LEVEL, margin_abs: float = MARGIN_ABS,
                      margin_rel: float = MARGIN_REL, min_units: int = MIN_DSDL_PER_SESSION,
                      n_mitigations: int = N_MITIGATIONS, holm_in_sim: bool = HOLM_IN_SIM,
                      holm_alpha: float = HOLM_ALPHA) -> dict:
    """Power of the SETTLED H5 non-inferiority rule under NO true inferiority (true mean per-session ΔMR = 0), §7.

    The rule (``HOLM_IN_SIM = True``): in EACH simulated dataset, the three mitigations M1/M2/M3 are simulated
    TOGETHER on the same INCLUDED sessions (those with >= ``min_units`` projected DS-DL turns); for each
    mitigation the non-inferiority p-value is obtained by INVERTING the one-sided ``level`` (95%) BCa upper bound
    at the margin ``min(margin_abs, margin_rel * s_mr)`` (the smallest alpha at which the 1 - alpha BCa upper
    bound is below the margin, ``_ni_pvalue_vec``); Holm is applied over the three mitigations; a mitigation is
    declared NON-INFERIOR iff its Holm-adjusted p < ``holm_alpha`` (.05).

    Each mitigation's per-session ΔMR draws a session random effect ~ Normal(0, sqrt(``between_var``)) PLUS
    within-session noise ~ Normal(0, sqrt(within_var_s)) per session. ``within_var`` is a scalar, a per-session
    array shared by every mitigation, or (review round 3, m3) a 2-D array with one row per mitigation, so each
    mitigation is sized with its OWN within-session variance: the variance of the per-session difference
    estimated from that mitigation's discordant DS-DL counts (``model_variance_parts`` / ``_project_within_var``).
    A per-session array, or a row, covers either every projected session (the kept ones are taken) or only the
    kept ones. The mitigations draw INDEPENDENT session effects. ``s_mr`` sets the margin only.

    Returns per-mitigation power (share of datasets declaring that mitigation non-inferior) and the FAMILY power
    (share of datasets where ALL three are declared non-inferior), each with a Monte Carlo SE, plus the margin,
    the design effect, and the variance parts used. Because the session effects are independent across
    mitigations, the family power is a conservative FLOOR (review round 3, m2): in the real test every
    mitigation is differenced against the same B0 on the same sessions, which correlates the three positively and
    makes "all three non-inferior" more likely than the product of independent draws. Per-mitigation power, which
    sets the default verdict, does not depend on this.
    """
    import numpy as np
    counts = np.asarray([c for c in per_session_counts if c >= min_units], dtype=np.float64)
    k = len(counts)
    margin = min(margin_abs, margin_rel * s_mr) if s_mr == s_mr else float("nan")
    mask = np.asarray([c >= min_units for c in per_session_counts], dtype=bool)

    def _align(v):
        # scalar broadcast, or a per-session array over ALL projected sessions (keep those >= min_units) or over
        # the kept sessions only; non-finite or non-positive entries contribute no within-session noise.
        v = np.asarray(v, dtype=np.float64)
        if v.ndim == 0:
            out = np.full(k, float(v))
        else:
            out = v[mask] if v.shape[0] == mask.shape[0] else v[:k]
        return np.where(np.isfinite(out) & (out > 0), out, 0.0)

    wv = np.asarray(within_var, dtype=np.float64)
    if wv.ndim == 2:
        if wv.shape[0] != n_mitigations:
            raise ValueError(f"simulate_h5_power: within_var has {wv.shape[0]} rows for {n_mitigations} mitigations")
        within_var_mat = np.stack([_align(wv[mi]) for mi in range(n_mitigations)]) if n_mitigations \
            else np.zeros((0, k))
    else:
        within_var_mat = np.tile(_align(wv), (n_mitigations, 1))
    within_sd_mat = np.sqrt(within_var_mat)
    between_sd = math.sqrt(max(0.0, between_var)) if between_var == between_var else 0.0
    per_mit_within_mean = [float(np.mean(within_var_mat[mi])) if k else float("nan") for mi in range(n_mitigations)]
    base = {"margin": margin, "margin_abs": margin_abs, "margin_rel_times_s_mr": margin_rel * s_mr,
            "s_mr": s_mr, "n_sessions_included": int(k), "n_sim": n_sim, "n_boot": n_boot,
            "level": level, "holm_in_sim": holm_in_sim, "holm_alpha": holm_alpha,
            "n_mitigations": n_mitigations, "between_var": between_var,
            "within_var_mean": float(np.mean(within_var_mat)) if k and n_mitigations else float("nan"),
            "within_var_mean_per_mitigation": per_mit_within_mean,
            "family_power_assumption": "independent session effects per mitigation, so power_family is a "
                                       "conservative floor for the real design, where the shared B0 correlates "
                                       "the mitigations positively; per-mitigation power is unaffected"}
    if k < 5 or not (margin == margin):
        # §7: with fewer than 5 included sessions the one-sided BCa cannot reach .05 -- not assessable.
        return {**base, "power_family": float("nan"), "power_per_mitigation": [float("nan")] * n_mitigations,
                "mc_se_family": float("nan"), "design_effect": float("nan"),
                "not_assessable": f"only {k} projected sessions clear {min_units} DS-DL turns (< 5)"
                if k < 5 else "margin undefined (S_MR is nan)"}

    naive_var = float(np.mean(within_var_mat))
    design_effect = (naive_var + between_sd ** 2) / naive_var if naive_var > 0 else float("nan")
    design_effect_per_mitigation = [(w + between_sd ** 2) / w if w > 0 else float("nan")
                                    for w in per_mit_within_mean]

    # simulate all three mitigations jointly per dataset; collect per-mitigation NI p-values. Review round 2, m1:
    # the bootstrap is drawn in batches of datasets so the (batch, n_boot, k) resample-index array stays near
    # SIM_BATCH_ELEMENTS entries whatever n_sim and n_boot are (10,000 resamples, as the confirmatory test uses).
    sim_batch = max(1, SIM_BATCH_ELEMENTS // max(1, n_boot * k))
    per_mit_reject = np.zeros((n_sim, n_mitigations), dtype=bool) if not holm_in_sim else None
    pvals = np.empty((n_sim, n_mitigations), dtype=np.float64)
    for mi in range(n_mitigations):
        sess_eff = rng.normal(0.0, between_sd, size=(n_sim, k))
        noise = rng.normal(0.0, 1.0, size=(n_sim, k)) * within_sd_mat[mi][None, :]
        data = sess_eff + noise                                       # true mean 0 (no inferiority)
        for lo in range(0, n_sim, sim_batch):
            hi = min(n_sim, lo + sim_batch)
            if holm_in_sim:
                pvals[lo:hi, mi] = _ni_pvalue_vec(data[lo:hi], margin, rng, n_boot)
            else:
                upper = _bca_upper_vec(data[lo:hi], level, rng, n_boot)
                per_mit_reject[lo:hi, mi] = np.isfinite(upper) & (upper < margin)

    if holm_in_sim:
        reject = _holm_reject_batch(pvals, alpha=holm_alpha)          # (n_sim, m) after Holm
    else:
        reject = per_mit_reject

    per_mit_power = reject.mean(axis=0).tolist()
    family_reject = reject.all(axis=1)
    family_power = float(family_reject.mean())
    mc_se_family = math.sqrt(family_power * (1.0 - family_power) / n_sim) if n_sim > 0 else float("nan")
    per_mit_se = [math.sqrt(p * (1.0 - p) / n_sim) if n_sim > 0 else float("nan") for p in per_mit_power]
    return {**base, "power_per_mitigation": per_mit_power, "power_per_mitigation_se": per_mit_se,
            "power_family": family_power, "mc_se_family": mc_se_family, "design_effect": design_effect,
            "design_effect_per_mitigation": design_effect_per_mitigation}


# ===================================================================================================== #
# 5. SS-DL check (PREREGISTRATION_M2B.md §7 last sentence) -- reported only
# ===================================================================================================== #
def simulate_ssdl_power(per_session_ssdl_counts: Sequence[float], s_delta_far: float,
                        between_session_sd: float, rng, n_sim: int = N_SIM, n_boot: int = N_BOOT,
                        sesoi_rel: float = 0.25, min_units: int = MIN_UNITS) -> dict:
    """Matching power for the conversation H2/H4 SS-DL excess (ΔFAR), from dev SS-DL counts (§7). REPORTED ONLY.

    This mirrors the H4 test's sign-flip-over-sessions on the per-session cell-standardised ΔFAR (a mean > 0 with
    a >= 25% relative effect), not a non-inferiority bound: a session enters with >= ``min_units`` SS-DL points
    (with cell controls), the per-session ΔFAR has a between-session SD ``between_session_sd`` and within-session
    SD ``sqrt(p(1-p)/n)`` with ``p = s_delta_far`` as a crude fire-rate proxy, and the true effect is
    ``s_delta_far`` (the smallest-effect ΔFAR). Power is the share of datasets whose one-sided sign-flip p-value
    (``stats.signflip``) is < .05. Never sets the H5 verdict.
    """
    import numpy as np
    from csdiar.analysis.stats import signflip
    counts = np.asarray([c for c in per_session_ssdl_counts if c >= min_units], dtype=np.float64)
    k = len(counts)
    base = {"s_delta_far": s_delta_far, "sesoi_rel": sesoi_rel, "n_sessions_included": int(k),
            "n_sim": n_sim, "between_session_sd": between_session_sd,
            # review round 3, m5: the within-session part is an assumption, not derived from dev
            "within_sd_source": "assumed_binomial_proxy",
            "within_sd_model": "sqrt(p(1-p)/n) with p = s_delta_far and n the session's SS-DL points"}
    if k < 5:
        return {**base, "power": float("nan"), "mc_se": float("nan"),
                "not_assessable": f"only {k} dev sessions clear {min_units} SS-DL points (< 5)"}
    p = min(max(s_delta_far, 1e-6), 1 - 1e-6)
    within_sd = np.sqrt(p * (1.0 - p) / counts)
    true_mean = s_delta_far
    hits = 0
    for _ in range(n_sim):
        vals = true_mean + rng.normal(0.0, between_session_sd, size=k) + rng.normal(0.0, 1.0, size=k) * within_sd
        res = signflip(vals, rng)
        if res["p_one"] < 0.05:
            hits += 1
    power = hits / n_sim
    mc_se = math.sqrt(power * (1.0 - power) / n_sim) if n_sim > 0 else float("nan")
    return {**base, "power": power, "mc_se": mc_se}


def ssdl_between_sd_from_dev(per_session_delta_far: dict) -> dict:
    """Between-session SD of the SS-DL check from dev's per-session cell-standardised ΔFAR (review m4).

    ``per_session_delta_far`` maps dev session -> its ``delta_far_std`` at B0's operating point (from the tuning
    record's per-session values over the H4-included sessions). Returns ``{between_sd, source, n_sessions}``:
    when at least two finite dev values are present the between-session SD is their sample SD (``source`` =
    'dev_delta_far'); otherwise a flagged assumed prior (``source`` = 'assumed', between_sd = nan so the caller
    substitutes its ``--ssdl-between-sd`` default and records it as an ASSUMPTION, not dev-derived).
    """
    import numpy as np
    vals = [float(v) for v in per_session_delta_far.values() if v == v]
    if len(vals) >= 2:
        return {"between_sd": float(np.std(np.asarray(vals, dtype=np.float64), ddof=1)),
                "source": "dev_delta_far", "n_sessions": len(vals)}
    return {"between_sd": float("nan"), "source": "assumed", "n_sessions": len(vals)}


# ===================================================================================================== #
# 6. inputs: dev record + per-session dev scores (PREREGISTRATION_M2B.md §3, §7)
# ===================================================================================================== #
def b0_mr_from_tuning(tuning_record: dict) -> float:
    """B0's dev MR_DS-DL at its frozen threshold, from a ``tune_m2b`` record's operating points. S_MR stand-in."""
    op = tuning_record.get("operating_points", {}).get("B0", {})
    mr = op.get("mr_dsdl")
    if mr is None:
        raise ValueError("b0_mr_from_tuning: tuning record has no B0 operating point MR_DS-DL")
    return float(mr)


def between_session_sd_of(per_session_mr: dict) -> float:
    """Between-session SD of B0's per-session dev MR_DS-DL -- the FALLBACK diff-spread proxy
    (DIFF_SPREAD_MODEL='b0_mr_sd', used only when a mitigation's paired dev counts are missing)."""
    import numpy as np
    vals = np.asarray([float(v) for v in per_session_mr.values()], dtype=np.float64)
    if len(vals) < 2:
        return 0.0
    return float(np.std(vals, ddof=1))


# ----- read the dev inputs the simulation needs FROM the tuning record (review m1: no out-of-band file) -----
def b0_per_session_mr_from_tuning(tuning_record: dict) -> dict:
    """B0's per-session dev MR_DS-DL at its frozen operating point, read from the tuning record's new
    ``per_session_at_operating_points`` (``tune_m2b`` now writes it; review m1 drops the out-of-band file).
    Returns ``{recording: mr_dsdl}`` over every scored dev session (nan values kept so the caller can filter)."""
    ps = tuning_record.get("per_session_at_operating_points", {}).get("B0", {}).get("per_session")
    if ps is None:
        raise ValueError("b0_per_session_mr_from_tuning: tuning record has no "
                         "per_session_at_operating_points.B0.per_session (regenerate the tuning record with the "
                         "updated tune_m2b that writes per-session dev values -- review m1)")
    return {rid: float(v["mr_dsdl"]) for rid, v in ps.items()}


def h5_included_sessions_from_tuning(tuning_record: dict) -> list:
    """The H5 (DS-DL) inclusion list B0's operating point produced, from the tuning record's per-session block."""
    inc = tuning_record.get("per_session_at_operating_points", {}).get("B0", {}).get("inclusion", {})
    return list(inc.get("h5_dsdl", []))


def check_b0_mr_consistency(tuning_record: dict, tol: float = 1e-9) -> dict:
    """Refuse a tuning record whose B0 operating-point MR_DS-DL disagrees with the MEAN of its per-session MR
    over the H5-included sessions (review m1). Both are the session-mean at B0's frozen threshold, so they must
    agree to ``tol``; a mismatch means a stale or mis-scoped record and the sizing simulation must not run on it.

    Returns ``{s_mr, mean_over_included, n_included, diff}`` on success; raises ValueError on a mismatch."""
    import numpy as np
    s_mr = b0_mr_from_tuning(tuning_record)
    per_session = b0_per_session_mr_from_tuning(tuning_record)
    included = h5_included_sessions_from_tuning(tuning_record)
    vals = [per_session[s] for s in included if s in per_session and per_session[s] == per_session[s]]
    if not vals:
        raise ValueError("check_b0_mr_consistency: no H5-included session has a finite per-session MR; "
                         "cannot verify B0's S_MR against the per-session values")
    mean_over_included = float(np.mean(vals))
    diff = abs(mean_over_included - s_mr)
    if diff > tol:
        raise ValueError(f"check_b0_mr_consistency: B0 operating-point MR_DS-DL {s_mr!r} disagrees with the mean "
                         f"of its per-session MR over the {len(vals)} H5-included sessions "
                         f"({mean_over_included!r}); |diff| = {diff:.3e} > tol {tol:.0e}. The record is stale or "
                         "mis-scoped -- regenerate it; the sizing simulation refuses to run on it (review m1).")
    return {"s_mr": s_mr, "mean_over_included": mean_over_included, "n_included": len(vals), "diff": diff}


def paired_counts_for_mitigation(tuning_record: dict, mitigation: str) -> Optional[dict]:
    """A mitigation's per-session paired DS-DL turn counts vs B0, from the tuning record
    (``per_session_at_operating_points[<mit>].paired_turns_vs_b0``), or None when the record did not supply them
    (then the caller falls back to the B0-spread proxy and FLAGS it, review M1/M2)."""
    entry = tuning_record.get("per_session_at_operating_points", {}).get(mitigation, {})
    return entry.get("paired_turns_vs_b0")


def b0_per_session_delta_far_from_tuning(tuning_record: dict) -> dict:
    """B0's per-session cell-standardised ΔFAR at its operating point, for the SS-DL check's between-session
    spread (review m4). Keyed by recording; nan values kept."""
    ps = tuning_record.get("per_session_at_operating_points", {}).get("B0", {}).get("per_session", {})
    return {rid: float(v.get("delta_far_std")) for rid, v in ps.items() if v.get("delta_far_std") is not None}


def dev_counts_from_summary(summary: dict) -> dict:
    """Pull dev per-session DS-DL and SS-DL counts and the recording list from ``ingest_displace``'s summary.json.

    Returns per-session dicts keyed by recording id. Never reads an eval reference -- the summary must be the DEV
    split (a guard refuses otherwise)."""
    if summary.get("split") != "dev":
        raise ValueError(f"dev_counts_from_summary: summary is split {summary.get('split')!r}, expected 'dev' "
                         "(the H5 sizing simulation reads DEV counts only, §7)")
    recs = summary.get("recordings", [])
    dsdl = {r["recording"]: int(r.get("n_dsdl", 0)) for r in recs}
    ssdl = {r["recording"]: int(r.get("n_ssdl", 0)) for r in recs}
    return {"dsdl_per_session": dsdl, "ssdl_per_session": ssdl, "n_recordings": len(recs),
            "dsdl_total": sum(dsdl.values()), "ssdl_total": sum(ssdl.values())}


# ===================================================================================================== #
# 7. hashing
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
    return sha256_file(_repo_root() / GUARDED_SELF)


def _open_choices() -> dict:
    return {"diff_spread_model": DIFF_SPREAD_MODEL, "diff_spread_factor": DIFF_SPREAD_FACTOR,
            "project_per": PROJECT_PER, "power_scope": POWER_SCOPE, "holm_in_sim": HOLM_IN_SIM,
            "holm_alpha": HOLM_ALPHA, "between_sensitivity": list(BETWEEN_SENSITIVITY),
            "n_sim": N_SIM, "n_boot": N_BOOT, "n_mitigations": N_MITIGATIONS}


# ===================================================================================================== #
# 8. the whole power/size check -> the record with the verdict
# ===================================================================================================== #
def run_power_check(summary: dict, tuning_record: dict, eval_listing: Sequence[tuple],
                    *, dev_speech_hours: Optional[float] = None,
                    dev_audio_hours: Optional[float] = None, s_delta_far: float = 0.05,
                    ssdl_between_sd: float = 0.02, n_sim: int = N_SIM, n_boot: int = N_BOOT,
                    n_sim_sensitivity: Optional[int] = None, n_boot_sensitivity: Optional[int] = None,
                    byte_rate: int = WAV_BYTE_RATE, header_bytes: int = WAV_HEADER_BYTES,
                    input_hashes: Optional[dict] = None, seed: int = SEED,
                    mitigations: Sequence[str] = ("M1", "M2", "M3")) -> dict:
    """The H5 sample-size simulation and the SS-DL check, end to end, returning the record with ``verdict``.

    Inputs (review m1 drops the out-of-band ``--b0-per-session-mr`` file; everything dev comes from the record):
      * ``summary`` -- the DEV reader ``summary.json`` (per-session DS-DL / SS-DL counts);
      * ``tuning_record`` -- the ``tune_m2b`` dev record. B0's operating-point MR_DS-DL is S_MR; B0's per-session
        MR, each mitigation's per-session paired DS-DL turn counts vs B0, the H4/H5 inclusion lists and B0's
        per-session ΔFAR all come from its ``per_session_at_operating_points`` block. The record is REFUSED when
        B0's S_MR disagrees with the mean of its per-session MR over the H5-included sessions (tol 1e-9);
      * ``eval_listing`` -- the eval archive FILE LISTING as ``(name, uncompressed_size_bytes)`` (eval hours;
        never an eval reference count);
      * ``dev_speech_hours`` / ``dev_audio_hours`` -- dev hours for the per-speech-hour projection (§7); when
        absent the projection falls back to AUDIO hours and is flagged.

    The per-session MR DIFFERENCE is modelled per mitigation from dev's paired DS-DL counts (review M2): a
    WITHIN-session variance from the discordant turns (missed by M only + missed by B0 only) and a BETWEEN-session
    variance by method of moments. A mitigation whose paired counts are missing falls back to the B0-spread
    proxy, FLAGGED in the record. The settled H5 rule (``HOLM_IN_SIM = True``) simulates the three mitigations
    together, inverts each one's BCa bound to a non-inferiority p, Holm over three, non-inferior iff Holm-adj
    p < .05 -- so per-mitigation AND family power are reported, plus a sensitivity table of power at 0.5x / 1x /
    2x the between-session part (the lever the freeze author reads). S_MR is the SESSION MEAN (§7). The verdict is
    "confirmatory" iff the chosen-scope power >= ``POWER_TARGET`` (0.8), else "descriptive".
    """
    import numpy as np
    if len(mitigations) != N_MITIGATIONS:
        raise ValueError(f"run_power_check: H5's family is {N_MITIGATIONS} mitigations (M1-M3), got {list(mitigations)}")
    # Random streams (review round 3, m1). The primary H5 simulation and every row of the between-variance
    # sensitivity table start from the SAME fresh generator (common random numbers): the rows then differ only in
    # the between-session variance, and the 1x row reproduces the primary exactly when it runs at the primary's
    # n_sim / n_boot. The SS-DL check draws from its own child stream of the same seed.
    def h5_rng():
        return np.random.default_rng(seed)
    ssdl_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])

    counts = dev_counts_from_summary(summary)
    # review m1: read S_MR and B0's per-session MR from the record, and refuse a record whose B0 MR disagrees
    # with the mean of its per-session MR over the H5-included sessions (tol 1e-9).
    consistency = check_b0_mr_consistency(tuning_record, tol=1e-9)
    s_mr = consistency["s_mr"]
    b0_per_session_mr = b0_per_session_mr_from_tuning(tuning_record)
    h5_included = h5_included_sessions_from_tuning(tuning_record)

    diff_flags: list = []

    # eval hours from the listing (AUDIO hours); dev hours for the projection base
    eval_seconds = listing_to_seconds(eval_listing, byte_rate=byte_rate, header_bytes=header_bytes)
    eval_audio_hours = sum(eval_seconds.values()) / 3600.0
    n_eval_sessions = len(eval_seconds)

    projection_flags = []
    frac = dev_speech_fraction(dev_speech_hours, dev_audio_hours)
    if PROJECT_PER == "speech" and frac is not None:
        base_eval_hours = eval_audio_hours * frac
        base_dev_hours = dev_speech_hours
        project_base = "speech"
    else:
        base_eval_hours = eval_audio_hours
        base_dev_hours = dev_audio_hours if dev_audio_hours else eval_audio_hours  # best-effort
        project_base = "audio"
        if PROJECT_PER == "speech":
            projection_flags.append("dev/eval speech hours not supplied; projected on AUDIO hours instead")

    if base_dev_hours is None or base_dev_hours <= 0:
        # with no dev hours at all we cannot form a rate; fall back to dev-hours = eval-hours ratio of 1 and flag
        base_dev_hours = base_eval_hours if base_eval_hours > 0 else 1.0
        projection_flags.append("dev hours not supplied; projection rate uses eval hours as the base (flagged)")

    per_session = project_per_session_counts(counts["dsdl_per_session"], base_dev_hours, base_eval_hours,
                                             n_eval_sessions)
    projected_counts = per_session["per_session"]

    # ----- per-mitigation variance modelling from dev's PAIRED DS-DL counts (review M2), with the fallback -----
    b0_proxy_sd = between_session_sd_of(b0_per_session_mr)
    b0_proxy_var = (DIFF_SPREAD_FACTOR * b0_proxy_sd) ** 2
    # B0's per-session difference ΔMR vs itself is identically 0, so the MoM "observed difference variance" the
    # paired model needs is per MITIGATION; for the fallback we use B0's own between-session MR spread.
    mitigation_models: dict = {}
    for mit in mitigations:
        paired = paired_counts_for_mitigation(tuning_record, mit)
        if paired:
            # per-session dev ΔMR for the MoM between-session estimate: (misses_M - misses_B0)/n per session.
            per_session_diff = {}
            for s, pc in paired.items():
                n = float(pc.get("n_dsdl", 0) or 0)
                if n > 0:
                    per_session_diff[s] = (float(pc.get("missed_by_mit_only", 0))
                                           - float(pc.get("missed_by_b0_only", 0))) / n
            vp = model_variance_parts(paired, per_session_diff)
            mitigation_models[mit] = {"source": "paired_dev", "within_var_mean": vp["within_var_mean"],
                                      "between_var": vp["between_var"],
                                      "discordance_rate": vp["discordance_rate"],
                                      "observed_diff_var": vp["observed_diff_var"],
                                      "n_dev_sessions": vp["n_sessions"]}
        else:
            diff_flags.append(f"{mit}: paired dev DS-DL counts missing; fell back to the B0-spread proxy "
                              f"(DIFF_SPREAD_MODEL='b0_mr_sd', between_var={b0_proxy_var:.3e})")
            mitigation_models[mit] = {"source": "b0_mr_sd_fallback", "within_var_mean": float("nan"),
                                      "between_var": b0_proxy_var, "discordance_rate": float("nan"),
                                      "observed_diff_var": float("nan"), "n_dev_sessions": 0}

    # the projected within-session variance per session, PER MITIGATION from its own dev discordance rate (review
    # round 3, m3: averaging the rates under-sized the noisiest mitigation, the one most likely to set the
    # verdict). A mitigation without a paired rate takes the mean rate of those that have one, flagged. When no
    # paired counts exist at all, fall back to the binomial scale s_mr(1-s_mr)/n * 2 for every mitigation, flagged.
    paired_rates = {mit: m["discordance_rate"] for mit, m in mitigation_models.items()
                    if m["source"] == "paired_dev" and m["discordance_rate"] == m["discordance_rate"]}
    rate_used: dict = {}
    if paired_rates:
        disc_rate = float(np.mean(list(paired_rates.values())))
        rows = []
        for mit in mitigations:
            if mit in paired_rates:
                rate_used[mit] = paired_rates[mit]
            else:
                rate_used[mit] = disc_rate
                diff_flags.append(f"{mit}: no paired dev discordance rate; its within-session variance uses the "
                                  f"mean rate of the mitigations that have one ({disc_rate:.4g})")
            rows.append(_project_within_var(projected_counts, rate_used[mit]))
        within_var_proj = np.stack(rows)                     # (n_mitigations, n_projected_sessions)
        within_model = "paired_dev_rate_per_mitigation"
    else:
        counts_arr = np.asarray(projected_counts, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            within_var_proj = np.where(counts_arr > 0,
                                       np.maximum(s_mr * (1.0 - s_mr), 0.0) * 2.0 / counts_arr, np.nan)
        disc_rate = float("nan")
        within_model = "b0_binomial_fallback"
        diff_flags.append("no mitigation supplied paired dev counts; within-session variance fell back to the "
                          "binomial scale s_mr(1-s_mr)/n * 2 (the old proxy)")

    def _nanmean_or_nan(a):
        a = np.asarray(a, dtype=np.float64)
        return float(np.nanmean(a)) if a.size and np.isfinite(a).any() else float("nan")

    within_mean_per_mit = ({mit: _nanmean_or_nan(within_var_proj[i]) for i, mit in enumerate(mitigations)}
                           if within_var_proj.ndim == 2 else
                           {mit: _nanmean_or_nan(within_var_proj) for mit in mitigations})

    # between-session variance: the weakest (largest) across mitigations with a finite estimate drives the
    # family verdict conservatively; report every mitigation's own value too.
    between_vars = [m["between_var"] for m in mitigation_models.values() if m["between_var"] == m["between_var"]]
    between_var = max(between_vars) if between_vars else b0_proxy_var

    # the settled H5 simulation: three mitigations jointly, Holm, non-inferior iff Holm-adj p < .05
    h5 = simulate_h5_power(projected_counts, s_mr, within_var_proj, between_var, h5_rng(),
                           n_sim=n_sim, n_boot=n_boot, n_mitigations=len(mitigations))
    per_mitigation_power = h5.get("power_per_mitigation", [float("nan")])
    family_power = h5.get("power_family", float("nan"))
    chosen_power = (min(per_mitigation_power) if per_mitigation_power else float("nan")) \
        if POWER_SCOPE == "per_mitigation" else family_power

    not_assessable = h5.get("not_assessable")
    if not_assessable:
        verdict = "descriptive"
    else:
        verdict = "confirmatory" if (chosen_power == chosen_power and chosen_power >= POWER_TARGET) else "descriptive"

    # sensitivity table: power at 0.5x / 1x / 2x the between-session VARIANCE part (the lever, review M2/M3).
    # The table is ADVISORY (it never sets the verdict), so it may run at fewer sims than the primary check.
    n_sim_sens = n_sim_sensitivity if n_sim_sensitivity is not None else n_sim
    n_boot_sens = n_boot_sensitivity if n_boot_sensitivity is not None else n_boot
    sensitivity = {}
    for mult in BETWEEN_SENSITIVITY:
        sim = simulate_h5_power(projected_counts, s_mr, within_var_proj, between_var * mult,
                                h5_rng(), n_sim=n_sim_sens, n_boot=n_boot_sens, n_mitigations=len(mitigations))
        sensitivity[f"{mult:g}x"] = {"between_var": between_var * mult,
                                     "power_family": sim.get("power_family"),
                                     "power_per_mitigation": sim.get("power_per_mitigation"),
                                     "not_assessable": sim.get("not_assessable")}

    # SS-DL check (reported only): between-session spread DERIVED from dev's per-session ΔFAR (review m4),
    # flagging the assumed-prior fallback.
    ssdl_delta_far = b0_per_session_delta_far_from_tuning(tuning_record)
    ssdl_sd_info = ssdl_between_sd_from_dev(ssdl_delta_far)
    if ssdl_sd_info["source"] == "dev_delta_far":
        ssdl_sd = ssdl_sd_info["between_sd"]
        ssdl_sd_source = "dev_delta_far"
    else:
        ssdl_sd = ssdl_between_sd
        ssdl_sd_source = "assumed_prior"
        diff_flags.append(f"SS-DL between-session SD is an ASSUMED prior ({ssdl_between_sd}); dev per-session "
                          "ΔFAR had < 2 finite values to derive it from (review m4)")
    ssdl = simulate_ssdl_power(list(counts["ssdl_per_session"].values()), s_delta_far, ssdl_sd, ssdl_rng,
                               n_sim=n_sim, n_boot=n_boot)
    ssdl["between_sd_source"] = ssdl_sd_source
    ssdl["between_sd_n_dev_sessions"] = ssdl_sd_info["n_sessions"]

    record = {
        "seed": seed,
        "code_sha256": {GUARDED_SELF: self_sha256()},
        "input_sha256": dict(input_hashes or {}),
        "choices": _open_choices(),
        "constants": {"margin_abs": MARGIN_ABS, "margin_rel": MARGIN_REL, "one_sided_level": ONE_SIDED_LEVEL,
                      "power_target": POWER_TARGET, "min_dsdl_per_session": MIN_DSDL_PER_SESSION,
                      "plan_reference_turns": list(PLAN_REFERENCE_TURNS),
                      "wav": {"sample_rate_hz": WAV_SAMPLE_RATE_HZ, "channels": WAV_CHANNELS,
                              "bytes_per_sample": WAV_BYTES_PER_SAMPLE, "header_bytes": WAV_HEADER_BYTES,
                              "byte_rate": WAV_BYTE_RATE}},
        "dev_counts": counts,
        "s_mr": s_mr,
        "s_mr_note": "S_MR is B0's SESSION MEAN MR_DS-DL over the H5-included dev sessions (§7), not a pooled "
                     "rate; the binomial n and the rate are on the same footing.",
        "b0_mr_consistency": consistency,
        "h5_included_sessions": sorted(h5_included),
        "variance_model": {"model": DIFF_SPREAD_MODEL, "within_model": within_model,
                           "within_var_mean_projected": _nanmean_or_nan(within_var_proj),
                           "within_var_mean_projected_per_mitigation": within_mean_per_mit,
                           "between_var_used": between_var, "discordance_rate": disc_rate,
                           "discordance_rate_per_mitigation": rate_used,
                           "per_mitigation": mitigation_models, "b0_proxy_var": b0_proxy_var,
                           "flags": diff_flags},
        "sensitivity_between_var": sensitivity,
        "sensitivity_note": "every row and the primary H5 run use the same fresh seed (common random numbers), so "
                            "rows differ only in the between-session variance; at the primary's n_sim/n_boot the "
                            "1x row equals the primary",
        "rng_streams": {"h5_primary_and_sensitivity": f"default_rng({seed})",
                        "ssdl": f"default_rng(SeedSequence({seed}).spawn(1)[0])"},
        "projection": {"base": project_base, "dev_hours": base_dev_hours, "eval_audio_hours": eval_audio_hours,
                       "eval_base_hours": base_eval_hours, "dev_speech_fraction": frac,
                       "n_eval_sessions": n_eval_sessions,
                       "rate_per_hour": per_session.get("rate_per_hour"),
                       "projected_eval_dsdl": per_session.get("eval_total"),
                       "projected_per_session": projected_counts,
                       "n_sessions_ge_min_projected": per_session.get("n_sessions_ge_min"),
                       "flags": projection_flags},
        "h5_power": h5,
        "power_per_mitigation": per_mitigation_power,
        "power_family": family_power,
        "power_family_note": "computed with independent session effects per mitigation: a conservative floor for "
                             "the real design, where the shared B0 correlates the three; per-mitigation power, "
                             "which sets the default verdict, does not depend on it",
        "power_scope_used": POWER_SCOPE,
        "ssdl_power": ssdl,
        "verdict": verdict,
    }
    return record


# ===================================================================================================== #
# 9. CLI
# ===================================================================================================== #
def _read_listing(path: Path) -> list:
    """Read the eval file listing JSON. Accepts a list of [name, size] pairs, a list of {name,size} objects, or a
    {name: size} mapping. Never opens audio -- the sizes come from the archive central directory (§2)."""
    doc = json.loads(Path(path).read_text())
    if isinstance(doc, dict):
        return [(k, v) for k, v in doc.items()]
    out = []
    for e in doc:
        if isinstance(e, dict):
            out.append((e.get("name"), e.get("size", e.get("uncompressed_size", e.get("bytes")))))
        else:
            out.append((e[0], e[1]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--summary", type=Path, required=True, help="DEV ingest_displace summary.json")
    ap.add_argument("--tuning", type=Path, required=True, help="tune_m2b dev record JSON (B0 operating point)")
    ap.add_argument("--eval-listing", type=Path, required=True,
                    help="eval archive file listing JSON ([name,size] / {name:size}); NO eval reference count")
    ap.add_argument("--dev-speech-hours", type=float, default=None, help="dev SPEECH hours (§7 projection base)")
    ap.add_argument("--dev-audio-hours", type=float, default=None, help="dev AUDIO hours (for the speech frac)")
    ap.add_argument("--s-delta-far", type=float, default=0.05, help="SS-DL check smallest-effect ΔFAR")
    ap.add_argument("--ssdl-between-sd", type=float, default=0.02,
                    help="SS-DL check between-session SD -- ASSUMED prior, used only when dev per-session ΔFAR "
                         "cannot derive it (review m4)")
    ap.add_argument("--n-sim", type=int, default=N_SIM)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, required=True, help="output JSON path")
    args = ap.parse_args(argv)

    summary = json.loads(args.summary.read_text())
    tuning_record = json.loads(args.tuning.read_text())
    eval_listing = _read_listing(args.eval_listing)

    input_hashes = {str(args.summary): sha256_file(args.summary),
                    str(args.tuning): sha256_file(args.tuning),
                    str(args.eval_listing): sha256_file(args.eval_listing)}

    record = run_power_check(summary, tuning_record, eval_listing,
                             dev_speech_hours=args.dev_speech_hours, dev_audio_hours=args.dev_audio_hours,
                             s_delta_far=args.s_delta_far, ssdl_between_sd=args.ssdl_between_sd,
                             n_sim=args.n_sim, n_boot=args.n_boot, input_hashes=input_hashes, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1, default=_json_default))
    print(json.dumps({"written": str(args.out), "verdict": record["verdict"],
                      "power_family": record["h5_power"].get("power_family"),
                      "power_per_mitigation": record["h5_power"].get("power_per_mitigation"),
                      "projected_eval_dsdl": record["projection"]["projected_eval_dsdl"]}, default=str))
    return 0


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


if __name__ == "__main__":
    sys.exit(main())
