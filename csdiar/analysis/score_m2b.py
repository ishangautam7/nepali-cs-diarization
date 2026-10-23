#!/usr/bin/env python3
"""DISPLACE 2024 conversation scoring library (PREREGISTRATION_M2B.md §2, §3, §5, §7).

Pure scoring functions for the conversation tests: given a system hypothesis (its predicted speaker-change times
per recording) and the reference units produced by ``csdiar.data.ingest_displace`` (SS-DL points, DS-DL/DS-SL
turns and controls), produce the per-session numbers the confirmatory statistics consume -- FAR_SS-DL, FAR_CTRL,
the cell-standardised delta-FAR, the phrase-control FAR, MR_DS-DL / MR_DS-SL and changes per minute -- plus the
DET helpers and a thin DER wrapper.

This module ONLY turns a hypothesis and the reference rows into per-session statistics. The sign-flip, BCa and
outcome machinery lives in ``csdiar.analysis.stats`` and is not reimplemented here; the cell-standardised
delta-FAR arithmetic is ported verbatim from ``csdiar.analysis.pilot.h2_cell`` with DISPLACE cells = (session,
pause bin) (§2). The reader's column names and derived flags are used exactly and never recomputed: ``kind``
(ssdl / dsdl / dssl / ctrl), ``recording``, ``t``, ``pause_bin``, ``phrase``, the run-length ``stratum`` and the
``left_s`` / ``right_s`` run speech times (``csdiar/data/ingest_displace.py``).

Nothing heavy is imported at module import time; numpy is imported lazily inside the functions that need it, and
pyannote.metrics only inside ``der`` and the SegmentationRecall cross-check, so importing this module (e.g. in a
host with no numpy) is cheap. No DISPLACE data is read here: callers pass the already-ingested rows.

Interpretive choices (where the supplement does not pin a rule; the final report lists these):
  - change time: a change between two consecutive hypothesis segments with different labels is the shared
    boundary when they touch (gap <= EPS), else the middle of the gap between them, mirroring the reference
    turn time = middle of the gap (§2);
  - matching pool for MR: EVERY reference DS turn in the turns file (dsdl and dssl rows) competes one-to-one for
    hypothesis changes, so one change never counts for two turns (§2, brief);
  - collar endpoints are CLOSED: a point fires / a turn matches on |delta| <= c, endpoints included;
  - delta-FAR cell standardisation is pilot.h2_cell's: each scored SS-DL point's fire minus the mean fire rate
    of the controls of its own (session, pause bin) cell, controls reused, averaged over the session's scored
    points (§2);
  - DET pooling is reported BOTH pooled over units and averaged over sessions, because §5 fixes only the
    threshold-order crossing rule for reading a DET at a target and does not settle which (see ``det_summary``).
"""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from statistics import fmean
from typing import NamedTuple, Sequence

EPS = 1e-9

# --- frozen constants (PREREGISTRATION_M2B.md §2, §5, §7) ------------------------------------------------
C = 0.5                 # primary collar c (§2 "primary collar is c = 0.5 s")
COLLAR_CURVE = tuple(round(0.1 * k, 1) for k in range(1, 11))  # 0.1 .. 1.0 s descriptive curve (§5, §9.1)
MIN_UNITS = 10          # §7 inclusion rule: a session counts for a statistic only with >= 10 units of it
# Phrase-boundary controls (pause > 150 ms, §2) are read from the reader's `phrase` column, never re-derived.
_DS_KINDS = ("dsdl", "dssl")

# review m3: the per-session H2 RR (§7 "ratio of sums of fires over SS-DL points and over their cell controls")
# is computed in `confirm_m2b` as ssdl_fires_with_cell_controls / cell_ctrl_rate_sum. Either side being zero
# makes the ratio (or its log) undefined -- 0/0, x/0 or a 0 numerator -- so such a session cannot form a finite
# log RR and is dropped from the H2 sign-flip. The threshold for "zero" is this named constant, documented so
# the drop rule is explicit rather than an inline `<= 0` comparison (H2_CONTROL_NEVER_FIRES='drop_session').
RR_ZERO_SUM_EPS = 0.0   # a numerator or denominator at or below this is treated as zero and the session dropped


# --- RTTM parsing of a hypothesis ------------------------------------------------------------------------
@dataclass(frozen=True)
class HypSegment:
    start: float
    end: float
    label: str


def parse_hypothesis_rttm(text: str) -> list[HypSegment]:
    """Parse an exclusive-diarization hypothesis RTTM into (start, end, label) segments.

    Lines are "<TYPE> <file> <chan> <onset> <dur> <NA> <NA> <label> ...", as ``ingest_displace.parse_rttm``:
    any TYPE, blank/comment (';' / '#') lines skipped, non-positive duration dropped, field 8 (1-based) the
    label. Segments are returned sorted by (start, end). This parses the HYPOTHESIS only; the reference tracks
    are read by ``csdiar.data.ingest_displace``.
    """
    segs: list[HypSegment] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] in ";#":
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            onset = float(parts[3])
            dur = float(parts[4])
        except ValueError:
            continue
        label = parts[7]
        if dur <= 0 or label in ("", "<NA>"):
            continue
        segs.append(HypSegment(onset, onset + dur, label))
    segs.sort(key=lambda s: (s.start, s.end))
    return segs


def _assert_no_overlap(segments: Sequence[HypSegment]) -> None:
    prev_end = None
    for s in sorted(segments, key=lambda s: (s.start, s.end)):
        if prev_end is not None and s.start < prev_end - EPS:
            raise ValueError("hypothesis_changes requires non-overlapping (exclusive) input segments; "
                             f"segment starting at {s.start} overlaps previous ending at {prev_end}")
        prev_end = s.end if prev_end is None else max(prev_end, s.end)


def hypothesis_changes(segments: Sequence[HypSegment]) -> list[float]:
    """Sorted hypothesized speaker-change times from an EXCLUSIVE diarization hypothesis.

    REFUSES overlapping input: the change-scoring input is the exclusive diarization, so an overlap is a caller
    error, not something to silently resolve. For two consecutive segments with different labels the change is
    the shared boundary when they touch (gap <= EPS) and the middle of the gap when they do not, mirroring the
    reference turn time (ingest_displace: turn time = middle of the gap). Consecutive SAME-label segments are
    not a change (one speaker continuing across a pause).
    """
    segs = sorted(segments, key=lambda s: (s.start, s.end))
    _assert_no_overlap(segs)
    changes: list[float] = []
    for a, b in zip(segs, segs[1:]):
        if a.label == b.label:
            continue
        gap = b.start - a.end
        changes.append(b.start if gap <= EPS else (a.end + b.start) / 2.0)
    changes.sort()
    return changes


# --- firing (FAR_SS-DL, FAR_CTRL): no one-to-one constraint ----------------------------------------------
def fires(times: Sequence[float], changes: Sequence[float], c: float = C) -> list[bool]:
    """For each point time t, True when at least one change lies within the CLOSED collar [t - c, t + c].

    Used for SS-DL points and controls alike (FAR_SS-DL, FAR_CTRL): a point may be covered by any number of
    changes, there is no one-to-one constraint (§2 "a point fires when at least one hypothesized speaker change
    lies within +-c of it").
    """
    ch = sorted(changes)
    out: list[bool] = []
    for t in times:
        lo = bisect_left(ch, t - c - EPS)
        out.append(lo < len(ch) and ch[lo] <= t + c + EPS)
    return out


# --- one-to-one matching of reference turns to changes (MR_DS-DL, MR_DS-SL) ------------------------------
def match_turns(ref_times: Sequence[float], changes: Sequence[float], c: float = C) -> dict:
    """One-to-one matching of reference turn times to hypothesis changes within +-c (closed collar).

    Maximises the number of matched turns; among maximum matchings prefers the one with the smaller total
    absolute distance -- exactly what pyannote.metrics SegmentationRecall with a tolerance computes. Because the
    graph is an interval bipartite graph (each turn connects to a contiguous run of sorted changes), a left-to-
    right sweep that always assigns the earliest still-available change not before the turn's window, with a
    single backward "steal" when a later change would be forced to go unmatched, is optimal. We instead solve it
    directly with a small DP over the sorted turns and sorted changes, which is transparent and the tests pin.
    When maximum matchings also tie on total distance, the DP's fill precedence picks one deterministically (it
    prefers matching the LATER turn); which one carries no meaning, and exact ties need equal distances.

    Returns {'matched': int, 'total': len(ref_times), 'pairs': [(turn_index, change_index, distance), ...],
    'total_distance': float}. ``pairs`` indexes into the SORTED-by-time orderings.
    """
    turns = sorted(range(len(ref_times)), key=lambda i: ref_times[i])
    tt = [ref_times[i] for i in turns]
    ch = sorted(changes)
    n, m = len(tt), len(ch)
    INF = float("inf")
    # dp[j] over changes processed; we want to maximise matches, tie-break minimal distance.
    # best[i][j] = (num_matched, total_distance) considering turns[i:] and changes[j:].
    # Solve with memoised recursion; sizes here are per-recording (hundreds), fine for a DP but we keep it
    # O(n*m) with an explicit table to stay allocation-light and deterministic.
    # Each cell: maximise matches first (store as negative for a single min comparison), then minimise distance.
    neg_best = [[0] * (m + 1) for _ in range(n + 1)]       # number of matches (to maximise)
    dist_best = [[0.0] * (m + 1) for _ in range(n + 1)]    # total distance among max-match solutions (minimise)
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            # option 1: leave turn i unmatched
            best_match = neg_best[i + 1][j]
            best_dist = dist_best[i + 1][j]
            # option 2: skip change j (try to match turn i with a later change)
            if neg_best[i][j + 1] > best_match or (neg_best[i][j + 1] == best_match
                                                   and dist_best[i][j + 1] < best_dist - EPS):
                best_match = neg_best[i][j + 1]
                best_dist = dist_best[i][j + 1]
            # option 3: match turn i to change j if within collar
            d = abs(tt[i] - ch[j])
            if d <= c + EPS:
                cand_match = neg_best[i + 1][j + 1] + 1
                cand_dist = dist_best[i + 1][j + 1] + d
                if cand_match > best_match or (cand_match == best_match and cand_dist < best_dist - EPS):
                    best_match = cand_match
                    best_dist = cand_dist
            neg_best[i][j] = best_match
            dist_best[i][j] = best_dist
    # reconstruct
    pairs: list[tuple[int, int, float]] = []
    i = j = 0
    while i < n and j < m:
        # Mirror the fill's precedence EXACTLY: leave (option 1) is the base, skip (option 2) overrides only
        # when strictly better, take (option 3) overrides only when strictly better. Recomputing the winner the
        # same way here avoids brittle float '==' against the stored optimum.
        best_match = neg_best[i + 1][j]
        best_dist = dist_best[i + 1][j]
        move = "leave"
        if neg_best[i][j + 1] > best_match or (neg_best[i][j + 1] == best_match
                                               and dist_best[i][j + 1] < best_dist - EPS):
            best_match, best_dist, move = neg_best[i][j + 1], dist_best[i][j + 1], "skip"
        d = abs(tt[i] - ch[j])
        if d <= c + EPS:
            cand_match = neg_best[i + 1][j + 1] + 1
            cand_dist = dist_best[i + 1][j + 1] + d
            if cand_match > best_match or (cand_match == best_match and cand_dist < best_dist - EPS):
                move = "take"
        if move == "take":
            pairs.append((turns[i], j, d))
            i += 1
            j += 1
        elif move == "leave":
            i += 1
        else:  # skip change j
            j += 1
    matched = len(pairs)
    total_distance = sum(p[2] for p in pairs)
    return {"matched": matched, "total": n, "pairs": pairs, "total_distance": total_distance}


def miss_rate(ref_times: Sequence[float], changes: Sequence[float], c: float = C) -> float:
    """Share of reference turns in ``ref_times`` left unmatched under one-to-one matching (MR). nan if no turns."""
    n = len(ref_times)
    if n == 0:
        return float("nan")
    return (n - match_turns(ref_times, changes, c)["matched"]) / n


# --- per-session scoring of one system at one operating point --------------------------------------------
@dataclass
class SessionScore:
    session: str
    n_ssdl: int                 # scored SS-DL points (after min-run-length filter)
    n_ctrl: int
    n_phrase_ctrl: int
    n_dsdl: int                 # DS-DL turns in the matching pool of this session (scored)
    n_dssl: int
    far_ssdl: float             # mean fire rate over this session's SS-DL points
    far_ctrl: float             # mean fire rate over all controls
    far_phrase_ctrl: float      # mean fire rate over phrase controls (pause > 150 ms)
    delta_far_std: float        # cell-standardised SS-DL fire minus cell control fire rate (h2_cell style)
    # m4: an INCLUDED H4 session always has a finite delta_far_std. The H4 inclusion gate is
    # n_ssdl_with_cell_controls >= MIN_UNITS, which is exactly len(deltas); delta_far_std is nan only when
    # deltas is empty (no SS-DL point had a cell control), i.e. the session is excluded. So a future change to
    # the inclusion unit must keep "included => n_ssdl_with_cell_controls > 0", or it would feed nan to the
    # §7 sign-flip.
    n_ssdl_with_cell_controls: int
    n_ssdl_without_cell_controls: int
    # m3: a nan mr_dsdl / mr_dssl means this session has NO turns of that kind (n_dsdl / n_dssl == 0), so the
    # miss rate is undefined, not zero. The §7 MIN_UNITS inclusion gate (>= 10 DS-DL / DS-SL turns) excludes
    # such a session from MR_DS-DL / MR_DS-SL before any mean is taken, so stats.py must never average these
    # nan values as numbers -- filter on inclusion first.
    mr_dsdl: float
    mr_dssl: float
    changes_per_min: float
    duration_s: float
    # §7 / brief §5: the two per-session quantities the conversation H2 RR is the ratio of -- computed in the
    # same delta_far_std loop below, over the SS-DL points that HAVE a cell control. RR = (SS-DL points with a
    # cell control that fire) / (sum of those points' own cell-control fire rates); the >= 1 pp support check
    # uses delta_far_std, not these. Defaults keep every existing SessionScore(...) constructor working (a
    # session built without them reports no H2 contribution): a 0 numerator / 0 denominator is exactly the
    # RR_ZERO_SUM_EPS zero-sum drop case, so confirm_m2b drops such a session from the H2 sign-flip.
    ssdl_fires_with_cell_controls: float = 0.0   # count of SS-DL-with-cell-control points that fired
    cell_ctrl_rate_sum: float = 0.0              # sum over those points of their own cell's control fire rate


def _cell(row: dict) -> tuple:
    """DISPLACE cell for cell standardisation: (session, pause bin) (§2).

    The session is implicit (scoring is per session), so inside a session the cell key is the pause bin; the
    docstring name keeps the (session, pause bin) framing of the supplement.
    """
    return (row["pause_bin"],)


def _is_true(v) -> bool:
    """Reader rows come from a CSV, so booleans arrive as the strings 'True'/'False'; accept either."""
    return v is True or v == "True"


def _run_ok(row: dict, min_run_s: float | None) -> bool:
    """Minimum-run-length filter for SS-DL points (§2, §9.6): both runs >= min_run_s. Default off (None).

    Uses the reader's ``left_s`` / ``right_s`` run speech times; never recomputed. The threshold is a PARAMETER
    because its value is TBD from the dev timing distribution.
    """
    if min_run_s is None:
        return True
    return float(row["left_s"]) >= min_run_s - EPS and float(row["right_s"]) >= min_run_s - EPS


def score_session(session: str,
                  rows: Sequence[dict],
                  changes: Sequence[float],
                  duration_s: float,
                  c: float = C,
                  min_run_s: float | None = None) -> SessionScore:
    """Score one system at one operating point on ONE recording session.

    ``rows`` are this session's reader rows (ssdl / dsdl / dssl / ctrl), ``changes`` the system's hypothesized
    change times for the recording, ``duration_s`` the recording duration (reader summary) for changes/minute.
    The cell-standardised delta-FAR is pilot.h2_cell's arithmetic with cells = pause bin within the session:
    each scored SS-DL point's fire minus the mean fire rate of the controls of its own cell, controls reused,
    averaged over the session's points that HAVE at least one cell control (§2); points with no cell control are
    counted separately. FAR_CTRL is over all controls; FAR on phrase controls (pause > 150 ms) is separate.
    """
    ssdl_rows = [r for r in rows if r["kind"] == "ssdl" and _run_ok(r, min_run_s)]
    ctrl_rows = [r for r in rows if r["kind"] == "ctrl"]
    dsdl_rows = [r for r in rows if r["kind"] == "dsdl"]
    dssl_rows = [r for r in rows if r["kind"] == "dssl"]
    phrase_rows = [r for r in ctrl_rows if _is_true(r["phrase"])]   # the reader's flag: pause > 150 ms

    ch = sorted(changes)
    ssdl_fire = fires([float(r["t"]) for r in ssdl_rows], ch, c)
    ctrl_fire = fires([float(r["t"]) for r in ctrl_rows], ch, c)
    phrase_fire = fires([float(r["t"]) for r in phrase_rows], ch, c)

    far_ssdl = _mean(ssdl_fire)
    far_ctrl = _mean(ctrl_fire)
    far_phrase = _mean(phrase_fire)

    # cell control fire RATE per cell (controls reused), then per SS-DL point its own cell's rate (h2_cell)
    cell_ctrl: dict[tuple, list[bool]] = defaultdict(list)
    for r, f in zip(ctrl_rows, ctrl_fire):
        cell_ctrl[_cell(r)].append(f)
    cell_rate = {cell: _mean(v) for cell, v in cell_ctrl.items()}
    deltas: list[float] = []
    without = 0
    # brief §5: accumulate the two per-session H2 RR terms over the SAME SS-DL points that enter delta_far_std
    # (those WITH a cell control): the numerator is how many of them fire, the denominator the sum of their own
    # cell's control fire rate. RR = ssdl_fires_with_cell_controls / cell_ctrl_rate_sum (formed in confirm_m2b).
    ssdl_fires_with_cell_controls = 0.0
    cell_ctrl_rate_sum = 0.0
    for r, f in zip(ssdl_rows, ssdl_fire):
        rate = cell_rate.get(_cell(r))
        if rate is None:
            without += 1
            continue
        deltas.append((1.0 if f else 0.0) - rate)
        ssdl_fires_with_cell_controls += 1.0 if f else 0.0
        cell_ctrl_rate_sum += rate
    delta_far_std = _mean(deltas) if deltas else float("nan")

    # MR over the full DS pool of this session (dsdl and dssl rows compete one-to-one for the same changes)
    ds_times = [float(r["t"]) for r in dsdl_rows] + [float(r["t"]) for r in dssl_rows]
    match = match_turns(ds_times, ch, c)
    matched_idx = {p[0] for p in match["pairs"]}  # indices into ds_times (dsdl first, then dssl)
    n_dsdl = len(dsdl_rows)
    n_dssl = len(dssl_rows)
    dsdl_matched = sum(1 for i in matched_idx if i < n_dsdl)
    dssl_matched = sum(1 for i in matched_idx if i >= n_dsdl)
    mr_dsdl = (n_dsdl - dsdl_matched) / n_dsdl if n_dsdl else float("nan")
    mr_dssl = (n_dssl - dssl_matched) / n_dssl if n_dssl else float("nan")

    cpm = (len(ch) / (duration_s / 60.0)) if duration_s and duration_s > 0 else float("nan")

    return SessionScore(
        session=session,
        n_ssdl=len(ssdl_rows), n_ctrl=len(ctrl_rows), n_phrase_ctrl=len(phrase_rows),
        n_dsdl=n_dsdl, n_dssl=n_dssl,
        far_ssdl=far_ssdl, far_ctrl=far_ctrl, far_phrase_ctrl=far_phrase,
        delta_far_std=delta_far_std,
        n_ssdl_with_cell_controls=len(deltas), n_ssdl_without_cell_controls=without,
        mr_dsdl=mr_dsdl, mr_dssl=mr_dssl,
        changes_per_min=cpm, duration_s=duration_s,
        ssdl_fires_with_cell_controls=ssdl_fires_with_cell_controls,
        cell_ctrl_rate_sum=cell_ctrl_rate_sum,
    )


def _mean(xs: Sequence[bool | float]) -> float:
    xs = list(xs)
    if not xs:
        return float("nan")
    return sum(1.0 if x is True else (0.0 if x is False else float(x)) for x in xs) / len(xs)


def score_system(rows_by_session: dict[str, Sequence[dict]],
                 changes_by_recording: dict[str, Sequence[float]],
                 durations: dict[str, float],
                 c: float = C,
                 min_run_s: float | None = None) -> dict:
    """Score one system at one operating point over every session, and apply the §7 inclusion rule.

    ``rows_by_session`` maps recording/session id -> its reader rows; ``changes_by_recording`` maps the same ids
    to hypothesized change times; ``durations`` maps them to recording seconds (reader summary). Returns per-
    session ``SessionScore`` objects plus, for each statistic, the list of sessions INCLUDED (>= MIN_UNITS units
    of that statistic) and the sessions EXCLUDED with their unit count, so the caller reports exclusions (§7).

    Inclusion units per §7: H4/FAR statistics count SS-DL points that have a cell control
    (``n_ssdl_with_cell_controls``); MR_DS-DL counts DS-DL turns; MR_DS-SL counts DS-SL turns.
    """
    scores: dict[str, SessionScore] = {}
    for sess, rows in rows_by_session.items():
        ch = changes_by_recording.get(sess, [])
        dur = durations.get(sess, float("nan"))
        scores[sess] = score_session(sess, rows, ch, dur, c=c, min_run_s=min_run_s)

    def split(unit_of) -> dict:
        inc, exc = [], {}
        for sess, s in scores.items():
            n = unit_of(s)
            (inc.append(sess) if n >= MIN_UNITS else exc.__setitem__(sess, n))
        return {"included": sorted(inc), "excluded": exc}

    inclusion = {
        "ssdl": split(lambda s: s.n_ssdl_with_cell_controls),   # H4 / FAR_SS-DL, delta-FAR (confirmatory, §7)
        "dsdl": split(lambda s: s.n_dsdl),                       # MR_DS-DL (H5, confirmatory, §7)
        # m2: MR_DS-SL is the §8 SECONDARY, not a §7 confirmatory family -- §7 names only the H4 (SS-DL) and H5
        # (DS-DL) inclusion gates. This "dssl" split is ADVISORY ONLY: it reports which sessions have >= MIN_UNITS
        # DS-SL turns for the secondary MR_DS-SL, and must never be fed to a confirmatory test as a gate.
        "dssl": split(lambda s: s.n_dssl),                       # MR_DS-SL (§8 secondary; advisory, not a gate)
    }
    return {"sessions": scores, "inclusion": inclusion, "min_units": MIN_UNITS, "collar": c,
            "min_run_s": min_run_s}


def collar_curve(rows_by_session: dict[str, Sequence[dict]],
                 changes_by_recording: dict[str, Sequence[float]],
                 durations: dict[str, float],
                 collars: Sequence[float] = COLLAR_CURVE,
                 min_run_s: float | None = None) -> dict[float, dict]:
    """Descriptive FAR/MR-vs-collar curve over 0.1-1.0 s (§5, §9.1). Never replaces the primary c = 0.5 s."""
    return {c: score_system(rows_by_session, changes_by_recording, durations, c=c, min_run_s=min_run_s)
            for c in collars}


# --- DET helpers -----------------------------------------------------------------------------------------
def _probit(p: float):
    """Inverse standard-normal CDF (probit), stdlib only. p clamped to the open (0,1) so +-inf never appears."""
    from statistics import NormalDist
    p = min(max(p, 1e-9), 1 - 1e-9)
    return NormalDist().inv_cdf(p)


def _interp_probit_pair(a_from: float, a_to: float, a_target: float,
                        b_from: float, b_to: float) -> float:
    """Reading on the OTHER axis where a segment crosses ``a_target`` on the "along" axis, probit on both axes.

    ``a_*`` are the along-axis (target) values at the two segment endpoints, ``b_*`` the other-axis values.
    The weight is placed by probit on the along axis; the reading is probit-interpolated on the other axis and
    mapped back through the normal CDF. Endpoints are handled by the caller (node hits); here the segment is
    known to straddle ``a_target`` strictly, so ``a_from != a_to``.
    """
    pa0, pa1, pat = _probit(a_from), _probit(a_to), _probit(a_target)
    w = (pat - pa0) / (pa1 - pa0) if pa1 != pa0 else 0.0
    from statistics import NormalDist
    pb = _probit(b_from) + w * (_probit(b_to) - _probit(b_from))
    return NormalDist().cdf(pb)


class DetReading(NamedTuple):
    """A DET read at a target: the BEST reading on the other axis and how many times the target was crossed."""
    value: float
    n_crossings: int


def read_det(points: Sequence[tuple[float, float, float]], target: float, along: str = "far") -> DetReading:
    """Read a DET at a target by the threshold-order crossing rule (PREREGISTRATION_M2B.md §5).

    ``points`` are (threshold, FAR_SS-DL, MR_DS-DL) triples from a threshold sweep. The DET is the polyline
    through the operating points IN THRESHOLD ORDER (NOT sorted by FAR or MR); the sweep need not be monotone,
    so the target axis can be crossed more than once. ``along`` names the axis the target is on: ``"far"`` reads
    MR at a target FAR (the primary), ``"mr"`` reads FAR at a target MR (the secondary). The reading is the
    OTHER axis, probit-interpolated on both axes.

    Crossings, walking adjacent thresholds low to high:
      - a sweep point that HITS the target (|along - target| <= EPS) is a node crossing, giving the raw other-
        axis value at that node (a curve that only touches the target at a local extremum node still reads);
      - a run of CONSECUTIVE sweep points at the target counts as ONE crossing (its best node);
      - a segment whose two endpoints STRADDLE the target (one strictly below, one strictly above) crosses it
        once in its interior, read by probit interpolation on that segment.
    Each crossing yields one reading on the other axis. If the target is crossed more than once the summary is
    the BEST reading -- the LOWEST other-axis value (lowest MR at a target FAR, lowest FAR at a target MR), an
    operating point the sweep actually reaches -- and ``n_crossings`` reports how many there were. A target the
    sweep never crosses is NOT reached: ``value`` is nan with ``n_crossings == 0``, never extrapolated.

    Input is validated: duplicate thresholds and any non-finite threshold / FAR / MR raise ValueError. Order of
    the input triples does not matter; they are sorted by threshold here.
    """
    if along not in ("far", "mr"):
        raise ValueError(f"along must be 'far' or 'mr', got {along!r}")
    triples = list(points)
    for pt in triples:
        if len(pt) != 3:
            raise ValueError(f"DET points must be (threshold, FAR, MR) triples; got a {len(pt)}-item point "
                             f"{tuple(pt)!r} (the public shape is 3-tuples, not (FAR, MR) pairs)")
    for t, far, mr in triples:
        if not (isfinite(t) and isfinite(far) and isfinite(mr)):
            raise ValueError(f"DET points must be finite; got threshold/FAR/MR ({t}, {far}, {mr})")
    thr = [t for t, _, _ in triples]
    if len(set(thr)) != len(thr):
        raise ValueError("DET points have duplicate thresholds; each sweep threshold must be unique")
    triples.sort(key=lambda p: p[0])
    # along axis = target axis, other axis = reading
    along_idx, other_idx = (1, 2) if along == "far" else (2, 1)
    a = [p[along_idx] for p in triples]
    b = [p[other_idx] for p in triples]
    n = len(triples)

    readings: list[float] = []

    # 1) node crossings: runs of consecutive points AT the target count once, reading the run's best (lowest b).
    i = 0
    hit = [abs(a[k] - target) <= EPS for k in range(n)]
    while i < n:
        if hit[i]:
            j = i
            best = b[i]
            while j + 1 < n and hit[j + 1]:
                j += 1
                if b[j] < best:
                    best = b[j]
            readings.append(best)
            i = j + 1
        else:
            i += 1

    # 2) interior crossings: adjacent segments whose endpoints STRADDLE the target (neither endpoint a hit).
    for k in range(n - 1):
        if hit[k] or hit[k + 1]:
            continue                       # a node hit is already counted in step 1
        lo, hi = a[k], a[k + 1]
        if (lo - target) * (hi - target) < 0:   # strictly opposite sides -> one interior crossing
            readings.append(_interp_probit_pair(lo, hi, target, b[k], b[k + 1]))

    if not readings:
        return DetReading(float("nan"), 0)
    return DetReading(min(readings), len(readings))


def mr_at_far(points: Sequence[tuple[float, float, float]], far: float = 0.10) -> float:
    """BEST MR_DS-DL at a target FAR_SS-DL by the §5 threshold-order crossing rule (primary = 10%).

    Thin wrapper over ``read_det(points, far, along="far")`` returning only the value; use ``read_det`` when the
    crossing count is needed. ``points`` are (threshold, FAR, MR) triples.
    """
    return read_det(points, far, along="far").value


def far_at_mr(points: Sequence[tuple[float, float, float]], mr: float = 0.05) -> float:
    """BEST FAR_SS-DL at a target MR_DS-DL by the §5 threshold-order crossing rule (secondary = 5% and 10%).

    Thin wrapper over ``read_det(points, mr, along="mr")`` returning only the value. ``points`` are
    (threshold, FAR, MR) triples.
    """
    return read_det(points, mr, along="mr").value


def det_summary(det_points_pooled: Sequence[tuple[float, float, float]],
                det_points_by_session: Sequence[Sequence[tuple[float, float, float]]] | None = None) -> dict:
    """DET summary at the pre-registered operating points, reported BOTH pooled and averaged over sessions.

    §5 fixes the threshold-order crossing rule (``read_det``) and names MR at FAR = 10% (primary) and FAR at
    MR = 5% / 10% (secondary), but does NOT settle whether the DET is pooled over units or averaged over
    sessions, so both are returned. ``points`` everywhere are (threshold, FAR, MR) triples.

    ``pooled`` holds the best reading at each operating point from the pooled sweep (nan = the sweep never
    crosses that point, also listed in ``pooled_not_reached``). ``pooled_crossings`` holds the crossing count at
    each operating point, and ``pooled_multi_crossing`` lists the operating points crossed more than once (the
    non-monotone ones whose best reading was chosen among several). ``by_session`` gives, per operating point,
    the ``statistics.fmean`` over the sessions whose own sweep reaches it, ``n_reached`` / ``n_not_reached``, and
    ``n_multi_crossing`` (sessions whose sweep crossed that point more than once). Sessions that do not reach a
    point are counted, never averaged in, and the caller must report them. ``statistics.fmean`` keeps the module
    stdlib-only (no numpy) so it imports and tests on a host without numpy.
    """
    targets = {"mr_at_far10": ("far", 0.10), "far_at_mr05": ("mr", 0.05),
               "far_at_mr10": ("mr", 0.10)}
    pooled_read = {name: read_det(det_points_pooled, x, along=along) for name, (along, x) in targets.items()}
    pooled = {name: r.value for name, r in pooled_read.items()}
    out = {
        "pooled": pooled,
        "pooled_not_reached": sorted(k for k, v in pooled.items() if v != v),
        "pooled_crossings": {name: r.n_crossings for name, r in pooled_read.items()},
        "pooled_multi_crossing": sorted(name for name, r in pooled_read.items() if r.n_crossings > 1),
    }
    if det_points_by_session is not None:
        by_session = {}
        for name, (along, x) in targets.items():
            reads = [read_det(p, x, along=along) for p in det_points_by_session]
            reached = [r.value for r in reads if r.value == r.value]
            n_multi = sum(1 for r in reads if r.n_crossings > 1)
            by_session[name] = {"mean": fmean(reached) if reached else float("nan"),
                                "n_reached": len(reached), "n_not_reached": len(reads) - len(reached),
                                "n_multi_crossing": n_multi}
        out["by_session"] = by_session
    return out


# --- DER wrapper -----------------------------------------------------------------------------------------
def der(reference_rttm_path, hypothesis_rttm_path) -> float:
    """Diarization error rate via pyannote.metrics (§2, §8), collar 0.0, overlap scored.

    Thin wrapper around ``pyannote.metrics.diarization.DiarizationErrorRate(collar=0.0, skip_overlap=False)``;
    pyannote is imported lazily inside the function so the module imports with no heavy dependency. Returns the
    DER as a float.
    """
    from pyannote.metrics.diarization import DiarizationErrorRate
    from pyannote.database.util import load_rttm

    def _one(path):
        ann = load_rttm(str(path))
        if not ann:
            raise ValueError(f"no annotation parsed from {path}")
        # der() is per recording (§2, §8): load_rttm maps uri -> Annotation, and a single-recording RTTM has
        # exactly one entry. m5: refuse anything else loudly rather than silently scoring the first URI, which
        # would be the wrong recording.
        if len(ann) != 1:
            raise ValueError(f"der() scores exactly one recording, but {path} holds {len(ann)} "
                             f"({sorted(ann.keys())}); pass a single-recording RTTM")
        return next(iter(ann.values()))

    reference = _one(reference_rttm_path)
    hypothesis = _one(hypothesis_rttm_path)
    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    return float(metric(reference, hypothesis))
