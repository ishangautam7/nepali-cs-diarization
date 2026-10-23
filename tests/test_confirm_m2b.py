#!/usr/bin/env python3
"""Unit tests for csdiar.analysis.confirm_m2b (PREREGISTRATION_M2B.md §7, §8, §11).

Model-free and fast. A synthetic world of 6 sessions where the verdicts are KNOWN is built directly from
`score_m2b` inputs (reader rows + hypothesis change times), so the confirmatory machinery is exercised without a
model, audio, or the GPU box:
  * M1 = a NET IMPROVEMENT: it lowers the SS-DL excess (delta-FAR) and does not raise DS-DL misses;
  * M2 = REDUCES FALSE SPLITS AT THE COST OF MISSED TURNS: lower delta-FAR but higher MR_DS-DL;
  * M3 = NO EFFECT: delta-FAR and MR like B0.
The bootstrap counts are kept small through a monkeypatched BCA_N_BOOT so the whole file runs in well under a
second.

Run on the box (numpy present):
  .venv-pyannote4/bin/python -m pytest -q -p no:cacheprovider tests/test_confirm_m2b.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from csdiar.analysis import confirm_m2b as C
from csdiar.analysis import score_m2b


@pytest.fixture(autouse=True)
def _small_bootstrap(monkeypatch):
    monkeypatch.setattr(C, "BCA_N_BOOT", 300)


# ======================================================================================================= #
# synthetic reader rows + hypotheses
# ======================================================================================================= #
def _rows_for_session(rid: str, n_ssdl=10, n_ctrl=10, n_dsdl=10) -> list[dict]:
    """A session's reader rows: n_ssdl SS-DL points, n_ctrl controls (all in the 'none' pause-bin cell, so every
    point has a cell control), n_dsdl DS-DL turns. Ten of each (the MIN_UNITS inclusion floor) makes the intended
    rates 0.6 / 0.2 / 0.3 / 0.7 exact fractions. Times are spread 3 s apart so a hypothesis change placed on
    one never fires on another (collar 0.5 s)."""
    rows: list[dict] = []
    t = 1.0
    for i in range(n_ssdl):
        rows.append({"recording": rid, "split": "eval", "kind": "ssdl", "t": round(t, 3),
                     "speaker": "s1", "speaker2": "", "lang_left": "en", "lang_right": "hi",
                     "left_s": 2.0, "right_s": 2.0, "stratum": "ge1", "elig_p0L1": True,
                     "pause_ms": 0.0, "pause_bin": "none", "phrase": False})
        t += 3.0
    for i in range(n_ctrl):
        rows.append({"recording": rid, "split": "eval", "kind": "ctrl", "t": round(t, 3),
                     "speaker": "s1", "speaker2": "", "lang_left": "en", "lang_right": "en",
                     "left_s": 2.0, "right_s": 2.0, "stratum": "ge1", "elig_p0L1": False,
                     "pause_ms": 0.0, "pause_bin": "none", "phrase": False})
        t += 3.0
    for i in range(n_dsdl):
        rows.append({"recording": rid, "split": "eval", "kind": "dsdl", "t": round(t, 3),
                     "speaker": "s1", "speaker2": "s2", "lang_left": "en", "lang_right": "hi",
                     "left_s": 2.0, "right_s": 2.0, "stratum": "ge1", "elig_p0L1": False,
                     "pause_ms": 0.0, "pause_bin": "none", "phrase": False})
        t += 3.0
    return rows


def _changes_hitting(rows, kind, frac) -> list[float]:
    """A change at every row of `kind` for the first `frac` fraction of them -- deterministic firing."""
    times = [float(r["t"]) for r in rows if r["kind"] == kind]
    k = int(round(frac * len(times)))
    return sorted(times[:k])


def _system_changes(rows, far_frac, mr_frac):
    """Hypothesis change times giving FAR_SS-DL ~ far_frac (points fired) and MR_DS-DL ~ (1 - mr_frac matched).
    Controls are never hit (so delta-FAR = far_frac). A DS-DL turn is matched when a change sits on it: we hit
    `1 - mr_frac` of the DS-DL turns, so MR = mr_frac."""
    ssdl = _changes_hitting(rows, "ssdl", far_frac)
    dsdl = _changes_hitting(rows, "dsdl", 1.0 - mr_frac)
    return sorted(ssdl + dsdl)


def _world(sessions=6):
    """Rows per session + per-system change times. B0: far 0.6, mr 0.3. M1: far 0.2 (big drop), mr 0.3 (same).
    M2: far 0.2 (drop) but mr 0.7 (worse). M3: far 0.6, mr 0.3 (no effect)."""
    rids = [f"EVA{i:02d}" for i in range(sessions)]
    rows_by_session = {rid: _rows_for_session(rid) for rid in rids}
    durations = {rid: 300.0 for rid in rids}
    systems = {"B0": (0.60, 0.30), "M1": (0.20, 0.30), "M2": (0.20, 0.70), "M3": (0.60, 0.30)}
    changes = {sys: {rid: _system_changes(rows_by_session[rid], far, mr) for rid in rids}
               for sys, (far, mr) in systems.items()}
    return rids, rows_by_session, durations, changes


def _score(rows_by_session, durations, changes):
    return score_m2b.score_system(rows_by_session, changes, durations)


# ======================================================================================================= #
# score-level sanity: the synthetic world has the FAR/MR we intend
# ======================================================================================================= #
def test_synthetic_world_has_intended_rates():
    _, rbs, dur, ch = _world()
    b0 = _score(rbs, dur, ch["B0"])
    s = next(iter(b0["sessions"].values()))
    assert s.far_ssdl == pytest.approx(0.6, abs=1e-9)
    assert s.far_ctrl == pytest.approx(0.0, abs=1e-9)
    assert s.delta_far_std == pytest.approx(0.6, abs=1e-9)
    assert s.mr_dsdl == pytest.approx(0.3, abs=1e-9)
    m1 = next(iter(_score(rbs, dur, ch["M1"])["sessions"].values()))
    assert m1.delta_far_std == pytest.approx(0.2, abs=1e-9)
    assert m1.mr_dsdl == pytest.approx(0.3, abs=1e-9)


# ======================================================================================================= #
# a minimal frozen record for run_analysis (no freeze check -- run_analysis is split-agnostic)
# ======================================================================================================= #
def _rec(rids):
    return {"split": {"eval": rids},
            "tuning": {"thresholds": {"B0": 0.6, "M1": 0.6, "M2": 0.6, "M3": 0.6}}}


def _run(descriptive=False, sessions=6):
    rids, rbs, dur, ch = _world(sessions)
    rec = _rec(rids)
    rng = np.random.default_rng(C.SEED)

    # patch _frozen_tag + _score_system_sweep + _op_score to use our in-memory change dicts instead of RTTMs
    import csdiar.analysis.confirm_m2b as M

    def fake_tag(rec_, name, spec):
        return name

    def fake_sweep(rows_by_session, durations, hyp_root, split, tag, rids_):
        # one DET point at threshold 0.6 from the chosen operating point (enough for read_det nodes)
        res = _score(rows_by_session, durations, ch[tag])
        det_points = []
        det_by_session = {}
        inc_far = set(res["inclusion"]["ssdl"]["included"])
        inc_mr = set(res["inclusion"]["dsdl"]["included"])
        far = C.mean_stat([res["sessions"][s].far_ssdl for s in inc_far])
        mr = C.mean_stat([res["sessions"][s].mr_dsdl for s in inc_mr])
        # two sweep points straddling FAR=0.10 so read_det can cross it: (0.55, far_lo, mr) and (0.60, far, mr)
        for s in inc_far & inc_mr:
            fs, ms = res["sessions"][s].far_ssdl, res["sessions"][s].mr_dsdl
            det_by_session[s] = [(0.55, max(fs, 0.11), ms), (0.60, min(fs, 0.09), ms)]
        det_points = [(0.55, max(far, 0.11), mr), (0.60, min(far, 0.09), mr)]
        return {"by_threshold": {0.6: res}, "det_points": det_points, "det_by_session": det_by_session}

    def fake_op(rows_by_session, durations, hyp_root, split, tag, threshold, rids_):
        return _score(rows_by_session, durations, ch[tag])

    def fake_native_changes(hyp_root, split, tag, rids_):
        # native P1 fires at 60% of SS-DL points and 20% of controls (RR = 3, delta-FAR = 0.4): a defined log RR,
        # since a session whose controls never fire is dropped (H2_CONTROL_NEVER_FIRES). P2 never fires, so every
        # session drops and P2 is not assessable.
        if tag == C.NATIVE_TAG["P1"]:
            return {rid: sorted(_changes_hitting(rbs[rid], "ssdl", 0.6) + _changes_hitting(rbs[rid], "ctrl", 0.2))
                    for rid in rids_}
        return {rid: [] for rid in rids_}

    M_frozen_tag = M._frozen_tag
    monkey = pytest.MonkeyPatch()
    monkey.setattr(M, "_frozen_tag", fake_tag)
    monkey.setattr(M, "_score_system_sweep", fake_sweep)
    monkey.setattr(M, "_op_score", fake_op)
    monkey.setattr(M, "_native_changes", fake_native_changes)
    try:
        res = M.run_analysis(rbs, dur, Path("/none"), "eval", rec, rng, descriptive,
                             [7.62939453125e-06] * 3)
    finally:
        monkey.undo()
    return res


# ======================================================================================================= #
# the known verdicts
# ======================================================================================================= #
def test_intersection_union_labels_match_the_known_world():
    res = _run()
    iu = res["intersection_union"]
    assert iu["M1"]["label"] == "net improvement"
    assert iu["M1"]["net_improvement"] is True
    assert iu["M2"]["label"] == "reduces false splits at the cost of missed turns"
    assert iu["M2"]["net_improvement"] is False
    # M3 has no reduction (delta-FAR == B0), so H4 not supported -> "no reduction shown"
    assert iu["M3"]["label"] == "no reduction shown"


def test_h4_supported_for_m1_and_m2_not_m3():
    res = _run()
    assert res["H4"]["M1"]["outcome"]["category"] == "supported"
    assert res["H4"]["M2"]["outcome"]["category"] == "supported"   # M2 also cuts false splits
    assert res["H4"]["M3"]["outcome"]["category"] != "supported"
    # relative reduction: M1 drops delta-FAR from 0.6 to 0.2 -> 0.4/0.6 ~ 0.667 >= 0.25 SESOI
    assert res["H4"]["M1"]["relative_reduction"] == pytest.approx((0.6 - 0.2) / 0.6, abs=1e-6)


def test_h5_non_inferior_for_m1_not_m2():
    res = _run()
    m = res["H5"]["members"]
    assert m["M1"]["non_inferior"] is True       # delta MR = 0 < margin
    assert m["M2"]["non_inferior"] is False      # delta MR = +0.4 >> margin
    # S_MR is B0's mean MR over H5 sessions (0.3), margin = min(0.02, 0.25*0.3) = 0.02
    assert res["H5"]["margin"] == pytest.approx(0.02, abs=1e-9)


# ======================================================================================================= #
# descriptive-H5 path
# ======================================================================================================= #
def test_descriptive_verdict_withholds_net_improvement_claim():
    res = _run(descriptive=True)
    assert res["power_verdict"] == "descriptive"
    assert res["H5"]["members"]["M1"]["non_inferior"] is None
    assert res["H5"]["members"]["M1"]["claim_withheld"] is True
    assert res["intersection_union"]["M1"]["net_improvement"] is None
    assert res["intersection_union"]["M1"]["claim_withheld"] is True


# ======================================================================================================= #
# nothing tuned on eval: operating points come only from the frozen record
# ======================================================================================================= #
def test_operating_points_are_the_frozen_thresholds():
    res = _run()
    for name in ("B0", "M1", "M2", "M3"):
        assert res["operating_points"][name]["threshold"] == 0.6


# ======================================================================================================= #
# not-assessable: too few sessions with enough units
# ======================================================================================================= #
def test_not_assessable_with_too_few_sessions():
    # 4 sessions < MIN_SESSIONS = 5 -> every family member is not assessable
    res = _run(sessions=4)
    assert res["H4"]["M1"].get("not_assessable") is True
    assert res["H5"]["members"]["M1"].get("not_assessable") is True
    assert res["intersection_union"]["M1"]["label"] == "not assessable"


def test_session_excluded_when_under_min_units():
    # a session with < 10 DS-DL turns is excluded from H5 (MIN_UNITS gate inside score_m2b)
    rids = [f"EVA{i:02d}" for i in range(6)]
    rbs = {rid: _rows_for_session(rid) for rid in rids}
    # strip one session's DS-DL rows down to 3 turns
    rbs[rids[0]] = [r for r in rbs[rids[0]] if r["kind"] != "dsdl"] + \
        [r for r in rbs[rids[0]] if r["kind"] == "dsdl"][:3]
    dur = {rid: 300.0 for rid in rids}
    b0 = _score(rbs, dur, {rid: [] for rid in rids})
    assert rids[0] in b0["inclusion"]["dsdl"]["excluded"]
    assert rids[0] not in b0["inclusion"]["dsdl"]["included"]


# ======================================================================================================= #
# H2 conversation: Holm m = 5 uses the P0 p-values from the file unchanged
# ======================================================================================================= #
def test_h2_holm_family_m5_uses_frozen_p0_pvalues():
    res = _run()
    h2 = res["H2_conversation"]
    assert h2["holm_family_m"] == 5
    assert h2["p0_d1_one_sided_pvalues"] == {"e1": 7.62939453125e-06, "e2": 7.62939453125e-06,
                                             "e3": 7.62939453125e-06}
    # native P1 fires at 60% of SS-DL points and 20% of controls -> RR 3, delta-FAR 0.4 -> supported (6 sessions,
    # exact sign-flip p = 1/64, Holm rank 4 of m = 5 -> 2/64 < .05); P2 never fires -> not assessable
    assert h2["members"]["P1"]["rr"] == pytest.approx(3.0, rel=1e-9)
    assert h2["members"]["P1"]["delta_far"] == pytest.approx(0.4, abs=1e-9)
    assert h2["members"]["P1"]["p_one_holm"] == pytest.approx(2 / 64, abs=1e-12)
    assert h2["members"]["P1"]["supported"] is True
    assert h2["members"]["P2"].get("not_assessable") is True


def test_h2_pvalues_typed_from_file_not_hardcoded(tmp_path):
    # confirm the reader pulls the three p-values from the M2a JSON structure
    p = tmp_path / "m2a.json"
    p.write_text(json.dumps({"h2_p0": {"primary": {
        "e1": {"signflip": {"p_one": 0.001}}, "e2": {"signflip": {"p_one": 0.002}},
        "e3": {"signflip": {"p_one": 0.003}}}}}))
    assert C._p0_pvalues_from_m2a(p) == [0.001, 0.002, 0.003]


# ======================================================================================================= #
# H6 determinism under the seed
# ======================================================================================================= #
def test_h6_is_deterministic_under_the_seed():
    a = _run()["H6"]
    b = _run()["H6"]
    assert a["ordering_significant"] == b["ordering_significant"]
    for key in a["pairs"]:
        if "p_one_sided" in a["pairs"][key]:
            assert a["pairs"][key]["p_one_sided"] == b["pairs"][key]["p_one_sided"]
            assert a["pairs"][key]["p_two_sided"] == b["pairs"][key]["p_two_sided"]


# ======================================================================================================= #
# H2 three-way support gate (review m5 / brief fix 10): the RR and delta-FAR arms must be able to fail
# independently of the Holm p. We drive _h2_native directly with hand-built per-session H2 quantities.
# ======================================================================================================= #
def _native_score(per_session_fields):
    """A minimal score_m2b.score_system-shaped dict for H2: `per_session_fields` maps session -> dict with
    ssdl_fires_with_cell_controls, cell_ctrl_rate_sum, delta_far_std. Every session is SS-DL-included."""
    sessions = {}
    for s, f in per_session_fields.items():
        sessions[s] = score_m2b.SessionScore(
            session=s, n_ssdl=10, n_ctrl=10, n_phrase_ctrl=0, n_dsdl=0, n_dssl=0,
            far_ssdl=float("nan"), far_ctrl=float("nan"), far_phrase_ctrl=float("nan"),
            delta_far_std=f["delta_far_std"], n_ssdl_with_cell_controls=10, n_ssdl_without_cell_controls=0,
            mr_dsdl=float("nan"), mr_dssl=float("nan"), changes_per_min=float("nan"), duration_s=300.0,
            ssdl_fires_with_cell_controls=f["ssdl_fires_with_cell_controls"],
            cell_ctrl_rate_sum=f["cell_ctrl_rate_sum"])
    return {"sessions": sessions, "inclusion": {"ssdl": {"included": sorted(sessions), "excluded": {}}}}


def test_h2_not_supported_when_rr_below_125_though_holm_significant():
    # 6 sessions, each: numerator 7, denominator 6 -> RR = 7/6 ~ 1.167 (< 1.25), delta-FAR = 0.10 (>= 1 pp),
    # log RR > 0 in every session -> exact sign-flip p = 1/64, Holm rank 4 of m = 5 -> 2/64 < .05.
    fields = {f"S{i}": {"ssdl_fires_with_cell_controls": 7.0, "cell_ctrl_rate_sum": 6.0,
                        "delta_far_std": 0.10} for i in range(6)}
    rng = np.random.default_rng(C.SEED)
    h2 = C._h2_native({"P1": _native_score(fields), "P2": None}, [7.62939453125e-06] * 3, rng)
    m = h2["members"]["P1"]
    assert m["p_one_holm"] < 0.05           # Holm-significant
    assert m["delta_far"] >= C.H2_DFAR_MIN_PP
    assert m["rr"] < C.H2_RR_MIN
    assert m["supported"] is False          # fails on the RR arm alone


def test_h2_not_supported_when_delta_far_below_1pp_though_holm_significant_and_rr_ok():
    # numerator 5, denominator 1 -> RR = 5 (>> 1.25); but delta-FAR fixed at 0.005 (< 1 pp). log RR > 0 every
    # session -> Holm-significant. Supported must still be False because the delta-FAR arm fails.
    fields = {f"S{i}": {"ssdl_fires_with_cell_controls": 5.0, "cell_ctrl_rate_sum": 1.0,
                        "delta_far_std": 0.005} for i in range(6)}
    rng = np.random.default_rng(C.SEED)
    h2 = C._h2_native({"P1": _native_score(fields), "P2": None}, [7.62939453125e-06] * 3, rng)
    m = h2["members"]["P1"]
    assert m["p_one_holm"] < 0.05
    assert m["rr"] >= C.H2_RR_MIN
    assert m["delta_far"] < C.H2_DFAR_MIN_PP
    assert m["supported"] is False          # fails on the delta-FAR arm alone


def test_h2_zero_sum_session_dropped():
    # a session whose numerator or denominator is zero forms no finite log RR and is dropped.
    fields = {f"S{i}": {"ssdl_fires_with_cell_controls": 7.0, "cell_ctrl_rate_sum": 6.0,
                        "delta_far_std": 0.1} for i in range(6)}
    fields["S0"] = {"ssdl_fires_with_cell_controls": 0.0, "cell_ctrl_rate_sum": 6.0, "delta_far_std": -0.6}
    sc = _native_score(fields)
    log_rr = C._h2_log_rr_per_session(sc)
    assert "S0" not in log_rr and len(log_rr) == 5


# ======================================================================================================= #
# H6 ordering: not significant when one of the three pairs fails (brief fix 10)
# ======================================================================================================= #
def test_h6_ordering_not_significant_when_one_pair_fails():
    # Build per-session MR-at-FAR readings so M1>M2>M3 holds for two pairs but M2 vs M3 is a tie (no sign).
    # mr_at_far_by_m maps system -> {session: reading}. 6 sessions.
    sess = [f"S{i}" for i in range(6)]
    mr = {
        "M1": {s: 0.40 for s in sess},
        "M2": {s: 0.20 for s in sess},
        "M3": {s: 0.20 for s in sess},   # equal to M2 -> M2 vs M3 diff == 0, not significant
    }
    rng = np.random.default_rng(C.SEED)
    h6 = C._h6(mr, {}, rng)
    assert h6["pairs"]["M1_vs_M2"]["significant"] is True
    assert h6["pairs"]["M1_vs_M3"]["significant"] is True
    assert h6["pairs"]["M2_vs_M3"]["significant"] is False
    assert h6["ordering_significant"] is False      # all_pairs: one failing pair sinks the ordering


def test_two_stage_bootstrap_widens_h6_when_m3_seeds_disagree():
    # The genuine two-stage bootstrap must add the between-seed variance: with M3's seeds in wide disagreement
    # per session, the M1-vs-M3 one-sided p is LARGER (wider / less certain) than when M3's seeds all agree on
    # the same per-session mean. Same observed per-session means in both cases, only the seed spread differs.
    sess = [f"S{i}" for i in range(8)]
    # M1 misses more than M3 by 0.10 per session (predict M1 - M3 > 0).
    mr_m1 = {s: 0.30 for s in sess}
    mr_m3_mean = {s: 0.20 for s in sess}

    def run(seed_spread):
        # three M3 seeds; their per-session mean is 0.20 either way, but the spread around it is seed_spread.
        seeds = {f"m3_s{j}": {s: 0.20 + (j - 1) * seed_spread for s in sess} for j in range(3)}
        mr = {"M1": mr_m1, "M3": dict(mr_m3_mean)}
        rng = np.random.default_rng(C.SEED)
        return C._h6(mr, seeds, rng)["pairs"]["M1_vs_M3"]

    tight = run(0.0)       # seeds agree exactly
    wide = run(0.50)       # seeds disagree strongly (M3 per-seed in {-0.30, 0.20, 0.70})
    # the observed mean diff is identical (M3's per-session seed-MEAN is 0.20 in both); the wide-seed bootstrap
    # carries the between-seed variance the session-only bootstrap ignores, so its one-sided p is STRICTLY
    # larger -- the seed stage genuinely widens H6's interval (brief fix 3 / review M3).
    assert wide["mean_mr_diff"] == pytest.approx(tight["mean_mr_diff"], abs=1e-9)
    assert tight["p_one_sided"] == pytest.approx(0.0, abs=1e-12)   # session-only: never crosses with 0.10 gap
    assert wide["p_one_sided"] > tight["p_one_sided"]              # seed disagreement adds uncertainty


# ======================================================================================================= #
# H5: the reported bound equals bca_ci on the LIVE path, and p + bound come from the same replicates (M4)
# ======================================================================================================= #
def test_h5_bound_equals_bca_ci_on_live_path_and_p_matches_bound():
    from csdiar.analysis.stats import bca_ci
    # One member M1 with a known per-session delta-MR; M2/M3 not assessable (too few sessions in their dicts).
    sess = [f"S{i}" for i in range(8)]
    dmr_m1 = {s: (-0.01 + 0.002 * i) for i, s in enumerate(sess)}   # mean slightly negative
    dmr = {"M1": dmr_m1, "M2": {}, "M3": {}}
    s_mr = 0.30
    rng = np.random.default_rng(C.SEED)
    h5 = C._h5_family(dmr, s_mr, rng, set(), descriptive=False)
    m = h5["members"]["M1"]
    # recompute the one-sided 95% upper bound with a FRESH generator at the SAME seed+call order: the first
    # bca_ci call in _h5_family consumed the rng from C.SEED state, so a fresh default_rng(SEED) matches it.
    x = [dmr_m1[s] for s in sorted(dmr_m1)]
    rng2 = np.random.default_rng(C.SEED)
    ref = bca_ci(x, C.mean_stat, 2 * C.H5_ONE_SIDED_LEVEL - 1.0, rng2)
    assert m["one_sided_95_upper"] == pytest.approx(ref["hi"], abs=1e-12)   # bit-for-bit bca_ci on the live path
    # the p inverts THAT bound at the margin: at a margin equal to the reported bound, p ~ 0.05 (the level whose
    # upper bound equals the margin), confirming p and bound share one resampling.
    rng3 = np.random.default_rng(C.SEED)
    ci = bca_ci(x, C.mean_stat, 2 * C.H5_ONE_SIDED_LEVEL - 1.0, rng3)       # same replicates as the bound
    rng3.bit_generator.state = np.random.default_rng(C.SEED).bit_generator.state
    p_at_bound = C._ni_pvalue(x, ci["hi"], np.random.default_rng(C.SEED))
    assert p_at_bound == pytest.approx(0.05, abs=0.03)


def test_h5_two_sided_interval_reported_at_983():
    sess = [f"S{i}" for i in range(8)]
    dmr = {"M1": {s: (-0.01 + 0.002 * i) for i, s in enumerate(sess)}, "M2": {}, "M3": {}}
    rng = np.random.default_rng(C.SEED)
    h5 = C._h5_family(dmr, 0.30, rng, set(), descriptive=False)
    assert h5["members"]["M1"]["two_sided_ci"]["level"] == pytest.approx(C.H4_CI_LEVEL, abs=1e-9)
    assert h5["two_sided_level"] == pytest.approx(1 - 0.05 / 3, abs=1e-9)


def test_h5_non_inferior_decision_uses_holm_adjusted_p():
    # M1 clearly non-inferior (mean delta MR ~ -0.01, bound well below the 0.02 margin) -> Holm p < .05 -> True.
    sess = [f"S{i}" for i in range(10)]
    dmr = {"M1": {s: -0.01 for s in sess}, "M2": {}, "M3": {}}
    rng = np.random.default_rng(C.SEED)
    h5 = C._h5_family(dmr, 0.30, rng, set(), descriptive=False)
    m = h5["members"]["M1"]
    assert m["non_inferior"] == bool(m["ni_pvalue_holm"] < C.ALPHA)
    assert m["non_inferior"] is True


def test_h5_second_member_bound_and_p_share_resamples_after_first(monkeypatch):
    """review-v2 m4: with TWO assessable members, the per-member `state` capture is inside the loop, so the
    second member's one-sided bound must equal `stats.bca_ci` recomputed at the generator state LEFT BY THE
    FIRST member, and its inverted-p must invert THAT bound (same replicates). A refactor that hoisted the
    `state` capture out of the loop would move the second member's draws and break both assertions."""
    from csdiar.analysis.stats import bca_ci
    sess = [f"S{i}" for i in range(8)]
    # M1 and M2 both assessable, with DISTINCT per-session deltas so their bounds differ; M3 empty.
    dmr_m1 = {s: (-0.012 + 0.002 * i) for i, s in enumerate(sess)}
    dmr_m2 = {s: (-0.004 + 0.001 * i) for i, s in enumerate(sess)}
    dmr = {"M1": dmr_m1, "M2": dmr_m2, "M3": {}}
    s_mr = 0.30
    margin = min(C.H5_MARGIN_PP, C.H5_MARGIN_REL * s_mr)

    h5 = C._h5_family(dmr, s_mr, np.random.default_rng(C.SEED), set(), descriptive=False)
    m2 = h5["members"]["M2"]
    assert "one_sided_95_upper" in m2 and "one_sided_95_upper" in h5["members"]["M1"]

    # Replay the generator through member M1 exactly as _h5_family drives it (one-sided bca_ci, rewind,
    # _ni_pvalue, two-sided bca_ci), leaving the rng at the state member M2 starts from.
    rng = np.random.default_rng(C.SEED)
    x1 = [dmr_m1[s] for s in sorted(dmr_m1)]
    st1 = rng.bit_generator.state
    bca_ci(x1, C.mean_stat, 2 * C.H5_ONE_SIDED_LEVEL - 1.0, rng)   # M1 one-sided
    rng.bit_generator.state = st1
    C._ni_pvalue(x1, margin, rng)                                  # M1 inverted-p (same replicates)
    bca_ci(x1, C.mean_stat, C.H5_TWO_SIDED_LEVEL, rng)            # M1 two-sided
    # now rng is at the state member M2 begins at. Its reported one-sided bound must equal bca_ci here.
    x2 = [dmr_m2[s] for s in sorted(dmr_m2)]
    st2 = rng.bit_generator.state
    ref2 = bca_ci(x2, C.mean_stat, 2 * C.H5_ONE_SIDED_LEVEL - 1.0, rng)
    assert m2["one_sided_95_upper"] == pytest.approx(ref2["hi"], abs=1e-12)
    # and M2's inverted p comes from the SAME replicates: inverting at a margin equal to the reported bound
    # yields p ~ 0.05 (the level whose one-sided upper bound equals the margin), computed from st2's draw.
    rng.bit_generator.state = st2
    p_at_bound = C._ni_pvalue(x2, ref2["hi"], rng)
    assert p_at_bound == pytest.approx(0.05, abs=0.03)


# ======================================================================================================= #
# M1: confirm reads only the frozen hypotheses; M2: native hash refusal and the --native-absent path
# (exercised through freeze_m2b in tests/test_freeze_m2b.py; here we cover the run-time behaviour)
# ======================================================================================================= #
def test_tuning_not_assessable_system_dropped_from_eval_hyps_and_reported():
    """review-v2 m2: a P1-AHC system the tuner marked not_assessable has NO frozen hypotheses
    (`eval_hypothesis_files`/`_frozen_systems` drop it) and is reported not-assessable by H4 and H5."""
    rids = [f"EVA{i:02d}" for i in range(6)]
    # a record where M1 has a threshold BUT is also listed not_assessable by the tuner. (No bare "M3" system:
    # a real M3 operating point is keyed "M3:<ckpt>"; here only the P1-AHC systems B0/M1/M2 are frozen.)
    rec = {"split": {"eval": rids},
           "tuning": {"thresholds": {"B0": 0.6, "M1": 0.6, "M2": 0.6},
                      "not_assessable": {"M1": {"reason": "degenerate operating point"}}}}
    # _frozen_systems / eval_hypothesis_files must omit M1 entirely.
    assert "M1" not in C._frozen_systems(rec)
    hyp_files = C.eval_hypothesis_files(Path("/none"), rec)
    from csdiar.pipelines import run_m2b as R
    m1_tag = R.system_tag("M1", delta=0.0, alpha=0.0, tau0=0.0, m3_ckpt=None)
    assert not any(f"/{m1_tag}/" in str(p) for p in hyp_files)   # no M1 hypothesis path at all

    # and run_analysis reports M1 not-assessable in H4/H5 and the intersection-union.
    _, rbs, dur, ch = _world()
    rng = np.random.default_rng(C.SEED)
    import csdiar.analysis.confirm_m2b as M
    monkey = pytest.MonkeyPatch()
    monkey.setattr(M, "_frozen_tag", lambda r, n, s: n)
    monkey.setattr(M, "_op_score", lambda a, b, c, d, tag, th, e: _score(rbs, dur, ch[tag]))
    monkey.setattr(M, "_score_system_sweep",
                   lambda a, b, c, d, tag, e: {"by_threshold": {0.6: _score(rbs, dur, ch[tag])},
                                               "det_points": [], "det_by_session": {}})
    monkey.setattr(M, "_native_changes", lambda a, b, tag, rids_: {rid: [] for rid in rids_})
    try:
        res = M.run_analysis(rbs, dur, Path("/none"), "eval", rec, rng, False, [7.62939453125e-06] * 3)
    finally:
        monkey.undo()
    assert res["not_assessable"].get("M1")                       # flagged degenerate/missing operating point
    assert res["H4"]["M1"].get("not_assessable") is True
    assert res["H5"]["members"]["M1"].get("not_assessable") is True
    assert res["intersection_union"]["M1"]["label"] == "not assessable"
def test_native_absent_reports_pipeline_h2_not_done():
    rids, rbs, dur, ch = _world()
    rec = _rec(rids)
    rec["native_absent"] = True
    rng = np.random.default_rng(C.SEED)
    import csdiar.analysis.confirm_m2b as M
    monkey = pytest.MonkeyPatch()
    monkey.setattr(M, "_frozen_tag", lambda r, n, s: n)
    monkey.setattr(M, "_op_score", lambda a, b, c, d, tag, th, e: _score(rbs, dur, ch[tag]))
    monkey.setattr(M, "_score_system_sweep",
                   lambda a, b, c, d, tag, e: {"by_threshold": {0.6: _score(rbs, dur, ch[tag])},
                                               "det_points": [], "det_by_session": {}})

    def _boom_native(*a, **k):
        raise AssertionError("native dir read under --native-absent")
    monkey.setattr(M, "_native_changes", _boom_native)
    try:
        res = M.run_analysis(rbs, dur, Path("/none"), "eval", rec, rng, False, [7.62939453125e-06] * 3)
    finally:
        monkey.undo()
    for sys_name in ("P1", "P2"):
        assert res["H2_conversation"]["members"][sys_name]["not_done"] is True
        assert res["H2_conversation"]["members"][sys_name]["reason"] == "native hypotheses not frozen"


def test_confirm_refuses_hyp_file_outside_frozen_set():
    # the frozen allow-set is enforced by _check_frozen_read: a path not in the set is refused.
    import csdiar.analysis.confirm_m2b as M
    monkey = pytest.MonkeyPatch()
    monkey.setattr(M, "_FROZEN_HYP_PATHS", {"/frozen/only.rttm"})
    try:
        with pytest.raises(SystemExit, match="not among the frozen eval hypotheses"):
            M._check_frozen_read(Path("/some/other/file.exclusive.rttm"))
    finally:
        monkey.undo()


# ======================================================================================================= #
# _ni_pvalue reproduces bca_ci's bound bit-for-bit on a shared seed
# ======================================================================================================= #
def test_ni_pvalue_bound_matches_bca_ci_for_shared_seed():
    from csdiar.analysis.stats import bca_ci
    deltas = [(-0.01 + 0.002 * i) for i in range(8)]
    # the one-sided 95% upper bound bca_ci returns (two-sided level 0.90 -> its hi is the 95% one-sided upper)
    rng1 = np.random.default_rng(12345)
    ci = bca_ci(deltas, C.mean_stat, 2 * 0.95 - 1.0, rng1, n_boot=2000)
    # _ni_pvalue, at a margin equal to that bound, should return p ~ 0.05 (the level whose upper bound == margin)
    rng2 = np.random.default_rng(12345)
    p = C._ni_pvalue(deltas, ci["hi"], rng2, n_boot=2000)
    assert p == pytest.approx(0.05, abs=0.02)


def test_ni_pvalue_returns_one_when_point_estimate_exceeds_margin():
    deltas = [0.10 + 0.001 * i for i in range(8)]   # mean ~ 0.10, far above a 0.02 margin
    p = C._ni_pvalue(deltas, 0.02, np.random.default_rng(1), n_boot=500)
    assert p == 1.0


# ======================================================================================================= #
# --dry-run never opens an eval path
# ======================================================================================================= #
def test_dry_run_never_touches_eval(tmp_path, monkeypatch):
    """The --dry-run CLI path reads only the dev units and the M2a result, writes confirm_m2b_dryrun.json, and
    never calls the eval in-process ingest or reads an eval path."""
    rids = [f"DEV{i:02d}" for i in range(6)]
    units_dev = tmp_path / "units/dev"
    units_dev.mkdir(parents=True)
    # write dev reader rows + a summary with durations
    import csv
    import gzip
    rows = []
    for rid in rids:
        rows.extend(_rows_for_session(rid))
    for kind in ("ssdl", "turns", "ctrl"):
        subset = [r for r in rows if (r["kind"] in ("dsdl", "dssl") if kind == "turns" else r["kind"] == kind)]
        with gzip.open(units_dev / f"{kind}_dev.csv.gz", "wt", encoding="utf-8", newline="") as fh:
            from csdiar.data.ingest_displace import FIELDS
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in subset:
                w.writerow(r)
    (units_dev / "summary.json").write_text(json.dumps(
        {"split": "dev", "recordings": [{"recording": rid, "duration_s": 300.0} for rid in rids]}))
    m2a = tmp_path / "m2a.json"
    m2a.write_text(json.dumps({"h2_p0": {"primary": {m: {"signflip": {"p_one": 1e-5}}
                                                     for m in ("e1", "e2", "e3")}}}))
    out = tmp_path / "out"
    # a FREEZE_M2B.json with a dev split and frozen thresholds; dry-run reads tuning from it
    record = tmp_path / "FREEZE_M2B.json"
    record.write_text(json.dumps({"split": {"dev": rids},
                                  "tuning": {"thresholds": {"B0": 0.6, "M1": 0.6, "M2": 0.6, "M3": 0.6}}}))
    monkeypatch.setattr(C, "RECORD", record)

    # poison the eval in-process ingest and any eval unit read: the dry run must call NEITHER
    def _boom(*_a, **_k):
        raise AssertionError("dry-run touched an eval path")
    monkeypatch.setattr(C, "_ingest_eval_in_process", _boom)

    # fake the per-system scoring so no RTTM is needed
    rbs = {rid: _rows_for_session(rid) for rid in rids}
    dur = {rid: 300.0 for rid in rids}
    ch = {s: {rid: _system_changes(rbs[rid], f, m) for rid in rids}
          for s, (f, m) in {"B0": (0.6, 0.3), "M1": (0.2, 0.3), "M2": (0.2, 0.7), "M3": (0.6, 0.3)}.items()}
    monkeypatch.setattr(C, "_frozen_tag", lambda rec, name, spec: name)
    monkeypatch.setattr(C, "_op_score", lambda rbs_, dur_, hr, sp, tag, th, rids_: _score(rbs_, dur_, ch[tag]))
    monkeypatch.setattr(C, "_score_system_sweep",
                        lambda rbs_, dur_, hr, sp, tag, rids_: {"by_threshold": {0.6: _score(rbs_, dur_, ch[tag])},
                                                                "det_points": [], "det_by_session": {}})
    monkeypatch.setattr(C, "_native_changes", lambda hr, sp, tag, rids_: {rid: [] for rid in rids_})

    rc = C.main(["--units-dev", str(units_dev), "--out", str(out), "--confirm-m2a", str(m2a),
                 "--hyp-root", str(tmp_path / "hyp"), "--dry-run"])
    assert rc == 0
    res = json.loads((out / "confirm_m2b_dryrun.json").read_text())
    assert res["dry_run"] is True and res["split"] == "dev"
    assert not (out / "confirm_m2b.json").exists()
    assert not (out / "confirm_m2b.started").exists()
