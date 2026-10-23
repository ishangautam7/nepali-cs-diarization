#!/usr/bin/env python3
"""Unit tests for csdiar.analysis.tune_p0_m2b (PREREGISTRATION_H4.md §2, §4, §5; PREREGISTRATION_M2B.md §2, §3,
§6).

Model-free (except one optional CPU smoke test of the dense grid), no GPU, no DISPLACE archive: every input is
synthetic. Covered:
  * GATE EQUALITY with the D1 code: `gated_curve` reproduces `confirm_h4.system_curves("M2", ...)` on shared
    inputs, so the DISPLACE P0 gate is the same code as the D1 gate (brief);
  * alpha = 0 reproduces B0 exactly (the ungated d curve);
  * lambda on the grid against `m2_lambda` on the same posteriors (pair (0, 1) == the D1 (ne, en) case);
  * the P0 hit rule at the +-0.5 s edges (reused `pilot.peak_strength`);
  * B0's dev tau calibration (reused `pilot.calibrate_tau`), including the degenerate -inf case;
  * the tie rules (smaller alpha, then lower MR, then smaller tau0) and the +1 pp bound boundary;
  * the not-assessable cases (< 5 two-language sessions with >= 10 DS-DL turns; degenerate dev tau);
  * a guard that only the dev split is read;
  * an OPTIONAL CPU smoke test of the dense grid on pyannote's bundled sample.wav, niced, skipped when the model
    is absent.

Run under .venv-pyannote4 (numpy present) on the box:
  CUDA_VISIBLE_DEVICES= .venv-pyannote4/bin/python -m pytest -q -p no:cacheprovider tests/test_tune_p0_m2b.py
"""
import json
from pathlib import Path

import numpy as np
import pytest

from csdiar.analysis import confirm_h4
from csdiar.analysis import m2_lambda
from csdiar.analysis import tune_p0_m2b as T
from csdiar.analysis.pilot import HOP, unit


# ======================================================================================================= #
# helpers to build a synthetic dense grid + posteriors
# ======================================================================================================= #
def _random_grid(n: int, dim: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return unit(rng.standard_normal((n, dim)).astype(np.float32))


def _random_posteriors(n: int, n_langs: int = 5, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.random((n, n_langs)) + 1e-3
    return p / p.sum(1, keepdims=True)


# ======================================================================================================= #
# GATE EQUALITY with the D1 code (brief: the DISPLACE gate is the same code as the D1 gate)
# ======================================================================================================= #
class TestGateEquality:
    def test_gated_curve_is_confirm_h4_system_curves_on_shared_inputs(self):
        """On a 2-language posterior (so the pair is (0, 1) and no renorm changes it), `gated_curve` and a direct
        `confirm_h4.system_curves("M2", ...)` call on the SAME unit grid and renormalised posteriors agree
        bit-for-bit -- the DISPLACE gate and the D1 gate are the same code path."""
        n = 60
        e = _random_grid(n, seed=3)
        # a genuine 2-language posterior, already summing to 1 (pair (0,1), renorm is the identity)
        two = _random_posteriors(n, n_langs=2, seed=4)
        alpha = 0.45
        got = T.gated_curve(e, two, (0, 1), alpha)
        # the D1 path: confirm_h4 expects an already-renormalised 2-column lid array (its load_lid renorms)
        want = confirm_h4.system_curves("M2", {"rec"}, {"rec": unit(e)}, {"rec": two}, {"alpha": alpha})["rec"]
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

    def test_gate_equality_with_full_posterior_pair_slice(self):
        """With a 107-wide posterior and an arbitrary reference pair, `gated_curve`'s renormalised 2-column slice
        equals feeding that same renormalised slice straight to `confirm_h4.system_curves`."""
        n = 80
        e = _random_grid(n, seed=5)
        post = _random_posteriors(n, n_langs=7, seed=6)
        i, j = 2, 5
        alpha = 0.3
        got = T.gated_curve(e, post, (i, j), alpha)
        two = post[:, (i, j)]
        two = two / two.sum(1, keepdims=True)
        want = confirm_h4.system_curves("M2", {"rec"}, {"rec": unit(e)}, {"rec": two}, {"alpha": alpha})["rec"]
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


# ======================================================================================================= #
# alpha = 0 reproduces B0 exactly
# ======================================================================================================= #
class TestAlphaZeroIsB0:
    def test_gated_alpha0_equals_d_curve(self):
        n = 50
        e = _random_grid(n, seed=7)
        post = _random_posteriors(n, n_langs=4, seed=8)
        d = np.asarray(T.d_curve(e))
        g0 = np.asarray(T.gated_curve(e, post, (0, 1), 0.0))
        np.testing.assert_array_equal(d[~np.isnan(d)], g0[~np.isnan(g0)])
        assert np.isnan(d[: T.LAG_K]).all() and np.isnan(g0[: T.LAG_K]).all()

    def test_d_curve_matches_confirm_h4_B0(self):
        """`d_curve` is exactly confirm_h4's ungated B0 curve on the same unit grid."""
        n = 40
        e = _random_grid(n, seed=9)
        got = np.asarray(T.d_curve(e))
        want = confirm_h4.system_curves("B0", {"rec"}, {"rec": unit(e)}, {}, {})["rec"]
        np.testing.assert_array_equal(got[~np.isnan(got)], np.asarray(want)[~np.isnan(want)])


# ======================================================================================================= #
# lambda on the grid against m2_lambda on the same posteriors
# ======================================================================================================= #
class TestLambdaOnGrid:
    def test_matches_m2_lambda_pair01(self):
        """`lambda_on_grid` at window k equals `m2_lambda.lam` at t = 0.1k on the SAME posteriors, for the D1
        (ne, en) = pair (0, 1) case (m2_lambda reads columns 1:3 of a [centre, ne, en, ...] array, which are the
        two language columns; we build a posterior whose columns 1 and 2 are the pair)."""
        n = 70
        # m2_lambda.lam reads post[k, 1:3] (ne, en). Build a [centre, p0, p1, other] layout so that the pair
        # columns (1, 2) carry a real 2-language distribution.
        rng = np.random.default_rng(11)
        p01 = rng.random((n, 2)) + 1e-3
        p01 = p01 / p01.sum(1, keepdims=True)
        ml_post = np.column_stack([np.arange(n) * HOP, p01[:, 0], p01[:, 1], np.zeros(n)])
        # our lambda reads pair indices into the SAME two columns (1, 2) of ml_post
        lam_ours = T.lambda_on_grid(ml_post, (1, 2))
        for k in range(T.LAG_K, n):
            t = k * HOP
            want = m2_lambda.lam(ml_post, t)
            assert want is not None
            assert lam_ours[k] == pytest.approx(want, abs=1e-12)

    def test_first_lag_k_are_nan(self):
        post = _random_posteriors(30, n_langs=3, seed=12)
        lam = T.lambda_on_grid(post, (0, 1))
        assert np.isnan(lam[: T.LAG_K]).all()
        assert np.isfinite(lam[T.LAG_K:]).all()

    def test_lambda_bounded_by_one_bit(self):
        post = _random_posteriors(60, n_langs=6, seed=13)
        lam = T.lambda_on_grid(post, (1, 4))
        fin = lam[np.isfinite(lam)]
        assert (fin >= -1e-12).all() and (fin <= 1.0 + 1e-9).all()


# ======================================================================================================= #
# the P0 hit rule at the +-0.5 s edges (reused pilot.peak_strength + the > tau comparison)
# ======================================================================================================= #
class TestHitRule:
    def _curve_with_peak_at(self, n: int, peak_k: int, peak_val: float) -> np.ndarray:
        d = np.full(n, 0.0, np.float32)
        d[: T.LAG_K] = np.nan                     # P0 has no left window for the first LAG_K
        d[peak_k] = peak_val
        return d

    def test_peak_just_inside_half_second_window_fires(self):
        """C = 0.5 s = 5 grid steps. A peak 5 steps away (exactly +-c) is inside the CLOSED window and fires."""
        assert abs(T.C - 0.5) < 1e-9
        r = int(round(T.C / HOP))                 # 5
        n = 60
        t_k = 30
        d = self._curve_with_peak_at(n, t_k + r, 0.9)
        assert T.hit_on_curve(d, t_k * HOP, tau=0.5) is True

    def test_peak_just_outside_half_second_window_does_not_fire(self):
        r = int(round(T.C / HOP))
        n = 60
        t_k = 30
        d = self._curve_with_peak_at(n, t_k + r + 1, 0.9)     # 0.6 s away: outside +-c
        assert T.hit_on_curve(d, t_k * HOP, tau=0.5) is False

    def test_peak_below_tau_does_not_fire(self):
        n = 60
        t_k = 30
        d = self._curve_with_peak_at(n, t_k, 0.4)
        assert T.hit_on_curve(d, t_k * HOP, tau=0.5) is False

    def test_nan_left_window_is_not_a_hit(self):
        n = 60
        d = np.full(n, 0.9, np.float32)
        d[: T.LAG_K] = np.nan
        # a point at the very start (k < LAG_K) has NaN d and cannot hit
        assert T.hit_on_curve(d, 0.1, tau=0.1) is False


# ======================================================================================================= #
# dev tau calibration (reused pilot.calibrate_tau), incl. the degenerate -inf case
# ======================================================================================================= #
def _ctrl_rows(times):
    return [{"kind": "ctrl", "t": t, "pause_bin": "none"} for t in times]


def _ssdl_rows(times, pause="none"):
    return [{"kind": "ssdl", "t": t, "pause_bin": pause, "left_s": 1.0, "right_s": 1.0} for t in times]


def _dsdl_rows(times):
    return [{"kind": "dsdl", "t": t, "pause_bin": "none", "left_s": 1.0, "right_s": 1.0} for t in times]


class TestDevTau:
    def test_tau_at_10pct_control_fire(self):
        """20 controls spread out, each sitting on its own peak of a known height; the tau at which 10% fire is
        the 90th-percentile peak height (pilot.calibrate_tau's quantile)."""
        n = 400
        d = np.full(n, 0.0, np.float32)
        d[: T.LAG_K] = np.nan
        times = [HOP * (T.LAG_K + 15 * k) for k in range(20)]   # well separated so NMS peaks don't collide
        heights = np.linspace(0.10, 0.90, 20)
        for t, h in zip(times, heights):
            d[int(round(t / HOP))] = h
        tau, dropped = T.dev_tau({"rec": _ctrl_rows(times)}, {"rec": d})
        # 10% of 20 controls = 2 fire above tau; tau is pilot's quantile(peaks, 1 - 0.10) -> near the top heights
        assert np.isfinite(tau)
        assert dropped == 0                         # every control has a defined, finite d(t)
        fired = sum(T.hit_on_curve(d, t, tau) for t in times)
        assert fired <= 2

    def test_degenerate_tau_minus_inf(self):
        """When <= 10% of controls have a FINITE peak (pilot refuses the degenerate case, PREREGISTRATION_H4.md
        §5), dev_tau is -inf and tune() raises InsufficientData. Here the controls sit past LAG_K but the curve
        is all-NaN, so every control is DROPPED (M1): calibrate_tau then sees an empty peak set -> share 0 -inf.
        """
        n = 200
        d = np.full(n, np.nan, np.float32)        # no finite peaks anywhere -> finite share 0 <= 10%
        times = [HOP * (T.LAG_K + 10 * k) for k in range(12)]
        tau, dropped = T.dev_tau({"rec": _ctrl_rows(times)}, {"rec": d})
        assert tau == float("-inf")
        assert dropped == 12                        # all 12 dropped (NaN d), none passed as -inf

    def test_m1_edge_controls_dropped_not_passed_as_minus_inf(self):
        """M1 (review v1), the reviewer's trace: 12 controls, 2 of them in the first second (t < 1.0 s, so
        k < LAG_K -> d[k] NaN, no left window). dev_tau must DROP those 2 exactly as pilot.score_points does and
        calibrate tau on the 10 finite ones -- NOT pass them to calibrate_tau as -inf.

        We check two things the old -inf handling got wrong:
          (a) tau equals the tau computed from the 10 finite controls alone (share = 1.0), not the shifted
              quantile the old code produced from share = 10/12;
          (b) a case the old -inf handling would have pushed into the degenerate branch is NOT degenerate here.
        """
        n = 400
        d = np.full(n, 0.0, np.float32)
        d[: T.LAG_K] = np.nan
        # 2 edge controls inside the first second (k < LAG_K == 10 windows == 1.0 s)
        edge_times = [HOP * 3, HOP * 6]
        # 10 finite controls well past the edge, each on its own peak of known height
        finite_times = [HOP * (T.LAG_K + 15 * k) for k in range(10)]
        heights = np.linspace(0.10, 0.90, 10)
        for t, h in zip(finite_times, heights):
            d[int(round(t / HOP))] = h
        all_times = edge_times + finite_times
        tau_all, dropped = T.dev_tau({"rec": _ctrl_rows(all_times)}, {"rec": d})
        tau_finite, dropped_finite = T.dev_tau({"rec": _ctrl_rows(finite_times)}, {"rec": d})
        # (a) the 2 edge controls are dropped, and tau is the 10-finite-control tau, bit-for-bit
        assert dropped == 2 and dropped_finite == 0
        assert tau_all == tau_finite
        assert np.isfinite(tau_all)

    def test_m1_edge_controls_do_not_fabricate_degenerate_case(self):
        """M1 (review v1): only 1 of 12 controls has a finite peak, the other 11 are edge/NaN. After the DROP the
        finite share is 1/1 = 1.0 (not degenerate). The OLD -inf handling would have given share = 1/12 <= 0.10
        -> tau = -inf (a fabricated degenerate case). With the fix, dropping the 11 leaves a finite tau."""
        n = 300
        d = np.full(n, np.nan, np.float32)
        # 11 edge/NaN controls (curve is NaN at their k), 1 finite control on a real peak
        edge_times = [HOP * k for k in range(1, 12)]           # all within / on NaN region or NaN cells
        finite_t = HOP * (T.LAG_K + 50)
        d[int(round(finite_t / HOP))] = 0.7
        times = edge_times + [finite_t]
        tau, dropped = T.dev_tau({"rec": _ctrl_rows(times)}, {"rec": d})
        assert dropped == 11                                   # the 11 NaN controls dropped, not passed as -inf
        assert np.isfinite(tau)                                # share = 1/1 = 1.0 -> NOT degenerate


# ======================================================================================================= #
# scoring + aggregation + the joint selection / tie / bound-boundary rules
# ======================================================================================================= #
def _make_curve_from_point_fires(point_times, fire_set, n=600):
    """Build a d curve where exactly the points in `fire_set` (by time) sit on a tall peak above tau=0.5, the
    rest on a 0.0 floor; LAG_K prefix NaN."""
    d = np.full(n, 0.0, np.float32)
    d[: T.LAG_K] = np.nan
    for t in point_times:
        if t in fire_set:
            d[int(round(t / HOP))] = 0.9
    return d


class TestScoreSessionP0:
    def test_far_std_and_mr(self):
        ssdl_t = [5.0, 7.0, 9.0]
        ctrl_t = [12.0, 14.0, 16.0, 18.0]
        dsdl_t = [20.0, 22.0]
        rows = _ssdl_rows(ssdl_t) + _ctrl_rows(ctrl_t) + _dsdl_rows(dsdl_t)
        # fire 2 of 3 ssdl, 1 of 4 ctrl (pause bin 'none' cell), 1 of 2 dsdl
        fire = {5.0, 7.0, 12.0, 20.0}
        d = _make_curve_from_point_fires(ssdl_t + ctrl_t + dsdl_t, fire)
        sc = T.score_session_p0("rec", rows, d, tau=0.5)
        assert sc.n_ssdl_with_cell_controls == 3 and sc.n_dsdl == 2 and sc.n_ctrl == 4
        # ctrl fire rate in the single 'none' cell = 1/4 = 0.25; ssdl fires = [1,1,0]; delta = each - 0.25
        assert sc.far_ssdl == pytest.approx((1 - 0.25 + 1 - 0.25 + 0 - 0.25) / 3)
        assert sc.ctrl_fire_rate == pytest.approx(0.25)
        assert sc.mr_dsdl == pytest.approx(1 - 1 / 2)      # 1 of 2 dsdl fired -> miss rate 0.5

    def test_points_without_defined_d_are_dropped_like_score_points(self):
        # review round 2, m1: a point in the first second (NaN d) or past the curve end is neither a fire nor a
        # non-fire; it leaves every rate and count, exactly as pilot.score_points drops it.
        ssdl_t = [5.0, 7.0]
        ctrl_t = [12.0, 14.0]
        dsdl_t = [20.0]
        edge_ssdl, edge_ctrl, edge_dsdl = [0.3], [0.5, 999.0], [0.2]     # k < LAG_K, or k >= len(curve)
        rows = (_ssdl_rows(ssdl_t + edge_ssdl) + _ctrl_rows(ctrl_t + edge_ctrl)
                + _dsdl_rows(dsdl_t + edge_dsdl))
        d = _make_curve_from_point_fires(ssdl_t + ctrl_t + dsdl_t, {5.0, 12.0, 20.0})
        sc = T.score_session_p0("rec", rows, d, tau=0.5)
        assert sc.n_ctrl == 2 and sc.n_dsdl == 1 and sc.n_ssdl_with_cell_controls == 2
        assert sc.ctrl_fire_rate == pytest.approx(0.5)                     # 1 of the 2 defined controls
        assert sc.far_ssdl == pytest.approx(((1 - 0.5) + (0 - 0.5)) / 2)
        assert sc.mr_dsdl == pytest.approx(0.0)                            # the defined DS-DL turn fired

    def test_ssdl_without_cell_control_counted_not_scored(self):
        # an ssdl in a pause bin with no control falls into without-cell-control
        rows = _ssdl_rows([5.0], pause="gt300") + _ctrl_rows([12.0])     # control is 'none' bin only
        d = _make_curve_from_point_fires([5.0, 12.0], set())
        sc = T.score_session_p0("rec", rows, d, tau=0.5)
        assert sc.n_ssdl_with_cell_controls == 0 and sc.n_ssdl_without_cell_controls == 1
        assert np.isnan(sc.far_ssdl)


class TestSelectAlphaTau0:
    def _cell(self, alpha, tau0, far, mr, feasible, far_defined=True):
        return T.GridCell(alpha=alpha, tau0=tau0, far_ssdl=far, mr_dsdl=mr,
                          delta_mr_upper_bound=(0.0 if feasible else 0.5), feasible=feasible,
                          n_sessions_far=6, n_sessions_mr=6, far_defined=far_defined)

    def test_minimise_far(self):
        cells = [self._cell(0.0, 0.0, 0.30, 0.10, True),
                 self._cell(0.5, 0.0, 0.10, 0.20, True),
                 self._cell(0.8, 0.05, 0.25, 0.15, True)]
        out = T.select_alpha_tau0(cells)
        assert out["alpha"] == 0.5 and out["far_ssdl"] == pytest.approx(0.10)

    def test_tie_far_goes_to_smaller_alpha(self):
        cells = [self._cell(0.8, 0.0, 0.12, 0.10, True),
                 self._cell(0.3, 0.02, 0.12, 0.10, True)]
        out = T.select_alpha_tau0(cells)
        assert out["alpha"] == 0.3

    def test_tie_far_and_alpha_goes_to_lower_mr(self):
        cells = [self._cell(0.3, 0.0, 0.12, 0.18, True),
                 self._cell(0.3, 0.05, 0.12, 0.11, True)]
        out = T.select_alpha_tau0(cells)
        assert out["mr_dsdl"] == pytest.approx(0.11)

    def test_tie_far_alpha_mr_goes_to_smaller_tau0(self):
        cells = [self._cell(0.3, 0.05, 0.12, 0.11, True),
                 self._cell(0.3, -0.03, 0.12, 0.11, True)]
        out = T.select_alpha_tau0(cells)
        assert out["tau0"] == pytest.approx(-0.03)

    def test_infeasible_cells_excluded(self):
        cells = [self._cell(0.0, 0.0, 0.30, 0.10, True),
                 self._cell(0.5, 0.0, 0.05, 0.40, False)]     # lowest FAR but INFEASIBLE
        out = T.select_alpha_tau0(cells)
        assert out["alpha"] == 0.0                            # the only feasible one

    def test_no_feasible_raises(self):
        cells = [self._cell(0.5, 0.0, 0.05, 0.40, False)]
        with pytest.raises(T.InsufficientData):
            T.select_alpha_tau0(cells)

    def test_m2_all_undefined_far_feasible_set_is_visible(self):
        """m2 (review v1): when every feasible cell has an UNDEFINED (NaN) FAR, the choice falls to the alpha/mr/
        tau0 tie fields (all NaN FAR coerce to +inf equally), and that is surfaced -- the chosen cell's
        `far_defined` is False and `n_feasible_far_undefined` counts them, so the alpha is never a silent
        NaN->inf artifact."""
        nan = float("nan")
        cells = [self._cell(0.0, 0.0, nan, 0.10, True, far_defined=False),
                 self._cell(0.5, 0.0, nan, 0.20, True, far_defined=False)]
        out = T.select_alpha_tau0(cells)
        assert out["alpha"] == 0.0                            # smaller alpha wins the FAR tie (both +inf)
        assert out["far_defined"] is False
        assert out["n_feasible_far_undefined"] == 2


class TestBoundBoundary:
    def test_bound_at_exactly_plus_one_pp_is_feasible(self):
        """deltaMR whose upper 80% BCa bound is exactly +1 pp is feasible (<= bound, closed)."""
        rng = np.random.default_rng(0)
        # construct per-session deltas whose bca upper bound lands at ~+0.01; use a constant just below, and
        # verify the feasibility comparison itself at the boundary.
        assert (0.01 <= T.MR_BOUND_PP + 1e-12)                # the comparison select_alpha_tau0 / tune use
        assert not (0.0101 <= T.MR_BOUND_PP + 1e-12)

    def test_mr_upper_bound_all_zero_deltas(self):
        """All-zero per-session deltaMR -> upper bound 0 (feasible). Exercises stats.bca_ci through the helper."""
        rng = np.random.default_rng(1)
        deltas = {f"s{k}": 0.0 for k in range(8)}
        ub = T.mr_upper_bound(deltas, rng)
        assert ub["n_sessions"] == 8
        assert ub["bound"] == pytest.approx(0.0, abs=1e-9)

    def test_mr_upper_bound_few_sessions_is_nan(self):
        ub = T.mr_upper_bound({"s0": 0.01}, np.random.default_rng(2))
        assert ub["n_sessions"] == 1 and np.isnan(ub["bound"])


# ======================================================================================================= #
# assessability (< 5 two-language sessions with >= 10 DS-DL turns)
# ======================================================================================================= #
class TestAssess:
    def test_tunable_when_enough(self):
        out = T.assess_tunable({f"s{k}": 12 for k in range(5)})
        assert out["tunable"] is True and out["n_qualifying"] == 5

    def test_not_tunable_too_few_sessions(self):
        out = T.assess_tunable({f"s{k}": 12 for k in range(4)})
        assert out["tunable"] is False and out["n_qualifying"] == 4

    def test_not_tunable_too_few_turns_per_session(self):
        out = T.assess_tunable({f"s{k}": 9 for k in range(10)})     # 10 sessions but each < 10 turns
        assert out["tunable"] is False and out["n_qualifying"] == 0

    def test_two_language_sessions_from_summary(self):
        """M2 (review v1): two-language is decided from RESOLVED VoxLingua107 codes via a label map and the
        recording's LID label list, not the raw LD-label count. All labels here map cleanly to codes present in
        the LID list; r2's three labels resolve to three codes -> excluded."""
        summary = {"recordings": [
            {"recording": "r1", "languages": ["eng", "hin"], "n_dsdl": 15},
            {"recording": "r2", "languages": ["eng", "hin", "ben"], "n_dsdl": 20},   # 3 codes -> excluded
            {"recording": "r3", "languages": ["eng", "ben"], "n_dsdl": 5},
        ]}
        label_map = {"eng": "en", "hin": "hi", "ben": "bn"}
        lid = {"r1": ["en", "hi"], "r2": ["en", "hi", "bn"], "r3": ["en", "bn"]}
        c = T.two_language_sessions(summary, label_map, lambda rid: lid[rid])
        assert c["n_two_language_recordings"] == 2
        assert set(c["dsdl_per_two_lang_session"]) == {"r1", "r3"}
        assert c["n_two_lang_sessions_ge10_dsdl"] == 1
        assert c["pair_by_recording"]["r1"] == ("en", "hi")
        assert c["pair_by_recording"]["r3"] == ("bn", "en")      # distinct, sorted order of the two labels
        assert "r2" in c["excluded"] and c["n_excluded"] == 1

    def test_m2_unmapped_label_excluded_not_aborted(self):
        """M2: a recording with an LD label absent from the label map is EXCLUDED, flagged and counted -- not an
        abort."""
        summary = {"recordings": [{"recording": "r1", "languages": ["eng", "xyz"], "n_dsdl": 15}]}
        label_map = {"eng": "en", "hin": "hi"}                  # 'xyz' unmapped
        c = T.two_language_sessions(summary, label_map, lambda rid: ["en", "hi"])
        assert c["dsdl_per_two_lang_session"] == {}
        assert "r1" in c["excluded"]
        assert "unmapped" in c["excluded"]["r1"]["reason"]

    def test_m2_label_maps_to_code_absent_from_lid_labels_excluded(self):
        """M2: a label mapping to a VoxLingua107 code that is NOT in the recording's LID label list is
        unresolvable -> the recording is excluded, flagged and counted."""
        summary = {"recordings": [{"recording": "r1", "languages": ["eng", "hin"], "n_dsdl": 15}]}
        label_map = {"eng": "en", "hin": "hi"}
        c = T.two_language_sessions(summary, label_map, lambda rid: ["en", "bn"])   # 'hi' not in LID labels
        assert c["dsdl_per_two_lang_session"] == {}
        assert "r1" in c["excluded"]
        assert "not in the recording's LID label list" in c["excluded"]["r1"]["reason"]

    def test_m2_two_labels_same_code_not_two_language(self):
        """M2: two distinct LD labels that map to the SAME VoxLingua107 code collapse to one distinct code, so
        the recording is NOT two-language -> excluded, flagged and counted."""
        summary = {"recordings": [{"recording": "r1", "languages": ["hin", "hindi"], "n_dsdl": 15}]}
        label_map = {"hin": "hi", "hindi": "hi"}                # both -> 'hi'
        c = T.two_language_sessions(summary, label_map, lambda rid: ["en", "hi"])
        assert c["dsdl_per_two_lang_session"] == {}
        assert "r1" in c["excluded"]
        assert "not exactly two" in c["excluded"]["r1"]["reason"]

    def test_m2_resolve_recording_pair_resolved(self):
        """M2: a clean two-language recording resolves to its code pair and the LID-label indices."""
        res = T.resolve_recording_pair(["eng", "hin"], ["en", "hi", "bn"], {"eng": "en", "hin": "hi"})
        assert res["resolved"] is True
        assert res["codes"] == ("en", "hi")
        assert res["indices"] == (0, 1)


# ======================================================================================================= #
# the pure tune() core end-to-end on synthetic curves (degenerate + assessable)
# ======================================================================================================= #
def _session_rows(seed, n_ssdl=12, n_ctrl=12, n_dsdl=12):
    rng = np.random.default_rng(seed)
    t = 5.0
    rows = []
    for _ in range(n_ssdl):
        rows.append({"kind": "ssdl", "t": t, "pause_bin": "none", "left_s": 1.0, "right_s": 1.0}); t += 2.0
    for _ in range(n_ctrl):
        rows.append({"kind": "ctrl", "t": t, "pause_bin": "none"}); t += 2.0
    for _ in range(n_dsdl):
        rows.append({"kind": "dsdl", "t": t, "pause_bin": "none", "left_s": 1.0, "right_s": 1.0}); t += 2.0
    return rows


class TestTuneCore:
    def test_assessability_rechecked_on_scorable_dsdl_counts(self):
        # review round 3, M1: five sessions with 10 DS-DL turns each on the reader's count, but one session's first
        # DS-DL turn sits in the first second (no defined d) -> 9 scorable turns -> only 4 sessions qualify on the
        # scored counts, so tune() must report M2 not assessable instead of freezing alpha on 4 sessions.
        sessions = [f"s{k}" for k in range(5)]
        rows = {s: _session_rows(k, n_dsdl=10) for k, s in enumerate(sessions)}
        rows["s0"][-10]["t"] = 0.6                          # s0's first DS-DL turn moves into the first second
        n = 900
        d = {}
        for s in sessions:
            c = np.full(n, 0.0, np.float32)
            c[: T.LAG_K] = np.nan
            for r in rows[s]:
                if r["kind"] == "ctrl":
                    c[int(round(r["t"] / HOP))] = 0.2        # finite control peaks -> finite tau
            d[s] = c
        assert T.assess_tunable({s: 10 for s in sessions})["tunable"] is True   # the reader-count verdict
        with pytest.raises(T.InsufficientData, match="scorable DS-DL") as exc:
            T.tune(rows, d, lambda s, a: d[s], np.random.default_rng(0))
        # review round 4, m2: the decisive numbers travel as structured details, not only inside the message
        assert exc.value.details["scored_dsdl_per_session"] == {"s0": 9, "s1": 10, "s2": 10, "s3": 10, "s4": 10}
        assert exc.value.details["n_scored_two_lang_sessions_ge10_dsdl"] == 4

    @staticmethod
    def _five_sessions_with_edge_points(n=900):
        """Five sessions of 12 SS-DL / 12 controls / 12 DS-DL, plus in s0 an SS-DL point, a control and a DS-DL
        turn with no defined d(t) (first second, or past the grid). B0 curves carry finite control peaks."""
        sessions = [f"s{k}" for k in range(5)]
        rows = {s: _session_rows(k) for k, s in enumerate(sessions)}
        rows["s0"] += [{"kind": "ssdl", "t": 0.3, "pause_bin": "none", "left_s": 1.0, "right_s": 1.0},
                       {"kind": "ctrl", "t": 999.0, "pause_bin": "none"},
                       {"kind": "dsdl", "t": 0.6, "pause_bin": "none", "left_s": 1.0, "right_s": 1.0}]
        d = {}
        for s in sessions:
            c = np.full(n, 0.0, np.float32)
            c[: T.LAG_K] = np.nan
            for r in rows[s]:
                k = int(round(r["t"] / HOP))
                if r["kind"] == "ctrl" and k < n:
                    c[k] = 0.2
            d[s] = c
        return sessions, rows, d

    def test_scored_population_fixed_across_operating_points(self):
        # review round 4, m1: the counts the re-check and the inclusion sets read off the offset-0 B0 cell are the
        # same at every tau and on a gated curve (gating subtracts finite values only after LAG_K).
        sessions, rows, d = self._five_sessions_with_edge_points()
        rng = np.random.default_rng(3)
        gated = {s: d[s].copy() for s in sessions}
        for s in sessions:
            gated[s][T.LAG_K:] -= 0.6 * rng.random(len(d[s]) - T.LAG_K).astype(np.float32)

        def counts(sc):
            return (sc.n_ssdl_with_cell_controls, sc.n_ssdl_without_cell_controls, sc.n_dsdl, sc.n_ctrl)

        ref = {s: counts(sc) for s, sc in T._score_curve_over_sessions(d, rows, 0.1).items()}
        assert ref["s0"] == (12, 0, 12, 12) and ref["s1"] == (12, 0, 12, 12)     # the edge points left
        for curves in (d, gated):
            for tau in (-0.5, 0.1, 0.15, 0.9, 5.0):
                got = {s: counts(sc) for s, sc in T._score_curve_over_sessions(curves, rows, tau).items()}
                assert got == ref

    def test_gating_that_changes_the_scored_population_fails_loudly(self):
        # review round 4, m1: a gated curve with a NaN where B0 is defined (e.g. NaN posteriors) would score a
        # different population than B0's; tune() raises instead of mixing the two.
        sessions, rows, d = self._five_sessions_with_edge_points()
        k_turn = int(round(rows["s1"][-1]["t"] / HOP))              # s1's last DS-DL turn

        def gated(s, alpha):
            g = d[s].copy()
            if s == "s1":
                g[k_turn] = np.nan
            return g

        with pytest.raises(RuntimeError, match="scored population changed"):
            T.tune(rows, d, gated, np.random.default_rng(0))

    def test_degenerate_tau_raises(self):
        rows = {"rec": _session_rows(0)}
        d = {"rec": np.full(400, np.nan, np.float32)}       # no finite peaks -> dev tau -inf
        with pytest.raises(T.InsufficientData):
            T.tune(rows, d, lambda s, a: d[s], np.random.default_rng(0))

    def test_alpha_zero_always_feasible_and_chosen_when_it_minimises_far(self):
        """Five sessions. The ungated B0 fires FEW SS-DL points (low FAR), and every alpha > 0 only lowers more
        -- but here we rig the gated curves to fire MORE (higher FAR) so B0 (alpha 0) wins on the minimise-FAR
        objective, proving alpha 0 is feasible and selectable."""
        sessions = [f"s{k}" for k in range(5)]
        rows = {s: _session_rows(k) for k, s in enumerate(sessions)}
        # grid long enough to cover the latest point time (36 points * 2 s + 5 s offset ~= 77 s -> k ~= 770)
        n = 900
        # B0 curve: no peaks -> FAR 0 everywhere, but we need tau finite, so give controls finite peaks below the
        # ssdl. Put a modest peak under every control so 10% calibration yields a finite tau.
        d = {}
        for s in sessions:
            c = np.full(n, 0.0, np.float32)
            c[: T.LAG_K] = np.nan
            for r in rows[s]:
                if r["kind"] == "ctrl":
                    c[int(round(r["t"] / HOP))] = 0.2        # finite control peaks -> finite tau
            d[s] = c

        # gated curves (alpha > 0): raise EVERY ssdl point onto a tall peak, so FAR is HIGHER than B0.
        def gated(s, alpha):
            g = d[s].copy()
            for r in rows[s]:
                if r["kind"] == "ssdl":
                    g[int(round(r["t"] / HOP))] = 5.0        # fires above any tau -> high FAR
            return g

        out = T.tune(rows, d, gated, np.random.default_rng(0))
        assert out["chosen"]["alpha"] == 0.0                 # B0 minimises FAR and is always feasible
        assert np.isfinite(out["dev_tau"])
        # every cell carries a feasibility verdict and alpha-0/tau0-offset-0 is feasible by construction.
        # tau0 is stored rounded to 6 dp, so pick the alpha-0 cell whose tau0 is closest to the dev tau.
        alpha0 = [c for c in out["cells"] if c.alpha == 0.0]
        b0 = min(alpha0, key=lambda c: abs(c.tau0 - out["dev_tau"]))
        assert b0.feasible is True
        assert b0.delta_mr_upper_bound == pytest.approx(0.0, abs=1e-9)


# ======================================================================================================= #
# eval-split guard
# ======================================================================================================= #
class TestEvalGuard:
    def test_read_reader_rows_refuses_eval(self, tmp_path):
        with pytest.raises(ValueError, match="DEV split only"):
            T.read_reader_rows(tmp_path, "eval")

    def test_run_p0_tuning_refuses_non_dev_summary(self, tmp_path):
        units = tmp_path / "units"
        units.mkdir()
        (units / "summary.json").write_text(json.dumps({"split": "eval", "recordings": []}))
        inp = T.TuningInputs(units_dir=units, cache_dir=tmp_path, wav_dir=tmp_path, out_path=tmp_path / "o.json")
        with pytest.raises(ValueError, match="not the dev split"):
            T.run_p0_tuning(inp, write=False)


# ======================================================================================================= #
# run_p0_tuning records not-assessable through fakes (no model, no audio)
# ======================================================================================================= #
class TestRunP0Tuning:
    def _write_units(self, units, recordings, dsdl_counts, langs):
        import csv
        import gzip
        units.mkdir(parents=True, exist_ok=True)
        # minimal reader CSVs (one row each is enough; the not-assessable path never scores)
        for kind in ("ssdl", "turns", "ctrl"):
            with gzip.open(units / f"{kind}_dev.csv.gz", "wt", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["recording", "kind", "t", "pause_bin", "left_s", "right_s"])
                w.writeheader()
                for rid in recordings:
                    k = {"ssdl": "ssdl", "turns": "dsdl", "ctrl": "ctrl"}[kind]
                    w.writerow({"recording": rid, "kind": k, "t": 5.0, "pause_bin": "none",
                                "left_s": 1.0, "right_s": 1.0})
        summary = {"split": "dev", "recordings": [
            {"recording": rid, "languages": langs[rid], "n_dsdl": dsdl_counts[rid]} for rid in recordings]}
        (units / "summary.json").write_text(json.dumps(summary))

    def test_not_assessable_recorded_not_raised(self, tmp_path):
        units = tmp_path / "units"
        recs = ["r1", "r2", "r3"]
        self._write_units(units, recs, {"r1": 15, "r2": 15, "r3": 15},
                          {r: ["eng", "hin"] for r in recs})      # 3 two-lang sessions -> < 5, not tunable
        inp = T.TuningInputs(units_dir=units, cache_dir=tmp_path, wav_dir=tmp_path, out_path=tmp_path / "o.json",
                             label_map={"eng": "en", "hin": "hi"})
        # fake LID loader so two-language resolution works with no cache on disk (model-free)
        lid = lambda rid: {"post": np.zeros((1, 2)), "labels": ["en", "hi"]}
        rec = T.run_p0_tuning(inp, write=True, lid_loader=lid)
        assert rec["m2_assessable"]["tunable"] is False
        assert rec["alpha"] is None and "not tested" in rec["d1_stage2"]
        assert "not_assessable" in rec["flags"]
        # the three recordings resolved to a two-language pair from the label map (M2)
        assert set(rec["counts"]["pair_by_recording"]) == {"r1", "r2", "r3"}
        assert rec["counts"]["n_excluded"] == 0
        # the output JSON was written and carries the code hash + input hashes
        out = json.loads((tmp_path / "o.json").read_text())
        assert out["code_sha256"] and out["input_sha256"]
        assert out["label_map"] == {"eng": "en", "hin": "hi"}

    def test_m2_unresolved_recording_excluded_in_run(self, tmp_path):
        """M2 end-to-end: a recording whose LD label is unmapped is excluded, flagged and counted in the record,
        not aborted."""
        units = tmp_path / "units"
        recs = ["r1", "r2"]
        self._write_units(units, recs, {"r1": 15, "r2": 15},
                          {"r1": ["eng", "hin"], "r2": ["eng", "zzz"]})   # r2 'zzz' unmapped
        inp = T.TuningInputs(units_dir=units, cache_dir=tmp_path, wav_dir=tmp_path, out_path=tmp_path / "o.json",
                             label_map={"eng": "en", "hin": "hi"})
        lid = lambda rid: {"post": np.zeros((1, 2)), "labels": ["en", "hi"]}
        rec = T.run_p0_tuning(inp, write=False, lid_loader=lid)
        assert "r2" in rec["excluded_recordings"]
        assert set(rec["counts"]["pair_by_recording"]) == {"r1"}
        assert "excluded_recordings" in rec["flags"]

    def test_run_requires_label_map(self, tmp_path):
        units = tmp_path / "units"
        self._write_units(units, ["r1"], {"r1": 15}, {"r1": ["eng", "hin"]})
        inp = T.TuningInputs(units_dir=units, cache_dir=tmp_path, wav_dir=tmp_path, out_path=tmp_path / "o.json")
        with pytest.raises(ValueError, match="label-map"):
            T.run_p0_tuning(inp, write=False)

    def test_scored_counts_recorded_when_scorable_turns_fall_short(self, tmp_path):
        # review round 4, m2: the reader counts say tunable (5 x 10 DS-DL turns) but r0's first turn has no
        # defined d(t), so only 4 sessions keep 10 scorable turns. The record must carry the scored per-session
        # counts and the difference flag as structured fields, not only inside the not-assessable message.
        import csv
        import gzip
        units = tmp_path / "units"
        units.mkdir(parents=True)
        recs = [f"r{k}" for k in range(5)]
        rows = {rid: _session_rows(k, n_dsdl=10) for k, rid in enumerate(recs)}
        rows["r0"][-10]["t"] = 0.6                                  # r0's first DS-DL turn: k = 6 < LAG_K
        for kind, name in (("ssdl", "ssdl_dev.csv.gz"), ("dsdl", "turns_dev.csv.gz"), ("ctrl", "ctrl_dev.csv.gz")):
            with gzip.open(units / name, "wt", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["recording", "kind", "t", "pause_bin", "left_s", "right_s"])
                w.writeheader()
                for rid in recs:
                    for r in rows[rid]:
                        if r["kind"] == kind:
                            w.writerow({"recording": rid, **r})
        summary = {"split": "dev", "recordings": [
            {"recording": rid, "languages": ["eng", "hin"], "n_dsdl": 10} for rid in recs]}
        (units / "summary.json").write_text(json.dumps(summary))
        n = 900
        grids = {rid: _random_grid(n, seed=10 + k) for k, rid in enumerate(recs)}
        posts = {rid: _random_posteriors(n, n_langs=2, seed=20 + k) for k, rid in enumerate(recs)}
        inp = T.TuningInputs(units_dir=units, cache_dir=tmp_path, wav_dir=tmp_path, out_path=tmp_path / "o.json",
                             label_map={"eng": "en", "hin": "hi"})
        rec = T.run_p0_tuning(inp, write=True, grid_loader=lambda rid: grids[rid],
                              lid_loader=lambda rid: {"post": posts[rid], "labels": ["en", "hi"]})
        assert rec["m2_assessable"]["tunable"] is True and rec["m2_assessable"]["n_qualifying"] == 5
        assert rec["alpha"] is None and "scorable DS-DL" in rec["flags"]["not_assessable"]
        assert rec["scored_dsdl_per_session"] == {"r0": 9, "r1": 10, "r2": 10, "r3": 10, "r4": 10}
        assert rec["n_scored_two_lang_sessions_ge10_dsdl"] == 4
        assert "scored_vs_summary_sessions" in rec["flags"]
        out = json.loads((tmp_path / "o.json").read_text())
        assert out["scored_dsdl_per_session"]["r0"] == 9 and out["n_scored_two_lang_sessions_ge10_dsdl"] == 4

    def test_caller_pair_is_refused(self, tmp_path):
        # review round 2, M1: a recording's pair comes only from the label map; a caller pair (here a language
        # outside the recording's reference content) would re-renormalise lambda and is refused outright.
        units = tmp_path / "units"
        self._write_units(units, ["r1"], {"r1": 15}, {"r1": ["eng", "hin"]})
        inp = T.TuningInputs(units_dir=units, cache_dir=tmp_path, wav_dir=tmp_path, out_path=tmp_path / "o.json",
                             label_map={"eng": "en", "hin": "hi"}, pair_by_recording={"r1": ("en", "bn")})
        lid = lambda rid: {"post": np.zeros((1, 3)), "labels": ["en", "hi", "bn"]}
        with pytest.raises(ValueError, match="label map"):
            T.run_p0_tuning(inp, write=False, lid_loader=lid)

    def test_cli_has_no_pair_override(self, tmp_path):
        import contextlib
        import io
        err = io.StringIO()
        with pytest.raises(SystemExit), contextlib.redirect_stderr(err):
            T.main(["tune", "--units", str(tmp_path), "--cache-dir", str(tmp_path), "--wav-dir", str(tmp_path),
                    "--out", str(tmp_path / "o.json"), "--pair", "r1=en,bn"])
        assert "unrecognized arguments: --pair" in err.getvalue()

    def test_excluded_recordings_lid_caches_are_hashed(self, tmp_path):
        # review round 2, m2: the two-language decision reads every summary recording's LID labels, so every
        # consulted LID cache is in input_sha256, the excluded recording's included.
        from csdiar.pipelines import run_m2b as R
        units = tmp_path / "units"
        recs = ["r1", "r2"]
        self._write_units(units, recs, {"r1": 15, "r2": 15},
                          {"r1": ["eng", "hin"], "r2": ["eng", "zzz"]})   # r2 excluded (unmapped label)
        for rid in recs:
            p = Path(R.lid_cache_path(tmp_path, rid))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(f"lid cache for {rid}".encode())
        inp = T.TuningInputs(units_dir=units, cache_dir=tmp_path, wav_dir=tmp_path, out_path=tmp_path / "o.json",
                             label_map={"eng": "en", "hin": "hi"})
        lid = lambda rid: {"post": np.zeros((1, 2)), "labels": ["en", "hi"]}
        rec = T.run_p0_tuning(inp, write=False, lid_loader=lid)
        assert "r2" in rec["excluded_recordings"]
        for rid in recs:
            assert str(Path(R.lid_cache_path(tmp_path, rid))) in rec["input_sha256"]


# ======================================================================================================= #
# m4 (review v1): load_dense_grid checks code_hash and model on the READ path (model-free; writes a cache npz)
# ======================================================================================================= #
class TestLoadDenseGridValidation:
    def _write_cache(self, tmp_path, path_model, stored_model, code_hash, n=20, dim=8):
        """Write a dense-grid npz at the `path_model` cache path, stamping `stored_model` / `code_hash` inside."""
        path = T.dense_grid_cache_path(tmp_path, path_model, "rid")
        path.parent.mkdir(parents=True, exist_ok=True)
        grid = np.zeros((n, dim), np.float16)
        np.savez_compressed(path, version=np.int64(T.CACHE_VERSION), rid="rid", model=stored_model,
                            code_hash=code_hash, provenance="{}", grid=grid,
                            hop=np.float64(T.HOP), win=np.float64(T.WIN), padded=np.int64(0))
        return path

    def test_correct_code_hash_loads(self, tmp_path):
        self._write_cache(tmp_path, "e3", "e3", T._code_hash())
        grid, meta = T.load_dense_grid(tmp_path, "e3", "rid")
        assert grid.shape == (20, 8) and meta["model"] == "e3"

    def test_code_hash_mismatch_raises(self, tmp_path):
        self._write_cache(tmp_path, "e3", "e3", "deadbeef" * 8)   # not the current _code_hash()
        with pytest.raises(ValueError, match="code_hash"):
            T.load_dense_grid(tmp_path, "e3", "rid")

    def test_model_mismatch_raises(self, tmp_path):
        # npz sits at the e3 cache path but stamps model='e2' internally -> load for 'e3' must refuse
        self._write_cache(tmp_path, "e3", "e2", T._code_hash())
        with pytest.raises(ValueError, match="model"):
            T.load_dense_grid(tmp_path, "e3", "rid")

    def test_version_mismatch_raises(self, tmp_path):
        path = T.dense_grid_cache_path(tmp_path, "e3", "rid")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, version=np.int64(T.CACHE_VERSION + 1), rid="rid", model="e3",
                            code_hash=T._code_hash(), provenance="{}", grid=np.zeros((5, 8), np.float16),
                            hop=np.float64(T.HOP), win=np.float64(T.WIN), padded=np.int64(0))
        with pytest.raises(ValueError, match="cache version"):
            T.load_dense_grid(tmp_path, "e3", "rid")


# ======================================================================================================= #
# m5 (review v1): the stored bound records BOTH levels (one_sided 0.80 and bca_two_sided 0.60)
# ======================================================================================================= #
class TestBoundLevelsRecorded:
    def test_both_levels_in_bca(self):
        ub = T.mr_upper_bound({f"s{k}": 0.0 for k in range(8)}, np.random.default_rng(0))
        assert ub["bca"]["one_sided_level"] == pytest.approx(0.80)
        assert ub["bca"]["bca_two_sided_level"] == pytest.approx(0.60)

    def test_both_levels_when_too_few_sessions(self):
        ub = T.mr_upper_bound({"s0": 0.01}, np.random.default_rng(0))
        assert ub["bca"]["one_sided_level"] == pytest.approx(0.80)
        assert ub["bca"]["bca_two_sided_level"] == pytest.approx(0.60)


# ======================================================================================================= #
# OPTIONAL CPU smoke test of the dense grid (few seconds of pyannote's bundled sample.wav, niced, skipped if
# models are absent). It never uses the GPU (device="cpu") and reads no DISPLACE data.
# ======================================================================================================= #
def _sample_wav() -> Path | None:
    try:
        import pyannote.audio
    except Exception:
        return None
    base = Path(pyannote.audio.__file__).resolve().parent
    for cand in base.rglob("sample.wav"):
        return cand
    return None


@pytest.mark.filterwarnings("ignore")
def test_dense_grid_cpu_smoke(tmp_path):
    import os
    try:
        import soundfile  # noqa: F401
        import torch       # noqa: F401
        from csdiar.models.e3 import CKPT
    except Exception:
        pytest.skip("torch / soundfile / e3 not available")
    wav = _sample_wav()
    if wav is None:
        pytest.skip("pyannote sample.wav not found")
    if not CKPT.exists():
        pytest.skip("E3 checkpoint not present on this host")
    os.nice(10)
    # trim to a few seconds so the smoke test stays fast and CPU-only
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(str(wav), dtype="float32")
    if data.ndim > 1:
        data = data.mean(1)
    short = tmp_path / "short.wav"
    sf.write(str(short), data[: int(3 * sr)], sr)
    try:
        meta = T.build_dense_grid(short, tmp_path, rid="smoke", model="e3", device="cpu")
    except (ModuleNotFoundError, ImportError, OSError, RuntimeError) as e:
        pytest.skip(f"E3 model cannot be loaded in this venv (expected; E3 lives in .venv-e3): {e}")
    assert meta["n_windows"] > 0
    grid, _ = T.load_dense_grid(tmp_path, "e3", "smoke")
    d = np.asarray(T.d_curve(grid))
    assert np.isnan(d[: T.LAG_K]).all()
    assert np.isfinite(d[T.LAG_K:]).any()
