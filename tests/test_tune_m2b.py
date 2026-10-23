#!/usr/bin/env python3
"""Unit tests for csdiar.analysis.tune_m2b (PREREGISTRATION_M2B.md §3, §4, §5, §7).

These cover, WITHOUT loading any model, reading any audio, or touching the GPU box:
  * the three dev fits (fit_b1_means, fit_m1_v_hat, fit_b2_delta) on tiny synthetic embeddings with known
    answers -- the B2 clamp at BOTH ends, the M1 sign convention, and the too-little-data refusals;
  * the POOLED (corpus) DER helper, including the reviewer's two-recording trace where pooled DER picks 0.35
    while a session mean picks 0.30 (major M1);
  * every selection rule with hand-made numbers -- B0 ties both ways, the mitigation match-to-B0 tie to lower
    FAR, the M2 alpha +1 pp bound boundary, alpha 0 always feasible, the alpha-0 BCa key parity (minor m1), the
    _per_session_delta_mr session intersection (minor m2), and the not-assessable cases;
  * the orchestration (run_dev_tuning) end to end through fakes -- a hand-made sweep whose chosen B0 threshold,
    matched thresholds, chosen alpha and fitted values are asserted (major M2); and that two runs differing only
    in one cache's embeddings give different records (major M3);
  * m4 (per-session DET in the curve) and m5 (non-finite duration refused);
  * a guard that nothing reads an eval split.

Run under .venv-pyannote4 (numpy present; tune_m2b's fits and BCa use it):
  .venv-pyannote4/bin/python -m pytest -q -p no:cacheprovider tests/test_tune_m2b.py
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from csdiar.analysis import tune_m2b as T
from csdiar.analysis.score_m2b import SessionScore


# ======================================================================================================= #
# FITS
# ======================================================================================================= #
class TestFitB1Means:
    def test_mean_per_language_with_counts(self):
        slots = {
            "en": [[1.0, 0.0], [3.0, 0.0]],          # mean (2, 0)
            "hi": [[0.0, 2.0], [0.0, 4.0], [0.0, 6.0]],  # mean (0, 4)
        }
        out = T.fit_b1_means(slots)
        np.testing.assert_allclose(out["means"]["en"], [2.0, 0.0])
        np.testing.assert_allclose(out["means"]["hi"], [0.0, 4.0])
        assert out["counts"] == {"en": 2, "hi": 3}
        assert out["dim"] == 2

    def test_keys_are_passed_through_unchanged_for_pair_indices(self):
        # the orchestration keys slot embeddings by PAIR INDEX (ints), not codes; fit_b1_means is agnostic.
        slots = {0: [[1.0, 0.0]], 7: [[0.0, 2.0]]}
        out = T.fit_b1_means(slots)
        assert set(out["means"]) == {0, 7}
        assert out["counts"] == {0: 1, 7: 1}

    def test_empty_language_refused(self):
        with pytest.raises(T.InsufficientData):
            T.fit_b1_means({"en": [[1.0, 0.0]], "hi": []})

    def test_dim_mismatch_refused(self):
        with pytest.raises(T.InsufficientData):
            T.fit_b1_means({"en": [[1.0, 0.0]], "hi": [[1.0, 2.0, 3.0]]})


class TestFitM1VHat:
    def test_sign_english_minus_other(self):
        # one unit, English mean (2,0,0), other (Hindi) mean (0,0,0): diff = (2,0,0) -> v_hat = (1,0,0)
        units = [{"speaker": "A", "recording": "R1",
                  "by_lang": {"en": [2.0, 0.0, 0.0], "hi": [0.0, 0.0, 0.0]}}]
        out = T.fit_m1_v_hat(units)
        np.testing.assert_allclose(out["v_hat"], [1.0, 0.0, 0.0], atol=1e-12)
        assert out["n_units"] == 1
        assert out["n_skipped_no_english"] == 0

    def test_sign_is_english_minus_other_not_reverse(self):
        # English below the other language on axis 0 -> v points NEGATIVE on axis 0 (en - other)
        units = [{"speaker": "A", "recording": "R1",
                  "by_lang": {"en": [0.0, 0.0], "bn": [5.0, 0.0]}}]
        out = T.fit_m1_v_hat(units)
        assert out["v_hat"][0] < 0  # en - other = (-5, 0) -> unit (-1, 0)
        np.testing.assert_allclose(out["v_hat"], [-1.0, 0.0], atol=1e-12)

    def test_consistent_sign_across_language_pairs(self):
        # Two units with DIFFERENT non-English languages; both oriented en - other, so they add, not cancel.
        units = [
            {"speaker": "A", "recording": "R1", "by_lang": {"en": [1.0, 0.0], "hi": [0.0, 0.0]}},
            {"speaker": "B", "recording": "R2", "by_lang": {"en": [1.0, 0.0], "bn": [0.0, 0.0]}},
        ]
        out = T.fit_m1_v_hat(units)
        np.testing.assert_allclose(out["v_hat"], [1.0, 0.0], atol=1e-12)
        assert out["n_units"] == 2

    def test_unit_without_english_is_skipped_and_counted(self):
        units = [
            {"speaker": "A", "recording": "R1", "by_lang": {"en": [2.0, 0.0], "hi": [0.0, 0.0]}},
            {"speaker": "B", "recording": "R2", "by_lang": {"hi": [1.0, 1.0], "bn": [2.0, 2.0]}},
        ]
        out = T.fit_m1_v_hat(units)
        assert out["n_units"] == 1
        assert out["n_skipped_no_english"] == 1

    def test_no_usable_unit_refused(self):
        units = [{"speaker": "A", "recording": "R1", "by_lang": {"en": [1.0, 0.0]}}]  # only one language
        with pytest.raises(T.InsufficientData):
            T.fit_m1_v_hat(units)

    def test_zero_direction_refused(self):
        # en == other for every unit -> all diffs zero -> averaged difference is the zero vector
        units = [{"speaker": "A", "recording": "R1", "by_lang": {"en": [1.0, 1.0], "hi": [1.0, 1.0]}}]
        with pytest.raises(T.InsufficientData):
            T.fit_m1_v_hat(units)


class TestFitB2Delta:
    def test_positive_shift_unclamped(self):
        out = T.fit_b2_delta(same_lang_cosines=[0.9, 0.8], cross_lang_cosines=[0.5, 0.5])
        assert out["raw"] == pytest.approx(0.85 - 0.5)
        assert out["delta"] == pytest.approx(0.35)
        assert out["clamped_low"] is False and out["clamped_high"] is False
        assert out["n_same"] == 2 and out["n_cross"] == 2

    def test_clamp_low_at_zero(self):
        # same < cross -> raw negative -> clamped to 0 (the low end)
        out = T.fit_b2_delta(same_lang_cosines=[0.2], cross_lang_cosines=[0.8])
        assert out["raw"] < 0
        assert out["delta"] == 0.0
        assert out["clamped_low"] is True

    def test_clamp_high_at_delta_max(self):
        out = T.fit_b2_delta(same_lang_cosines=[1.0], cross_lang_cosines=[0.0], delta_max=0.5)
        assert out["raw"] == pytest.approx(1.0)
        assert out["delta"] == 0.5
        assert out["clamped_high"] is True

    def test_empty_pool_refused(self):
        with pytest.raises(T.InsufficientData):
            T.fit_b2_delta([], [0.5])
        with pytest.raises(T.InsufficientData):
            T.fit_b2_delta([0.5], [])


# ======================================================================================================= #
# POOLED (corpus) DER  -- major M1
# ======================================================================================================= #
class TestPooledDer:
    def test_pooled_is_sum_error_over_sum_total(self):
        comps = {"A": {"error": 3.0, "total": 10.0}, "B": {"error": 240.0, "total": 600.0}}
        assert T.pooled_der(comps) == pytest.approx((3.0 + 240.0) / (10.0 + 600.0))

    def test_reviewer_two_recording_trace_pooled_picks_035(self):
        # The reviewer's M1 trace: two dev recordings, thresholds {0.30, 0.35}.
        # Rec A (10 s ref): DER 0.30 at .30 (error 3), 0.90 at .35 (error 9).
        # Rec B (600 s ref): DER 0.40 at .30 (error 240), 0.38 at .35 (error 228).
        comp_30 = {"A": {"error": 3.0, "total": 10.0}, "B": {"error": 240.0, "total": 600.0}}
        comp_35 = {"A": {"error": 9.0, "total": 10.0}, "B": {"error": 228.0, "total": 600.0}}
        der_30 = T.pooled_der(comp_30)      # (3+240)/610 = 0.3984
        der_35 = T.pooled_der(comp_35)      # (9+228)/610 = 0.3885
        assert der_30 == pytest.approx(243.0 / 610.0)
        assert der_35 == pytest.approx(237.0 / 610.0)
        # pooled picks .35 (lower pooled DER), where an unweighted session MEAN of ratios would pick .30
        assert der_35 < der_30
        session_mean_30 = (0.30 + 0.40) / 2      # 0.35
        session_mean_35 = (0.90 + 0.38) / 2      # 0.64
        assert session_mean_30 < session_mean_35      # the session mean would pick .30 -- the bug M1 fixes

        # and select_b0_threshold on a pooled-DER curve chooses .35
        pts = [T.ThresholdPoint(threshold=0.30, der=der_30, far_ssdl=0.2, mr_dsdl=0.2),
               T.ThresholdPoint(threshold=0.35, der=der_35, far_ssdl=0.2, mr_dsdl=0.2)]
        assert T.select_b0_threshold(pts)["threshold"] == 0.35

    def test_refuses_nonfinite_or_missing_components(self):
        with pytest.raises(ValueError):
            T.pooled_der({"A": {"error": float("nan"), "total": 10.0}, "B": {"error": 5.0, "total": 20.0}})
        with pytest.raises(ValueError):
            T.pooled_der({"B": {"error": 5.0, "total": 20.0}, "C": None})
        with pytest.raises(ValueError):
            T.pooled_der({"D": {}})
        assert T.pooled_der({"B": {"error": 5.0, "total": 20.0}}) == pytest.approx(5.0 / 20.0)
        assert T.pooled_der({}) != T.pooled_der({})   # nan when there is nothing to pool


# ======================================================================================================= #
# SELECTION RULES
# ======================================================================================================= #
def _pt(th, der=float("nan"), far=float("nan"), mr=float("nan"), per_session_mr=None):
    return T.ThresholdPoint(threshold=th, der=der, far_ssdl=far, mr_dsdl=mr,
                            per_session_mr_dsdl=per_session_mr or {})


class TestSelectB0:
    def test_lowest_der(self):
        pts = [_pt(0.30, der=0.40), _pt(0.35, der=0.25), _pt(0.40, der=0.33)]
        out = T.select_b0_threshold(pts)
        assert out["threshold"] == 0.35
        assert out["der"] == 0.25

    def test_tie_to_lower_threshold(self):
        pts = [_pt(0.30, der=0.20), _pt(0.35, der=0.20), _pt(0.40, der=0.50)]
        out = T.select_b0_threshold(pts, tie="lower_threshold")
        assert out["threshold"] == 0.30
        assert out["ties"] == [0.30, 0.35]

    def test_tie_to_higher_threshold(self):
        pts = [_pt(0.30, der=0.20), _pt(0.35, der=0.20)]
        out = T.select_b0_threshold(pts, tie="higher_threshold")
        assert out["threshold"] == 0.35

    def test_degenerate_refused(self):
        pts = [_pt(0.30), _pt(0.35)]  # all nan DER
        assert T.degenerate_operating_point(pts) is True
        with pytest.raises(T.InsufficientData):
            T.select_b0_threshold(pts)


class TestSelectMatched:
    def test_closest_mr_to_b0(self):
        pts = [_pt(0.30, far=0.20, mr=0.10), _pt(0.35, far=0.18, mr=0.22), _pt(0.40, far=0.15, mr=0.31)]
        out = T.select_matched_threshold(pts, b0_mr_dsdl=0.20)   # closest to 0.20 is 0.22 at 0.35
        assert out["threshold"] == 0.35

    def test_tie_on_mr_distance_goes_to_lower_far(self):
        # two thresholds equidistant from B0's MR (0.20): 0.15 and 0.25, both |d| = 0.05. Lower FAR wins.
        pts = [_pt(0.30, far=0.30, mr=0.15), _pt(0.35, far=0.10, mr=0.25)]
        out = T.select_matched_threshold(pts, b0_mr_dsdl=0.20)
        assert out["threshold"] == 0.35      # same |MR - 0.20|, lower FAR (0.10)
        assert out["far_ssdl"] == 0.10

    def test_degenerate_refused(self):
        with pytest.raises(T.InsufficientData):
            T.select_matched_threshold([_pt(0.30)], b0_mr_dsdl=0.2)


class TestAssessM2:
    def test_tunable_when_enough_sessions(self):
        turns = {f"s{i}": 12 for i in range(5)}      # 5 sessions, each >= 10 DS-DL turns
        out = T.assess_m2_tunable(turns)
        assert out["tunable"] is True
        assert out["n_qualifying"] == 5

    def test_not_tunable_too_few_sessions(self):
        turns = {"s0": 50, "s1": 40, "s2": 30, "s3": 20}  # only 4 qualify
        out = T.assess_m2_tunable(turns)
        assert out["tunable"] is False
        assert out["n_qualifying"] == 4

    def test_not_tunable_sessions_below_turn_minimum(self):
        turns = {f"s{i}": 9 for i in range(10)}      # 10 sessions but each < 10 turns
        out = T.assess_m2_tunable(turns)
        assert out["tunable"] is False
        assert out["n_qualifying"] == 0

    def test_boundary_exactly_min(self):
        turns = {f"s{i}": 10 for i in range(5)}      # exactly 5 sessions, exactly 10 turns each
        assert T.assess_m2_tunable(turns)["tunable"] is True


class TestSelectM2Alpha:
    def _alpha_curve(self, alpha, far, per_session_mr):
        pt = T.ThresholdPoint(threshold=0.50, der=0.3, far_ssdl=far, mr_dsdl=float("nan"),
                              per_session_mr_dsdl=per_session_mr)
        return T.SystemCurve(system="M2", points=[pt], param_tag=f"a{alpha}")

    def test_alpha_zero_always_feasible_even_if_only_option(self):
        # B0 per-session MR, and alpha 0 == B0 (deltas 0). With no feasible alpha>0, alpha 0 still wins.
        b0 = T.ThresholdPoint(threshold=0.50, der=0.3, far_ssdl=0.30, mr_dsdl=0.20,
                              per_session_mr_dsdl={f"s{i}": 0.20 for i in range(6)})
        curves = {
            0.0: self._alpha_curve(0.0, far=0.30, per_session_mr={f"s{i}": 0.20 for i in range(6)}),
            # alpha 0.5: lowers FAR but raises MR hugely (+0.5 each) -> bound way over +1pp -> infeasible
            0.5: self._alpha_curve(0.5, far=0.05, per_session_mr={f"s{i}": 0.70 for i in range(6)}),
        }
        rng = np.random.default_rng(T.SEED)
        out = T.select_m2_alpha(curves, b0, rng)
        assert out["alpha"] == 0.0
        assert 0.0 in out["feasible"]
        assert out["per_alpha"][0.5]["feasible"] is False

    def test_alpha_zero_bca_stub_has_same_keys_as_bca_ci(self):
        # minor m1: the alpha-0 row's bound carries the SAME keys bca_ci returns, so reading bca["a"]/["z0"]
        # on it never KeyErrors.
        b0 = T.ThresholdPoint(threshold=0.50, der=0.3, far_ssdl=0.30, mr_dsdl=0.20,
                              per_session_mr_dsdl={f"s{i}": 0.20 for i in range(6)})
        curves = {0.0: self._alpha_curve(0.0, 0.30, {f"s{i}": 0.20 for i in range(6)})}
        out = T.select_m2_alpha(curves, b0, np.random.default_rng(T.SEED))
        bca0 = out["per_alpha"][0.0]["bca"]
        assert set(bca0) == {"estimate", "lo", "hi", "level", "z0", "a", "n_boot", "n_dropped"}

    def test_picks_lowest_far_among_feasible(self):
        b0 = T.ThresholdPoint(threshold=0.50, der=0.3, far_ssdl=0.30, mr_dsdl=0.20,
                              per_session_mr_dsdl={f"s{i}": 0.20 for i in range(6)})
        # alpha 0.3: FAR 0.20, deltaMR tiny (+0.002 each) -> feasible; alpha 0.6: FAR 0.10, deltaMR tiny -> feasible
        curves = {
            0.0: self._alpha_curve(0.0, far=0.30, per_session_mr={f"s{i}": 0.20 for i in range(6)}),
            0.3: self._alpha_curve(0.3, far=0.20, per_session_mr={f"s{i}": 0.202 for i in range(6)}),
            0.6: self._alpha_curve(0.6, far=0.10, per_session_mr={f"s{i}": 0.201 for i in range(6)}),
        }
        rng = np.random.default_rng(T.SEED)
        out = T.select_m2_alpha(curves, b0, rng)
        assert out["alpha"] == 0.6                   # lowest FAR among the feasible
        assert set(out["feasible"]) == {0.0, 0.3, 0.6}

    def test_bound_boundary_excludes_just_over(self):
        # Construct deltaMR so the one-sided 80% UPPER BCa bound sits just OVER +1pp -> infeasible.
        b0 = T.ThresholdPoint(threshold=0.50, der=0.3, far_ssdl=0.30, mr_dsdl=0.20,
                              per_session_mr_dsdl={f"s{i}": 0.20 for i in range(8)})
        # every session deltaMR = +0.03 (constant) -> the mean is 0.03 and the BCa bound collapses to 0.03 > 0.01
        over = self._alpha_curve(0.4, far=0.05, per_session_mr={f"s{i}": 0.23 for i in range(8)})
        curves = {0.0: self._alpha_curve(0.0, 0.30, {f"s{i}": 0.20 for i in range(8)}), 0.4: over}
        rng = np.random.default_rng(T.SEED)
        out = T.select_m2_alpha(curves, b0, rng)
        assert out["per_alpha"][0.4]["feasible"] is False
        assert out["alpha"] == 0.0                   # only alpha 0 remains feasible

    def test_bound_boundary_includes_just_under(self):
        b0 = T.ThresholdPoint(threshold=0.50, der=0.3, far_ssdl=0.30, mr_dsdl=0.20,
                              per_session_mr_dsdl={f"s{i}": 0.20 for i in range(8)})
        # constant deltaMR = +0.005 < +1pp everywhere -> bound 0.005 <= 0.01 -> feasible, and lower FAR than 0
        under = self._alpha_curve(0.4, far=0.05, per_session_mr={f"s{i}": 0.205 for i in range(8)})
        curves = {0.0: self._alpha_curve(0.0, 0.30, {f"s{i}": 0.20 for i in range(8)}), 0.4: under}
        rng = np.random.default_rng(T.SEED)
        out = T.select_m2_alpha(curves, b0, rng)
        assert out["per_alpha"][0.4]["feasible"] is True
        assert out["alpha"] == 0.4

    def test_per_session_delta_mr_uses_session_intersection(self):
        # minor m2: the alpha point includes a session (s9) B0 excluded, and B0 includes s0..s5 of which the
        # alpha point drops s5. Only the intersection {s0..s4} contributes a delta.
        b0 = T.ThresholdPoint(threshold=0.50, der=0.3, far_ssdl=0.30, mr_dsdl=0.20,
                              per_session_mr_dsdl={f"s{i}": 0.20 for i in range(6)})   # s0..s5
        alpha_pt_sessions = {f"s{i}": 0.25 for i in range(5)}    # s0..s4
        alpha_pt_sessions["s9"] = 0.25                           # extra session B0 never saw
        curve = self._alpha_curve(0.4, far=0.05, per_session_mr=alpha_pt_sessions)
        deltas = T._per_session_delta_mr(curve, b0)
        assert set(deltas) == {"s0", "s1", "s2", "s3", "s4"}      # intersection only
        assert all(abs(v - 0.05) < 1e-12 for v in deltas.values())


# ======================================================================================================= #
# dev_counts / assessability from a summary.json
# ======================================================================================================= #
class TestDevCounts:
    def test_counts_two_language_recordings(self):
        summary = {"recordings": [
            {"recording": "R1", "languages": ["en", "hi"], "n_dsdl": 15},
            {"recording": "R2", "languages": ["en", "hi", "bn"], "n_dsdl": 3},
            {"recording": "R3", "languages": ["en", "te"], "n_dsdl": 12},
        ]}
        out = T.dev_counts(summary)
        assert out["n_two_language_recordings"] == 2
        assert set(out["two_language_recordings"]) == {"R1", "R3"}
        assert out["dsdl_per_session"] == {"R1": 15, "R2": 3, "R3": 12}
        assert out["n_sessions_ge10_dsdl"] == 2


# ======================================================================================================= #
# EVAL GUARD
# ======================================================================================================= #
class TestEvalGuard:
    def test_read_reader_rows_refuses_eval(self, tmp_path):
        with pytest.raises(ValueError, match="DEV split only"):
            T.read_reader_rows(tmp_path, "eval")

    def test_read_reader_rows_refuses_arbitrary_split(self, tmp_path):
        with pytest.raises(ValueError):
            T.read_reader_rows(tmp_path, "test")

    def test_plan_refuses_non_dev_summary(self, tmp_path):
        (tmp_path / "summary.json").write_text(json.dumps({"split": "eval", "recordings": []}))
        inp = T.TuningInputs(units_dir=tmp_path, cache_dir=tmp_path, wav_dir=tmp_path,
                             out_path=tmp_path / "out.json")
        with pytest.raises(SystemExit, match="expected 'dev'"):
            T.plan(inp)


# ======================================================================================================= #
# CURVE building from score_m2b.score_system output + DER components  (incl. m4: per-session DET)
# ======================================================================================================= #
def _session_score(sess, far, mr, n_ssdl_cc=20, n_dsdl=20, dur=600.0):
    return SessionScore(
        session=sess, n_ssdl=n_ssdl_cc, n_ctrl=50, n_phrase_ctrl=5, n_dsdl=n_dsdl, n_dssl=0,
        far_ssdl=far, far_ctrl=0.05, far_phrase_ctrl=0.05, delta_far_std=far - 0.05,
        n_ssdl_with_cell_controls=n_ssdl_cc, n_ssdl_without_cell_controls=0,
        mr_dsdl=mr, mr_dssl=float("nan"), changes_per_min=10.0, duration_s=dur)


class TestCurveFromScores:
    def test_session_mean_far_mr_and_pooled_der(self):
        # two thresholds, two sessions; one session excluded for MR at th 0.35 (too few DS-DL)
        def score(th):
            s1 = _session_score("R1", far=0.10 + th, mr=0.20)
            s2 = _session_score("R2", far=0.20, mr=0.30, n_dsdl=(20 if th == 0.30 else 2))
            included_mr = ["R1", "R2"] if th == 0.30 else ["R1"]
            return {"sessions": {"R1": s1, "R2": s2},
                    "inclusion": {"ssdl": {"included": ["R1", "R2"], "excluded": {}},
                                  "dsdl": {"included": included_mr, "excluded": {}}}}
        threshold_scores = {0.30: score(0.30), 0.35: score(0.35)}
        # DER components per threshold (pooled): th .30 -> (20+40)/(100+200)=0.2 ; th .35 -> (30+30)/(100+200)=0.2
        der_comp = {0.30: {"R1": {"error": 20.0, "total": 100.0}, "R2": {"error": 40.0, "total": 200.0}},
                    0.35: {"R1": {"error": 30.0, "total": 100.0}, "R2": {"error": 30.0, "total": 200.0}}}
        curve = T.curve_from_scores("B0", threshold_scores, der_comp, aggregation="session_mean")
        p30 = next(p for p in curve.points if p.threshold == 0.30)
        p35 = next(p for p in curve.points if p.threshold == 0.35)
        # th 0.30: MR mean over R1,R2 = (0.20+0.30)/2 = 0.25; th 0.35: only R1 included -> 0.20
        assert p30.mr_dsdl == pytest.approx(0.25)
        assert p35.mr_dsdl == pytest.approx(0.20)
        assert p30.n_sessions_mr == 2 and p35.n_sessions_mr == 1
        # pooled DER (Sum error / Sum total), NOT a session mean of ratios
        assert p30.der == pytest.approx(60.0 / 300.0)
        assert p35.der == pytest.approx(60.0 / 300.0)

    def test_per_session_det_is_built(self):
        # m4: the curve's det summary carries the by_session reading (not only the pooled one).
        def score(th):
            s1 = _session_score("R1", far=0.10, mr=0.20)
            s2 = _session_score("R2", far=0.12, mr=0.18)
            return {"sessions": {"R1": s1, "R2": s2},
                    "inclusion": {"ssdl": {"included": ["R1", "R2"], "excluded": {}},
                                  "dsdl": {"included": ["R1", "R2"], "excluded": {}}}}
        ts = {th: score(th) for th in (0.30, 0.35, 0.40)}
        dc = {th: {"R1": {"error": 10.0, "total": 100.0}, "R2": {"error": 12.0, "total": 100.0}}
              for th in (0.30, 0.35, 0.40)}
        curve = T.curve_from_scores("B0", ts, dc)
        assert "by_session" in curve.det                     # per-session DET present (m4)
        assert "pooled" in curve.det


# ======================================================================================================= #
# ORCHESTRATION with monkeypatched runner + scorers + fit builders  (major M2, M3; minor m5)
# ======================================================================================================= #
def _write_dev_units(tmp_path, recordings, durations=None):
    """Write minimal dev ssdl/turns/ctrl csv.gz + summary.json so read_reader_rows + run_dev_tuning run."""
    from csdiar.data.ingest_displace import FIELDS
    units = tmp_path / "units"
    units.mkdir()
    for kind in ("ssdl", "turns", "ctrl"):
        with gzip.open(units / f"{kind}_dev.csv.gz", "wt", encoding="utf-8", newline="") as fh:
            import csv as _csv
            w = _csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            for rec in recordings:
                base = {"recording": rec, "split": "dev", "speaker": "A", "speaker2": "",
                        "lang_left": "en", "lang_right": "hi", "left_s": 1.0, "right_s": 1.0,
                        "stratum": "ge1", "elig_p0L1": True, "pause_ms": 0.0, "pause_bin": "none",
                        "phrase": False}
                if kind == "ssdl":
                    w.writerow({**base, "kind": "ssdl", "t": 5.0})
                elif kind == "turns":
                    w.writerow({**base, "kind": "dsdl", "t": 7.0})
                else:
                    w.writerow({**base, "kind": "ctrl", "t": 9.0, "lang_right": "en"})
    recs = []
    for r in recordings:
        rec = {"recording": r, "languages": ["en", "hi"], "n_dsdl": 12}
        if durations is not None:
            rec["duration_s"] = durations.get(r, 600.0)
        else:
            rec["duration_s"] = 600.0
        recs.append(rec)
    summary = {"split": "dev", "recordings": recs}
    (units / "summary.json").write_text(json.dumps(summary))
    return units


def _write_ref_rttms(tmp_path, recordings):
    d = tmp_path / "refs"
    d.mkdir()
    for rid in recordings:
        (d / f"{rid}_SPEAKER.rttm").write_text(
            f"SPEAKER {rid} 1 0.000 300.000 <NA> <NA> A <NA> <NA>\n")
    return d


class _FakeAnn:
    """A tiny stand-in for a pyannote Annotation that segments_from_annotation can flatten."""
    def __init__(self, uri, segs):
        self.uri = uri
        self._segs = segs

    def itertracks(self, yield_label=True):
        for i, (s, e, l) in enumerate(self._segs):
            yield type("Seg", (), {"start": s, "end": e})(), i, l


def _orchestration_fakes(recordings, b0_best_th=0.40, b0_der_by_th=None):
    """Build the full set of fakes for run_dev_tuning: a runner that encodes a hand-made sweep, a scorer that
    reads a planted FAR/MR per (system, threshold, recording), a DER-components fn keyed by threshold, cache/lid
    loaders, embedder and fit builders. Returns a dict of kwargs for run_dev_tuning plus the planted truth.
    """
    import numpy as np

    # planted per-threshold pooled DER components for B0 so pooled DER is lowest at b0_best_th.
    if b0_der_by_th is None:
        b0_der_by_th = {th: 0.5 for th in T.THRESHOLD_GRID}
        b0_der_by_th[b0_best_th] = 0.1

    # planted per-(system, threshold) FAR/MR. MR engineered so B0's MR at its best threshold = 0.20, and each
    # mitigation's MR crosses 0.20 at a known threshold.
    def far_mr(system, th):
        if system == "B0":
            return 0.30, (0.20 if th == b0_best_th else 0.40)
        if system == "B1":
            # B1 MR hits 0.20 exactly at 0.50
            return 0.25, abs(th - 0.50) * 2 + 0.20
        if system == "M1":
            return 0.22, abs(th - 0.45) * 2 + 0.20
        if system.startswith("M3"):
            return 0.18, abs(th - 0.55) * 2 + 0.20
        if system == "M2":
            # at B0's threshold, alpha is encoded into FAR/MR via the cache-less path below (handled in scorer)
            return 0.30, 0.20
        return 0.30, 0.30

    def fake_runner(pipe, cache, threshold, system, **kw):
        uri = cache["rid"]
        # encode system+threshold (+alpha for M2) into the returned annotation label so the scorer can read it
        tag = f"{system}|{threshold:.2f}"
        if system == "M2":
            tag += f"|a{kw.get('alpha', 0.0):.2f}"
        segs = [(0.0, 1.0, tag)]
        return _FakeAnn(uri, segs), _FakeAnn(uri, segs), {"system": system, "threshold": threshold}

    def fake_scorer(rows_by_session, changes_by_recording, durations, **kw):
        # recover (system, threshold, alpha) from the planted change list: hypothesis_changes of a single
        # 1-segment annotation is empty, so we instead stash the tag in a module-level via changes length trick.
        # Simpler: read it back from the recording's single ssdl tag we encoded. We cannot see labels here, so
        # the runner wrote the tag; we re-derive FAR/MR from a side channel the runner set on the scorer.
        sysname, th, alpha = fake_scorer._ctx
        sessions = {}
        for s in rows_by_session:
            if sysname == "M2":
                # M2 at B0's threshold: FAR decreases with alpha, MR rises slightly; alpha 0.20 is the lowest
                # FAR that still meets the +1pp bound in this plant.
                far = 0.30 - 0.5 * alpha
                mr = 0.20 + (0.002 if alpha <= 0.20 else 0.5) * (1 if alpha > 0 else 0)
            else:
                far, mr = far_mr(sysname, th)
            sessions[s] = _session_score(s, far=far, mr=mr)
        return {"sessions": sessions,
                "inclusion": {"ssdl": {"included": sorted(sessions), "excluded": {}},
                              "dsdl": {"included": sorted(sessions), "excluded": {}}}}
    fake_scorer._ctx = ("B0", 0.0, 0.0)

    # The scorer needs to know which (system, threshold, alpha) it is scoring. The runner sets it right before.
    orig_runner = fake_runner

    def runner_with_ctx(pipe, cache, threshold, system, **kw):
        fake_scorer._ctx = (system, float(threshold), float(kw.get("alpha", 0.0)))
        return orig_runner(pipe, cache, threshold, system, **kw)

    def fake_der_components(ref, hyp):
        # read the system+threshold out of the ctx the runner set; B0's pooled DER drives B0 selection.
        sysname, th, alpha = fake_scorer._ctx
        if sysname == "B0":
            d = b0_der_by_th.get(round(th, 2), 0.5)
        else:
            d = 0.3
        # return components with total=1.0 so pooled DER == d (single recording per call, pooled over recs
        # keeps the same ratio when every recording shares d)
        return {"error": d, "total": 1.0, "fa": d, "miss": 0.0, "conf": 0.0}

    def fake_cache_loader(cache_dir, model, rid):
        return {"rid": rid, "embeddings": np.zeros((1, 1, 2), dtype=np.float64)}

    def fake_lid_loader(cache_dir, rid):
        return {"post": np.zeros((1, 2)), "labels": ["en", "hi"], "hop": 0.1, "win": 1.0}

    return {
        "runner": runner_with_ctx,
        "scorer": fake_scorer,
        "der_components_fn": fake_der_components,
        "lid_loader": fake_lid_loader,
        "cache_loader": fake_cache_loader,
        "pipe_loader": lambda model, th: object(),
        "embedder_factory": lambda model, rid, wav, device="cpu": object(),
        # fits supplied via hooks so the builders are not needed; M2 assessability still reads counts.
        "fit_hooks": {"b1_means": {"delta": 0.0}, "m1_v_hat": {"note": "fitted"},
                      "b2_delta": {"delta": 0.1, "raw": 0.1, "n_same": 4, "n_cross": 4}},
    }, {"b0_best_th": b0_best_th}


class TestOrchestration:
    def test_run_end_to_end_selects_b0_matches_and_alpha(self, tmp_path):
        recordings = [f"s{i}" for i in range(6)]   # >=5 sessions so M2 is assessable
        units = _write_dev_units(tmp_path, recordings)
        refs = _write_ref_rttms(tmp_path, recordings)
        fakes, truth = _orchestration_fakes(recordings, b0_best_th=0.40)

        # B1 fit is supplied via a hook, but run_dev_tuning still needs means per recording for the B1 sweep;
        # since the runner is faked and ignores means, inject empty means via build_b1 returning a trivial fit.
        inp = T.TuningInputs(units_dir=units, cache_dir=tmp_path, wav_dir=tmp_path, ref_rttm_dir=refs,
                             out_path=tmp_path / "tune.json", fits_dir=tmp_path / "fits", model="e3",
                             m3_checkpoints=("m3:ckptA",),
                             pair_by_recording={r: ("en", "hi") for r in recordings})
        rec = T.run_dev_tuning(inp, write=True, **fakes)

        op = rec["operating_points"]
        assert op["B0"]["threshold"] == 0.40                 # lowest pooled DER (major M1 feeds major M2)
        assert op["B0"]["mr_dsdl"] == pytest.approx(0.20)
        # each mitigation matched to B0's MR (0.20) at its planted crossing threshold
        assert op["B1"]["threshold"] == pytest.approx(0.50)
        assert op["M1"]["threshold"] == pytest.approx(0.45)
        assert op["M3:m3:ckptA"]["threshold"] == pytest.approx(0.55)
        # M2 assessable and alpha chosen (lowest FAR meeting the +1pp bound in the plant = 0.20)
        assert rec["m2_assessable"]["tunable"] is True
        assert op["M2"]["threshold"] == pytest.approx(0.40)
        assert op["M2"]["alpha"] == pytest.approx(0.20)
        # fitted values recorded
        assert rec["fits"]["b2_delta"]["delta"] == pytest.approx(0.1)
        # curves + alpha table populated (not the empty record the review flagged)
        assert set(rec["curves"]) >= {"B0", "B1", "M1", "B2", "M3:m3:ckptA", "M2"}
        assert rec["alpha_table"]
        # on-disk record carries the same
        on_disk = json.loads((tmp_path / "tune.json").read_text())
        assert on_disk["operating_points"]["B0"]["threshold"] == 0.40
        assert on_disk["seed"] == T.SEED

    def test_two_runs_differing_in_one_cache_give_different_records(self, tmp_path):
        # major M3: input_sha256 covers the stage-1 caches, so a changed cache yields a different record.
        import numpy as np
        recordings = ["s0", "s1"]
        units = _write_dev_units(tmp_path, recordings)
        refs = _write_ref_rttms(tmp_path, recordings)

        # write two real stage-1 caches that differ only in embeddings, at the paths run_m2b expects.
        from csdiar.pipelines import run_m2b as R
        cache_dir_a = tmp_path / "cacheA"
        cache_dir_b = tmp_path / "cacheB"

        def write_caches(cache_dir, emb_value):
            for rid in recordings:
                p = R.stage1_cache_path(cache_dir, "e3", rid)
                p.parent.mkdir(parents=True, exist_ok=True)
                np.savez(p, rid=rid, embeddings=np.full((1, 1, 2), emb_value, dtype=np.float32))
                lp = R.lid_cache_path(cache_dir, rid)
                lp.parent.mkdir(parents=True, exist_ok=True)
                np.savez(lp, rid=rid, post=np.zeros((1, 2)), labels=np.asarray(["en", "hi"]))
        write_caches(cache_dir_a, 0.1)
        write_caches(cache_dir_b, 0.9)

        fakes, _ = _orchestration_fakes(recordings, b0_best_th=0.40)
        # make the cache loader read the real files so the hash covers them
        fakes["cache_loader"] = lambda cd, model, rid: {"rid": rid,
            "embeddings": np.load(R.stage1_cache_path(cd, model, rid))["embeddings"].astype(np.float64)}

        def run(cache_dir, out):
            inp = T.TuningInputs(units_dir=units, cache_dir=cache_dir, wav_dir=tmp_path, ref_rttm_dir=refs,
                                 out_path=out, fits_dir=out.parent / (out.stem + "_fits"), model="e3",
                                 pair_by_recording={r: ("en", "hi") for r in recordings})
            return T.run_dev_tuning(inp, write=False, **fakes)

        rec_a = run(cache_dir_a, tmp_path / "a.json")
        rec_b = run(cache_dir_b, tmp_path / "b.json")
        # the stage-1 cache files are hashed into input_sha256; different embeddings -> different hashes
        assert rec_a["input_sha256"] != rec_b["input_sha256"]

    def test_non_finite_duration_refused(self, tmp_path):
        # m5: an included recording with no usable duration (no summary duration, no timed rows) is refused.
        from csdiar.data.ingest_displace import FIELDS
        recordings = ["s0"]
        units = tmp_path / "units"
        units.mkdir()
        for kind in ("ssdl", "turns", "ctrl"):
            with gzip.open(units / f"{kind}_dev.csv.gz", "wt", encoding="utf-8", newline="") as fh:
                import csv as _csv
                w = _csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
                w.writeheader()
                # a row with an EMPTY time, so the duration fallback cannot find a max t
                w.writerow({"recording": "s0", "split": "dev", "kind": kind if kind != "turns" else "dsdl",
                            "t": "", "speaker": "A", "speaker2": "", "lang_left": "en", "lang_right": "hi",
                            "left_s": 1.0, "right_s": 1.0, "stratum": "ge1", "elig_p0L1": True,
                            "pause_ms": 0.0, "pause_bin": "none", "phrase": False})
        # summary without duration_s and without recordings carrying it
        (units / "summary.json").write_text(json.dumps(
            {"split": "dev", "recordings": [{"recording": "s0", "languages": ["en", "hi"], "n_dsdl": 12}]}))
        fakes, _ = _orchestration_fakes(recordings)
        refs = tmp_path / "refs"
        refs.mkdir()
        inp = T.TuningInputs(units_dir=units, cache_dir=tmp_path, wav_dir=tmp_path, ref_rttm_dir=refs,
                             out_path=tmp_path / "out.json", model="e3",
                             pair_by_recording={"s0": ("en", "hi")})
        with pytest.raises(ValueError, match="non-finite"):
            T.run_dev_tuning(inp, write=False, **fakes)

    def test_run_dev_tuning_refuses_missing_reference_dir(self, tmp_path):
        # B0 is chosen by the pooled dev DER over every dev recording, so a run without references is refused.
        recordings = ["s0", "s1"]
        units = _write_dev_units(tmp_path, recordings)
        fakes, _ = _orchestration_fakes(recordings)
        inp = T.TuningInputs(units_dir=units, cache_dir=tmp_path, wav_dir=tmp_path,
                             out_path=tmp_path / "out.json", model="e3",
                             pair_by_recording={r: ("en", "hi") for r in recordings})
        with pytest.raises(ValueError, match="ref_rttm_dir"):
            T.run_dev_tuning(inp, write=False, **fakes)

    def test_run_dev_tuning_refuses_eval_summary(self, tmp_path):
        units = tmp_path / "units"
        units.mkdir()
        (units / "summary.json").write_text(json.dumps({"split": "eval", "recordings": []}))
        inp = T.TuningInputs(units_dir=units, cache_dir=tmp_path, wav_dir=tmp_path,
                             out_path=tmp_path / "out.json")
        with pytest.raises(ValueError, match="dev split"):
            T.run_dev_tuning(inp, write=False)


# ======================================================================================================= #
# P0 gate stub
# ======================================================================================================= #
class TestP0GateStub:
    def test_p0_gate_alpha_not_implemented(self):
        with pytest.raises(T.NotImplementedYet):
            T.p0_gate_alpha()


# ======================================================================================================= #
# constants wired from the spec
# ======================================================================================================= #
class TestConstants:
    def test_threshold_grid_is_030_to_100_step_005(self):
        assert T.THRESHOLD_GRID[0] == 0.30
        assert T.THRESHOLD_GRID[-1] == 1.00
        assert len(T.THRESHOLD_GRID) == 15

    def test_alpha_grid_is_0_to_1_step_005(self):
        assert T.ALPHA_GRID[0] == 0.0
        assert T.ALPHA_GRID[-1] == 1.00
        assert len(T.ALPHA_GRID) == 21

    def test_frozen_rule_constants(self):
        assert T.M2_TAU0 == 0.0
        assert T.MR_BOUND_PP == 0.01
        assert T.BCA_ONE_SIDED_LEVEL == 0.80
        assert T.BCA_N_BOOT == 10000
        assert T.ASSESS_MIN_SESSIONS == 5
        assert T.ASSESS_MIN_DSDL_PER_SESSION == 10
        assert T.SEED == 20261018


# ======================================================================================================= #
# REFERENCE RESOLVER on hand-written RTTM text  (major M1 + M2 resolver units; minor m3)
# ======================================================================================================= #
def _rttm(rid, segs):
    """Build RTTM text from (onset, dur, label) triples."""
    return "".join(
        f"SPEAKER {rid} 1 {o:.3f} {d:.3f} <NA> <NA> {lab} <NA> <NA>\n" for (o, d, lab) in segs)


class TestDisplaceRefResolver:
    # Base layout: SD A[0,10] B[10,20]; LD en[0,5] hi[5,15] en[15,20].
    # Clean atoms: [0,5] A/en, [5,10] A/hi, [10,15] B/hi, [15,20] B/en.
    def _base(self):
        sd = _rttm("R", [(0.0, 10.0, "A"), (10.0, 10.0, "B")])
        ld = _rttm("R", [(0.0, 5.0, "en"), (5.0, 10.0, "hi"), (15.0, 5.0, "en")])
        return T.DisplaceRefResolver(sd, ld, rid="R")

    def test_clean_span_resolves_speaker_and_language(self):
        r = self._base()
        assert r.resolve(1.0, 4.0) == ("A", "en")
        assert r.resolve(6.0, 9.0) == ("A", "hi")
        assert r.resolve(11.0, 14.0) == ("B", "hi")

    def test_instant_resolve_single_arg(self):
        r = self._base()
        assert r.resolve(2.0) == ("A", "en")

    def test_language_change_inside_span_gives_none(self):
        r = self._base()
        # [1,7] crosses the en->hi language change at t=5 within speaker A: no single-language answer
        assert r.resolve(1.0, 7.0) is None

    def test_speaker_change_inside_span_gives_none(self):
        r = self._base()
        # [7,12] crosses the A->B speaker change at t=10: no single-speaker answer
        assert r.resolve(7.0, 12.0) is None

    def test_span_exactly_on_boundary_gives_none(self):
        r = self._base()
        # the instant t=5 sits exactly on the en/hi run boundary -> both languages touch -> None
        assert r.resolve(5.0) is None

    def test_overshoot_past_clean_run_end_gives_none(self):
        # m1: the reviewer's pinned case. A single clean atom A/en [0, 5] (no following run). A query ending
        # 2e-9 past 5 overshoots the run's end by 2·EPS of tolerance; resolve must give None, EXACTLY as the
        # reader's _window_in_clean_run(timeline, 0.0, 5.0+2e-9, "A", "en") does -- no extra ±EPS widening in
        # resolve makes it more permissive than the reader.
        from csdiar.data import ingest_displace as ING
        sd = _rttm("R", [(0.0, 5.0, "A")])
        ld = _rttm("R", [(0.0, 5.0, "en")])
        r = T.DisplaceRefResolver(sd, ld, rid="R")
        assert r.resolve(0.0, 5.0) == ("A", "en")                 # fully inside the clean run
        assert r.resolve(0.0, 5.0 + 2e-9) is None                 # 2e-9 past the run end -> no coverage
        # resolve answers bit-identically to the reader's own call on the same raw bounds
        assert ING._window_in_clean_run(r._timeline, 0.0, 5.0 + 2e-9, "A", "en") is False
        assert ING._window_in_clean_run(r._timeline, 0.0, 5.0, "A", "en") is True

    def test_overlap_inside_span_gives_none(self):
        # SD A[0,10] B[8,12]; LD en[0,12]. Atom [8,10] has two speakers (overlap).
        sd = _rttm("R", [(0.0, 10.0, "A"), (8.0, 4.0, "B")])
        ld = _rttm("R", [(0.0, 12.0, "en")])
        r = T.DisplaceRefResolver(sd, ld, rid="R")
        assert r.resolve(6.0, 9.0) is None            # span runs into the [8,10] overlap
        assert r.resolve(2.0, 7.0) == ("A", "en")     # fully before the overlap: clean

    def test_ambiguous_no_language_interval_gives_none(self):
        # SD A[0,10]; LD en[0,4], hi[6,10]. Atom [4,6] is speech with NO language (ambiguous).
        sd = _rttm("R", [(0.0, 10.0, "A")])
        ld = _rttm("R", [(0.0, 4.0, "en"), (6.0, 4.0, "hi")])
        r = T.DisplaceRefResolver(sd, ld, rid="R")
        assert r.resolve(3.0, 5.0) is None            # span runs into the [4,6] ambiguous gap
        assert r.resolve(1.0, 3.0) == ("A", "en")

    def test_clean_spans_lists_single_speaker_single_language_runs(self):
        r = self._base()
        spans = r.clean_spans()
        assert spans == [("A", 0.0, 5.0, "en"), ("A", 5.0, 10.0, "hi"),
                         ("B", 10.0, 15.0, "hi"), ("B", 15.0, 20.0, "en")]

    def test_from_paths_reads_files(self, tmp_path):
        from csdiar.data.ingest_displace import SD_SUFFIX, LD_SUFFIX
        (tmp_path / f"R{SD_SUFFIX}").write_text(_rttm("R", [(0.0, 10.0, "A")]))
        (tmp_path / f"R{LD_SUFFIX}").write_text(_rttm("R", [(0.0, 10.0, "en")]))
        r = T.DisplaceRefResolver.from_paths(tmp_path / f"R{SD_SUFFIX}", tmp_path / f"R{LD_SUFFIX}", rid="R")
        assert r.resolve(2.0, 8.0) == ("A", "en")

    def test_default_factory_refuses_missing_lang_dir(self, tmp_path):
        inp = T.TuningInputs(units_dir=tmp_path, cache_dir=tmp_path, wav_dir=tmp_path,
                             out_path=tmp_path / "o.json", ref_rttm_dir=tmp_path, lang_rttm_dir=None)
        with pytest.raises(ValueError, match="lang_rttm_dir"):
            T.default_ref_resolver_factory(inp, "R")

    def test_default_factory_refuses_missing_file(self, tmp_path):
        refs = tmp_path / "sd"; refs.mkdir()
        langs = tmp_path / "ld"; langs.mkdir()
        inp = T.TuningInputs(units_dir=tmp_path, cache_dir=tmp_path, wav_dir=tmp_path,
                             out_path=tmp_path / "o.json", ref_rttm_dir=refs, lang_rttm_dir=langs)
        with pytest.raises(FileNotFoundError):
            T.default_ref_resolver_factory(inp, "R")


# ======================================================================================================= #
# DEFAULT FIT-INPUT BUILDERS on a tiny synthetic cache + resolver  (major M1: real fit path pieces)
# ======================================================================================================= #
class _CacheResolver:
    """A minimal DisplaceRefResolver stand-in with a planted resolve() and clean_spans(), so the builders can be
    driven with known slot speakers and languages without any RTTM text."""
    def __init__(self, resolve_map, spans):
        self._map = resolve_map          # {(round(lo,3), round(hi,3)): (speaker, lang)}
        self._spans = spans

    def resolve(self, lo, hi=None):
        if hi is None:
            hi = lo
        return self._map.get((round(float(lo), 3), round(float(hi), 3)))

    def clean_spans(self):
        return list(self._spans)


def _tiny_cache(embeddings):
    """A stage-1 cache with two slots in one chunk: slot0 active at frame mids {0.5,1.5}, slot1 at {2.5,3.5}.
    frame_mid[f] = rf_start + rf_dur/2 + rf_step*f = 0.5 + f (rf_start=0, rf_dur=1, rf_step=1)."""
    binarized = np.zeros((1, 4, 2), dtype=np.float64)   # (chunks, frames, speakers)
    binarized[0, 0, 0] = 1; binarized[0, 1, 0] = 1       # slot0 active at f0,f1 -> mids 0.5,1.5
    binarized[0, 2, 1] = 1; binarized[0, 3, 1] = 1       # slot1 active at f2,f3 -> mids 2.5,3.5
    return {"rid": "R", "embeddings": np.asarray(embeddings, dtype=np.float64),
            "binarized": binarized, "seg_start": 0.0, "seg_step": 10.0,
            "rf_start": 0.0, "rf_step": 1.0, "rf_duration": 1.0}


class TestDefaultBuilders:
    def test_build_m1_units_takes_speaker_and_language_from_resolver(self):
        # slot0 span [0.5,1.5] -> (A, en); slot1 span [2.5,3.5] -> (A, hi). embeddings: en=(2,0), hi=(0,0).
        cache = _tiny_cache([[[2.0, 0.0], [0.0, 0.0]]])     # (1,2,2)
        lid = {"post": np.ones((4, 2)), "labels": ["en", "hi"], "hop": 1.0, "win": 1.0}
        # the builder now resolves over the slot's TRUE extent (midpoint span widened by rf_duration/2 = 0.5
        # each side, m2): slot0 mids {0.5,1.5} -> extent [0.0,2.0]; slot1 mids {2.5,3.5} -> extent [2.0,4.0].
        resolver = _CacheResolver({(0.0, 2.0): ("A", "en"), (2.0, 4.0): ("A", "hi")}, spans=[])
        units = T._default_build_m1_units(None, ["R"], {"R": cache}, {"R": lid}, {"R": (0, 1)}, {"R": resolver})
        assert len(units) == 1
        u = units[0]
        assert u["speaker"] == "A" and u["recording"] == "R"
        np.testing.assert_allclose(u["by_lang"]["en"], [2.0, 0.0])
        np.testing.assert_allclose(u["by_lang"]["hi"], [0.0, 0.0])
        # and the fit on these units gives v_hat = (1,0) (en - hi, unit-normalised)
        out = T.fit_m1_v_hat(units)
        np.testing.assert_allclose(out["v_hat"], [1.0, 0.0], atol=1e-12)

    def test_build_m1_units_drops_slot_with_no_resolver_answer(self):
        cache = _tiny_cache([[[2.0, 0.0], [5.0, 5.0]]])
        lid = {"post": np.ones((4, 2)), "labels": ["en", "hi"]}
        # slot1's TRUE extent [2.0,4.0] gets None (straddles a boundary) and is dropped; only slot0 ('en')
        # contributes, so the unit has a single language -> the builder returns it, and fit_m1_v_hat then
        # refuses it (no contrast). Keys are the widened extents (m2), not the raw midpoint spans.
        resolver = _CacheResolver({(0.0, 2.0): ("A", "en"), (2.0, 4.0): None}, spans=[])
        units = T._default_build_m1_units(None, ["R"], {"R": cache}, {"R": lid}, {"R": (0, 1)}, {"R": resolver})
        assert len(units) == 1
        assert set(units[0]["by_lang"]) == {"en"}     # slot1 dropped: only the 'en' slot survives
        with pytest.raises(T.InsufficientData):
            T.fit_m1_v_hat(units)

    def test_build_m1_units_none_refs_returns_empty(self):
        assert T._default_build_m1_units(None, ["R"], {}, {}, {}, None) == []

    def test_build_m1_units_true_extent_crosses_reference_change_over_real_resolver(self):
        # m2: drive the REAL DisplaceRefResolver (hand-written RTTM, no stand-in) over a small cache so the
        # slot's TRUE acoustic extent -- not just its frame-midpoint span -- decides attribution.
        # _tiny_cache: rf_start=0, rf_dur=1, rf_step=1, frame mid[f] = 0.5 + f.
        #   slot0 active f0,f1 -> mids {0.5,1.5} -> MIDPOINT span [0.5,1.5], TRUE extent [0.0,2.0].
        #   slot1 active f2,f3 -> mids {2.5,3.5} -> MIDPOINT span [2.5,3.5], TRUE extent [2.0,4.0].
        # References: SD A[0,10]; LD en[0,1.8], hi[1.8,10]. The en/hi change is at t=1.8.
        #   slot0: midpoint span [0.5,1.5] lies wholly in en (the OLD midpoint attribution would keep it as en),
        #          but its TRUE extent [0.0,2.0] CROSSES the 1.8 change -> resolve() gives None -> DROPPED (m2).
        #   slot1: TRUE extent [2.0,4.0] is entirely inside hi -> KEPT as ('A','hi').
        cache = _tiny_cache([[[7.0, 7.0], [0.0, 3.0]]])        # slot1 ('hi') embedding = (0,3)
        lid = {"post": np.ones((4, 2)), "labels": ["en", "hi"], "hop": 1.0, "win": 1.0}
        sd = _rttm("R", [(0.0, 10.0, "A")])
        ld = _rttm("R", [(0.0, 1.8, "en"), (1.8, 8.2, "hi")])
        resolver = T.DisplaceRefResolver(sd, ld, rid="R")
        units = T._default_build_m1_units(None, ["R"], {"R": cache}, {"R": lid}, {"R": (0, 1)}, {"R": resolver})
        # slot0 dropped (true extent crosses the change), slot1 kept -> one unit, only the 'hi' language present
        assert len(units) == 1
        assert set(units[0]["by_lang"]) == {"hi"}
        np.testing.assert_allclose(units[0]["by_lang"]["hi"], [0.0, 3.0])

    def test_build_m1_units_true_extent_clear_of_change_is_kept(self):
        # m2 companion: with the same geometry but the reference change moved WELL clear of slot0's true extent
        # [0.0,2.0] (en runs to 3.0), slot0 is attributed to 'en' and the unit carries both languages.
        cache = _tiny_cache([[[2.0, 0.0], [0.0, 3.0]]])        # slot0 'en'=(2,0), slot1 'hi'=(0,3)
        lid = {"post": np.ones((4, 2)), "labels": ["en", "hi"], "hop": 1.0, "win": 1.0}
        sd = _rttm("R", [(0.0, 10.0, "A")])
        ld = _rttm("R", [(0.0, 3.0, "en"), (3.0, 7.0, "hi")])   # change at 3.0: clear of [0,2]; slot1 [2,4] now crosses
        # slot1's true extent [2.0,4.0] crosses 3.0 -> dropped; slot0 [0,2] inside en -> kept. One language.
        resolver = T.DisplaceRefResolver(sd, ld, rid="R")
        units = T._default_build_m1_units(None, ["R"], {"R": cache}, {"R": lid}, {"R": (0, 1)}, {"R": resolver})
        assert len(units) == 1
        assert set(units[0]["by_lang"]) == {"en"}
        np.testing.assert_allclose(units[0]["by_lang"]["en"], [2.0, 0.0])

    def test_build_b2_cosines_from_clean_spans(self):
        # two same-language en segments (identical vectors -> cos 1) and one hi segment (orthogonal -> cos 0).
        class _Emb:
            def embed(self, s, e):
                return {(0.0, 1.0): np.array([1.0, 0.0]), (1.0, 2.0): np.array([1.0, 0.0]),
                        (2.0, 3.0): np.array([0.0, 1.0])}[(round(s, 3), round(e, 3))]
        spans = [("A", 0.0, 1.0, "en"), ("A", 1.0, 2.0, "en"), ("A", 2.0, 3.0, "hi")]
        resolver = _CacheResolver({}, spans=spans)
        same, cross = T._default_build_b2_cosines(None, ["R"], {}, {}, {}, {"R": resolver}, {"R": _Emb()})
        # same-language pair (en,en): cos 1.0 ; cross pairs (en,hi)x2: cos 0.0 each
        assert sorted(same) == pytest.approx([1.0])
        assert sorted(cross) == pytest.approx([0.0, 0.0])
        out = T.fit_b2_delta(same, cross)
        assert out["raw"] == pytest.approx(1.0)       # mean same (1.0) - mean cross (0.0)
        assert out["delta"] == 0.5                    # clamped to B2_DELTA_MAX

    def test_build_b2_cosines_none_refs_returns_empty(self):
        assert T._default_build_b2_cosines(None, ["R"], {}, {}, {}, None, {"R": object()}) == ([], [])
        assert T._default_build_b2_cosines(None, ["R"], {}, {}, {}, {"R": object()}, None) == ([], [])

    def test_build_b2_pair_cap_is_deterministic(self):
        # m4: four en segments -> C(4,2)=6 same-language pairs; cap at 2 must keep EXACTLY the two pairs the
        # documented per-(recording, speaker) crc32 seed selects, with their exact cosine values -- not merely
        # two-of-some and repeatable. The expected selection is recomputed here from the SAME stable key
        # (SEED, crc32('R'), crc32('A')), so a regression that changed the seeding (e.g. back to a corpus-wide
        # fixed seed, or Python's salted hash()) would change WHICH pairs survive and fail this test.
        import zlib
        angs = [0.0, 0.3, 0.6, 0.9]

        class _Emb:
            def embed(self, s, e):
                ang = {0.0: 0.0, 1.0: 0.3, 2.0: 0.6, 3.0: 0.9}[round(s, 3)]
                return np.array([np.cos(ang), np.sin(ang)])
        spans = [("A", float(i), float(i + 1), "en") for i in range(4)]
        resolver = _CacheResolver({}, spans=spans)
        a1, _ = T._default_build_b2_cosines(None, ["R"], {}, {}, {}, {"R": resolver}, {"R": _Emb()},
                                            max_pairs_per_unit=2)
        a2, _ = T._default_build_b2_cosines(None, ["R"], {}, {}, {}, {"R": resolver}, {"R": _Emb()},
                                            max_pairs_per_unit=2)
        assert len(a1) == 2
        assert a1 == pytest.approx(a2)                # fixed, stable seed -> reproducible truncation

        # recompute the EXACT surviving pairs from the documented stable key, then their cosines, and pin them.
        vecs = [np.array([np.cos(a), np.sin(a)]) for a in angs]
        pair_idx = [(a, b) for a in range(4) for b in range(a + 1, 4)]
        rng = np.random.default_rng([T.SEED, zlib.crc32(b"R"), zlib.crc32(b"A")])
        order = rng.permutation(len(pair_idx))[:2]
        kept = [pair_idx[k] for k in sorted(order)]
        expected = [float(vecs[i] @ vecs[j]) / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]))
                    for i, j in kept]
        assert len(kept) == 2                          # the known input truncates 6 -> 2
        assert a1 == pytest.approx(expected)           # EXACTLY the two cosines the stable seed selects (m4)

    def test_build_b2_pair_cap_sample_is_corpus_independent(self):
        # m5: a recording's capped sample must depend only on its own (rid, speaker), not on how many pairs
        # earlier recordings drew. Build with recording 'R2' ALONE, then with 'R1' processed before it (R1 also
        # exceeding the cap, so a shared corpus-wide RNG would have advanced its stream): R2's cosines are
        # identical both ways. Changing the recording ORDER likewise leaves each recording's sample unchanged.
        class _Emb:
            def __init__(self, base):
                self._base = base

            def embed(self, s, e):
                ang = self._base + 0.2 * round(s, 3)
                return np.array([np.cos(ang), np.sin(ang)])
        spans1 = [("A", float(i), float(i + 1), "en") for i in range(4)]     # R1: 6 pairs, cap bites
        spans2 = [("B", float(i), float(i + 1), "en") for i in range(4)]     # R2: 6 pairs, cap bites
        res = {"R1": _CacheResolver({}, spans=spans1), "R2": _CacheResolver({}, spans=spans2)}
        emb = {"R1": _Emb(0.0), "R2": _Emb(1.0)}

        r2_alone, _ = T._default_build_b2_cosines(None, ["R2"], {}, {}, {}, res, emb, max_pairs_per_unit=2)
        r2_after_r1, _ = T._default_build_b2_cosines(None, ["R1", "R2"], {}, {}, {}, res, emb,
                                                     max_pairs_per_unit=2)
        # R2's own two cosines are the LAST two of the [R1..., R2...] pooled list; they must equal R2-alone.
        assert r2_after_r1[-2:] == pytest.approx(r2_alone)

        # and reordering the corpus does not change either recording's sample
        ab, _ = T._default_build_b2_cosines(None, ["R1", "R2"], {}, {}, {}, res, emb, max_pairs_per_unit=2)
        ba, _ = T._default_build_b2_cosines(None, ["R2", "R1"], {}, {}, {}, res, emb, max_pairs_per_unit=2)
        assert sorted(ab) == pytest.approx(sorted(ba))


# ======================================================================================================= #
# REAL FIT PATH through run_dev_tuning (NO fit_hooks)  (major M2)
# ======================================================================================================= #
class TestRealFitOrchestration:
    def test_fits_b1_m1_b2_and_writes_hashed_artifacts(self, tmp_path):
        # Exercise the real fits (fit_b1_means / fit_m1_v_hat / fit_b2_delta), the npz/npy writing and the
        # means_by_index_by_rec wiring, with NO fit_hooks: the fit INPUTS are controlled through injected
        # builders so the fitted values are hand-computable, but every real-fit branch runs (review M2).
        recordings = ["s0", "s1"]
        units = _write_dev_units(tmp_path, recordings)
        refs = _write_ref_rttms(tmp_path, recordings)
        fakes, _ = _orchestration_fakes(recordings, b0_best_th=0.40)
        fakes.pop("fit_hooks")                      # <-- the point: no hooks, so the real fits run

        # B1 slot embeddings keyed by PAIR INDEX (0='en', 1='hi'): means en=(2,0), hi=(0,4).
        def build_b1(inputs, recs, caches, lids, pairs):
            return {0: [[1.0, 0.0], [3.0, 0.0]], 1: [[0.0, 2.0], [0.0, 6.0]]}

        # M1 units: one unit, en-hi diff (2,0)-(0,0)=(2,0) -> v_hat (1,0).
        def build_m1(inputs, recs, caches, lids, pairs, refs_):
            return [{"speaker": "A", "recording": "s0",
                     "by_lang": {"en": [2.0, 0.0], "hi": [0.0, 0.0]}}]

        # B2 cosines: same mean 0.9, cross mean 0.5 -> delta 0.4 (unclamped).
        def build_b2(inputs, recs, caches, lids, pairs, refs_, embs, max_pairs_per_unit=0):
            return [0.9, 0.9], [0.5, 0.5]

        # a fake resolver factory so run_dev_tuning builds `refs` (the real builders are overridden anyway).
        def ref_factory(inputs, rid):
            return _CacheResolver({}, spans=[])

        inp = T.TuningInputs(units_dir=units, cache_dir=tmp_path, wav_dir=tmp_path, ref_rttm_dir=refs,
                             lang_rttm_dir=refs, out_path=tmp_path / "tune.json", fits_dir=tmp_path / "fits",
                             model="e3", pair_by_recording={r: ("en", "hi") for r in recordings})
        rec = T.run_dev_tuning(inp, write=True, build_b1=build_b1, build_m1=build_m1, build_b2=build_b2,
                               ref_resolver_factory=ref_factory, **fakes)

        # fitted values recorded from the REAL fits
        assert rec["fits"]["b1_means"]["counts"] == {0: 2, 1: 2}
        assert rec["fits"]["b1_means"]["dim"] == 2
        assert rec["fits"]["m1_v_hat"]["n_units"] == 1
        assert rec["fits"]["b2_delta"]["delta"] == pytest.approx(0.4)
        assert rec["fits"]["b2_delta"]["raw"] == pytest.approx(0.4)
        assert "b1_unfittable" not in rec["flags"]
        assert "m1_unfittable" not in rec["flags"]

        # the npz / npy were written AND appear in input_sha256 (review M2)
        b1_npz = tmp_path / "fits" / "b1_means_dev.npz"
        v_npy = tmp_path / "fits" / "m1_v_hat_dev.npy"
        assert b1_npz.exists() and v_npy.exists()
        assert str(b1_npz) in rec["input_sha256"]
        assert str(v_npy) in rec["input_sha256"]

        # the fitted B1 means round-tripped to the runner's code-keyed npz
        with np.load(b1_npz) as z:
            np.testing.assert_allclose(z["en"], [2.0, 0.0])
            np.testing.assert_allclose(z["hi"], [0.0, 4.0])
        # the fitted v_hat is the unit en-hi direction
        np.testing.assert_allclose(np.load(v_npy), [1.0, 0.0], atol=1e-12)

    def test_real_run_with_references_does_not_land_unfittable(self, tmp_path):
        # major M1: with both reference dirs present and a resolver, M1/B2 must NOT be flagged unfittable just
        # because the default factory was auto-wired (it is, from lang_rttm_dir).
        recordings = ["s0", "s1"]
        units = _write_dev_units(tmp_path, recordings)
        refs = _write_ref_rttms(tmp_path, recordings)
        # write LANGUAGE rttms so the default resolver factory (auto-wired from lang_rttm_dir) can read them
        langs = tmp_path / "langs"; langs.mkdir()
        for rid in recordings:
            (langs / f"{rid}_LANGUAGE.rttm").write_text(_rttm(rid, [(0.0, 150.0, "en"), (150.0, 150.0, "hi")]))
        fakes, _ = _orchestration_fakes(recordings, b0_best_th=0.40)
        fakes.pop("fit_hooks")

        # builders fed known inputs so the fits succeed; the point is the auto-wired default factory + no flag.
        def build_b1(i, r, c, l, p):
            return {0: [[1.0, 0.0]], 1: [[0.0, 1.0]]}

        def build_m1(i, r, c, l, p, refs_):
            assert refs_ is not None               # the default factory was wired because lang_rttm_dir is set
            return [{"speaker": "A", "recording": "s0", "by_lang": {"en": [1.0, 0.0], "hi": [0.0, 0.0]}}]

        def build_b2(i, r, c, l, p, refs_, e, max_pairs_per_unit=0):
            assert refs_ is not None
            return [0.9], [0.5]

        inp = T.TuningInputs(units_dir=units, cache_dir=tmp_path, wav_dir=tmp_path, ref_rttm_dir=refs,
                             lang_rttm_dir=langs, out_path=tmp_path / "tune.json", fits_dir=tmp_path / "fits",
                             model="e3", pair_by_recording={r: ("en", "hi") for r in recordings})
        rec = T.run_dev_tuning(inp, write=False, build_b1=build_b1, build_m1=build_m1, build_b2=build_b2, **fakes)
        assert "m1_unfittable" not in rec["flags"]
        assert "b2_unfittable" not in rec["flags"]
        assert rec["fits"]["m1_v_hat"]["n_units"] == 1

    def test_b1_unfittable_when_a_recording_misses_a_pair_language_mean(self, tmp_path):
        # minor m1: a language pooled corpus-wide but absent in one recording would KeyError in the runner's B1
        # branch; run_dev_tuning must record B1 unfittable naming the language instead.
        recordings = ["s0", "s1"]
        units = _write_dev_units(tmp_path, recordings)
        refs = _write_ref_rttms(tmp_path, recordings)
        fakes, _ = _orchestration_fakes(recordings, b0_best_th=0.40)
        fakes.pop("fit_hooks")

        # B1 fit yields only language index 0 ('en'); index 1 ('hi') has no mean -> every recording misses 'hi'.
        def build_b1(i, r, c, l, p):
            return {0: [[1.0, 0.0], [2.0, 0.0]]}

        def build_m1(i, r, c, l, p, refs_):
            return [{"speaker": "A", "recording": "s0", "by_lang": {"en": [1.0, 0.0], "hi": [0.0, 0.0]}}]

        def build_b2(i, r, c, l, p, refs_, e, max_pairs_per_unit=0):
            return [0.9], [0.5]

        def ref_factory(inputs, rid):
            return _CacheResolver({}, spans=[])

        inp = T.TuningInputs(units_dir=units, cache_dir=tmp_path, wav_dir=tmp_path, ref_rttm_dir=refs,
                             lang_rttm_dir=refs, out_path=tmp_path / "tune.json", fits_dir=tmp_path / "fits",
                             model="e3", pair_by_recording={r: ("en", "hi") for r in recordings})
        rec = T.run_dev_tuning(inp, write=False, build_b1=build_b1, build_m1=build_m1, build_b2=build_b2,
                               ref_resolver_factory=ref_factory, **fakes)
        assert "b1_unfittable" in rec["flags"]
        assert "hi" in rec["flags"]["b1_unfittable"]
        # B1 then reported not assessable in the operating points (not a KeyError crash)
        assert "not_assessable" in rec["operating_points"]["B1"]


# ======================================================================================================= #
# m4: hours-only summary refused
# ======================================================================================================= #
class TestDurationUnits:
    def test_hours_only_summary_refused(self, tmp_path):
        from csdiar.data.ingest_displace import FIELDS
        recordings = ["s0"]
        units = tmp_path / "units"
        units.mkdir()
        for kind in ("ssdl", "turns", "ctrl"):
            with gzip.open(units / f"{kind}_dev.csv.gz", "wt", encoding="utf-8", newline="") as fh:
                import csv as _csv
                w = _csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
                w.writeheader()
        # a summary carrying `hours` but NOT `duration_s` must be refused (m4: no hours->seconds mis-read)
        (units / "summary.json").write_text(json.dumps(
            {"split": "dev", "recordings": [{"recording": "s0", "languages": ["en", "hi"],
                                             "n_dsdl": 12, "hours": 0.5}]}))
        fakes, _ = _orchestration_fakes(recordings)
        fakes.pop("fit_hooks", None)
        refs = tmp_path / "refs"; refs.mkdir()
        inp = T.TuningInputs(units_dir=units, cache_dir=tmp_path, wav_dir=tmp_path, ref_rttm_dir=refs,
                             out_path=tmp_path / "out.json", model="e3",
                             pair_by_recording={"s0": ("en", "hi")})
        with pytest.raises(ValueError, match="hours"):
            T.run_dev_tuning(inp, write=False, **fakes)


# ======================================================================================================= #
# m2: der_components through the REAL pyannote DiarizationErrorRate (skip when pyannote is missing)
# ======================================================================================================= #
class TestDerComponentsRealMetric:
    def test_two_tiny_rttms_confusion_two_total_twenty(self, tmp_path):
        pytest.importorskip("pyannote.metrics")
        # Brief's box values: reference A[0,10], B[10,20]; hypothesis x[0,12], y[12,20].
        # confusion 2.0, total 20.0, DER 0.1.
        ref = tmp_path / "ref.rttm"
        hyp = tmp_path / "hyp.rttm"
        ref.write_text(_rttm("R", [(0.0, 10.0, "A"), (10.0, 10.0, "B")]))
        hyp.write_text(_rttm("R", [(0.0, 12.0, "x"), (12.0, 8.0, "y")]))
        comp = T.der_components(ref, hyp)
        assert comp["total"] == pytest.approx(20.0)
        assert comp["conf"] == pytest.approx(2.0)
        assert comp["fa"] == pytest.approx(0.0)
        assert comp["miss"] == pytest.approx(0.0)
        assert comp["error"] == pytest.approx(2.0)
        # pooled DER over the single recording is 2/20 = 0.1
        assert T.pooled_der({"R": comp}) == pytest.approx(0.1)


# ======================================================================================================= #
# per-session record + paired-turn counts at the operating points (review m1 / old interface gap)
# ======================================================================================================= #
class TestPerSessionFromScores:
    def test_per_session_fields_and_inclusion_lists(self):
        # two sessions; the per_session dict carries every scored session, the inclusion lists are the §7 floors.
        res = {
            "sessions": {
                "R1": _session_score("R1", far=0.10, mr=0.20, n_ssdl_cc=20, n_dsdl=15),
                "R2": _session_score("R2", far=0.30, mr=0.40, n_ssdl_cc=3, n_dsdl=2),
            },
            "inclusion": {"ssdl": {"included": ["R1"], "excluded": {"R2": 3}},
                          "dsdl": {"included": ["R1"], "excluded": {"R2": 2}}},
        }
        ps = T.per_session_from_scores(res, res["inclusion"]["ssdl"]["included"],
                                       res["inclusion"]["dsdl"]["included"])
        assert set(ps["per_session"]) == {"R1", "R2"}        # every scored session visible
        assert ps["per_session"]["R1"]["mr_dsdl"] == pytest.approx(0.20)
        assert ps["per_session"]["R1"]["far_ssdl"] == pytest.approx(0.10)
        assert ps["per_session"]["R1"]["delta_far_std"] == pytest.approx(0.10 - 0.05)
        assert ps["per_session"]["R1"]["n_dsdl"] == 15
        assert ps["per_session"]["R1"]["n_ssdl_with_cell_controls"] == 20
        assert ps["inclusion"]["h4_ssdl"] == ["R1"]
        assert ps["inclusion"]["h5_dsdl"] == ["R1"]


class TestPairedTurnCountsVsB0:
    def test_missed_by_mit_only_b0_only_both(self):
        # One recording, 3 DS-DL turns at t = 10, 20, 30 (collar 0.5 s), no DS-SL turns.
        ds = {"R": {"times": [10.0, 20.0, 30.0], "n_dsdl": 3}}
        # B0 changes match turn 10 and 20 (not 30); mitigation matches 20 and 30 (not 10).
        b0 = {"R": [10.0, 20.0]}
        mit = {"R": [20.0, 30.0]}
        out = T.paired_turn_counts_vs_b0(ds, b0, mit)
        r = out["R"]
        assert r["n_dsdl"] == 3
        # turn 30 missed by B0 only; turn 10 missed by mitigation only; none missed by both
        assert r["missed_by_mit_only"] == 1      # turn 10 (B0 matched, mit missed)
        assert r["missed_by_b0_only"] == 1       # turn 30 (mit matched, B0 missed)
        assert r["missed_by_both"] == 0
        assert r["matched_by_both"] == 1         # turn 20

    def test_no_changes_means_all_missed_by_both(self):
        ds = {"R": {"times": [5.0, 15.0], "n_dsdl": 2}}
        out = T.paired_turn_counts_vs_b0(ds, {"R": []}, {"R": []})
        assert out["R"] == {"n_dsdl": 2, "missed_by_mit_only": 0, "missed_by_b0_only": 0,
                            "missed_by_both": 2, "matched_by_both": 0}

    def test_full_pool_matching_agrees_with_score_session_mr_dsdl(self):
        # review M1: on a session where a DS-SL turn sits INSIDE the collar of a DS-DL turn, the paired counts
        # must agree with score_session's mr_dsdl for BOTH systems -- the per-system missed count over n equals
        # that system's MR_DS-DL -- and the OLD DS-DL-only matching would have differed.
        from csdiar.analysis import score_m2b as S
        c = S.C  # 0.5 s collar
        # One DS-DL turn at t=10 and one DS-SL turn at t=10.2 (inside the DS-DL turn's collar). A single change
        # at t=10.15 is slightly closer to the DS-SL turn, so under one-to-one full-pool matching the change is
        # taken by the DS-SL turn and the DS-DL turn is MISSED -- exactly how score_session counts it.
        dsdl_t = 10.0
        dssl_t = 10.2
        change = 10.15
        rows = [{"kind": "dsdl", "t": str(dsdl_t), "pause_bin": "none"},
                {"kind": "dssl", "t": str(dssl_t), "pause_bin": "none"}]
        changes = [change]
        # score_session's mr_dsdl on this session (B0 == mitigation here; we verify one system at a time)
        sess = S.score_session("R", rows, changes, duration_s=600.0, c=c)
        assert sess.n_dsdl == 1
        # the DS-SL turn wins the change; the DS-DL turn is missed -> mr_dsdl == 1.0
        assert sess.mr_dsdl == pytest.approx(1.0)

        # paired_turn_counts_vs_b0 on the FULL pool reproduces that: with B0 = the same single change and the
        # mitigation making NO change, B0 misses the DS-DL turn (DS-SL stole its change) and so does the mit.
        ds_pool = T._ds_turn_times_by_recording({"R": rows})
        assert ds_pool["R"] == {"times": [dsdl_t, dssl_t], "n_dsdl": 1}
        out = T.paired_turn_counts_vs_b0(ds_pool, {"R": changes}, {"R": []})
        r = out["R"]
        # B0's missed DS-DL over n == B0's MR_DS-DL (1/1 = 1.0); same for the mitigation (no changes -> missed)
        b0_missed = r["missed_by_b0_only"] + r["missed_by_both"]
        mit_missed = r["missed_by_mit_only"] + r["missed_by_both"]
        assert b0_missed / r["n_dsdl"] == pytest.approx(sess.mr_dsdl)   # agrees with score_session
        assert mit_missed / r["n_dsdl"] == pytest.approx(1.0)

        # the OLD DS-DL-only matching would have DIFFERED: with the DS-SL turn removed, the lone change at 10.15
        # is within collar of the DS-DL turn at 10.0, so B0 would MATCH it (missed = 0, MR = 0) -- the bug.
        b0_matched_dsdl_only = {p[0] for p in S.match_turns([dsdl_t], changes, c)["pairs"]}
        assert b0_matched_dsdl_only == {0}                              # DS-DL-only: matched -> MR 0, disagrees

    def test_ds_turn_times_returns_full_pool_dsdl_first(self):
        # the full DS pool, DS-DL turns FIRST then DS-SL (score_session's order), with n_dsdl; empty times dropped
        rows = {"R": [{"kind": "dsdl", "t": "7.0"}, {"kind": "dssl", "t": "9.0"},
                      {"kind": "ssdl", "t": "5.0"}, {"kind": "dsdl", "t": ""},
                      {"kind": "dssl", "t": "11.0"}]}
        out = T._ds_turn_times_by_recording(rows)
        assert out["R"] == {"times": [7.0, 9.0, 11.0], "n_dsdl": 1}   # dsdl(7.0) first, then dssl(9.0,11.0)


# ======================================================================================================= #
# per-session record is written at the operating points, and adding it does NOT move any selection (m1)
# ======================================================================================================= #
class TestPerSessionRecordInOrchestration:
    def test_operating_points_identical_with_and_without_per_session_fields(self, tmp_path):
        # The per-session record is additive: the chosen B0 threshold, the matched thresholds, the chosen alpha
        # and every operating point must be byte-identical to a run that ignores the new fields. We run once and
        # assert the selection results stand on their own next to the new `per_session_at_operating_points`.
        recordings = [f"s{i}" for i in range(6)]
        units = _write_dev_units(tmp_path, recordings)
        refs = _write_ref_rttms(tmp_path, recordings)
        fakes, _ = _orchestration_fakes(recordings, b0_best_th=0.40)
        inp = T.TuningInputs(units_dir=units, cache_dir=tmp_path, wav_dir=tmp_path, ref_rttm_dir=refs,
                             out_path=tmp_path / "tune.json", fits_dir=tmp_path / "fits", model="e3",
                             m3_checkpoints=("m3:ckptA",),
                             pair_by_recording={r: ("en", "hi") for r in recordings})
        rec = T.run_dev_tuning(inp, write=False, **fakes)

        # selection unchanged from the existing end-to-end test
        op = rec["operating_points"]
        assert op["B0"]["threshold"] == 0.40
        assert op["B0"]["mr_dsdl"] == pytest.approx(0.20)
        assert op["B1"]["threshold"] == pytest.approx(0.50)
        assert op["M1"]["threshold"] == pytest.approx(0.45)
        assert op["M3:m3:ckptA"]["threshold"] == pytest.approx(0.55)
        assert op["M2"]["alpha"] == pytest.approx(0.20)

        # the new per-session record exists, carries B0 and each mitigation, with the required fields
        ps = rec["per_session_at_operating_points"]
        assert "B0" in ps
        for name in ("B1", "M1", "M3:m3:ckptA", "M2"):
            assert name in ps, name
            sess0 = next(iter(ps[name]["per_session"].values()))
            assert set(sess0) == {"mr_dsdl", "delta_far_std", "far_ssdl", "n_dsdl",
                                  "n_ssdl_with_cell_controls"}
            assert set(ps[name]["inclusion"]) == {"h4_ssdl", "h5_dsdl"}
            # paired-turn counts vs B0 present for each mitigation
            assert ps[name]["paired_turns_vs_b0"] is not None
            pc0 = next(iter(ps[name]["paired_turns_vs_b0"].values()))
            assert set(pc0) == {"n_dsdl", "missed_by_mit_only", "missed_by_b0_only",
                                "missed_by_both", "matched_by_both"}
        # B0 itself has no paired-turn counts (it is the reference)
        assert "paired_turns_vs_b0" not in ps["B0"]
        # the record is JSON-serialisable with the new fields
        json.dumps(rec, default=T._json_default)
