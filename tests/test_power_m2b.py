#!/usr/bin/env python3
"""Unit tests for csdiar.analysis.power_m2b (PREREGISTRATION_M2B.md §7).

Model-free and fast (no model, no audio, no GPU box, no DISPLACE reference). Small simulation counts are passed
through parameters. Covered:
  * the listing-to-seconds helper (header subtraction, extension filter, duplicate guard, byte-rate arithmetic);
  * the projection arithmetic (dev rate * eval hours; per-session shape + rescale; projection base);
  * the vectorised BCa upper bound matches stats.bca_ci's upper endpoint on shared inputs;
  * power near 1 for a large clean design and near the nominal 0.05 level at the margin (size check);
  * determinism under the seed;
  * that no eval reference path or eval count is read (the dev-only summary guard; listing carries sizes only);
  * the verdict threshold (confirmatory iff power >= 0.8).

Run under .venv-pyannote4 (numpy present):
  .venv-pyannote4/bin/python -m pytest -q -p no:cacheprovider tests/test_power_m2b.py
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from csdiar.analysis import power_m2b as P
from csdiar.analysis.stats import bca_ci


# --------------------------------------------------------------------------------------------------- #
# listing -> seconds
# --------------------------------------------------------------------------------------------------- #
def test_wav_seconds_from_size_header_and_rate():
    # 16 kHz mono 16-bit: 32000 bytes/s. 1 second of audio + 44-byte header.
    assert P.WAV_BYTE_RATE == 32000
    assert P.wav_seconds_from_size(32000 + 44) == pytest.approx(1.0)
    assert P.wav_seconds_from_size(44) == 0.0            # header only
    assert P.wav_seconds_from_size(10) == 0.0            # below header -> clamped, never negative
    assert P.wav_seconds_from_size(16000 + 44) == pytest.approx(0.5)


def test_wav_seconds_from_size_bad_rate():
    with pytest.raises(ValueError):
        P.wav_seconds_from_size(1000, byte_rate=0)


def test_listing_to_seconds_filters_and_stems():
    listing = [("Dev/Track1_SD/B001.wav", 32000 + 44),
               ("B002.flac", 64000 + 44),
               ("README.txt", 9999),                     # non-audio: skipped
               ("B003.WAV", 44)]                         # case-insensitive extension, 0 s
    secs = P.listing_to_seconds(listing)
    assert set(secs) == {"B001", "B002", "B003"}
    assert secs["B001"] == pytest.approx(1.0)
    assert secs["B002"] == pytest.approx(2.0)
    assert secs["B003"] == 0.0


def test_listing_to_seconds_duplicate_id_raises():
    with pytest.raises(ValueError):
        P.listing_to_seconds([("a/B001.wav", 1000), ("b/B001.wav", 2000)])


def test_listing_hours():
    listing = [("B001.wav", 3600 * 32000 + 44)]          # exactly 1 hour
    assert P.listing_hours(listing) == pytest.approx(1.0)


# --------------------------------------------------------------------------------------------------- #
# projection
# --------------------------------------------------------------------------------------------------- #
def test_project_eval_dsdl_rate():
    # dev: 400 turns over 20 h -> 20 turns/h; eval 18 h -> 360 projected.
    proj = P.project_eval_dsdl(400, 20.0, 18.0)
    assert proj["rate_per_hour"] == pytest.approx(20.0)
    assert proj["projected_eval_dsdl"] == pytest.approx(360.0)


def test_project_eval_dsdl_bad_dev_hours():
    with pytest.raises(ValueError):
        P.project_eval_dsdl(100, 0.0, 10.0)


def test_project_per_session_counts_shape_and_total():
    dev = {"d1": 30, "d2": 20, "d3": 10, "d4": 0}        # dev total 60
    res = P.project_per_session_counts(dev, dev_hours=20.0, eval_hours=18.0, n_eval_sessions=4)
    # projected total = (60 / 20) * 18 = 54, distributed in dev's descending shape, rescaled to 54
    assert res["eval_total"] == pytest.approx(54.0)
    assert sum(res["per_session"]) == pytest.approx(54.0)
    # descending dev shape preserved: first session largest
    assert res["per_session"] == sorted(res["per_session"], reverse=True)


def test_dev_speech_fraction():
    assert P.dev_speech_fraction(12.0, 20.0) == pytest.approx(0.6)
    assert P.dev_speech_fraction(None, 20.0) is None
    assert P.dev_speech_fraction(12.0, 0.0) is None


# --------------------------------------------------------------------------------------------------- #
# vectorised BCa upper bound vs stats.bca_ci on the SAME resample indices (review m2)
# --------------------------------------------------------------------------------------------------- #
def _shared_idx(n_sim, n_boot, k, seed):
    """One shared (n_sim, n_boot, k) index draw, so the vectorised path and a per-row stats.bca_ci use the
    IDENTICAL bootstrap resamples (not two independent RNG streams)."""
    return np.random.default_rng(seed).integers(0, k, size=(n_sim, n_boot, k))


def test_bca_upper_vec_matches_stats_bca_ci_on_shared_indices():
    # review m2: drive both off the SAME resample indices and require agreement to bootstrap rounding, not 5e-3.
    level = 0.95
    n_boot = 3000
    rows = np.array([[0.01, -0.02, 0.03, 0.00, 0.02, -0.01, 0.015, 0.005],
                     [-0.03, 0.04, -0.01, 0.02, 0.00, 0.05, -0.02, 0.03]], dtype=np.float64)
    idx = _shared_idx(rows.shape[0], n_boot, rows.shape[1], P.SEED)
    vec = P._bca_upper_vec(rows, level, None, n_boot, idx=idx)
    for i in range(rows.shape[0]):
        clusters = [[v] for v in rows[i]]
        # feed bca_ci a stat that resamples via the SAME per-row index rows, in order
        order = {"ptr": 0}

        class _RNG:
            def integers(self, low, high, size):
                j = order["ptr"]
                order["ptr"] += 1
                return idx[i, j]
        ref = bca_ci(clusters, lambda cs: float(np.mean([c[0] for c in cs])), 2 * level - 1,
                     _RNG(), n_boot=n_boot)
        assert vec[i] == pytest.approx(ref["hi"], abs=1e-9)


def test_bca_decision_agrees_on_a_batch():
    # review m2: the decision that feeds the verdict (bound < margin) must agree between the vectorised path and
    # stats.bca_ci on a batch of rows, driven off the same indices.
    level = 0.95
    n_boot = 2000
    margin = 0.02
    rng = np.random.default_rng(7)
    rows = rng.normal(0.0, 0.01, size=(20, 12))
    idx = _shared_idx(rows.shape[0], n_boot, rows.shape[1], P.SEED)
    vec = P._bca_upper_vec(rows, level, None, n_boot, idx=idx)
    for i in range(rows.shape[0]):
        clusters = [[v] for v in rows[i]]
        order = {"ptr": 0}

        class _RNG:
            def integers(self, low, high, size):
                j = order["ptr"]; order["ptr"] += 1
                return idx[i, j]
        ref = bca_ci(clusters, lambda cs: float(np.mean([c[0] for c in cs])), 2 * level - 1, _RNG(), n_boot=n_boot)
        assert (vec[i] < margin) == (ref["hi"] < margin)


# --------------------------------------------------------------------------------------------------- #
# vectorised NI p-value vs confirm_m2b._ni_pvalue on the SAME resample indices (review m2/m3)
# --------------------------------------------------------------------------------------------------- #
def test_ni_pvalue_vec_matches_confirm_m2b_on_shared_indices():
    from csdiar.analysis import confirm_m2b as C
    n_boot = 3000
    margin = 0.02
    rows = np.array([[0.00, 0.01, -0.01, 0.005, -0.005, 0.02, -0.02, 0.0],
                     [0.03, 0.04, 0.05, 0.02, 0.06, 0.03, 0.05, 0.04]], dtype=np.float64)  # row 2: clearly over
    idx = _shared_idx(rows.shape[0], n_boot, rows.shape[1], P.SEED)
    pv = P._ni_pvalue_vec(rows, margin, None, n_boot, idx=idx)
    for i in range(rows.shape[0]):
        order = {"ptr": 0}

        class _RNG:
            def integers(self, low, high, size):
                j = order["ptr"]; order["ptr"] += 1
                return idx[i, j]
        ref = C._ni_pvalue(list(rows[i]), margin, _RNG(), n_boot=n_boot)
        assert pv[i] == pytest.approx(ref, abs=5e-3)   # same construction; bisection resolution ~1e-2 on level
    # the clearly-inferior row (mean 0.04 >> margin) gets p = 1 both ways
    assert pv[1] == pytest.approx(1.0, abs=1e-6)


def test_vectorised_power_matches_full_bca_power_within_mc_se():
    # review m3: compute H5 power on a small design BOTH ways (the vectorised sim and a loop over stats.bca_ci on
    # the same seeded draws) and require they agree within the Monte Carlo SE.
    s_mr = 0.05
    margin = min(P.MARGIN_ABS, P.MARGIN_REL * s_mr)
    k = 12
    within_var = (0.02 ** 2) * np.ones(k)
    between_var = 0.005 ** 2
    n_sim, n_boot = 150, 500
    # vectorised path (single mitigation, no Holm so it is a pure BCa-bound power to compare to the loop)
    rng = np.random.default_rng(P.SEED)
    vec = P.simulate_h5_power([100.0] * k, s_mr, within_var, between_var, rng, n_sim=n_sim, n_boot=n_boot,
                              n_mitigations=1, holm_in_sim=False)
    vec_power = vec["power_per_mitigation"][0]
    # reference loop with stats.bca_ci on the SAME drawn datasets
    rng2 = np.random.default_rng(P.SEED)
    within_sd = np.sqrt(within_var); between_sd = np.sqrt(between_var)
    hits = 0
    for _ in range(n_sim):
        sess = rng2.normal(0.0, between_sd, size=k)
        noise = rng2.normal(0.0, 1.0, size=k) * within_sd
        data = sess + noise
        ci = bca_ci([[v] for v in data], lambda cs: float(np.mean([c[0] for c in cs])),
                    2 * P.ONE_SIDED_LEVEL - 1, np.random.default_rng(P.SEED), n_boot=n_boot)
        if ci["hi"] < margin:
            hits += 1
    ref_power = hits / n_sim
    se = (vec_power * (1 - vec_power) / n_sim) ** 0.5
    assert abs(vec_power - ref_power) <= 3 * max(se, 1e-3) + 0.05   # same family; different draw order tolerated


# --------------------------------------------------------------------------------------------------- #
# the new H5 simulation: joint Holm, per-mitigation + family power
# --------------------------------------------------------------------------------------------------- #
def test_holm_reject_batch_matches_scalar_holm():
    from csdiar.analysis.stats import holm
    pv = np.array([[0.01, 0.2, 0.004], [0.5, 0.6, 0.7], [0.02, 0.02, 0.02]])
    rej = P._holm_reject_batch(pv, alpha=0.05)
    for i in range(pv.shape[0]):
        adj = holm(pv[i])
        assert list(rej[i]) == [a < 0.05 for a in adj]


def test_holm_in_sim_rejects_less_than_raw_bound_at_three_mitigations():
    # review round 2, m3: both paths draw the same datasets and resamples from the same seed, and a Holm-adjusted
    # p < .05 implies a raw p < .05 (bound below the margin), so Holm can only reject less; on a borderline design
    # it rejects strictly less for the family.
    k = 12
    kw = dict(s_mr=0.20, within_var=(0.027 ** 2) * np.ones(k), between_var=(0.008 ** 2), n_sim=200, n_boot=400,
              n_mitigations=3)
    holm = P.simulate_h5_power([60.0] * k, rng=np.random.default_rng(P.SEED), holm_in_sim=True, **kw)
    raw = P.simulate_h5_power([60.0] * k, rng=np.random.default_rng(P.SEED), holm_in_sim=False, **kw)
    for h, r in zip(holm["power_per_mitigation"], raw["power_per_mitigation"]):
        assert h <= r + 1.0 / kw["n_sim"]
    assert 0.05 < raw["power_family"] < 0.95                    # borderline, so the comparison means something
    assert holm["power_family"] < raw["power_family"]


def test_h5_power_high_for_large_clean_design():
    # many sessions, lots of turns each, tiny spread: true diff 0 << margin -> family + per-mitigation power ~ 1.
    rng = np.random.default_rng(P.SEED)
    k = 40
    within_var = (0.002 ** 2) * np.ones(k)   # tiny within-session noise
    res = P.simulate_h5_power([200.0] * k, s_mr=0.05, within_var=within_var, between_var=(0.002 ** 2), rng=rng,
                              n_sim=300, n_boot=600)
    assert res["power_family"] > 0.9
    assert min(res["power_per_mitigation"]) > 0.9
    assert res["design_effect"] >= 1.0


def test_h5_not_assessable_too_few_sessions():
    rng = np.random.default_rng(P.SEED)
    res = P.simulate_h5_power([200.0, 200.0, 200.0], s_mr=0.05, within_var=(0.01 ** 2), between_var=(0.01 ** 2),
                              rng=rng, n_sim=100, n_boot=200)
    assert "not_assessable" in res
    assert res["power_family"] != res["power_family"]    # nan


def test_h5_margin_is_min_of_abs_and_rel():
    rng = np.random.default_rng(P.SEED)
    res_big = P.simulate_h5_power([100.0] * 10, s_mr=0.20, within_var=(0.01 ** 2), between_var=(0.01 ** 2),
                                  rng=rng, n_sim=40, n_boot=100)
    assert res_big["margin"] == pytest.approx(0.02)      # 0.25*0.20=0.05 > 0.02 -> abs binds
    res_small = P.simulate_h5_power([100.0] * 10, s_mr=0.04, within_var=(0.01 ** 2), between_var=(0.01 ** 2),
                                    rng=rng, n_sim=40, n_boot=100)
    assert res_small["margin"] == pytest.approx(0.01)    # 0.25*0.04=0.01 < 0.02 -> rel binds


def test_h5_size_near_nominal_when_true_effect_at_margin():
    # review m3: with the TRUE per-session mean sitting on the non-inferiority boundary, a single mitigation's
    # BCa-bound rejection probability is ~ the nominal one-sided level 0.05. Tightened band vs the old 0.01-0.12.
    rng = np.random.default_rng(P.SEED)
    s_mr = 0.08
    margin = min(P.MARGIN_ABS, P.MARGIN_REL * s_mr)      # 0.02
    k = 30
    within = np.sqrt((0.02 ** 2)) * np.ones(k)
    n_sim, n_boot = 800, 800
    data = margin + rng.normal(0.0, 1.0, size=(n_sim, k)) * within[None, :]
    upper = P._bca_upper_vec(data, P.ONE_SIDED_LEVEL, rng, n_boot)
    size = (np.isfinite(upper) & (upper < margin)).mean()
    assert 0.02 <= size <= 0.09                          # tightened around the nominal 0.05


def test_h5_deterministic_under_seed():
    k = 20
    wv = (0.01 ** 2) * np.ones(k)
    a = P.simulate_h5_power([120.0] * k, 0.06, wv, (0.01 ** 2), np.random.default_rng(P.SEED),
                            n_sim=200, n_boot=300)
    b = P.simulate_h5_power([120.0] * k, 0.06, wv, (0.01 ** 2), np.random.default_rng(P.SEED),
                            n_sim=200, n_boot=300)
    assert a["power_family"] == b["power_family"]
    assert a["power_per_mitigation"] == b["power_per_mitigation"]


# --------------------------------------------------------------------------------------------------- #
# variance modelling from paired dev counts (review M1/M2)
# --------------------------------------------------------------------------------------------------- #
def test_model_variance_parts_within_and_between():
    # 6 sessions, n=20 turns each; 2 discordant turns per session -> within-var = 2/400 = 0.005 each.
    paired = {f"s{i}": {"n_dsdl": 20, "missed_by_mit_only": 1, "missed_by_b0_only": 1,
                        "missed_by_both": 2, "matched_by_both": 16} for i in range(6)}
    # per-session ΔMR = (missed_M - missed_B0)/n = 0 for all -> observed diff var 0 -> between floored at 0.
    per_diff = {f"s{i}": 0.0 for i in range(6)}
    vp = P.model_variance_parts(paired, per_diff)
    assert vp["within_var_mean"] == pytest.approx(2.0 / 400.0)
    assert vp["between_var"] == pytest.approx(0.0)        # MoM: observed (0) - within (>0), floored at 0
    assert vp["discordance_rate"] == pytest.approx(2.0 / 20.0)


def test_model_variance_parts_positive_between():
    # inject real per-session spread in the difference; MoM between = observed var - mean within, floored.
    paired = {f"s{i}": {"n_dsdl": 100, "missed_by_mit_only": 1, "missed_by_b0_only": 1} for i in range(8)}
    per_diff = {f"s{i}": 0.10 * (i - 3.5) for i in range(8)}   # wide spread -> positive between
    vp = P.model_variance_parts(paired, per_diff)
    assert vp["within_var_mean"] == pytest.approx(2.0 / 10000.0)
    assert vp["between_var"] > 0.0


def test_project_within_var_scales_inverse_with_turns():
    wv = P._project_within_var([10.0, 100.0, 0.0], discordance_rate=0.1)
    assert wv[0] == pytest.approx(0.1 / 10.0)
    assert wv[1] == pytest.approx(0.1 / 100.0)
    assert not np.isfinite(wv[2])                        # zero turns -> nan


# --------------------------------------------------------------------------------------------------- #
# SS-DL check between-session SD derived from dev per-session ΔFAR (review m4)
# --------------------------------------------------------------------------------------------------- #
def test_ssdl_between_sd_from_dev():
    dfar = {"a": 0.02, "b": 0.04, "c": 0.06}
    out = P.ssdl_between_sd_from_dev(dfar)
    assert out["source"] == "dev_delta_far"
    assert out["between_sd"] == pytest.approx(np.std([0.02, 0.04, 0.06], ddof=1))
    assert out["n_sessions"] == 3


def test_ssdl_between_sd_assumed_when_too_few():
    out = P.ssdl_between_sd_from_dev({"a": 0.02})
    assert out["source"] == "assumed"
    assert out["between_sd"] != out["between_sd"]        # nan


def test_simulate_ssdl_determinism_and_not_assessable():
    rng = np.random.default_rng(P.SEED)
    a = P.simulate_ssdl_power([50.0] * 8, 0.05, 0.02, np.random.default_rng(P.SEED), n_sim=100, n_boot=100)
    b = P.simulate_ssdl_power([50.0] * 8, 0.05, 0.02, np.random.default_rng(P.SEED), n_sim=100, n_boot=100)
    assert a["power"] == b["power"]
    na = P.simulate_ssdl_power([50.0] * 3, 0.05, 0.02, rng, n_sim=50, n_boot=50)
    assert "not_assessable" in na


def test_simulate_ssdl_power_rises_with_effect_and_falls_with_spread():
    # review round 2, m4: a sign error in the loop would break these orderings.
    def power(effect, between_sd):
        return P.simulate_ssdl_power([50.0] * 8, effect, between_sd, np.random.default_rng(P.SEED),
                                     n_sim=80)["power"]
    assert power(0.10, 0.02) > power(0.005, 0.02) + 0.3
    assert power(0.05, 0.005) > power(0.05, 0.20) + 0.3


def test_verdict_follows_power_scope(monkeypatch):
    # review round 2, m2: each mitigation powered (weakest 0.85) but all three together only 0.60, so the verdict
    # is confirmatory under "per_mitigation" and descriptive under the stricter "family" scope.
    fixed = {"power_per_mitigation": [0.85, 0.90, 0.88], "power_per_mitigation_se": [0.01] * 3,
             "power_family": 0.60, "mc_se_family": 0.01, "design_effect": 1.2, "margin": 0.02,
             "n_sessions_included": 32}
    monkeypatch.setattr(P, "simulate_h5_power", lambda *a, **k: dict(fixed))
    recs = [f"D{i:03d}" for i in range(35)]
    tuning = _tuning_with_per_session(recs, {r: 0.05 for r in recs})
    summary = {"split": "dev", "recordings": [{"recording": r, "n_dsdl": 300, "n_ssdl": 400} for r in recs]}
    eval_listing = [(f"E{i:03d}.wav", int(0.5 * 3600 * 32000) + 44) for i in range(32)]
    verdicts = {}
    for scope in ("per_mitigation", "family"):
        monkeypatch.setattr(P, "POWER_SCOPE", scope)
        rec = P.run_power_check(summary, tuning, eval_listing, dev_speech_hours=None, dev_audio_hours=None,
                                n_sim=20, n_boot=40, n_sim_sensitivity=10, n_boot_sensitivity=20)
        verdicts[scope] = (rec["verdict"], rec["power_scope_used"])
    assert verdicts == {"per_mitigation": ("confirmatory", "per_mitigation"), "family": ("descriptive", "family")}


# --------------------------------------------------------------------------------------------------- #
# dev-only guard: no eval reference / count is read
# --------------------------------------------------------------------------------------------------- #
def test_dev_counts_refuses_eval_split():
    with pytest.raises(ValueError):
        P.dev_counts_from_summary({"split": "eval", "recordings": []})


def test_dev_counts_from_dev_summary():
    summary = {"split": "dev", "recordings": [
        {"recording": "B001", "n_dsdl": 30, "n_ssdl": 50},
        {"recording": "B002", "n_dsdl": 12, "n_ssdl": 20}]}
    c = P.dev_counts_from_summary(summary)
    assert c["dsdl_total"] == 42 and c["ssdl_total"] == 70
    assert c["dsdl_per_session"] == {"B001": 30, "B002": 12}


# --------------------------------------------------------------------------------------------------- #
# tuning-record readers + the B0 consistency refusal (review m1)
# --------------------------------------------------------------------------------------------------- #
def _tuning_with_per_session(recs, mr, dfar=None, n_dsdl=300, paired_mo=2, paired_bo=2, paired_n=300,
                             with_paired=True, b0_mr=None):
    dfar = dfar or {r: 0.03 for r in recs}
    b0_ps = {r: {"mr_dsdl": mr[r], "delta_far_std": dfar[r], "far_ssdl": 0.1,
                 "n_dsdl": n_dsdl, "n_ssdl_with_cell_controls": 40} for r in recs}
    pso = {"B0": {"per_session": b0_ps, "inclusion": {"h4_ssdl": recs, "h5_dsdl": recs}}}
    for mit in ("M1", "M2", "M3"):
        entry = {"per_session": b0_ps, "inclusion": {"h4_ssdl": recs, "h5_dsdl": recs}}
        if with_paired:
            entry["paired_turns_vs_b0"] = {r: {"n_dsdl": paired_n, "missed_by_mit_only": paired_mo,
                                               "missed_by_b0_only": paired_bo, "missed_by_both": 3,
                                               "matched_by_both": paired_n - paired_mo - paired_bo - 3}
                                           for r in recs}
        else:
            entry["paired_turns_vs_b0"] = None
        pso[mit] = entry
    mean_mr = sum(mr.values()) / len(mr)
    return {"operating_points": {"B0": {"mr_dsdl": b0_mr if b0_mr is not None else mean_mr}},
            "per_session_at_operating_points": pso}


def test_check_b0_mr_consistency_passes_and_refuses():
    recs = [f"D{i:03d}" for i in range(6)]
    mr = {r: 0.05 for r in recs}
    good = _tuning_with_per_session(recs, mr)                 # b0_mr = mean = 0.05
    out = P.check_b0_mr_consistency(good)
    assert out["diff"] <= 1e-9
    # a record whose B0 MR disagrees with the per-session mean is refused
    bad = _tuning_with_per_session(recs, mr, b0_mr=0.07)
    with pytest.raises(ValueError, match="disagrees"):
        P.check_b0_mr_consistency(bad)


def test_run_power_check_refuses_inconsistent_record():
    recs = [f"D{i:03d}" for i in range(12)]
    mr = {r: 0.05 for r in recs}
    tuning = _tuning_with_per_session(recs, mr, b0_mr=0.09)   # inconsistent
    summary = {"split": "dev", "recordings": [{"recording": r, "n_dsdl": 300, "n_ssdl": 400} for r in recs]}
    eval_listing = [(f"E{i:03d}.wav", int(0.6 * 3600 * 32000) + 44) for i in range(16)]
    with pytest.raises(ValueError, match="disagrees"):
        P.run_power_check(summary, tuning, eval_listing, dev_speech_hours=15.0, dev_audio_hours=20.0,
                          n_sim=50, n_boot=100)


# --------------------------------------------------------------------------------------------------- #
# end to end: listing-only projection, no eval count read; verdict threshold; fallback flag
# --------------------------------------------------------------------------------------------------- #
def test_run_power_check_uses_listing_sizes_not_counts():
    recs = [f"D{i:03d}" for i in range(35)]
    mr = {r: 0.05 for r in recs}
    tuning = _tuning_with_per_session(recs, mr)
    summary = {"split": "dev", "recordings": [{"recording": r, "n_dsdl": 300, "n_ssdl": 400} for r in recs]}
    eval_listing = [(f"E{i:03d}.wav", int(0.5 * 3600 * 32000) + 44) for i in range(32)]   # 16 audio h
    rec = P.run_power_check(summary, tuning, eval_listing, dev_speech_hours=None, dev_audio_hours=None,
                            n_sim=40, n_boot=80, n_sim_sensitivity=20, n_boot_sensitivity=50)
    assert rec["projection"]["eval_audio_hours"] == pytest.approx(16.0)
    assert any("dev hours" in f or "speech hours" in f for f in rec["projection"]["flags"])
    assert rec["verdict"] in ("confirmatory", "descriptive")


def test_run_power_check_speech_projection():
    recs = [f"D{i:03d}" for i in range(20)]
    mr = {r: 0.05 for r in recs}
    tuning = _tuning_with_per_session(recs, mr, n_dsdl=25)
    summary = {"split": "dev", "recordings": [{"recording": r, "n_dsdl": 25, "n_ssdl": 40} for r in recs]}
    eval_listing = [(f"E{i:03d}.wav", int(0.5 * 3600 * 32000) + 44) for i in range(20)]   # 10 audio h
    rec = P.run_power_check(summary, tuning, eval_listing, dev_speech_hours=12.0, dev_audio_hours=20.0,
                            n_sim=60, n_boot=120, n_sim_sensitivity=30, n_boot_sensitivity=60)
    assert rec["projection"]["base"] == "speech"
    assert rec["projection"]["dev_speech_fraction"] == pytest.approx(0.6)
    assert rec["projection"]["eval_base_hours"] == pytest.approx(6.0)
    assert rec["projection"]["projected_eval_dsdl"] == pytest.approx((500 / 12.0) * 6.0)


def test_run_power_check_falls_back_and_flags_when_paired_missing():
    # review M1/M2: a mitigation with no paired dev counts falls back to the B0-spread proxy, FLAGGED.
    recs = [f"D{i:03d}" for i in range(35)]
    mr = {r: 0.05 + 0.01 * ((i % 5) - 2) for i, r in enumerate(recs)}   # real B0 spread for the proxy
    tuning = _tuning_with_per_session(recs, mr, with_paired=False)
    summary = {"split": "dev", "recordings": [{"recording": r, "n_dsdl": 300, "n_ssdl": 400} for r in recs]}
    eval_listing = [(f"E{i:03d}.wav", int(0.6 * 3600 * 32000) + 44) for i in range(32)]
    rec = P.run_power_check(summary, tuning, eval_listing, dev_speech_hours=15.0, dev_audio_hours=20.0,
                            n_sim=40, n_boot=100, n_sim_sensitivity=20, n_boot_sensitivity=50)
    flags = rec["variance_model"]["flags"]
    assert any("paired dev DS-DL counts missing" in f for f in flags)
    for mit in ("M1", "M2", "M3"):
        assert rec["variance_model"]["per_mitigation"][mit]["source"] == "b0_mr_sd_fallback"


def test_verdict_confirmatory_for_well_powered_design():
    recs = [f"D{i:03d}" for i in range(35)]
    mr = {r: 0.05 for r in recs}
    tuning = _tuning_with_per_session(recs, mr, n_dsdl=300, paired_mo=2, paired_bo=2, paired_n=300)
    summary = {"split": "dev", "recordings": [{"recording": r, "n_dsdl": 300, "n_ssdl": 400} for r in recs]}
    eval_listing = [(f"E{i:03d}.wav", int(0.6 * 3600 * 32000) + 44) for i in range(32)]
    rec = P.run_power_check(summary, tuning, eval_listing, dev_speech_hours=15.0, dev_audio_hours=20.0,
                            n_sim=60, n_boot=120, n_sim_sensitivity=20, n_boot_sensitivity=50)
    assert rec["power_family"] >= 0.8
    assert rec["verdict"] == "confirmatory"
    assert set(rec["sensitivity_between_var"]) == {"0.5x", "1x", "2x"}
    assert len(rec["power_per_mitigation"]) == 3


def test_verdict_descriptive_when_not_assessable():
    recs = [f"D{i:03d}" for i in range(35)]
    mr = {r: 0.05 for r in recs}
    tuning = _tuning_with_per_session(recs, mr, n_dsdl=5)
    summary = {"split": "dev", "recordings": [{"recording": r, "n_dsdl": 5, "n_ssdl": 8} for r in recs]}
    eval_listing = [(f"E{i:03d}.wav", 44) for i in range(32)]   # 0 s each -> 0 projected turns
    rec = P.run_power_check(summary, tuning, eval_listing, dev_speech_hours=15.0, dev_audio_hours=20.0,
                            n_sim=100, n_boot=200)
    assert rec["verdict"] == "descriptive"


# --------------------------------------------------------------------------------------------------- #
# m5: n_sessions_ge_min straddling the floor, and the k<5 end-to-end descriptive path
# --------------------------------------------------------------------------------------------------- #
def test_project_per_session_counts_n_sessions_ge_min_straddles_floor():
    # review m5: a dev shape whose projection puts some sessions above and some below the 10-turn floor.
    dev = {f"d{i}": v for i, v in enumerate([40, 30, 20, 10, 5, 2])}   # dev total 107
    # project onto 6 eval sessions, rescale so the total lands where ~half clear the floor
    res = P.project_per_session_counts(dev, dev_hours=20.0, eval_hours=20.0, n_eval_sessions=6)
    per = res["per_session"]
    assert res["n_sessions_ge_min"] == sum(1 for c in per if c >= P.MIN_DSDL_PER_SESSION)
    assert 0 < res["n_sessions_ge_min"] < 6          # genuinely straddles the floor


def test_run_power_check_k_below_5_is_descriptive_with_reason():
    # review m5 end to end: a projection yielding < 5 included sessions gives verdict 'descriptive' with the
    # not_assessable reason surfaced in h5_power.
    recs = [f"D{i:03d}" for i in range(6)]
    mr = {r: 0.05 for r in recs}
    tuning = _tuning_with_per_session(recs, mr, n_dsdl=20)
    summary = {"split": "dev", "recordings": [{"recording": r, "n_dsdl": 20, "n_ssdl": 40} for r in recs]}
    # 4 eval sessions, small hours -> few projected turns, < 5 clear the floor
    eval_listing = [(f"E{i:03d}.wav", int(0.05 * 3600 * 32000) + 44) for i in range(4)]
    rec = P.run_power_check(summary, tuning, eval_listing, dev_speech_hours=15.0, dev_audio_hours=20.0,
                            n_sim=40, n_boot=80, n_sim_sensitivity=20, n_boot_sensitivity=50)
    assert rec["verdict"] == "descriptive"
    assert "not_assessable" in rec["h5_power"]


# --------------------------------------------------------------------------------------------------- #
# Reviewer M1: power at realistic projected counts near the 10-30 turn floor
# --------------------------------------------------------------------------------------------------- #
def test_power_at_realistic_low_n_projected_counts():
    # review M1: pin how power behaves at a dev-shaped tile near the 10-30 turn floor. With few turns per
    # session the within-session noise is large relative to the 0.02 margin, so power is well BELOW 1 -- the
    # low-n regime that decides a borderline freeze, which the large-n test never exercises.
    rng = np.random.default_rng(P.SEED)
    k = 15
    counts = [12.0 + (i % 4) * 5 for i in range(k)]   # 12..27 projected DS-DL turns per session
    # within-session variance at these counts from a dev discordance rate of ~0.1 turns/turn
    within_var = P._project_within_var(counts, discordance_rate=0.1)
    low = P.simulate_h5_power(counts, s_mr=0.05, within_var=within_var, between_var=(0.01 ** 2), rng=rng,
                              n_sim=400, n_boot=600)
    # a large-n, low-discordance clean design for contrast (many turns AND a small discordance rate, so the
    # within-session noise is well below the 0.0125 margin and the design is well powered).
    rng2 = np.random.default_rng(P.SEED)
    big_counts = [400.0] * k
    big_within = P._project_within_var(big_counts, discordance_rate=0.01)
    big = P.simulate_h5_power(big_counts, s_mr=0.05, within_var=big_within, between_var=(0.002 ** 2), rng=rng2,
                              n_sim=400, n_boot=600)
    assert low["power_family"] < big["power_family"]          # low-n is strictly less powered
    assert big["power_family"] > 0.8                          # the clean large-n design is well powered
    # and the low-n family power is materially under 1 -- the regime that decides a borderline freeze
    assert low["power_family"] < 0.95


# --------------------------------------------------------------------------------------------------- #
# record carries hashes, seed, params, and the new fields
# --------------------------------------------------------------------------------------------------- #
def test_record_has_hashes_seed_and_params():
    recs = [f"D{i:03d}" for i in range(20)]
    mr = {r: 0.05 for r in recs}
    tuning = _tuning_with_per_session(recs, mr, n_dsdl=25)
    summary = {"split": "dev", "recordings": [{"recording": r, "n_dsdl": 25, "n_ssdl": 40} for r in recs]}
    eval_listing = [(f"E{i:03d}.wav", int(0.5 * 3600 * 32000) + 44) for i in range(20)]
    rec = P.run_power_check(summary, tuning, eval_listing, dev_speech_hours=12.0, dev_audio_hours=20.0,
                            n_sim=40, n_boot=100, n_sim_sensitivity=20, n_boot_sensitivity=50,
                            input_hashes={"x": "deadbeef"})
    assert rec["seed"] == P.SEED
    assert P.GUARDED_SELF in rec["code_sha256"]
    assert rec["input_sha256"]["x"] == "deadbeef"
    assert rec["choices"]["project_per"] == "speech"
    assert rec["choices"]["holm_in_sim"] is True
    assert "margin_abs" in rec["constants"]
    assert "variance_model" in rec and "sensitivity_between_var" in rec
    assert rec["ssdl_power"]["between_sd_source"] in ("dev_delta_far", "assumed_prior")
    json.dumps(rec, default=P._json_default)


# --------------------------------------------------------------------------------------------------- #
# review round 3: shared seed (m1), family floor (m2), per-mitigation within variance (m3),
# projection shape (m4), SS-DL within-SD flag (m5)
# --------------------------------------------------------------------------------------------------- #
def test_sensitivity_1x_row_reproduces_primary():
    # m1: the primary run and every sensitivity row start from one fresh seed, so at the primary's n_sim/n_boot
    # the 1x row IS the primary.
    recs = [f"D{i:03d}" for i in range(35)]
    mr = {r: 0.05 for r in recs}
    tuning = _tuning_with_per_session(recs, mr, n_dsdl=25, paired_mo=3, paired_bo=3, paired_n=25)
    summary = {"split": "dev", "recordings": [{"recording": r, "n_dsdl": 25, "n_ssdl": 40} for r in recs]}
    eval_listing = [(f"E{i:03d}.wav", int(0.6 * 3600 * 32000) + 44) for i in range(32)]
    rec = P.run_power_check(summary, tuning, eval_listing, dev_speech_hours=15.0, dev_audio_hours=20.0,
                            n_sim=40, n_boot=80)
    one = rec["sensitivity_between_var"]["1x"]
    assert one["power_family"] == rec["power_family"]
    assert one["power_per_mitigation"] == rec["power_per_mitigation"]


def test_h5_identical_within_rows_reproduce_shared_vector():
    # m3: a 2-D within-variance input with identical rows gives exactly the shared-vector result.
    k = 12
    counts = [60.0] * k
    wv = P._project_within_var(counts, discordance_rate=0.1)
    kw = dict(s_mr=0.05, between_var=(0.004 ** 2), n_sim=60, n_boot=120)
    a = P.simulate_h5_power(counts, within_var=wv, rng=np.random.default_rng(P.SEED), **kw)
    b = P.simulate_h5_power(counts, within_var=np.stack([wv, wv, wv]), rng=np.random.default_rng(P.SEED), **kw)
    assert a["power_per_mitigation"] == b["power_per_mitigation"]
    assert a["power_family"] == b["power_family"]


def test_h5_noisier_mitigation_gets_lower_power():
    # m3: each mitigation is sized with its OWN within-session variance, so the noisy one is less powered.
    k = 20
    counts = [200.0] * k
    quiet = P._project_within_var(counts, discordance_rate=0.02)
    noisy = P._project_within_var(counts, discordance_rate=0.5)
    res = P.simulate_h5_power(counts, s_mr=0.05, within_var=np.stack([quiet, quiet, noisy]),
                              between_var=(0.002 ** 2), rng=np.random.default_rng(P.SEED), n_sim=200, n_boot=300)
    p = res["power_per_mitigation"]
    assert p[2] < p[0] - 0.3 and p[2] < p[1] - 0.3
    assert res["within_var_mean_per_mitigation"][2] > res["within_var_mean_per_mitigation"][0]
    with pytest.raises(ValueError):
        P.simulate_h5_power(counts, s_mr=0.05, within_var=np.stack([quiet, noisy]), between_var=0.0,
                            rng=np.random.default_rng(P.SEED), n_sim=10, n_boot=20)


def test_run_power_check_sizes_each_mitigation_from_its_own_rate():
    # m3 end to end: M3 discords with B0 ten times as often as M1; M2 has no paired counts and takes the mean
    # rate of M1 and M3, flagged.
    recs = [f"D{i:03d}" for i in range(35)]
    mr = {r: 0.05 for r in recs}
    tuning = _tuning_with_per_session(recs, mr, n_dsdl=300, paired_mo=2, paired_bo=2, paired_n=300)
    pso = tuning["per_session_at_operating_points"]
    pso["M3"]["paired_turns_vs_b0"] = {r: {"n_dsdl": 300, "missed_by_mit_only": 20, "missed_by_b0_only": 20,
                                           "missed_by_both": 3, "matched_by_both": 257} for r in recs}
    pso["M2"]["paired_turns_vs_b0"] = None
    summary = {"split": "dev", "recordings": [{"recording": r, "n_dsdl": 300, "n_ssdl": 400} for r in recs]}
    eval_listing = [(f"E{i:03d}.wav", int(0.6 * 3600 * 32000) + 44) for i in range(32)]
    rec = P.run_power_check(summary, tuning, eval_listing, dev_speech_hours=15.0, dev_audio_hours=20.0,
                            n_sim=20, n_boot=40, n_sim_sensitivity=10, n_boot_sensitivity=20)
    vm = rec["variance_model"]
    assert vm["within_model"] == "paired_dev_rate_per_mitigation"
    rates = vm["discordance_rate_per_mitigation"]
    assert rates["M1"] == pytest.approx(4 / 300)
    assert rates["M3"] == pytest.approx(40 / 300)
    assert rates["M2"] == pytest.approx((4 / 300 + 40 / 300) / 2)
    assert any(f.startswith("M2:") and "mean rate" in f for f in vm["flags"])
    w = vm["within_var_mean_projected_per_mitigation"]
    assert w["M3"] > w["M2"] > w["M1"]


def test_projection_interpolates_dev_shape_onto_fewer_eval_sessions():
    # m4: 35 dev -> 32 eval sessions keeps the turn-poor tail; tiling dropped three of the five small sessions.
    dev = {f"d{i:02d}": (100 if i < 30 else 1) for i in range(35)}
    res = P.project_per_session_counts(dev, dev_hours=20.0, eval_hours=20.0, n_eval_sessions=32)
    per = res["per_session"]
    assert len(per) == 32 and sum(per) == pytest.approx(float(sum(dev.values())))
    assert sum(1 for c in per if c < 0.05 * max(per)) >= 4
    assert per == sorted(per, reverse=True)
    assert res["shape_method"].startswith("quantile_interpolation")
    same = P.project_per_session_counts({"a": 30, "b": 20, "c": 10}, dev_hours=10.0, eval_hours=10.0,
                                        n_eval_sessions=3)
    assert same["per_session"] == pytest.approx([30.0, 20.0, 10.0])
    assert same["shape_method"] == "dev_counts"


def test_record_marks_assumptions_and_refuses_other_families():
    # m2 / m5: the record says the family power is a floor and the SS-DL within-session SD is assumed; m1: the
    # streams are named. H5's family is M1-M3, so another list of mitigations is refused.
    recs = [f"D{i:03d}" for i in range(20)]
    mr = {r: 0.05 for r in recs}
    tuning = _tuning_with_per_session(recs, mr, n_dsdl=25)
    summary = {"split": "dev", "recordings": [{"recording": r, "n_dsdl": 25, "n_ssdl": 40} for r in recs]}
    eval_listing = [(f"E{i:03d}.wav", int(0.5 * 3600 * 32000) + 44) for i in range(20)]
    kw = dict(dev_speech_hours=12.0, dev_audio_hours=20.0, n_sim=20, n_boot=40, n_sim_sensitivity=10,
              n_boot_sensitivity=20)
    rec = P.run_power_check(summary, tuning, eval_listing, **kw)
    assert "floor" in rec["power_family_note"]
    assert "independent" in rec["h5_power"]["family_power_assumption"]
    assert rec["ssdl_power"]["within_sd_source"] == "assumed_binomial_proxy"
    assert rec["rng_streams"]["h5_primary_and_sensitivity"] == f"default_rng({P.SEED})"
    json.dumps(rec, default=P._json_default)
    with pytest.raises(ValueError, match="family"):
        P.run_power_check(summary, tuning, eval_listing, mitigations=("M1", "M2"), **kw)
