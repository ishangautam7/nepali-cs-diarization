"""Tests for scripts/review_channel_exclusion.py (post hoc leave-one-channel-out sensitivity)."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from csdiar.analysis.pilot import stratified_auc

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("review_channel_exclusion",
                                               ROOT / "scripts" / "review_channel_exclusion.py")
rce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rce)


def _toy_sets(seed=1, k=5):
    rng = np.random.default_rng(seed)
    return {f"c{i}": rng.normal(0.03 + 0.01 * i, 0.1, size=12 + 7 * i) for i in range(k)}


def _summary(sets):
    return {c: {"n": len(v), "mean": float(v.mean()), "sd": float(v.std(ddof=1))} for c, v in sets.items()}


def test_h1_estimates_match_the_frozen_definitions():
    sets = _toy_sets()
    est = rce.h1_estimates(_summary(sets))
    clusters = list(sets.values())
    within = math.sqrt(sum((len(c) - 1) * np.var(c, ddof=1) for c in clusters) / sum(len(c) - 1 for c in clusters))
    assert est["channel_weighted"] == pytest.approx(np.mean([c.mean() for c in clusters]) / within, abs=1e-12)
    allv = np.concatenate(clusters)
    assert est["switch_weighted"] == pytest.approx(allv.mean() / allv.std(ddof=1), abs=1e-12)
    assert est["n_units"] == len(allv) and est["n_channels"] == 5


def test_switch_weighted_auc_equals_the_auc_pooled_over_switch_points():
    rng = np.random.default_rng(2)
    chans, pooled_pos, pooled_neg = {}, {}, {}
    for i in range(4):
        pos = {("v", b): list(rng.normal(0.2, 1, 5 + 3 * i + b)) for b in range(3)}
        neg = {("v", b): list(rng.normal(0.0, 1, 20)) for b in range(3)}
        auc, n = stratified_auc(pos, neg)
        chans[f"c{i}"] = {"auc": auc, "n_ssdl": n}
        pooled_pos.update({(f"c{i}",) + k: v for k, v in pos.items()})
        pooled_neg.update({(f"c{i}",) + k: v for k, v in neg.items()})
    est = rce.auc_estimates(chans)
    assert est["switch_weighted"] == pytest.approx(stratified_auc(pooled_pos, pooled_neg)[0], abs=1e-12)
    assert est["channel_weighted"] == pytest.approx(np.mean([v["auc"] for v in chans.values()]), abs=1e-12)


def test_loco_fits_every_channel_once_and_summary_flags_crossings():
    chans = {"a": {"auc": 0.62, "n_ssdl": 10}, "b": {"auc": 0.56, "n_ssdl": 30}, "c": {"auc": 0.52, "n_ssdl": 20},
             "d": {"auc": 0.54, "n_ssdl": 15}}
    res = rce.loco(chans, rce.auc_estimates)
    assert sorted(res["leave_one_out"]) == ["a", "b", "c", "d"]
    assert res["leave_one_out"]["a"]["channel_weighted"] == pytest.approx((0.56 + 0.52 + 0.54) / 3)
    s = rce.summarise(res, "h2p0_auc", {c: v["n_ssdl"] for c, v in chans.items()})
    cw = s["channel_weighted"]
    assert cw["full"] == pytest.approx(0.56) and cw["full_meets_sesoi"]
    assert cw["min_without"] == "a" and cw["max_without"] == "c"
    assert cw["sesoi_crossings"] == ["a"]  # without a: 0.54 < 0.55
    assert cw["direction_changes"] == []
    assert s["units_of_left_out_channel"] == {"a": 10, "b": 30, "c": 20, "d": 15}


def test_direction_change_is_relative_to_the_full_sample():
    chans = {"a": {"auc": 0.40, "n_ssdl": 10}, "b": {"auc": 0.48, "n_ssdl": 10}, "c": {"auc": 0.58, "n_ssdl": 10}}
    s = rce.summarise(rce.loco(chans, rce.auc_estimates), "det_auc", {})["channel_weighted"]
    assert s["full"] < 0.5
    assert s["direction_changes"] == ["a"]  # without a the mean rises above 0.5


def test_h3_estimates_scale_and_test_the_predicted_negative_direction():
    est = rce.h3_estimates({"a": -0.02, "b": -0.03, "c": -0.01, "d": -0.04}, 0.1)
    assert est["channel_weighted"] == pytest.approx(-0.25)
    assert est["p_one"] == pytest.approx(1 / 16)  # all four negative: only the observed pattern is as extreme
    assert est["switch_weighted"] is None


def test_rel_never_returns_a_home_path():
    assert rce.rel(ROOT / "results" / "x.json") == "results/x.json"
    assert rce.rel(ROOT.parent / "other" / "y.npz") == "../other/y.npz"
