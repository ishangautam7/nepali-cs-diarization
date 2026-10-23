"""Tests for the confirmatory helpers in csdiar.analysis.confirm and csdiar.analysis.attribution."""
import math

import numpy as np

from csdiar.analysis.attribution import adjusted_effects
from csdiar.analysis.confirm import check_freeze, dz_stat, included, plain_dz, pooled_sd


def test_pooled_sd_and_dz_by_hand():
    a, b = np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0])  # variances 1 and 2, weights 2 and 1
    assert abs(pooled_sd([a, b]) - math.sqrt((2 * 1 + 1 * 2) / 3)) < 1e-12
    assert abs(dz_stat([a, b]) - ((2.0 + 3.0) / 2) / math.sqrt(4 / 3)) < 1e-12
    allv = np.array([1.0, 2.0, 3.0, 2.0, 4.0])
    assert abs(plain_dz([a, b]) - allv.mean() / allv.std(ddof=1)) < 1e-12


def test_included_splits_by_minimum():
    inc, exc = included({"a": list(range(10)), "b": list(range(9))}, minimum=10)
    assert list(inc) == ["a"] and exc == {"b": 9}


def test_check_freeze_dry_run_on_validation():
    assert check_freeze(["validation"])["mode"].startswith("dry run")


def test_adjusted_effects_recovers_condition_coefficient():
    rng = np.random.default_rng(0)
    rows = []
    for ch in range(6):
        for s in range(30):
            base = rng.normal()
            for cond in (1, 0, 0, 0):
                cov = {f"d_{m}": float(rng.normal()) for m in ("f0", "energy", "rate", "tilt", "snr")}
                y = base - 0.5 * cond + 0.3 * cov["d_tilt"] + 0.01 * rng.normal()
                rows.append({"y": y, "cond": cond, "speaker": f"c{ch}", "set": f"c{ch}s{s}", **cov})
    r = adjusted_effects(rows, -1, rng)
    assert r["n_channels"] == 6 and r["assessable"]
    assert all(abs(v["beta_cond"] + 0.5) < 0.02 for v in r["channels"].values())
    assert abs(r["signflip"]["p_one"] - 1 / 64) < 1e-12  # all 6 channels in the predicted direction
