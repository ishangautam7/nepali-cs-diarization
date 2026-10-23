"""Tests for the H4 analysis (csdiar/analysis/confirm_h4.py) and its folds (csdiar/mitigations/folds.py)."""
from __future__ import annotations

import numpy as np
import pytest

from csdiar.analysis import confirm_h4 as h4
from csdiar.mitigations import folds


def test_curve_compares_windows_one_second_apart():
    e = h4.unit(np.eye(12, 4)[np.arange(12) % 4].astype(np.float32))
    d = h4.curve(e)
    assert np.isnan(d[: h4.LAG_K]).all()
    # windows k and k-10 are rows (k % 4) and ((k - 10) % 4): different unit axes -> distance 1
    assert np.allclose(d[h4.LAG_K:], 1.0)


def test_b2_gate_only_where_predicted_language_changes():
    rng = np.random.default_rng(0)
    e = h4.unit(rng.standard_normal((30, 8)).astype(np.float32))
    lid = np.zeros((30, 2)); lid[:, 0] = 1.0           # all ne ...
    lid[20:, :] = [0.1, 0.9]                            # ... then en from window 20
    out = h4.system_curves("B2", {"c"}, {"c": e}, {"c": lid}, {"delta": 0.05})["c"]
    base = h4.curve(e)
    changed = np.zeros(30, bool); changed[20:30] = True   # k - 10 < 20 <= k
    assert np.allclose(out[changed], base[changed] - 0.05)
    assert np.allclose(out[h4.LAG_K:20], base[h4.LAG_K:20])


def test_m2_gate_scales_with_jsd_and_vanishes_for_equal_posteriors():
    rng = np.random.default_rng(1)
    e = h4.unit(rng.standard_normal((25, 8)).astype(np.float32))
    lid = np.tile([0.5, 0.5], (25, 1))
    out = h4.system_curves("M2", {"c"}, {"c": e}, {"c": lid}, {"alpha": 2.0})["c"]
    assert np.allclose(out[h4.LAG_K:], h4.curve(e)[h4.LAG_K:])
    lid2 = lid.copy(); lid2[15:] = [1.0, 0.0]; lid2[:15] = [0.0, 1.0]
    out2 = h4.system_curves("M2", {"c"}, {"c": e}, {"c": lid2}, {"alpha": 2.0})["c"]
    k = 20                                               # windows 10 (en) and 20 (ne): JSD = 1 bit
    assert out2[k] == pytest.approx(h4.curve(e)[k] - 2.0, abs=1e-5)


def test_predicted_lang_threshold_and_length_check():
    lid = np.array([[0.5, 0.5], [0.49, 0.51], [0.9, 0.1]])
    assert h4.predicted_lang(lid, 3).tolist() == [0, 1, 0]
    with pytest.raises(ValueError):
        h4.predicted_lang(lid, 4)


def test_family_staged_member_enters_as_p_one_and_categories():
    rng = np.random.default_rng(0)
    chans = [f"c{i}" for i in range(15)]
    strong = {c: -0.02 for c in chans}                   # every channel reduced by 2 pp
    null = {c: (0.001 if i % 2 else -0.001) for i, c in enumerate(chans)}
    res = h4.test_family({"M1": strong, "M3": null}, s_b0=0.03, rng=rng, staged={"M2"})
    assert res["M2"] == {"staged": True}
    assert res["M1"]["signflip"]["p_one"] == pytest.approx(2 ** -15)
    # Holm over three members with M2 as p = 1: M1 is the smallest p, multiplied by 3
    assert res["M1"]["p_one_holm"] == pytest.approx(3 * 2 ** -15)
    assert res["M1"]["outcome"]["category"] == "supported"          # 0.02 >= 0.25 * 0.03
    assert res["M3"]["outcome"]["category"] in ("inconclusive", "equivalent")


def test_family_degenerate_baseline_is_inconclusive():
    rng = np.random.default_rng(0)
    res = h4.test_family({"M1": {f"c{i}": -0.01 for i in range(15)}}, s_b0=-0.001, rng=rng, staged={"M2", "M3"})
    assert res["M1"]["outcome"]["category"] == "inconclusive"
    assert res["M1"]["relative_reduction"] is None


def test_folds_are_fixed_disjoint_triples():
    chans = [f"UC{i:02d}" for i in range(15)]
    f = folds.h4_folds(chans)
    assert [len(x) for x in f] == [3] * 5
    assert sorted(sum(f, [])) == sorted(chans)
    assert f == folds.h4_folds(list(reversed(chans)))          # order of the input does not matter
    assert set(folds.fitting_channels(chans, 0)) == set(chans) - set(f[0])
    with pytest.raises(ValueError):
        folds.h4_folds(chans[:14])
