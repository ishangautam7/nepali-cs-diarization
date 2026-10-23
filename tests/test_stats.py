"""Tests for csdiar.analysis.stats."""
import numpy as np

from csdiar.analysis.stats import bca_ci, holm, outcome, signflip


def test_signflip_exact_one_and_two_sided():
    rng = np.random.default_rng(0)
    r = signflip([1.0, 2.0, 3.0], rng)
    assert r["exact"] and abs(r["p_one"] - 1 / 8) < 1e-12 and abs(r["p_two"] - 2 / 8) < 1e-12
    r = signflip([-1.0, -2.0, -3.0], rng)
    assert r["p_one"] == 1.0 and abs(r["p_two"] - 2 / 8) < 1e-12


def test_holm_matches_hand_computation():
    adj = holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])
    assert holm([0.5, 0.5]) == [1.0, 1.0]


def test_bca_covers_the_mean_of_normal_clusters():
    rng = np.random.default_rng(1)
    hits = 0
    for _ in range(60):
        clusters = list(rng.normal(0.3, 1.0, 17))
        ci = bca_ci(clusters, lambda c: float(np.mean(c)), 0.90, rng, n_boot=1500)
        hits += ci["lo"] <= 0.3 <= ci["hi"]
        assert ci["lo"] <= ci["estimate"] <= ci["hi"]
    assert 45 <= hits <= 60  # nominal 90% coverage, loose bounds for 60 replicates


def test_outcome_categories_and_precedence():
    kw = dict(sesoi=0.2, direction=1)
    assert outcome(0.01, 0.02, 0.35, 0.25, 0.45, **kw)["category"] == "supported"
    assert outcome(0.01, 0.02, 0.10, 0.05, 0.15, **kw)["category"] == "nonzero but trivial"
    assert outcome(0.9, 0.01, -0.30, -0.40, -0.20, **kw)["category"] == "reversed"
    assert outcome(0.4, 0.8, 0.02, -0.10, 0.12, **kw)["category"] == "equivalent"
    assert outcome(0.2, 0.4, 0.15, -0.05, 0.35, **kw)["category"] == "inconclusive"
    # a predicted negative effect (H3) is mirrored
    assert outcome(0.01, 0.02, -0.30, -0.40, -0.25, sesoi=0.2, direction=-1)["category"] == "supported"
    assert outcome(0.99, 0.01, 0.30, 0.20, 0.40, sesoi=0.2, direction=-1)["category"] == "reversed"
    assert outcome(0.40, 0.80, -0.02, -0.10, 0.08, sesoi=0.2, direction=-1)["category"] == "equivalent"
    # significant but inside the equivalence bounds: the higher category wins, the flag is still reported
    r = outcome(0.01, 0.02, 0.10, 0.05, 0.15, sesoi=0.2, direction=1)
    assert r["category"] == "nonzero but trivial" and r["equivalent"]


def test_signflip_ties_and_zeros():
    rng = np.random.default_rng(0)
    r = signflip([0.0, 0.0, 1.0], rng)  # zeros flip to themselves: half the patterns reach the observed mean
    assert abs(r["p_one"] - 0.5) < 1e-12 and abs(r["p_two"] - 1.0) < 1e-12
    r = signflip([1.0, 1.0, 1.0], rng)
    assert abs(r["p_one"] - 1 / 8) < 1e-12


def test_bca_matches_scipy_reference():
    from scipy.stats import bootstrap
    x = np.random.default_rng(7).lognormal(0.0, 0.8, 17)
    ours = bca_ci(list(x), lambda c: float(np.mean(c)), 0.95, np.random.default_rng(1), n_boot=20000)
    ref = bootstrap((x,), np.mean, confidence_level=0.95, n_resamples=20000, method="BCa",
                    random_state=np.random.default_rng(2)).confidence_interval
    width = ref.high - ref.low
    assert abs(ours["lo"] - ref.low) < 0.05 * width and abs(ours["hi"] - ref.high) < 0.05 * width


def test_bca_degenerate_inputs():
    rng = np.random.default_rng(0)
    assert np.isnan(bca_ci([], lambda c: float(np.mean(c)), 0.9, rng)["lo"])
    assert np.isnan(bca_ci([1.0], lambda c: float(np.mean(c)), 0.9, rng)["hi"])


def test_signflip_is_exact_for_17_channels():
    x = np.random.default_rng(3).normal(0.2, 1.0, 17)
    r = signflip(x, np.random.default_rng(0))
    assert r["exact"] and r["n"] == 17
    assert abs(r["p_one"] * 2 ** 17 - round(r["p_one"] * 2 ** 17)) < 1e-6


def test_bca_with_the_dz_ratio_statistic():
    from csdiar.analysis.confirm import dz_stat
    g = np.random.default_rng(5)
    clusters = [g.normal(0.3, 1.0, int(n)) for n in g.integers(15, 60, 17)]
    ci = bca_ci(clusters, dz_stat, 0.95, np.random.default_rng(1), n_boot=4000)
    assert ci["lo"] < dz_stat(clusters) < ci["hi"] and ci["n_dropped"] == 0
    assert 0.1 < ci["lo"] and ci["hi"] < 0.5  # true dz 0.3; about 600 sets
