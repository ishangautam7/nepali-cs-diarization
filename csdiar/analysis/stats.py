"""Confirmatory statistics (PLAN.md §7; decisions 13-17).

- signflip: exact sign-flip test over clusters (channels), one-sided in the predicted direction and two-sided.
- bca_ci: BCa bootstrap confidence interval over clusters, with jackknife acceleration.
- holm: Holm step-down adjustment of a family of p-values.
- outcome: the pre-registered outcome categories, with every underlying condition reported.
All functions are deterministic given the numpy Generator passed in.
"""
from __future__ import annotations

from statistics import NormalDist
from typing import Callable, Sequence

import numpy as np

_N = NormalDist()
_SIGNS: dict[int, np.ndarray] = {}


def signflip(x: Sequence[float], rng: np.random.Generator, n_mc: int = 2 ** 17) -> dict:
    """Sign-flip test of mean(x) > 0 over clusters. Exact (all 2^n patterns) when 2^n <= n_mc.

    Returns p_one (H1: mean > 0) and p_two (|mean| as large as observed), both counting the observed pattern.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return {"p_one": float("nan"), "p_two": float("nan"), "exact": True, "n": 0}
    if 2 ** n <= n_mc:
        if n not in _SIGNS:
            _SIGNS[n] = (((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1).astype(np.float64)
        signs, exact = _SIGNS[n], True
    else:
        signs, exact = rng.choice([-1.0, 1.0], size=(n_mc, n)), False
    means = signs @ x / n
    obs = x.mean()
    eps = 1e-12 * max(1.0, abs(obs))
    return {"p_one": float((means >= obs - eps).mean()), "p_two": float((np.abs(means) >= abs(obs) - eps).mean()),
            "exact": exact, "n": n}


def bca_ci(clusters: list, stat: Callable[[list], float], level: float, rng: np.random.Generator,
           n_boot: int = 10000) -> dict:
    """BCa interval for stat(clusters), resampling whole clusters with replacement."""
    k = len(clusters)
    nan = {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan"), "level": level, "z0": float("nan"),
           "a": float("nan"), "n_boot": 0, "n_dropped": 0}
    if k < 2:
        return nan
    theta = float(stat(clusters))
    boots = np.array([stat([clusters[i] for i in rng.integers(0, k, k)]) for _ in range(n_boot)], dtype=np.float64)
    n_dropped = int((~np.isfinite(boots)).sum())
    boots = boots[np.isfinite(boots)]
    if len(boots) == 0 or not np.isfinite(theta):
        return {**nan, "estimate": theta, "n_dropped": n_dropped}
    frac = ((boots < theta).sum() + 0.5 * (boots == theta).sum()) / len(boots)
    frac = min(max(frac, 1 / (len(boots) + 1)), len(boots) / (len(boots) + 1))
    z0 = _N.inv_cdf(frac)
    jack = np.array([stat(clusters[:i] + clusters[i + 1:]) for i in range(k)], dtype=np.float64)
    d = jack.mean() - jack
    denom = 6 * (d ** 2).sum() ** 1.5
    a = float((d ** 3).sum() / denom) if denom > 0 else 0.0
    alpha = 1 - level
    qs = []
    for z in (_N.inv_cdf(alpha / 2), _N.inv_cdf(1 - alpha / 2)):
        qs.append(_N.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))
    lo, hi = np.quantile(boots, qs)
    return {"estimate": theta, "lo": float(lo), "hi": float(hi), "level": level, "z0": z0, "a": a,
            "n_boot": int(len(boots)), "n_dropped": n_dropped}


def holm(p: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in the input order."""
    p = np.asarray(p, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def outcome(p_one_adj: float, p_two_adj: float, estimate: float, tost_lo: float, tost_hi: float,
            sesoi: float, direction: int = 1, alpha: float = 0.05) -> dict:
    """Pre-registered outcome category (decision 16).

    estimate, tost_lo, tost_hi are on the effect scale; direction = +1 if the prediction is a positive effect,
    -1 if negative. The TOST interval is the (1 - 2 alpha / m) interval. Precedence:
    supported > nonzero but trivial > reversed > equivalent > inconclusive. Every condition is returned.
    """
    e = direction * estimate
    lo, hi = sorted((direction * tost_lo, direction * tost_hi))
    flags = {"significant_predicted": p_one_adj < alpha, "meets_sesoi": e >= sesoi,
             "significant_reversed": p_two_adj < alpha and e < 0, "equivalent": -sesoi < lo and hi < sesoi}
    if flags["significant_predicted"] and flags["meets_sesoi"]:
        cat = "supported"
    elif flags["significant_predicted"]:
        cat = "nonzero but trivial"
    elif flags["significant_reversed"]:
        cat = "reversed"
    elif flags["equivalent"]:
        cat = "equivalent"
    else:
        cat = "inconclusive"
    return {"category": cat, **flags}
