"""Tests for csdiar.analysis.confirm_h7 helpers (no D4 data)."""
import numpy as np

from csdiar.analysis.confirm_h7 import N_MC, signflip_h7


def test_monte_carlo_branch_adds_one():
    x = np.full(30, 1.0)  # every speaker in the predicted direction: no sampled pattern beats it in practice
    r = signflip_h7(x, np.random.default_rng(0))
    assert not r["exact"] and r["monte_carlo_patterns"] == N_MC
    assert abs(r["p_one"] - 1 / (N_MC + 1)) < 1e-12
    assert r["p_two"] >= r["p_one"]


def test_exact_branch_unchanged():
    r = signflip_h7([1.0, 2.0, 3.0], np.random.default_rng(0))
    assert r["exact"] and abs(r["p_one"] - 1 / 8) < 1e-12


def _result(p_one: float, est: float, kind: str) -> dict:
    sf = {"p_one": p_one, "p_two": min(1.0, 2 * p_one)}
    if kind == "h1":
        return {"signflip": sf, "dz": est, "dz_tost": {"lo": est - 0.02, "hi": est + 0.02}}
    return {"signflip": sf, "auc_minus_half": est, "tost": {"lo": est - 0.01, "hi": est + 0.01}}


def test_grade_verdicts_and_holm():
    from csdiar.analysis.confirm_h7 import MODELS, PAIRS, grade
    block = {p: {"models": {m: {} for m in MODELS}} for p in PAIRS}
    for m in MODELS:
        # hi-en: H1 strong in all three embeddings; H2-P0 strong only for e1 and trivial in size
        block["hi-en"]["models"][m]["h1"] = _result(1e-5, 0.30, "h1")
        block["hi-en"]["models"][m]["h2_p0"] = _result(1e-5 if m == "e1" else 0.5, 0.02, "h2_p0")
        # bn-en: nothing significant
        block["bn-en"]["models"][m]["h1"] = _result(0.4, 0.05, "h1")
        block["bn-en"]["models"][m]["h2_p0"] = _result(0.4, 0.01, "h2_p0")
    v = grade(block)
    assert v["hi-en/h1"] == {"verdict": "replicated", "n_replicated": 3, "n_supported_at_d1_sesoi": 3}
    assert v["hi-en/h2_p0"]["verdict"] == "partly replicated" and v["hi-en/h2_p0"]["n_replicated"] == 1
    assert v["hi-en/h2_p0"]["n_supported_at_d1_sesoi"] == 0  # AUC 0.52 is below 0.55
    assert v["bn-en/h1"]["verdict"] == "not replicated"
    r = block["hi-en"]["models"]["e1"]["h1"]
    assert abs(r["p_one_holm"] - 12 * 1e-5) < 1e-12 and r["replicated"] is True  # smallest of 12 ties: Holm x12
