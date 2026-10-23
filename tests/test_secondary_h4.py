"""Tests for the H4 secondary analyses (csdiar/analysis/secondary_h4.py)."""
from __future__ import annotations

import numpy as np
import pytest

from csdiar.analysis import secondary_h4 as sec
from csdiar.mitigations.m1_subspace import project_out


def test_p0_points_keeps_only_what_the_p0L1_cell_and_tau_read():
    pts = {"a": {"kind": "ssdl", "elig_p0L1": True, "elig_det_w1": False},
           "b": {"kind": "ssdl", "elig_p0L1": False, "elig_det_w1": True},
           "c": {"kind": "ctrl", "elig_p0L1": False, "elig_det_w1": True},
           "d": {"kind": "ctrl", "elig_p0L1": True, "elig_det_w1": False}}
    assert sorted(sec.p0_points(pts)) == ["a", "c"]


def test_inner_folds_are_fixed_disjoint_triples_of_the_fitting_channels():
    fit = [f"UC{i:02d}" for i in range(12)]
    f = sec.inner_folds(fit)
    assert [len(x) for x in f] == [3] * 4
    assert sorted(sum(f, [])) == sorted(fit)
    assert f == sec.inner_folds(list(reversed(fit)))
    for bad in (fit[:11], []):
        with pytest.raises(ValueError):
            sec.inner_folds(bad)


def test_choose_k_takes_the_lowest_inner_dfar_ties_to_smaller_k_and_flags_a_fallback():
    assert sec.choose_k({1: 0.03, 2: 0.01, 4: 0.02, 8: None, 16: 0.05}) == (2, False)
    assert sec.choose_k({1: 0.02, 2: 0.01, 4: 0.01, 8: 0.01, 16: 0.03}) == (2, False)
    assert sec.choose_k({k: None for k in sec.KS}) == (1, True)


def _lambda_case(phrase_code: str = "True"):
    p = np.tile([1.0, 0.0], (40, 1))
    p[20:] = [0.0, 1.0]                                   # ne -> en at window 20: lambda(2.0 s) = 1 bit
    pt = lambda kind, t, pause, phrase: {"kind": kind, "speaker_id": "A", "video_id": "v1", "pause_bin": pause,
                                         "pos_bin": "0", "phrase": phrase, "clip_id": "c1", "t": t}
    pts = {f"s{i}": pt("ssdl", 2.0, "none", "False") for i in range(12)}
    pts.update({f"c{i}": pt("ctrl", 1.5, "none", "False") for i in range(12)})          # lambda 0
    pts.update({f"p{i}": pt("ctrl", 2.0, "gt300", phrase_code) for i in range(5)})      # phrase, lambda 1
    return {"lid": {"c1": p}}, [{"pts": pts, "scored": ["A"]}]


def test_part_lambda_end_to_end_on_a_synthetic_switch():
    d, us = _lambda_case()
    r = sec.part_lambda(d, us, None, np.random.default_rng(0), {})
    a = r["per_channel"]["A"]
    assert a["auc"] == pytest.approx(1.0) and a["n_ssdl"] == 12
    assert a["auc_phrase_vs_other"] == pytest.approx(1.0) and a["n_phrase"] == 5
    assert r["included"] == ["A"] and r["summary"]["auc"]["mean"] == pytest.approx(1.0)


def test_part_lambda_refuses_an_unexpected_phrase_encoding():
    d, us = _lambda_case(phrase_code="true")
    with pytest.raises(SystemExit):
        sec.part_lambda(d, us, None, np.random.default_rng(0), {})


def test_nap_basis_recovers_a_planted_subspace():
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((32, 2)))
    diffs = rng.standard_normal((200, 2)) * [3.0, 2.0] @ q.T + 0.01 * rng.standard_normal((200, 32))
    u = sec.nap_basis(diffs, 2)
    assert u.shape == (32, 2)
    assert np.allclose(u.T @ u, np.eye(2), atol=1e-5)
    # the planted directions lie in the recovered span
    assert np.allclose(np.linalg.norm(u.T @ q, axis=0), 1.0, atol=1e-3)
    with pytest.raises(SystemExit):
        sec.nap_basis(diffs[:3], 4)


def test_nap_basis_k1_is_the_mean_direction_when_the_shift_dominates():
    rng = np.random.default_rng(1)
    v = sec.h4.unit(rng.standard_normal(16))
    diffs = v + 0.05 * rng.standard_normal((100, 16))
    u = sec.nap_basis(diffs, 1)[:, 0]
    assert abs(float(u @ v)) > 0.999


def test_project_basis_removes_the_span_and_equals_project_out_for_k1():
    rng = np.random.default_rng(2)
    e = sec.h4.unit(rng.standard_normal((50, 16)))
    u = sec.random_basis(16, 3, rng)
    p = sec.project_basis(e, u)
    assert np.allclose(p @ u, 0.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(p, axis=1), 1.0, atol=1e-5)
    v = u[:, 0]
    assert np.allclose(sec.project_basis(e, v[:, None]), project_out(e, v), atol=1e-5)


def test_random_and_pca_bases():
    rng = np.random.default_rng(3)
    r = sec.random_basis(20, 4, rng)
    assert np.allclose(r.T @ r, np.eye(4), atol=1e-5)
    axis = np.zeros(20); axis[5] = 1.0
    x = rng.standard_normal((500, 20)) * 0.1 + rng.standard_normal((500, 1)) * 5.0 * axis
    assert abs(float(sec.pca_basis(x, 1)[5, 0])) > 0.99


def test_session_diffs_use_speakers_with_both_languages_only():
    by = {("s1", "v1"): {"ne": [np.array([1.0, 0.0])], "en": [np.array([0.0, 1.0]), np.array([0.0, 1.0])]},
          ("s1", "v2"): {"ne": [np.array([1.0, 0.0])], "en": []},
          ("s2", "v3"): {"ne": [np.array([1.0, 0.0])], "en": [np.array([1.0, 1.0])]}}
    d = sec.session_diffs(by, {"s1"})
    assert d.tolist() == [[-1.0, 1.0]]
    assert sec.session_windows(by, {"s1"}).shape == (4, 2)


def test_lam_at_is_one_bit_at_a_hard_switch_and_undefined_at_the_edge():
    p = np.tile([1.0, 0.0], (40, 1))
    p[20:] = [0.0, 1.0]
    assert sec.lam_at(p, 2.0) == pytest.approx(1.0)        # windows 10 (ne) and 20 (en)
    assert sec.lam_at(p, 1.5) == pytest.approx(0.0)        # windows 5 and 15, both ne
    assert sec.lam_at(p, 0.5) is None                      # no left window
    assert sec.lam_at(p, 4.0) is None                      # past the grid
    assert sec.lam_at(None, 2.0) is None


def test_compare_reports_mean_difference_and_unadjusted_signflip():
    rng = np.random.default_rng(0)
    chans = [f"c{i}" for i in range(10)]
    b = {c: 0.03 for c in chans}
    a = {c: 0.02 for c in chans}
    a["c0"] = None
    r = sec.compare(a, b, chans, s_b0=0.03, rng=rng)
    assert r["n_channels"] == 9
    assert r["mean_delta"] == pytest.approx(-0.01)
    assert r["relative_reduction"] == pytest.approx(1 / 3)
    assert r["p_one_unadjusted"] == pytest.approx(2 ** -9)
    assert sec.compare({}, b, chans, 0.03, rng) == {"n_channels": 0}


def test_summarise_includes_channels_by_b0_count_and_holds_s_b0():
    rng = np.random.default_rng(0)
    row = lambda n, x: {"n_ssdl": n, "auc": 0.6, "far_ssdl": 0.1 + x, "far_ctrl_std": 0.1, "delta_far_std": x}
    table = {"B0": {"c1": row(12, 0.03), "c2": row(10, 0.01), "c3": row(9, 0.5)},
             "M1": {"c1": row(12, 0.01), "c2": row(10, 0.01), "c3": row(9, 0.0)}}
    s = sec.summarise(table, [("M1", "B0")], rng)
    assert s["included"] == ["c1", "c2"] and s["excluded"] == ["c3"]
    assert s["S_B0"] == pytest.approx(0.02)
    assert s["comparisons"]["M1_vs_B0"]["mean_delta"] == pytest.approx(-0.01)
    assert s["mean_over_included"]["B0"]["delta_far_std"] == pytest.approx(0.02)


def test_mean_evals_averages_rates_over_draws():
    ev = lambda x: {"tau": x, "channels": {"c": {"n_ssdl": 11, "auc": x, "far_ssdl": x, "far_ctrl_std": x,
                                                 "delta_far_std": x}}}
    m = sec.mean_evals([ev(0.1), ev(0.3)])
    assert m["tau"] == [0.1, 0.3]
    assert m["channels"]["c"]["delta_far_std"] == pytest.approx(0.2)
    assert m["channels"]["c"]["n_ssdl"] == 11


def _windows():
    wins = []
    for ch, vids in (("A", ("a1", "a2")), ("B", ("b1", "b2"))):
        for v in vids:
            for i in range(30):
                wins.append({"speaker_id": ch, "video_id": v, "abs_t0": str(1.0 * i),
                             "lang": "ne" if i % 2 else "en", "clip_id": f"{v}_{i // 10}", "k": str(i % 10)})
    return wins


def test_draw_trials_follow_the_h3_rules():
    wins = _windows()
    tr = sec.draw_trials(wins, np.random.default_rng(0))
    a, b, cross = tr["tgt"]
    assert len(a) > 0
    vid = np.array([w["video_id"] for w in wins])
    t = np.array([float(w["abs_t0"]) for w in wins])
    lang = np.array([w["lang"] for w in wins])
    assert (vid[a] == vid[b]).all() and (np.abs(t[a] - t[b]) >= sec.LAG_S).all()
    assert (cross == (lang[a] != lang[b])).all() and cross.any() and (~cross).any()
    na, nb, ncross = tr["non"]
    chan = np.array([w["speaker_id"] for w in wins])
    assert (chan[na] != chan[nb]).all()
    assert (ncross == (lang[na] != lang[nb])).all()


def test_trial_eers_are_zero_for_perfectly_separated_speakers():
    wins = _windows()
    tr = sec.draw_trials(wins, np.random.default_rng(0))
    e = np.array([[1.0, 0.0] if w["speaker_id"] == "A" else [0.0, 1.0] for w in wins], np.float32)
    e += 1e-3 * np.random.default_rng(1).standard_normal(e.shape).astype(np.float32)
    r = sec.trial_eers(sec.h4.unit(e), tr)
    assert r["mono"] == pytest.approx(0.0, abs=1e-3) and r["cross"] == pytest.approx(0.0, abs=1e-3)


def test_probes_on_separable_data():
    rng = np.random.default_rng(0)
    wins = _windows()
    spk = [w["speaker_id"] for w in wins]
    vid = [w["video_id"] for w in wins]
    lang = [w["lang"] for w in wins]
    x = rng.standard_normal((len(wins), 8)).astype(np.float32) * 0.1
    x[:, 0] += [3.0 if s == "A" else -3.0 for s in spk]
    x[:, 1] += [3.0 if l == "en" else -3.0 for l in lang]
    assert sec.speaker_probe(x, spk, vid) == pytest.approx(1.0)
    idx = rng.permutation(len(wins))
    tr, te = idx[:60], idx[60:]
    lang_a = np.array(lang)
    assert sec.lang_probe(x[tr], lang_a[tr], x[te], lang_a[te]) == pytest.approx(1.0)
    assert sec.lang_probe(x[:2], ["en", "en"], x[2:], lang[2:]) is None
    assert sec.speaker_probe(x[:30], spk[:30], vid[:30]) is None          # one channel only


def test_m3_names_and_parse():
    names = sec.m3_names({"m3_tag": "f2"}, None)
    assert names["primary"] == [f"m3_f2_s{s}_w0.3_none" for s in range(5)]
    assert names["w1_none"] == ["m3_f2_s0_w1_none"] and names["w0.3_nod5"] == ["m3_f2_s0_w0.3_nod5"]
    assert list(sec.m3_names({"m3_tag": "all"}, None)) == ["primary"]
    assert sec.m3_names({"m3_tag": None}, "m3_f0_s0_w0.3_none") == {"primary": ["m3_f0_s0_w0.3_none"]}
    assert sec.m3_names({"m3_tag": None}, None) == {}
    assert sec.parse_m3("m3_f3_s0_w0.1_none") == {"fold": 3, "seed": 0, "w_dist": 0.1, "ablation": "none"}
    assert sec.parse_m3("m3_all_s4_w0.3_none")["fold"] is None


def test_check_stage1_accepts_a_reproduction_and_refuses_a_mismatch():
    stage1 = {"per_channel_extra": {"B0": {"c1": {"tau": 0.93}, "c2": {"tau": 0.93}},
                                    "M1": {"c1": {"tau": 0.91}},
                                    "M3_seeds": {"c1": [{"grid": "m3_f0_s0_w0.3_none", "tau": 0.95, "dfar": 0.01}]}},
              "dfar": {"B0": {"c1": 0.03}, "M1": {"c1": 0.02}}}
    u = {"fold": 0, "scored": ["c1"]}
    ok = lambda tau, x: {"tau": tau, "channels": {"c1": {"delta_far_std": x}}}
    sec.check_stage1(stage1, "B0", u, ok(0.93, 0.03))
    sec.check_stage1(stage1, "M1", u, ok(0.91, 0.02))
    sec.check_stage1(stage1, "M3", u, ok(0.95, 0.01), "m3_f0_s0_w0.3_none")
    sec.check_stage1(None, "B0", u, ok(0.5, 0.5))                         # dry run: nothing to check
    sec.check_stage1(stage1, "M1", {"fold": None, "scored": ["t1"]}, ok(0.5, 0.5))   # test scope: fold-free
    for args in (("B0", u, ok(0.94, 0.03)), ("B0", u, ok(0.93, 0.031)), ("M1", u, ok(0.91, 0.021)),
                 ("M3", u, ok(0.95, 0.02), "m3_f0_s0_w0.3_none")):
        with pytest.raises(SystemExit):
            sec.check_stage1(stage1, *args)
