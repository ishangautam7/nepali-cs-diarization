"""Tests for scripts/score_listening.py (stdlib + pytest + numpy; synthetic answers, no audio)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "score_listening", Path(__file__).resolve().parents[1] / "scripts/score_listening.py")
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)

T, F = "True", "False"
#   point  kind    stratum          split        emb1 det p0L1 det_w1  q1        q2           q3
D1 = [("P1", "ssdl", "ssdl:emb1", "train", T, T, T, F, "no", "same", "yes"),
      ("P2", "ssdl", "ssdl:emb1", "test", T, T, T, F, "yes", "different", "yes"),
      ("P10", "ssdl", "ssdl:emb1", "validation", T, T, T, F, "no", "same", "no"),   # pilot: outside the domain
      ("P3", "ssdl", "ssdl:det_only", "train", F, T, T, F, "unsure", "same", "no"),
      ("P4", "ssdl", "ssdl:det_only", "train", F, T, F, F, "no", "same", "yes"),
      ("P5", "ctrl", "ctrl:emb1", "train", T, T, F, T, "yes", "same", "yes"),
      ("P6", "ctrl", "ctrl:emb1", "test", T, T, F, T, "no", "same", "yes"),
      ("P7", "ctrl", "ctrl:det_only", "train", F, T, F, T, "no", "unsure", "unsure"),
      ("P8", "ctrl", "ctrl:det_only", "train", F, T, F, F, "yes", "same", "yes")]
POOLS = {"ssdl:emb1": 10, "ssdl:det_only": 4, "ctrl:emb1": 6, "ctrl:det_only": 2}   # weights 10/3, 2, 3, 1
W1 = 10 / 3


def d1_inputs():
    key = [{"item": f"L{k:03d}", "point_id": p, "condition": s, "split": sp}
           for k, (p, _, s, sp, *_) in enumerate(D1, 1)]
    answers = [{"item": f"L{k:03d}", "q1_language_change": a, "q2_same_speaker": b, "q3_boundary": c}
               for k, (*_, a, b, c) in enumerate(D1, 1)]
    points = {p: {"point_id": p, "kind": kind, "split": sp, "elig_emb1": e, "elig_det": d, "elig_p0L1": l,
                  "elig_det_w1": w} for p, kind, _, sp, e, d, l, w, *_ in D1}
    points["P9"] = {"point_id": "P9", "kind": "ssdl", "split": "train", "elig_emb1": F, "elig_det": F,
                    "elig_p0L1": T, "elig_det_w1": F}
    return answers, key, points


def test_primary_cell_rates_are_design_weighted_confirmatory_only_and_leave_out_unsure():
    answers, key, points = d1_inputs()
    r = S.score_d1(answers, key, points, POOLS, {"e3": 0.0273}, n_boot=200)
    pc = r["primary_cell"]
    assert pc["pi"]["share"] == pytest.approx(0.5) and pc["pi"]["n"] == 2 and pc["pi"]["n_left_out"] == 1
    assert pc["delta"]["share"] == pytest.approx(W1 / (2 * W1 + 2))               # P1 same, P2 different, P3 same
    assert pc["kappa"]["share"] == pytest.approx(3 / 7)                          # P5 yes, P6 no, P7 no; P8 not det_w1
    assert pc["boundary_ok_p0L1"]["share"] == pytest.approx(2 * W1 / (2 * W1 + 2))
    assert pc["boundary_ok_all_switch"]["share"] == pytest.approx((2 * W1 + 2) / (2 * W1 + 4))
    assert r["dfar"]["e3"]["corrected"] == pytest.approx(0.0273 / (1 - 0.5 - 3 / 7))
    assert r["coverage"] == {"p0L1_switch_points": {"population": 4, "outside_the_sampled_pools": 1},
                             "det_w1_controls": {"population": 3, "outside_the_sampled_pools": 0}}
    s = r["strata"]["ssdl:emb1"]
    assert s["n"] == 3 and s["n_confirmatory"] == 2 and s["q3"]["no"]["n"] == 1    # the pilot item is described


def test_the_bootstrap_is_seeded_and_brackets_the_estimate():
    answers, key, points = d1_inputs()
    a = S.score_d1(answers, key, points, POOLS, {"e3": 0.0273}, n_boot=300)
    assert a == S.score_d1(answers, key, points, POOLS, {"e3": 0.0273}, n_boot=300)
    lo, hi = a["primary_cell"]["pi"]["ci95_boot"]
    assert 0 <= lo <= 0.5 <= hi <= 1


def test_percentile_interval_interpolates():
    assert S.percentile_ci([float(x) for x in range(101)]) == pytest.approx((2.5, 97.5))
    assert S.percentile_ci([]) is None


def test_no_correction_when_contamination_swamps_the_signal():
    assert S.corrected(0.03, 0.6, 0.5) is None and S.corrected(0.03, None, 0.1) is None
    assert S.corrected(0.03, 0.1, 0.1) == pytest.approx(0.0375)


def test_mismatched_unfinished_or_unexpected_answers_are_refused():
    answers, key, points = d1_inputs()
    with pytest.raises(SystemExit, match="same items"):
        S.score_d1(answers[:-1], key, points, POOLS, {"e3": 0.03}, n_boot=10)
    answers[0]["q3_boundary"] = ""
    with pytest.raises(SystemExit, match="not fully answered"):
        S.score_d1(answers, key, points, POOLS, {"e3": 0.03}, n_boot=10)
    answers[0]["q3_boundary"] = "maybe"
    with pytest.raises(SystemExit, match="unexpected answers"):
        S.score_d1(answers, key, points, POOLS, {"e3": 0.03}, n_boot=10)
    for a in answers:
        del a["q3_boundary"]
    with pytest.raises(SystemExit, match="lacks columns"):
        S.score_d1(answers, key, points, POOLS, {"e3": 0.03}, n_boot=10)


def test_purity_shares_flagged_against_controls_on_usable_and_all_pairs():
    key = [{"item": f"P{k}", "kind": kind} for k, kind in enumerate(["flagged"] * 4 + ["control"] * 2)]
    q = [("different", "usable"), ("same", "unusable"), ("unsure", "usable"), ("different", "usable"),
         ("same", "usable"), ("different", "usable")]
    answers = [{"item": f"P{k}", "q1_same_speaker": a, "q2_usable": b} for k, (a, b) in enumerate(q)]
    r = S.score_purity(answers, key)
    assert r["flagged"]["usable_pairs"]["share_different"] == pytest.approx(1.0)      # P0, P3 (P2 unsure)
    assert r["flagged"]["all_pairs"]["share_different"] == pytest.approx(2 / 3)
    assert r["flagged"]["share_unusable"] == pytest.approx(1 / 4)
    assert r["control"]["usable_pairs"]["share_different"] == pytest.approx(1 / 2)
    assert r["difference_flagged_minus_control_usable_pairs"] == pytest.approx(0.5)
    assert r["difference_flagged_minus_control_all_pairs"] == pytest.approx(2 / 3 - 1 / 2)
    answers[0]["q2_usable"] = "fine"
    with pytest.raises(SystemExit, match="unexpected answers"):
        S.score_purity(answers, key)


def test_wilson_interval():
    assert S.wilson(0, 0) is None
    lo, hi = S.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    assert S.wilson(0, 10)[0] == 0.0


# ---- --first N (DEVIATIONS.md entry 11) ----------------------------------------------------------------------
def prefix_inputs(n):
    answers, key, points = d1_inputs()
    for a in answers[n:]:                                    # the listener stopped after item n
        a.update(q1_language_change="", q2_same_speaker="", q3_boundary="")
    return answers, key, points


def test_first_n_scores_only_the_prefix_with_its_own_weights():
    answers, key, points = prefix_inputs(4)                  # L001-L004 = P1, P2, P10, P3
    r = S.score_d1(answers, key, points, POOLS, {"e3": 0.03}, n_boot=200, first=4)
    assert r["n_items"] == 4 and r["first"] == 4 and r["n_q3_blank"] == 0
    pc = r["primary_cell"]
    # ssdl:emb1 keeps weight 10/3 (3 of 3 sampled); ssdl:det_only now has 1 scored item, so weight 4/1 = 4
    assert pc["boundary_ok_all_switch"]["share"] == pytest.approx((2 * W1) / (2 * W1 + 4))   # P1, P2 yes; P3 no
    assert pc["delta"]["share"] == pytest.approx(W1 / (2 * W1 + 4))                         # only P2 different
    assert pc["kappa"]["share"] is None and r["dfar"]["e3"]["corrected"] is None             # no control scored
    assert r["strata"]["ssdl:det_only"]["n"] == 1 and r["strata"]["ctrl:emb1"]["n"] == 0


def test_first_n_refuses_answers_after_the_prefix_and_a_bad_n():
    answers, key, points = d1_inputs()                       # every item answered
    with pytest.raises(SystemExit, match="after the first 4 have answers"):
        S.score_d1(answers, key, points, POOLS, {"e3": 0.03}, n_boot=10, first=4)
    for n in (0, len(key) + 1):
        with pytest.raises(SystemExit, match="--first must be between"):
            S.score_d1(answers, key, points, POOLS, {"e3": 0.03}, n_boot=10, first=n)
    assert S.score_d1(answers, key, points, POOLS, {"e3": 0.03}, n_boot=10, first=len(key))["n_items"] == len(key)


def test_first_n_leaves_a_blank_q3_out_but_still_requires_q1_and_q2():
    answers, key, points = prefix_inputs(4)
    answers[0]["q3_boundary"] = ""                           # P1
    r = S.score_d1(answers, key, points, POOLS, {"e3": 0.03}, n_boot=10, first=4)
    assert r["n_q3_blank"] == 1 and r["strata"]["ssdl:emb1"]["q3"]["blank"]["n"] == 1
    assert r["primary_cell"]["boundary_ok_all_switch"]["share"] == pytest.approx(W1 / (W1 + 4))   # P2 yes, P3 no
    assert r["primary_cell"]["boundary_ok_all_switch"]["n_left_out"] == 1
    answers[1]["q1_language_change"] = ""                    # P2
    with pytest.raises(SystemExit, match="not fully answered"):
        S.score_d1(answers, key, points, POOLS, {"e3": 0.03}, n_boot=10, first=4)


def test_without_first_the_frozen_rules_apply_unchanged():
    answers, key, points = prefix_inputs(4)
    with pytest.raises(SystemExit, match="not fully answered"):
        S.score_d1(answers, key, points, POOLS, {"e3": 0.03}, n_boot=10)
    full = S.score_d1(*d1_inputs(), POOLS, {"e3": 0.0273}, n_boot=50)
    assert full["first"] is None and full["n_q3_blank"] == 0 and full["n_items"] == len(D1)


def test_purity_first_n_scores_only_the_prefix_and_refuses_later_answers():
    key = [{"item": f"P{k:03d}", "kind": kind} for k, kind in enumerate(["flagged", "control", "flagged", "flagged"], 1)]
    q = [("different", "usable"), ("same", "usable"), ("same", "unusable"), ("", "")]
    answers = [{"item": f"P{k:03d}", "q1_same_speaker": a, "q2_usable": b} for k, (a, b) in enumerate(q, 1)]
    r = S.score_purity(answers, key, first=3)
    assert r["first"] == 3 and r["flagged"]["n"] == 2 and r["control"]["n"] == 1
    assert r["flagged"]["usable_pairs"]["share_different"] == pytest.approx(1.0)       # P001; P003 is unusable
    assert r["flagged"]["share_unusable"] == pytest.approx(0.5)
    assert r["control"]["usable_pairs"]["share_different"] == pytest.approx(0.0)
    with pytest.raises(SystemExit, match="not fully answered"):                       # without --first: all needed
        S.score_purity(answers, key)
    for n in (0, 5):
        with pytest.raises(SystemExit, match="--first must be between"):
            S.score_purity(answers, key, first=n)
    answers[3].update(q1_same_speaker="same", q2_usable="usable")
    with pytest.raises(SystemExit, match="after the first 3 have answers"):
        S.score_purity(answers, key, first=3)
    answers[1]["q2_usable"] = ""
    with pytest.raises(SystemExit, match="not fully answered"):
        S.score_purity(answers, key, first=4)


def test_purity_first_through_the_command_line(tmp_path):
    (tmp_path / "a.csv").write_text("item,q1_same_speaker,q2_usable\nP001,same,usable\nP002,,\n")
    (tmp_path / "k.csv").write_text("item,kind\nP001,flagged\nP002,control\n")
    S.main(["purity", "--answers", str(tmp_path / "a.csv"), "--key", str(tmp_path / "k.csv"),
            "--out", str(tmp_path / "out"), "--first", "1"])
    r = json.loads((tmp_path / "out" / "listening_purity.json").read_text())
    assert r["first"] == 1 and r["flagged"]["n"] == 1 and r["control"]["n"] == 0
    assert r["difference_flagged_minus_control_usable_pairs"] is None
