"""Tests for the extended two-rater scorer (scripts/score_human_subset.py), EXPLORATORY review issue 4.

Synthetic ratings only; no audio and no real labels. Run with:
  PYTHONPATH=. <venv>/bin/python -m pytest -q -p no:cacheprovider tests/test_review_human_subset.py
"""
import csv
import math

import pytest

from scripts.score_human_subset import (cohen_kappa, wilson, flag_rate, timing_agreement, resolved_label,
                                         verified_subset, read_answers, read_adjudication, refuse_key_columns)


def test_refuse_key_columns():
    # a plain rater header is fine
    refuse_key_columns("ok.csv", ["item_id", "audio_file", "same_speaker", "unusable", "notes"])
    with pytest.raises(SystemExit):
        refuse_key_columns("leak.csv", ["item_id", "point_id", "same_speaker"])
    with pytest.raises(SystemExit):
        refuse_key_columns("leak.csv", ["item_id", "kind", "lang_left"])


def _key(n_switch=3, n_control=3, channels=("A", "B", "C")):
    key = {}
    k = 1
    for i in range(n_switch):
        key[f"i{k:03d}"] = {"item_id": f"i{k:03d}", "kind": "switch", "speaker_id": channels[i % len(channels)]}
        k += 1
    for i in range(n_control):
        key[f"i{k:03d}"] = {"item_id": f"i{k:03d}", "kind": "control", "speaker_id": channels[i % len(channels)]}
        k += 1
    return key


def _ans(same=None, switch=None, boundary=None, unusable=None, t=None):
    return {"same_speaker": same, "audible_language_switch": switch, "boundary_within_0.1s": boundary,
            "unusable": unusable, "switch_time_s": t}


def test_timing_agreement_only_counts_both_switch_with_times():
    key = ["i001", "i002", "i003"]
    r1 = {"i001": _ans(switch="y", t=2.00), "i002": _ans(switch="y", t=2.50), "i003": _ans(switch="n")}
    r2 = {"i001": _ans(switch="y", t=2.05), "i002": _ans(switch="y", t=2.80), "i003": _ans(switch="y", t=1.0)}
    ta = timing_agreement(key, r1, r2)
    # i003 excluded (r1 said no switch); i001 diff 0.05 (within), i002 diff 0.30 (outside)
    assert ta["n"] == 2
    assert math.isclose(ta["mean_abs_diff_s"], (0.05 + 0.30) / 2, abs_tol=1e-9)
    assert math.isclose(ta["within_0.1s_rate"], 0.5)


def test_flag_rate_unusable():
    ids = ["i001", "i002", "i003", "i004"]
    r1 = {i: _ans(unusable="y" if i in ("i001", "i002") else "n") for i in ids}
    r2 = {i: _ans(unusable="y" if i == "i001" else "n") for i in ids}
    fr = flag_rate(ids, r1, r2, "unusable", "y", ("y", "n"))
    assert fr["rater1"]["k"] == 2 and fr["rater1"]["n"] == 4
    assert fr["rater2"]["k"] == 1
    # both agree unusable only on i001
    assert fr["both_agree"]["k"] == 1
    # either flags unusable on i001, i002
    assert fr["either_flags"]["k"] == 2


def test_resolved_label_prefers_adjudication_then_agreement():
    r1 = {"i001": _ans(same="y"), "i002": _ans(same="y")}
    r2 = {"i001": _ans(same="n"), "i002": _ans(same="y")}
    adj = {"i001": {"same_speaker": "y"}}
    assert resolved_label("i001", "same_speaker", r1, r2, adj) == "y"   # adjudicated
    assert resolved_label("i002", "same_speaker", r1, r2, {}) == "y"     # agreed
    assert resolved_label("i001", "same_speaker", r1, r2, {}) is None    # disagree, no adjudication


def test_verified_subset_and_minimum():
    # 2 switch, 2 control over 2 channels; both raters agree all pass
    key = {"i001": {"kind": "switch", "speaker_id": "A"}, "i002": {"kind": "switch", "speaker_id": "B"},
           "i003": {"kind": "control", "speaker_id": "A"}, "i004": {"kind": "control", "speaker_id": "B"}}
    for v in key.values():
        v["item_id"] = next(k for k, vv in key.items() if vv is v)
    sw_pass = _ans(same="y", switch="y", boundary="y", unusable="n")
    ct_pass = _ans(same="y", switch="n", boundary="n", unusable="n")
    r1 = {"i001": sw_pass, "i002": sw_pass, "i003": ct_pass, "i004": ct_pass}
    r2 = dict(r1)
    vs = verified_subset(key, r1, r2, {})
    assert vs["n_verified_switch"] == 2 and vs["n_verified_control"] == 2 and vs["n_channels"] == 2
    assert vs["minimum"]["met"] is False  # below 100/100/10
    # a switch marked unusable by adjudication drops out
    vs2 = verified_subset(key, r1, r2, {"i001": {"unusable": "y"}})
    assert vs2["n_verified_switch"] == 1


def test_read_answers_and_adjudication_roundtrip(tmp_path):
    key = {"i001": {}, "i002": {}}
    sheet = tmp_path / "r.csv"
    with open(sheet, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["item_id", "audio_file", "same_speaker", "audible_language_switch",
                                           "switch_time_s", "boundary_within_0.1s", "unusable", "notes"])
        w.writeheader()
        w.writerow({"item_id": "i001", "same_speaker": "y", "audible_language_switch": "y",
                    "switch_time_s": "2.02", "boundary_within_0.1s": "y", "unusable": "n"})
        w.writerow({"item_id": "i002", "same_speaker": "n", "audible_language_switch": "n",
                    "switch_time_s": "", "boundary_within_0.1s": "n", "unusable": "n"})
    ans = read_answers(sheet, set(key))
    assert ans["i001"]["switch_time_s"] == 2.02 and ans["i002"]["switch_time_s"] is None
    assert ans["i001"]["unusable"] == "n" and ans["i002"]["same_speaker"] == "n"

    adj_path = tmp_path / "adj.csv"
    with open(adj_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["item_id", "field", "rater1", "rater2", "resolved", "reason"])
        w.writeheader()
        w.writerow({"item_id": "i001", "field": "same_speaker", "rater1": "y", "rater2": "n",
                    "resolved": "y", "reason": "re-listened"})
    adj = read_adjudication(adj_path, set(key))
    assert adj["i001"]["same_speaker"] == "y"

    # a sheet that leaks a key column is refused
    bad = tmp_path / "bad.csv"
    with open(bad, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["item_id", "point_id", "same_speaker"])
        w.writeheader()
        w.writerow({"item_id": "i001", "point_id": "x:ssdl:1", "same_speaker": "y"})
    with pytest.raises(SystemExit):
        read_answers(bad, set(key))


def test_cohen_kappa_and_wilson_unchanged():
    # the existing API the other test file relies on still behaves
    a = ["y"] * 20 + ["n"] * 30
    assert math.isclose(cohen_kappa(a, a)["kappa"], 1.0)
    w = wilson(10, 20)
    assert math.isclose(w["lo"], 0.29930, abs_tol=1e-4) and math.isclose(w["hi"], 0.70070, abs_tol=1e-4)
