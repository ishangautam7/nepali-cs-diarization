"""Tests for the post-hoc review follow-up helpers (scripts/review_*.py, scripts/score_human_subset.py)."""
import math

import numpy as np

from scripts.review_fig1_reconciliation import cell_steps, pooled_raw
from scripts.review_human_subset import allocate
from scripts.review_pipeline_score import change_times, majority
from scripts.score_human_subset import cohen_kappa, wilson


def test_change_times_ignore_silence_and_count_a_joining_speaker():
    segs = [(0.0, 1.0, "A"), (1.5, 2.0, "A"), (2.0, 3.0, "B"), (2.5, 3.5, "C")]
    ch = change_times(segs, 4.0)
    # A -> silence -> A is no change; A -> B at 2.0; {B} -> {B, C} at 2.5; {B, C} -> {C} at 3.0
    assert np.allclose(ch, [2.0, 2.5, 3.0])
    assert len(change_times([(0.0, 3.0, "A")], 3.0)) == 0 and len(change_times([], 3.0)) == 0


def test_majority_label_by_overlap():
    segs = [(0.0, 1.0, "A"), (1.0, 1.5, "B"), (1.5, 4.0, "A")]
    assert majority(segs, 0.8, 1.6) == "B" and majority(segs, 0.0, 4.0) == "A" and majority(segs, 5.0, 6.0) is None


def test_allocate_is_proportional_with_one_per_stratum():
    sizes = {"a": 1000, "b": 500, "c": 20, "d": 3}
    alloc = allocate(sizes, 150)
    assert sum(alloc.values()) == 150
    assert all(1 <= alloc[k] <= sizes[k] for k in sizes)
    assert alloc["a"] > alloc["b"] > alloc["c"] >= alloc["d"] >= 1
    assert allocate({"x": 5, "y": 5}, 10) == {"x": 5, "y": 5}


def test_cohen_kappa_textbook_values():
    a = ["y"] * 20 + ["n"] * 30
    assert math.isclose(cohen_kappa(a, a)["kappa"], 1.0)
    # 2x2 table [[20, 5], [10, 15]]: po = 0.70, pe = 0.5*0.6 + 0.5*0.4 = 0.50, kappa = 0.40
    r1 = ["y"] * 25 + ["n"] * 25
    r2 = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    k = cohen_kappa(r1, r2)
    assert math.isclose(k["agreement"], 0.70) and math.isclose(k["kappa"], 0.40)
    assert cohen_kappa(["y"] * 5, ["y"] * 5)["kappa"] is None  # one category only: undefined


def test_wilson_interval():
    w = wilson(0, 20)
    assert w["rate"] == 0 and w["lo"] == 0 and math.isclose(w["hi"], 0.16113, abs_tol=1e-4)
    w = wilson(10, 20)
    assert math.isclose(w["lo"], 0.29930, abs_tol=1e-4) and math.isclose(w["hi"], 0.70070, abs_tol=1e-4)
    assert wilson(0, 0)["rate"] is None


def _pt(kind, chan, video, pause, pos, split="train"):
    return {"kind": kind, "speaker_id": chan, "video_id": video, "pause_bin": pause, "pos_bin": pos, "split": split,
            "elig_p0L1": kind == "ssdl", "elig_det_w1": kind == "ctrl"}


def test_cell_steps_point_weighted_mean_and_raw_rates():
    pts = {"s1": _pt("ssdl", "A", "v", "none", "0"), "s2": _pt("ssdl", "A", "v", "none", "1"),
           "s3": _pt("ssdl", "B", "w", "none", "0"),  # no control in its cell
           "c1": _pt("ctrl", "A", "v", "none", "0"), "c2": _pt("ctrl", "A", "v", "none", "0"),
           "c3": _pt("ctrl", "A", "v", "none", "1"), "c4": _pt("ctrl", "B", "w", "none", "1")}
    tau = 0.5
    vals = {"s1": {"peak": 0.9}, "s2": {"peak": 0.1}, "s3": {"peak": 0.9}, "c1": {"peak": 0.9},
            "c2": {"peak": -np.inf}, "c3": {"peak": 0.9}, "c4": {"peak": 0.1}}
    out = cell_steps(pts, vals, tau)
    # s1: 1 - mean(1, 0) = 0.5; s2: 0 - 1 = -1; s3 has no cell control
    assert out["c"]["n_ssdl"] == 2 and math.isclose(out["c"]["delta"], (0.5 - 1.0) / 2)
    assert math.isclose(out["b2"]["ssdl_rate"], 0.5) and math.isclose(out["b2"]["ctrl_rate"], 2 / 4)
    raw = pooled_raw(pts, vals, tau, lambda r: True)
    assert math.isclose(raw["all"]["ssdl_rate"], 2 / 3) and math.isclose(raw["all"]["ctrl_rate"], 2 / 4)
    # the old figure dropped points without a peak (-inf) from both denominators
    assert raw["figure"]["n_ctrl"] == 3 and math.isclose(raw["figure"]["ctrl_rate"], 2 / 3)
