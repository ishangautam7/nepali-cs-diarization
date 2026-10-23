#!/usr/bin/env python3
"""Hand-made cases with exact expected numbers for csdiar.analysis.score_m2b (run by pytest).

Every function gets small cases whose expected value is computed by hand (shown in a comment). An optional
cross-check against pyannote.metrics SegmentationRecall runs only when pyannote is importable in the box env;
otherwise ``match_turns`` is checked against a brute-force maximum matching on small cases.
"""
from __future__ import annotations

import itertools
import math
import os
import random
import unittest

from csdiar.analysis import score_m2b as sm
from csdiar.analysis.score_m2b import (C, HypSegment, det_summary, far_at_mr, fires, hypothesis_changes,
                                       match_turns, miss_rate, mr_at_far, parse_hypothesis_rttm, read_det,
                                       score_session, score_system)


def seg(start, end, label):
    return HypSegment(start, end, label)


def row(kind, t, pause_bin="none", phrase=False, left_s=2.0, right_s=2.0, rec="R1"):
    return {"kind": kind, "t": t, "pause_bin": pause_bin, "phrase": phrase,
            "left_s": left_s, "right_s": right_s, "recording": rec, "split": "dev"}


class TestHypothesisChanges(unittest.TestCase):
    def test_touching_vs_gapped(self):
        # A: [0,1] "s1"; B: [1,3] "s2" touch at 1.0 -> change = shared boundary 1.0
        # B: [1,3] "s2"; C: [4,5] "s3" gap [3,4] -> change = middle 3.5
        segs = [seg(0, 1, "s1"), seg(1, 3, "s2"), seg(4, 5, "s3")]
        self.assertEqual(hypothesis_changes(segs), [1.0, 3.5])

    def test_same_label_neighbours_not_counted(self):
        # same speaker "s1" across a pause [1,2] -> NOT a change; then "s2" at 2 touching -> change at 2.0
        segs = [seg(0, 1, "s1"), seg(2, 3, "s1"), seg(3, 4, "s2")]
        self.assertEqual(hypothesis_changes(segs), [3.0])

    def test_overlap_refused(self):
        # [0,2] and [1,3] overlap -> exclusive-input contract violated
        with self.assertRaises(ValueError):
            hypothesis_changes([seg(0, 2, "s1"), seg(1, 3, "s2")])

    def test_rttm_parse_then_changes(self):
        text = (
            "SPEAKER R1 1 0.000 1.000 <NA> <NA> s1 <NA> <NA>\n"
            "; a comment line\n"
            "SPEAKER R1 1 1.000 2.000 <NA> <NA> s2 <NA> <NA>\n"   # [1,3]
            "SPEAKER R1 1 4.000 1.000 <NA> <NA> s3 <NA> <NA>\n"   # [4,5]
        )
        segs = parse_hypothesis_rttm(text)
        self.assertEqual([(s.start, s.end, s.label) for s in segs],
                         [(0.0, 1.0, "s1"), (1.0, 3.0, "s2"), (4.0, 5.0, "s3")])
        self.assertEqual(hypothesis_changes(segs), [1.0, 3.5])


class TestFires(unittest.TestCase):
    def test_closed_endpoints(self):
        # c = 0.5; change at 1.0. point at 0.5 -> |0.5-1.0| = 0.5 == c -> fires (closed).
        # point at 1.5 -> 0.5 -> fires. point at 0.49 -> 0.51 > 0.5 -> no. point at 1.51 -> no.
        changes = [1.0]
        self.assertEqual(fires([0.5, 1.5, 0.49, 1.51, 1.0], changes, c=0.5),
                         [True, True, False, False, True])

    def test_at_least_one_change(self):
        # two changes; a point covered by either fires; no one-to-one constraint
        changes = [1.0, 5.0]
        self.assertEqual(fires([1.2, 5.4, 3.0], changes, c=0.5), [True, True, False])


class TestMatchTurns(unittest.TestCase):
    def test_change_between_two_turns_counted_once(self):
        # ref turns at 1.0 and 1.6; ONE change at 1.3 within +-0.5 of both.
        # one-to-one: the single change matches exactly one turn -> matched = 1, the other is a miss.
        # tie-break minimal distance: |1.3-1.0| = 0.3 vs |1.3-1.6| = 0.3 -> equal. Exact distance ties are resolved
        # deterministically toward the LATER turn (the DP's fill precedence); the direction carries no meaning.
        res = match_turns([1.0, 1.6], [1.3], c=0.5)
        self.assertEqual(res["matched"], 1)
        self.assertEqual(res["total"], 2)
        self.assertEqual([p[:2] for p in res["pairs"]], [(1, 0)])
        self.assertEqual(miss_rate([1.0, 1.6], [1.3], c=0.5), 0.5)

    def test_endpoints_closed(self):
        # turn at 0.0, change at 0.5 -> |0.5| == c -> matches.
        self.assertEqual(match_turns([0.0], [0.5], c=0.5)["matched"], 1)
        self.assertEqual(match_turns([0.0], [0.51], c=0.5)["matched"], 0)

    def test_minimal_distance_tie_break(self):
        # turns 1.0, 2.0; changes 1.4, 2.0. Max matching = 2.
        # Assign 1.0<-1.4 (0.4) and 2.0<-2.0 (0.0): total 0.4. Alternative 1.0<-? none closer; this is optimal.
        res = match_turns([1.0, 2.0], [1.4, 2.0], c=0.5)
        self.assertEqual(res["matched"], 2)
        self.assertAlmostEqual(res["total_distance"], 0.4, places=9)

    def test_steal_for_more_matches(self):
        # turns at 1.0 and 1.4; changes at 1.4 and 1.9 (c = 0.5).
        # Greedy earliest would match 1.0<-1.4, leaving 1.4 to 1.9 (0.5, ok) -> 2 matches.
        # Max matching = 2; ensure the DP finds it.
        res = match_turns([1.0, 1.4], [1.4, 1.9], c=0.5)
        self.assertEqual(res["matched"], 2)

    def test_no_change_all_missed(self):
        self.assertEqual(miss_rate([1.0, 2.0, 3.0], [], c=0.5), 1.0)

    def test_brute_force_small_cases(self):
        rng = random.Random(20261018)
        for _ in range(400):
            n = rng.randint(0, 5)
            m = rng.randint(0, 5)
            turns = [round(rng.uniform(0, 6), 2) for _ in range(n)]
            changes = [round(rng.uniform(0, 6), 2) for _ in range(m)]
            got = match_turns(turns, changes, c=0.5)["matched"]
            self.assertEqual(got, _brute_force_max_match(turns, changes, 0.5),
                             msg=f"turns={turns} changes={changes}")


def _brute_force_max_match(turns, changes, c):
    """Maximum one-to-one matches with |turn - change| <= c, by exhaustive search over small inputs."""
    best = 0
    idx = list(range(len(changes)))
    # try every injection of a subset of turns into distinct changes
    for r in range(min(len(turns), len(changes)), 0, -1):
        for tsub in itertools.combinations(range(len(turns)), r):
            for perm in itertools.permutations(idx, r):
                if all(abs(turns[tsub[k]] - changes[perm[k]]) <= c + 1e-9 for k in range(r)):
                    return r
    return best


class TestInclusionRule(unittest.TestCase):
    def test_min_units_10(self):
        # 12 ssdl points (each with a cell control) -> included; a session with 9 ds-dl turns -> excluded.
        rows = {}
        changes = {}
        durations = {}
        # session A: 12 ssdl points at distinct times, each in pause bin "none", plus 12 controls same bin
        ra = [row("ssdl", t=10 + i, left_s=2.0, right_s=2.0) for i in range(12)]
        ra += [row("ctrl", t=100 + i, pause_bin="none") for i in range(12)]
        rows["A"] = ra
        changes["A"] = []          # no hypothesis changes -> far = 0, but counts still apply
        durations["A"] = 600.0
        # session B: 9 ds-dl turns only
        rows["B"] = [row("dsdl", t=10 + i) for i in range(9)]
        changes["B"] = []
        durations["B"] = 600.0
        res = score_system(rows, changes, durations)
        self.assertIn("A", res["inclusion"]["ssdl"]["included"])
        self.assertIn("B", res["inclusion"]["dsdl"]["excluded"])
        self.assertEqual(res["inclusion"]["dsdl"]["excluded"]["B"], 9)
        self.assertNotIn("A", res["inclusion"]["dsdl"]["included"])  # A has 0 ds-dl turns


class TestH2CellParity(unittest.TestCase):
    def test_delta_far_std_by_hand(self):
        # Toy cell set, one session. Cells = pause bin.
        # Controls:
        #   bin "none":  three controls, two fire, one not -> cell fire rate = 2/3
        #   bin "gt300": two controls, one fires, one not  -> cell fire rate = 1/2
        # SS-DL points:
        #   p1 in "none", fires (1) -> delta = 1 - 2/3 = 1/3
        #   p2 in "none", not (0)   -> delta = 0 - 2/3 = -2/3
        #   p3 in "gt300", fires(1) -> delta = 1 - 1/2 = 1/2
        # mean delta = (1/3 - 2/3 + 1/2) / 3 = (1/6) / 3 = 1/18 = 0.055555...
        #
        # We realise "fires" by placing a change within +-0.5 of the points/controls we want to fire,
        # and far from the ones we do not. Collar c = 0.5.
        rows = []
        # changes placed exactly at the firing units' times
        fire_at = []
        # controls, bin none
        rows.append(row("ctrl", t=10.0, pause_bin="none")); fire_at.append(10.0)   # fires
        rows.append(row("ctrl", t=12.0, pause_bin="none")); fire_at.append(12.0)   # fires
        rows.append(row("ctrl", t=20.0, pause_bin="none"))                         # no (no change near 20)
        # controls, bin gt300 (the reader sets phrase=True for pause > 150 ms)
        rows.append(row("ctrl", t=30.0, pause_bin="gt300", phrase=True)); fire_at.append(30.0)  # fires
        rows.append(row("ctrl", t=40.0, pause_bin="gt300", phrase=True))                        # no
        # ss-dl points
        rows.append(row("ssdl", t=50.0, pause_bin="none")); fire_at.append(50.0)   # p1 fires
        rows.append(row("ssdl", t=60.0, pause_bin="none"))                         # p2 no
        rows.append(row("ssdl", t=70.0, pause_bin="gt300")); fire_at.append(70.0)  # p3 fires
        s = score_session("S", rows, fire_at, duration_s=600.0, c=0.5)
        self.assertEqual(s.n_ssdl, 3)
        self.assertEqual(s.n_ctrl, 5)
        self.assertEqual(s.n_phrase_ctrl, 2)              # the two gt300 controls (pause > 150 ms)
        self.assertEqual(s.n_ssdl_with_cell_controls, 3)
        self.assertAlmostEqual(s.far_ssdl, 2 / 3, places=9)   # 2 of 3 points fire
        self.assertAlmostEqual(s.far_ctrl, 3 / 5, places=9)   # 3 of 5 controls fire
        self.assertAlmostEqual(s.far_phrase_ctrl, 1 / 2, places=9)  # 1 of 2 phrase controls fire
        self.assertAlmostEqual(s.delta_far_std, 1 / 18, places=9)

    def test_ssdl_without_cell_control_counted(self):
        # an SS-DL point in a pause bin with NO control is skipped from delta and counted separately.
        rows = [
            row("ctrl", t=10.0, pause_bin="none"),
            row("ssdl", t=20.0, pause_bin="none"),     # has a cell control
            row("ssdl", t=30.0, pause_bin="gt300"),    # no gt300 control -> without
        ]
        s = score_session("S", rows, [], duration_s=600.0, c=0.5)
        self.assertEqual(s.n_ssdl_with_cell_controls, 1)
        self.assertEqual(s.n_ssdl_without_cell_controls, 1)

    def test_h2_rr_fields_by_hand(self):
        # brief fix 5: ssdl_fires_with_cell_controls / cell_ctrl_rate_sum, over the SS-DL points WITH a cell
        # control, in the same delta loop. Reuse the by-hand world of test_delta_far_std_by_hand:
        #   cell "none":  ctrl rate 2/3; cell "gt300": ctrl rate 1/2.
        #   p1 (none) fires, p2 (none) not, p3 (gt300) fires -> two of the three points fire.
        #   numerator  = fires over the three cell-matched points = 2
        #   denominator = 2/3 (p1) + 2/3 (p2) + 1/2 (p3) = 11/6
        rows = []
        fire_at = []
        rows.append(row("ctrl", t=10.0, pause_bin="none")); fire_at.append(10.0)
        rows.append(row("ctrl", t=12.0, pause_bin="none")); fire_at.append(12.0)
        rows.append(row("ctrl", t=20.0, pause_bin="none"))
        rows.append(row("ctrl", t=30.0, pause_bin="gt300", phrase=True)); fire_at.append(30.0)
        rows.append(row("ctrl", t=40.0, pause_bin="gt300", phrase=True))
        rows.append(row("ssdl", t=50.0, pause_bin="none")); fire_at.append(50.0)   # p1 fires
        rows.append(row("ssdl", t=60.0, pause_bin="none"))                         # p2 no
        rows.append(row("ssdl", t=70.0, pause_bin="gt300")); fire_at.append(70.0)  # p3 fires
        s = score_session("S", rows, fire_at, duration_s=600.0, c=0.5)
        self.assertAlmostEqual(s.ssdl_fires_with_cell_controls, 2.0, places=9)
        self.assertAlmostEqual(s.cell_ctrl_rate_sum, 2 / 3 + 2 / 3 + 1 / 2, places=9)
        # the point with no cell control (none here) never enters either sum
        self.assertEqual(s.n_ssdl_with_cell_controls, 3)

    def test_h2_rr_fields_default_on_bare_constructor(self):
        # the two fields have defaults so an existing positional/kw constructor without them still works.
        s = sm.SessionScore(session="S", n_ssdl=0, n_ctrl=0, n_phrase_ctrl=0, n_dsdl=0, n_dssl=0,
                            far_ssdl=0.0, far_ctrl=0.0, far_phrase_ctrl=0.0, delta_far_std=float("nan"),
                            n_ssdl_with_cell_controls=0, n_ssdl_without_cell_controls=0,
                            mr_dsdl=float("nan"), mr_dssl=float("nan"), changes_per_min=float("nan"),
                            duration_s=0.0)
        self.assertEqual(s.ssdl_fires_with_cell_controls, 0.0)
        self.assertEqual(s.cell_ctrl_rate_sum, 0.0)


class TestChangesPerMinute(unittest.TestCase):
    def test_rate(self):
        # 5 changes over 600 s = 10 min -> 0.5 changes/min
        rows = [row("ssdl", t=10.0)]
        s = score_session("S", rows, [1.0, 2.0, 3.0, 4.0, 5.0], duration_s=600.0)
        self.assertAlmostEqual(s.changes_per_min, 0.5, places=9)


class TestMRPoolShared(unittest.TestCase):
    def test_pool_is_shared_across_dsdl_and_dssl(self):
        # one DS-DL turn at 1.0 and one DS-SL turn at 1.2; ONE change at 1.1 within +-0.5 of both.
        # Shared pool, one-to-one: the single change matches EXACTLY ONE of the two turns, never both.
        # So exactly one of MR_DS-DL / MR_DS-SL is 0.0 and the other 1.0 (the change cannot count twice).
        rows = [row("dsdl", t=1.0), row("dssl", t=1.2)]
        s = score_session("S", rows, [1.1], duration_s=600.0, c=0.5)
        self.assertEqual(s.n_dsdl, 1)
        self.assertEqual(s.n_dssl, 1)
        self.assertEqual({s.mr_dsdl, s.mr_dssl}, {0.0, 1.0})  # one matched, one missed -- pool shared
        # exact tie (both 0.1 away): the later turn (the DS-SL one at 1.2) wins, deterministically
        self.assertEqual((s.mr_dsdl, s.mr_dssl), (1.0, 0.0))

    def test_direction_when_unambiguous(self):
        # change at 1.02 is strictly closer to the DS-DL turn (1.0, dist 0.02) than the DS-SL turn (1.3, 0.28),
        # so the DS-DL turn is the one matched: MR_DS-DL = 0, MR_DS-SL = 1.
        rows = [row("dsdl", t=1.0), row("dssl", t=1.3)]
        s = score_session("S", rows, [1.02], duration_s=600.0, c=0.5)
        self.assertAlmostEqual(s.mr_dsdl, 0.0, places=9)
        self.assertAlmostEqual(s.mr_dssl, 1.0, places=9)

    def test_two_changes_match_both(self):
        # two changes, one near each turn -> both matched, no miss.
        rows = [row("dsdl", t=1.0), row("dssl", t=5.0)]
        s = score_session("S", rows, [1.05, 4.95], duration_s=600.0, c=0.5)
        self.assertAlmostEqual(s.mr_dsdl, 0.0, places=9)
        self.assertAlmostEqual(s.mr_dssl, 0.0, places=9)


class TestProbitInterpolation(unittest.TestCase):
    # DET points are (threshold, FAR_SS-DL, MR_DS-DL) triples; the DET is read in THRESHOLD order (§5).

    def test_known_pair_monotone(self):
        # A monotone two-point sweep reads as before. Triples (thr, FAR, MR) = (0.4, 0.05, 0.20), (0.6, 0.20, 0.05).
        # mr_at_far(0.10): the single segment crosses FAR = 0.10 once; interpolate MR on probit axes.
        from statistics import NormalDist
        nd = NormalDist()
        p05, p10, p20 = nd.inv_cdf(0.05), nd.inv_cdf(0.10), nd.inv_cdf(0.20)
        w = (p10 - p05) / (p20 - p05)
        # segment FAR 0.05->0.20, MR 0.20->0.05 at target FAR 0.10
        expected_mr = nd.cdf(nd.inv_cdf(0.20) + w * (nd.inv_cdf(0.05) - nd.inv_cdf(0.20)))
        pts = [(0.4, 0.05, 0.20), (0.6, 0.20, 0.05)]
        r = read_det(pts, 0.10, along="far")
        self.assertEqual(r.n_crossings, 1)
        self.assertAlmostEqual(r.value, expected_mr, places=9)
        self.assertAlmostEqual(mr_at_far(pts, 0.10), expected_mr, places=9)

        # far_at_mr(0.10): target MR = 0.10 between MR 0.20 and 0.05; one crossing; read FAR on probit axes.
        w2 = (nd.inv_cdf(0.10) - nd.inv_cdf(0.20)) / (nd.inv_cdf(0.05) - nd.inv_cdf(0.20))
        expected_far = nd.cdf(nd.inv_cdf(0.05) + w2 * (nd.inv_cdf(0.20) - nd.inv_cdf(0.05)))
        r2 = read_det(pts, 0.10, along="mr")
        self.assertEqual(r2.n_crossings, 1)
        self.assertAlmostEqual(r2.value, expected_far, places=9)
        self.assertAlmostEqual(far_at_mr(pts, 0.10), expected_far, places=9)

    def test_reviewer_non_monotone_trace_best_of_two(self):
        # Reviewer's round-2 trace, as a threshold sweep (thr, FAR, MR): MR in threshold order is 0.30, 0.20, 0.40,
        # so target MR = 0.25 is crossed TWICE (0.30->0.20 and 0.20->0.40). The summary is the BEST (lowest) FAR,
        # which falls on the second segment, and n_crossings == 2.
        from statistics import NormalDist
        nd = NormalDist()
        pts = [(0.3, 0.30, 0.30), (0.4, 0.10, 0.20), (0.5, 0.05, 0.40)]
        # second crossing: MR 0.20->0.40 at 0.25, FAR 0.10->0.05
        w = (nd.inv_cdf(0.25) - nd.inv_cdf(0.20)) / (nd.inv_cdf(0.40) - nd.inv_cdf(0.20))
        expected_far = nd.cdf(nd.inv_cdf(0.10) + w * (nd.inv_cdf(0.05) - nd.inv_cdf(0.10)))
        r = read_det(pts, 0.25, along="mr")
        self.assertEqual(r.n_crossings, 2)
        self.assertAlmostEqual(r.value, expected_far, places=9)
        self.assertAlmostEqual(far_at_mr(pts, 0.25), expected_far, places=9)

    def test_threshold_order_not_x_sorting(self):
        # Threshold order and FAR-sorting DISAGREE. Thresholds 0.3/0.4/0.5/0.6 with (FAR, MR) =
        # (0.20, 0.10), (0.05, 0.15), (0.15, 0.30), (0.02, 0.50). Target FAR = 0.10.
        # FAR in threshold order: 0.20, 0.05, 0.15, 0.02 -> THREE segments straddle 0.10:
        #   0.3-0.4 (0.20->0.05), 0.4-0.5 (0.05->0.15), 0.5-0.6 (0.15->0.02). n_crossings == 3.
        # The best (lowest MR) is on the 0.3-0.4 segment (MR 0.10->0.15, lowest of the three).
        from statistics import NormalDist
        nd = NormalDist()
        pts = [(0.3, 0.20, 0.10), (0.4, 0.05, 0.15), (0.5, 0.15, 0.30), (0.6, 0.02, 0.50)]
        # expected on 0.3-0.4: FAR 0.20->0.05 at 0.10, MR 0.10->0.15
        w = (nd.inv_cdf(0.10) - nd.inv_cdf(0.20)) / (nd.inv_cdf(0.05) - nd.inv_cdf(0.20))
        expected_mr = nd.cdf(nd.inv_cdf(0.10) + w * (nd.inv_cdf(0.15) - nd.inv_cdf(0.10)))
        r = read_det(pts, 0.10, along="far")
        self.assertEqual(r.n_crossings, 3)
        self.assertAlmostEqual(r.value, expected_mr, places=9)
        self.assertAlmostEqual(mr_at_far(pts, 0.10), expected_mr, places=9)

    def test_node_hit_reads_raw_value(self):
        # A sweep point that sits exactly ON the target FAR reads the raw MR at that node. Here the 0.4 node has
        # FAR = 0.10 exactly (a local extremum in FAR: neighbours are 0.20 and 0.15, both above), so it is a node
        # crossing giving MR = 0.25 directly, n_crossings == 1.
        pts = [(0.3, 0.20, 0.30), (0.4, 0.10, 0.25), (0.5, 0.15, 0.40)]
        r = read_det(pts, 0.10, along="far")
        self.assertEqual(r.n_crossings, 1)
        self.assertAlmostEqual(r.value, 0.25, places=12)

    def test_run_of_points_at_target_counts_once(self):
        # A run of CONSECUTIVE sweep points at the target FAR counts as ONE crossing; the reading is the run's
        # best (lowest MR). Three consecutive nodes at FAR 0.10 with MR 0.30/0.20/0.35 -> one crossing, MR 0.20.
        pts = [(0.2, 0.40, 0.05), (0.3, 0.10, 0.30), (0.4, 0.10, 0.20), (0.5, 0.10, 0.35), (0.6, 0.02, 0.60)]
        r = read_det(pts, 0.10, along="far")
        self.assertEqual(r.n_crossings, 1)
        self.assertAlmostEqual(r.value, 0.20, places=12)

    def test_not_reached_is_nan_with_zero_crossings(self):
        # A sweep that never crosses the target FAR = 0.10: all FAR strictly below. nan, 0 crossings, no extrapolation.
        pts = [(0.3, 0.01, 0.6), (0.4, 0.02, 0.5), (0.5, 0.04, 0.4)]
        r = read_det(pts, 0.10, along="far")
        self.assertEqual(r.n_crossings, 0)
        self.assertTrue(math.isnan(r.value))
        self.assertTrue(math.isnan(mr_at_far(pts, 0.10)))

    def test_duplicate_thresholds_refused(self):
        with self.assertRaises(ValueError):
            read_det([(0.3, 0.10, 0.20), (0.3, 0.15, 0.25), (0.4, 0.05, 0.40)], 0.10, along="far")

    def test_non_finite_refused(self):
        with self.assertRaises(ValueError):
            read_det([(0.3, 0.10, 0.20), (0.4, float("nan"), 0.25)], 0.10, along="far")
        with self.assertRaises(ValueError):
            read_det([(float("inf"), 0.10, 0.20), (0.4, 0.05, 0.25)], 0.10, along="far")

    def test_old_style_pairs_refused_with_triple_message(self):
        # m1: the public DET shape changed from (FAR, MR) pairs to (threshold, FAR, MR) triples. A caller still
        # passing 2-tuples must get the module's own clear "triples" message, not an opaque unpack ValueError.
        with self.assertRaisesRegex(ValueError, "triples"):
            read_det([(0.10, 0.20), (0.30, 0.40)], 0.10, along="far")

    def test_shuffled_input_same_result(self):
        # Input order of the triples does not matter; read_det sorts by threshold.
        pts = [(0.3, 0.20, 0.10), (0.4, 0.05, 0.15), (0.5, 0.15, 0.30), (0.6, 0.02, 0.50)]
        shuffled = [pts[2], pts[0], pts[3], pts[1]]
        a = read_det(pts, 0.10, along="far")
        b = read_det(shuffled, 0.10, along="far")
        self.assertEqual(a.n_crossings, b.n_crossings)
        self.assertAlmostEqual(a.value, b.value, places=12)

    def test_det_summary_reports_not_reached_and_multi_crossings(self):
        # pooled never reaches FAR 10%; s1 reaches it (one crossing); s2 does not. A separate non-monotone sweep
        # (s3) crosses FAR 10% more than once so det_summary reports it under multi-crossing counts.
        pooled = [(0.3, 0.01, 0.6), (0.4, 0.02, 0.5), (0.5, 0.04, 0.4)]   # never reaches FAR 10%
        s1 = [(0.4, 0.05, 0.20), (0.6, 0.20, 0.05)]                       # reaches FAR 10%, one crossing
        s2 = [(0.3, 0.01, 0.6), (0.5, 0.04, 0.4)]                         # does not
        s3 = [(0.3, 0.20, 0.10), (0.4, 0.05, 0.15), (0.5, 0.15, 0.30), (0.6, 0.02, 0.50)]  # 3 crossings at FAR 10%
        out = det_summary(pooled, [s1, s2, s3])
        self.assertTrue(math.isnan(out["pooled"]["mr_at_far10"]))
        self.assertIn("mr_at_far10", out["pooled_not_reached"])
        self.assertEqual(out["pooled_crossings"]["mr_at_far10"], 0)
        bs = out["by_session"]["mr_at_far10"]
        self.assertEqual((bs["n_reached"], bs["n_not_reached"]), (2, 1))   # s1 and s3 reach it, s2 does not
        self.assertEqual(bs["n_multi_crossing"], 1)                        # s3 crosses 3 times
        self.assertAlmostEqual(bs["mean"],
                               (mr_at_far(s1, 0.10) + mr_at_far(s3, 0.10)) / 2.0, places=12)

    def test_det_summary_multi_crossing_pooled(self):
        # A non-monotone pooled sweep crossed more than once at a secondary FAR-at-MR point is reported.
        pooled = [(0.3, 0.30, 0.30), (0.4, 0.10, 0.20), (0.5, 0.05, 0.40)]  # MR 0.30,0.20,0.40 -> crosses 0.25 twice
        # pick targets that this sweep reaches; MR 0.05 -> far_at_mr05 crosses once, MR 0.10 -> far_at_mr10 ...
        out = det_summary(pooled)
        # far_at_mr05: MR in order 0.30,0.20,0.40 never dips to 0.05 -> not reached
        self.assertIn("far_at_mr05", out["pooled_not_reached"])
        self.assertEqual(out["pooled_crossings"]["far_at_mr05"], 0)

    def test_det_summary_pins_far_at_mr_against_standalone(self):
        # m1: pin the POOLED far_at_mr05 / far_at_mr10 against the standalone reader, so a role/sign swap between
        # the two secondaries cannot pass silently.
        pooled = [(0.3, 0.02, 0.02), (0.4, 0.08, 0.06), (0.5, 0.20, 0.14), (0.6, 0.35, 0.30)]
        out = det_summary(pooled)
        # these targets are reached (MR spans 0.02..0.30, so 0.05 and 0.10 are both crossed once)
        self.assertFalse(math.isnan(out["pooled"]["far_at_mr05"]))
        self.assertFalse(math.isnan(out["pooled"]["far_at_mr10"]))
        self.assertAlmostEqual(out["pooled"]["far_at_mr05"], far_at_mr(pooled, 0.05), places=12)
        self.assertAlmostEqual(out["pooled"]["far_at_mr10"], far_at_mr(pooled, 0.10), places=12)
        # and the two are distinct, so a swap would be caught
        self.assertNotAlmostEqual(out["pooled"]["far_at_mr05"], out["pooled"]["far_at_mr10"], places=6)

    def test_det_summary_by_session_far_at_mr_means_against_standalone(self):
        # m2: pin the BY_SESSION means of far_at_mr05 / far_at_mr10 against the standalone far_at_mr over the
        # reaching sessions, so a role/sign swap confined to the by_session branch cannot pass silently.
        # s1, s2 both reach MR 0.05 and 0.10 (MR spans 0.02..0.30); s3 never dips to 0.05 or 0.10 (MR >= 0.20),
        # so it is not-reached and must be excluded from the mean, not averaged as nan.
        s1 = [(0.3, 0.02, 0.02), (0.4, 0.08, 0.06), (0.5, 0.20, 0.14), (0.6, 0.35, 0.30)]
        s2 = [(0.3, 0.03, 0.03), (0.4, 0.07, 0.07), (0.5, 0.18, 0.12), (0.6, 0.30, 0.28)]
        s3 = [(0.3, 0.05, 0.20), (0.4, 0.10, 0.25), (0.5, 0.15, 0.30)]        # MR never reaches 0.05 / 0.10
        out = det_summary(s1, [s1, s2, s3])
        for name, target in (("far_at_mr05", 0.05), ("far_at_mr10", 0.10)):
            bs = out["by_session"][name]
            self.assertEqual((bs["n_reached"], bs["n_not_reached"]), (2, 1))   # s1, s2 reach it; s3 does not
            self.assertAlmostEqual(bs["mean"],
                                   (far_at_mr(s1, target) + far_at_mr(s2, target)) / 2.0, places=12)
        # the two secondaries are distinct in the by_session means too, so a swap between them would be caught
        self.assertNotAlmostEqual(out["by_session"]["far_at_mr05"]["mean"],
                                  out["by_session"]["far_at_mr10"]["mean"], places=6)


class TestSegmentationRecallParity(unittest.TestCase):
    """Verify match_turns on >= 200 random cases: against pyannote.metrics SegmentationRecall when importable
    (brief's preferred cross-check), and ALWAYS against a brute-force maximum matching (the fallback)."""

    def test_brute_force_200_cases(self):
        rng = random.Random(1234)
        for _ in range(250):
            turns = [round(rng.uniform(0, 8), 2) for _ in range(rng.randint(0, 6))]
            changes = [round(rng.uniform(0, 8), 2) for _ in range(rng.randint(0, 6))]
            self.assertEqual(match_turns(turns, changes, c=0.5)["matched"],
                             _brute_force_max_match(turns, changes, 0.5),
                             msg=f"turns={turns} changes={changes}")

    def test_against_pyannote_segmentation_recall(self):
        try:
            from pyannote.metrics.segmentation import SegmentationRecall  # noqa: F401
        except Exception:
            self.skipTest("pyannote.metrics not importable; brute-force parity covers match_turns")
            return
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.segmentation import SegmentationRecall

        # SegmentationRecall(tolerance=c) assigns reference boundaries to hypothesis boundaries one-to-one
        # (an internal maximum matching) and returns detected_ref / n_ref. To make both metrics unambiguous we
        # keep every boundary at least 2*tol + 0.2 apart from every other boundary (within and across the two
        # sets), so no change can lie within tol of two turns and no tie arises; recall * n_turns then equals
        # our one-to-one matched count. (Coincident / closely spaced boundaries are exercised by the brute-force
        # test above, where our one-to-one semantics are the reference.)
        tol = 0.5
        rng = random.Random(99)
        n_checked = 0
        T = 40.0
        for _ in range(600):
            k_turns = rng.randint(1, 5)
            k_ch = rng.randint(0, 5)
            picks = _well_separated(rng, k_turns + k_ch, T, min_gap=2 * tol + 0.2)
            if picks is None:
                continue
            turns = sorted(picks[:k_turns])
            changes = sorted(picks[k_turns:])
            ref = _boundaries_to_annotation(turns, T)
            hyp = _boundaries_to_annotation(changes, T)
            recall = SegmentationRecall(tolerance=tol)(ref, hyp)
            py_matched = round(recall * len(turns))
            self.assertEqual(match_turns(turns, changes, c=tol)["matched"], py_matched,
                             msg=f"turns={turns} changes={changes} recall={recall}")
            n_checked += 1
            if n_checked >= 200:
                break
        self.assertGreaterEqual(n_checked, 200, "expected at least 200 well-separated parity cases")


def _well_separated(rng, k, T, min_gap):
    """k times in (0.6, T-0.6), pairwise at least min_gap apart; None if it cannot be placed in a few tries."""
    for _ in range(50):
        pts = sorted(round(rng.uniform(0.6, T - 0.6), 2) for _ in range(k))
        if all(b - a >= min_gap for a, b in zip(pts, pts[1:])):
            return pts
    return None


def _boundaries_to_annotation(inner, T):
    from pyannote.core import Annotation, Segment
    ann = Annotation()
    cuts = [0.0] + list(inner) + [T]
    for i in range(len(cuts) - 1):
        ann[Segment(cuts[i], cuts[i + 1])] = f"s{i}"
    return ann


class TestDerSingleRecordingGuard(unittest.TestCase):
    """m3: der()'s m5 single-recording guard. Needs pyannote (load_rttm / DiarizationErrorRate), so it is
    skipped where pyannote is not importable (this host), and runs on the box."""

    def test_two_recording_rttm_raises(self):
        try:
            import pyannote.metrics.diarization  # noqa: F401
            import pyannote.database.util        # noqa: F401
        except Exception:
            self.skipTest("pyannote not importable; der() guard is exercised on the box")
            return
        import tempfile
        from csdiar.analysis.score_m2b import der

        # an RTTM holding TWO distinct recording URIs (REC_A and REC_B) -> der() must refuse it loudly rather
        # than silently scoring the first URI (the wrong recording).
        two_rec = (
            "SPEAKER REC_A 1 0.000 1.000 <NA> <NA> s1 <NA> <NA>\n"
            "SPEAKER REC_A 1 1.000 1.000 <NA> <NA> s2 <NA> <NA>\n"
            "SPEAKER REC_B 1 0.000 1.000 <NA> <NA> s1 <NA> <NA>\n"
        )
        one_rec = "SPEAKER REC_A 1 0.000 2.000 <NA> <NA> s1 <NA> <NA>\n"
        with tempfile.TemporaryDirectory() as d:
            two_path = os.path.join(d, "two.rttm")
            one_path = os.path.join(d, "one.rttm")
            with open(two_path, "w") as f:
                f.write(two_rec)
            with open(one_path, "w") as f:
                f.write(one_rec)
            # the two-recording RTTM as the REFERENCE triggers the guard in _one
            with self.assertRaisesRegex(ValueError, "exactly one recording"):
                der(two_path, one_path)
            # and as the HYPOTHESIS too
            with self.assertRaisesRegex(ValueError, "exactly one recording"):
                der(one_path, two_path)


if __name__ == "__main__":
    unittest.main()
