#!/usr/bin/env python3
"""Unit tests for csdiar.pipelines.m2b_hooks (PREREGISTRATION_M2B.md §3, §4).

Stdlib-only (math + unittest): the box's analysis environment has Python 3.12 without numpy or pytest.
Run with:  python3 -m unittest -v tests.test_m2b_hooks
"""
import math
import unittest

from csdiar.pipelines.m2b_hooks import (
    b2_merge,
    cosine_distance,
    lambda_at,
    m2_relabel,
    majority_language,
    two_language_jsd,
)


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


# reference base-2 JSD on 2-entry inputs, mirroring confirm_h4.jsd, for the cross-check
def _ref_jsd2(p, q):
    m = [0.5 * (a + b) for a, b in zip(p, q)]

    def kl(a, b):
        return sum(x * math.log2(max(x, 1e-12) / max(y, 1e-12)) for x, y in zip(a, b) if x > 0)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


class TestTwoLanguageJSD(unittest.TestCase):
    def test_one_hot_opposites_is_one_bit(self):
        self.assertAlmostEqual(two_language_jsd([1.0, 0.0], [0.0, 1.0], (0, 1)), 1.0, places=9)

    def test_equal_inputs_is_zero(self):
        self.assertAlmostEqual(two_language_jsd([0.6, 0.4], [0.6, 0.4], (0, 1)), 0.0, places=9)

    def test_hand_computed_value(self):
        # p=[0.75,0.25], q=[0.25,0.75], m=[0.5,0.5]
        # kl(p,m)=0.75*log2(1.5)+0.25*log2(0.5) = 0.4387219 - 0.25 = 0.1887219 (same for q by symmetry)
        expected = 0.75 * math.log2(1.5) + 0.25 * math.log2(0.5)
        self.assertAlmostEqual(two_language_jsd([0.75, 0.25], [0.25, 0.75], (0, 1)),
                               expected, places=9)
        self.assertAlmostEqual(expected, 0.18872187554086717, places=12)

    def test_matches_reference_jsd_on_two_entries(self):
        for p, q in ([[0.9, 0.1], [0.3, 0.7]], [[0.5, 0.5], [0.99, 0.01]]):
            self.assertAlmostEqual(two_language_jsd(p, q, (0, 1)), _ref_jsd2(p, q), places=9)

    def test_bounded_by_one_bit(self):
        for p, q in ([[1.0, 0.0], [0.0, 1.0]], [[0.8, 0.2], [0.1, 0.9]], [[0.5, 0.5], [0.5, 0.5]]):
            v = two_language_jsd(p, q, (0, 1))
            self.assertLessEqual(v, 1.0 + 1e-12)
            self.assertGreaterEqual(v, -1e-12)

    def test_renormalisation_ignores_other_entries(self):
        # entries 0 and 2 carry the (ne, en) mass; a big "other" entry at index 1 must not matter
        p = [0.75, 9.0, 0.25]   # unnormalised; renorm over (0,2) -> [0.75, 0.25]
        q = [0.25, 0.01, 0.75]  # renorm over (0,2) -> [0.25, 0.75]
        expected = 0.75 * math.log2(1.5) + 0.25 * math.log2(0.5)
        self.assertAlmostEqual(two_language_jsd(p, q, (0, 2)), expected, places=9)

    def test_zero_mass_over_pair_is_none(self):
        # left side has no mass on indices (0,1)
        self.assertIsNone(two_language_jsd([0.0, 0.0, 1.0], [0.5, 0.5, 0.0], (0, 1)))
        self.assertIsNone(two_language_jsd([0.5, 0.5, 0.0], [0.0, 0.0, 1.0], (0, 1)))

    def test_validation(self):
        with self.assertRaises(ValueError):
            two_language_jsd([], [1.0], (0, 1))
        with self.assertRaises(ValueError):
            two_language_jsd([0.5, 0.5], [0.5, 0.5], (0, 0))     # non-distinct pair
        with self.assertRaises(ValueError):
            two_language_jsd([0.5, 0.5], [0.5, 0.5], (0, 5))     # out of range


class TestLambdaAt(unittest.TestCase):
    def _post(self, n):
        # n windows on the 0.1 s grid; alternate one-hot (ne, en) so adjacent 1.0-s-apart windows differ
        out = []
        for k in range(n):
            out.append([1.0, 0.0] if (k // 10) % 2 == 0 else [0.0, 1.0])
        return out

    def test_in_range_value(self):
        # 30 windows; at t=1.5 -> k=15, kl=5: window5=[1,0], window15=[0,1] -> 1 bit
        post = self._post(30)
        self.assertAlmostEqual(lambda_at(1.5, post, (0, 1)), 1.0, places=9)

    def test_k_is_round_of_t_over_hop(self):
        post = self._post(30)
        # t=1.53 -> round(15.3)=15 ; t=1.57 -> round(15.7)=16
        self.assertAlmostEqual(lambda_at(1.53, post, (0, 1)), lambda_at(1.5, post, (0, 1)), places=9)

    def test_out_of_range_low(self):
        post = self._post(30)
        # k - lag < 0 : t=0.5 -> k=5, kl=-5 -> None
        self.assertIsNone(lambda_at(0.5, post, (0, 1)))

    def test_out_of_range_high(self):
        post = self._post(30)
        # k >= len(post): t=3.0 -> k=30 >= 30 -> None
        self.assertIsNone(lambda_at(3.0, post, (0, 1)))

    def test_edges_exact(self):
        post = self._post(30)
        # first in-range k is lag=10 -> t = 1.0
        self.assertIsNotNone(lambda_at(1.0, post, (0, 1)))
        # last in-range k is len-1 = 29 -> t = 2.9
        self.assertIsNotNone(lambda_at(2.9, post, (0, 1)))

    def test_validation(self):
        with self.assertRaises(ValueError):
            lambda_at(1.0, [], (0, 1))


class TestMajorityLanguage(unittest.TestCase):
    def test_most_duration(self):
        self.assertEqual(majority_language([(0.5, "ne"), (2.0, "en"), (0.3, "ne")]), "en")

    def test_tie_lexicographically_smallest(self):
        self.assertEqual(majority_language([(1.0, "en"), (1.0, "ne")]), "en")
        self.assertEqual(majority_language([(1.0, "zz"), (1.0, "aa")]), "aa")

    def test_validation(self):
        with self.assertRaises(ValueError):
            majority_language([])
        with self.assertRaises(ValueError):
            majority_language([(-1.0, "en")])


class TestB2Merge(unittest.TestCase):
    def _speakers(self, entries):
        return {lbl: {"centroid": _unit(cen), "lang_dur": ld} for lbl, cen, ld in entries}

    def test_merges_cross_language_split_below_threshold_plus_delta(self):
        # two near-identical centroids, different majority languages -> one speaker split across languages
        sp = self._speakers([
            ("A", [1.0, 0.02, 0.0], {"ne": 10.0}),
            ("B", [1.0, 0.0, 0.0], {"en": 6.0}),
        ])
        d = 1.0 - sum(x * y for x, y in zip(_unit([1.0, 0.02, 0.0]), [1.0, 0.0, 0.0]))
        self.assertLess(d, 0.001)
        # threshold just below d, delta lifts it over -> merge
        res = b2_merge(sp, threshold=d - 0.0005, delta=0.001)
        self.assertEqual(res["A"], res["B"])      # merged
        self.assertEqual(res["A"], "A")            # A has more duration -> keeps A

    def test_not_merged_when_distance_above_threshold_plus_delta(self):
        sp = self._speakers([
            ("A", [1.0, 0.0, 0.0], {"ne": 10.0}),
            ("B", [0.0, 1.0, 0.0], {"en": 6.0}),   # orthogonal -> distance 1.0
        ])
        res = b2_merge(sp, threshold=0.3, delta=0.2)   # 0.5 < 1.0 -> no merge
        self.assertNotEqual(res["A"], res["B"])

    def test_same_language_never_merged(self):
        # identical centroids, SAME language: must not merge even at a generous threshold+delta
        sp = self._speakers([
            ("A", [1.0, 0.0], {"en": 5.0}),
            ("B", [1.0, 0.0], {"en": 5.0}),
        ])
        res = b2_merge(sp, threshold=0.5, delta=0.5)
        self.assertNotEqual(res["A"], res["B"])

    def test_delta_zero_is_noop(self):
        sp = self._speakers([
            ("A", [1.0, 0.02, 0.0], {"ne": 10.0}),
            ("B", [1.0, 0.0, 0.0], {"en": 6.0}),
        ])
        d = 1.0 - sum(x * y for x, y in zip(_unit([1.0, 0.02, 0.0]), [1.0, 0.0, 0.0]))
        # threshold below d, delta 0 -> d not < threshold -> no change
        res = b2_merge(sp, threshold=d - 0.0005, delta=0.0)
        self.assertEqual(res, {"A": "A", "B": "B"})

    def test_iterative_merge_with_recomputed_centroids(self):
        # A (ne) and B (en) are close; C (en) is close only to the A+B merged centroid, far from A alone.
        # The merge must iterate: A+B first, then (A+B) with C once the centroid moves toward C.
        A = _unit([1.0, 0.0, 0.0])
        B = _unit([0.0, 1.0, 0.0])          # A-B distance = 1.0
        # merged A+B centroid ~ unit([1,1,0]); place C near that direction but with a different language
        C = _unit([1.0, 1.0, 0.0])
        sp = {
            "A": {"centroid": A, "lang_dur": {"ne": 5.0}},
            "B": {"centroid": B, "lang_dur": {"en": 5.0}},
            "C": {"centroid": C, "lang_dur": {"ne": 1.0}},
        }
        # distance A-B = 1.0; A-C = 1 - cos(0deg vs 45deg) = 1 - 0.7071 = 0.2929; B-C same.
        # With threshold+delta = 0.5: first merge is the SMALLEST cross-lang distance -> A-C (0.293) or B-C.
        # majority langs: A=ne, B=en, C=ne. Cross-language pairs: A-B, B-C. (A-C are both ne -> skipped.)
        # smallest cross-language distance is B-C (0.293) -> merge, then re-evaluate.
        res = b2_merge(sp, threshold=0.4, delta=0.1)
        # B and C merge (B more duration -> keep B). Then B(+C) majority: en 5 vs ne 1 -> en; A is ne.
        # new B centroid = weighted mean of B and C, still cross-language vs A; check all three collapse if close.
        self.assertEqual(res["B"], res["C"])
        # A stays separate from the merged cluster when still far (distance A-(B+C) > 0.5)
        merged_label = res["B"]
        self.assertIn("A", res)
        # sanity: result maps every original label
        self.assertEqual(set(res), {"A", "B", "C"})
        self.assertIn(merged_label, {"A", "B", "C"})

    def test_keep_label_more_duration_then_smaller(self):
        sp = self._speakers([
            ("Z", [1.0, 0.0], {"ne": 2.0}),
            ("A", [1.0, 0.0], {"en": 9.0}),   # same centroid, A has more duration
        ])
        res = b2_merge(sp, threshold=0.5, delta=0.0 + 0.5)
        self.assertEqual(res["Z"], "A")
        self.assertEqual(res["A"], "A")

    def test_validation(self):
        with self.assertRaises(ValueError):
            b2_merge({}, 0.5, 0.0)
        with self.assertRaises(ValueError):
            b2_merge({"A": {"centroid": [0.6, 0.6], "lang_dur": {"en": 1.0}}}, 0.5, 0.0)  # non-unit


class TestM2Relabel(unittest.TestCase):
    def _seg(self, start, end, label, emb):
        return {"start": start, "end": end, "label": label, "emb": _unit(emb)}

    def _lam_const(self, value):
        return lambda t: value

    def test_relabels_short_cross_language_segment(self):
        # S0 (spk A, long) | S1 (spk B, short, close emb) | S2 (spk A, long)
        segs = [
            self._seg(0.0, 3.0, "A", [1.0, 0.0, 0.0]),
            self._seg(3.1, 3.5, "B", [1.0, 0.02, 0.0]),   # short, embedding close to A
            self._seg(3.6, 6.0, "A", [1.0, 0.0, 0.0]),
        ]
        new, recs = m2_relabel(segs, self._lam_const(0.9), tau0=0.5, alpha=0.0)
        self.assertEqual([s["label"] for s in new], ["A", "A", "A"])
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["index"], 1)
        self.assertEqual((recs[0]["old"], recs[0]["new"]), ("B", "A"))

    def test_not_relabelled_when_lambda_below_min(self):
        segs = [
            self._seg(0.0, 3.0, "A", [1.0, 0.0, 0.0]),
            self._seg(3.1, 3.5, "B", [1.0, 0.02, 0.0]),
            self._seg(3.6, 6.0, "A", [1.0, 0.0, 0.0]),
        ]
        new, recs = m2_relabel(segs, self._lam_const(0.3), tau0=0.5, alpha=0.0)  # lam < 0.5
        self.assertEqual([s["label"] for s in new], ["A", "B", "A"])
        self.assertEqual(recs, [])

    def test_not_relabelled_when_distance_too_large(self):
        segs = [
            self._seg(0.0, 3.0, "A", [1.0, 0.0, 0.0]),
            self._seg(3.1, 3.5, "B", [0.0, 1.0, 0.0]),    # orthogonal -> distance 1.0
            self._seg(3.6, 6.0, "A", [1.0, 0.0, 0.0]),
        ]
        new, recs = m2_relabel(segs, self._lam_const(0.9), tau0=0.5, alpha=0.0)  # 1.0 !< 0.5
        self.assertEqual([s["label"] for s in new], ["A", "B", "A"])

    def test_respects_max_gap(self):
        segs = [
            self._seg(0.0, 3.0, "A", [1.0, 0.0, 0.0]),
            self._seg(6.0, 6.4, "B", [1.0, 0.02, 0.0]),   # gap 3.0 > max_gap 2.0
            self._seg(6.5, 9.0, "A", [1.0, 0.0, 0.0]),
        ]
        new, recs = m2_relabel(segs, self._lam_const(0.9), tau0=0.5, alpha=0.0, max_gap=2.0)
        # first pair gap (6.0-3.0=3.0) too large; second pair gap (6.5-6.4=0.1) ok -> B relabelled by S2
        self.assertEqual(new[0]["label"], "A")
        self.assertEqual(new[1]["label"], "A")  # relabelled via the S1-S2 pair
        self.assertEqual(new[2]["label"], "A")

    def test_double_claim_resolved_by_distance(self):
        # short middle B claimed by both neighbours; it takes the label of the closer neighbour.
        # here both neighbours are label "A" anyway, so craft distinct neighbour labels A and C.
        segs = [
            self._seg(0.0, 3.0, "A", [1.0, 0.05, 0.0]),   # distance to B larger
            self._seg(3.1, 3.5, "B", [1.0, 0.0, 0.0]),    # short -> the one relabelled
            self._seg(3.6, 6.0, "C", [1.0, 0.01, 0.0]),   # distance to B smaller -> wins
        ]
        new, recs = m2_relabel(segs, self._lam_const(0.9), tau0=0.5, alpha=0.0)
        self.assertEqual(new[1]["label"], "C")
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["index"], 1)
        self.assertEqual(recs[0]["new"], "C")

    def test_keeps_long_segments(self):
        # when B is the LONGER of a pair, the shorter neighbour is relabelled instead
        segs = [
            self._seg(0.0, 0.3, "A", [1.0, 0.02, 0.0]),   # short
            self._seg(0.4, 5.0, "B", [1.0, 0.0, 0.0]),    # long
        ]
        new, recs = m2_relabel(segs, self._lam_const(0.9), tau0=0.5, alpha=0.0)
        self.assertEqual([s["label"] for s in new], ["B", "B"])  # shorter A -> B
        self.assertEqual(recs[0]["index"], 0)

    def test_alpha_zero_tau0_below_all_is_noop(self):
        segs = [
            self._seg(0.0, 3.0, "A", [1.0, 0.0, 0.0]),
            self._seg(3.1, 3.5, "B", [1.0, 0.02, 0.0]),
            self._seg(3.6, 6.0, "A", [1.0, 0.0, 0.0]),
        ]
        # tau0 = 0.0 is below every positive distance -> nothing passes
        new, recs = m2_relabel(segs, self._lam_const(0.9), tau0=0.0, alpha=0.0)
        self.assertEqual([s["label"] for s in new], ["A", "B", "A"])
        self.assertEqual(recs, [])

    def test_decisions_on_original_labels_order_independent(self):
        # A chain A|B|A where relabelling S1 first would wrongly make S1-S2 same-label.
        # Decisions must use ORIGINAL labels, so both the S0-S1 and S1-S2 pairs are considered,
        # and S1 (short) is relabelled once by the nearer neighbour.
        segs = [
            self._seg(0.0, 3.0, "A", [1.0, 0.0, 0.0]),
            self._seg(3.1, 3.5, "B", [1.0, 0.03, 0.0]),
            self._seg(3.6, 6.0, "A", [1.0, 0.01, 0.0]),
        ]
        new, recs = m2_relabel(segs, self._lam_const(0.9), tau0=0.5, alpha=0.0)
        # S2 is closer to B than S0 is -> S1 takes A from S2 (both neighbours are "A" here)
        self.assertEqual(new[1]["label"], "A")
        self.assertEqual(len(recs), 1)

    def test_copies_not_mutated(self):
        segs = [
            self._seg(0.0, 3.0, "A", [1.0, 0.0, 0.0]),
            self._seg(3.1, 3.5, "B", [1.0, 0.02, 0.0]),
            self._seg(3.6, 6.0, "A", [1.0, 0.0, 0.0]),
        ]
        orig_labels = [s["label"] for s in segs]
        new, _ = m2_relabel(segs, self._lam_const(0.9), tau0=0.5, alpha=0.0)
        self.assertEqual([s["label"] for s in segs], orig_labels)  # input untouched
        self.assertIsNot(new[0], segs[0])

    def test_validation_unsorted(self):
        bad = [
            {"start": 0.0, "end": 3.0, "label": "A", "emb": [1.0, 0.0]},
            {"start": 2.0, "end": 4.0, "label": "B", "emb": [1.0, 0.0]},  # overlaps previous
        ]
        with self.assertRaises(ValueError):
            m2_relabel(bad, self._lam_const(0.9), tau0=0.5, alpha=0.0)

    def test_validation_non_unit_emb(self):
        bad = [{"start": 0.0, "end": 1.0, "label": "A", "emb": [0.6, 0.6]}]
        with self.assertRaises(ValueError):
            m2_relabel(bad, self._lam_const(0.9), tau0=0.5, alpha=0.0)

    def test_validation_dim_mismatch(self):
        bad = [
            {"start": 0.0, "end": 1.0, "label": "A", "emb": [1.0, 0.0]},
            {"start": 1.1, "end": 2.0, "label": "B", "emb": [1.0, 0.0, 0.0]},
        ]
        with self.assertRaises(ValueError):
            m2_relabel(bad, self._lam_const(0.9), tau0=0.5, alpha=0.0)


class TestCosineDistance(unittest.TestCase):
    def test_orthogonal_is_one(self):
        self.assertAlmostEqual(cosine_distance([1.0, 0.0], [0.0, 1.0]), 1.0, places=9)

    def test_identical_is_zero(self):
        self.assertAlmostEqual(cosine_distance([1.0, 0.0], [1.0, 0.0]), 0.0, places=9)

    def test_validation(self):
        with self.assertRaises(ValueError):
            cosine_distance([0.6, 0.6], [1.0, 0.0])        # non-unit
        with self.assertRaises(ValueError):
            cosine_distance([1.0, 0.0], [1.0, 0.0, 0.0])   # dim mismatch


if __name__ == "__main__":
    unittest.main()


class TestReviewRound1(unittest.TestCase):
    """Edge cases from the code review."""

    def test_b2_distance_tie_broken_by_sorted_label_pair(self):
        a, b = [1.0, 0.0], _unit([0.9, math.sqrt(1 - 0.81)])
        spk = {"S2": {"centroid": a, "lang_dur": {"ne": 10.0}},
               "S1": {"centroid": b, "lang_dur": {"en": 10.0}},
               "S3": {"centroid": b, "lang_dur": {"en": 10.0}}}
        # (S1,S2) and (S2,S3) are equally close and cross-language; the smaller label pair (S1,S2) merges first
        m = b2_merge(spk, threshold=0.05, delta=0.06)
        self.assertEqual(m["S1"], m["S2"])

    def test_b2_zero_duration_falls_back_to_equal_weights(self):
        a, b = [1.0, 0.0], _unit([0.95, math.sqrt(1 - 0.95 ** 2)])
        spk = {"A": {"centroid": a, "lang_dur": {"ne": 0.0}},
               "B": {"centroid": b, "lang_dur": {"en": 0.0}}}
        m = b2_merge(spk, threshold=0.0, delta=0.1)
        self.assertEqual(m["A"], m["B"])

    def test_b2_refuses_non_string_labels(self):
        with self.assertRaises(ValueError):
            b2_merge({1: {"centroid": [1.0, 0.0], "lang_dur": {"ne": 1.0}}}, 0.5, 0.1)

    def test_jsd_refuses_ragged_posteriors(self):
        with self.assertRaises(ValueError):
            two_language_jsd([0.5, 0.5], [0.5, 0.4, 0.1], (0, 1))

    def test_m2_length_tie_relabels_the_later_segment(self):
        e = [1.0, 0.0]
        segs = [{"start": 0.0, "end": 1.0, "label": "A", "emb": e},
                {"start": 1.1, "end": 2.1, "label": "B", "emb": e}]
        new, rec = m2_relabel(segs, lambda t: 1.0, tau0=0.5, alpha=0.0)
        self.assertEqual([s["label"] for s in new], ["A", "A"])
        self.assertEqual(rec[0]["index"], 1)

    def test_m2_does_not_cascade(self):
        e = [1.0, 0.0]
        segs = [{"start": 0.0, "end": 5.0, "label": "A", "emb": e},
                {"start": 5.1, "end": 5.9, "label": "B", "emb": e},
                {"start": 6.0, "end": 6.4, "label": "C", "emb": e},
                {"start": 6.5, "end": 12.0, "label": "A", "emb": e}]
        new, _ = m2_relabel(segs, lambda t: 1.0, tau0=0.5, alpha=0.0)
        # B -> A (B shorter than A); C is claimed by B (C shorter) and by the last A at equal distance, the earlier
        # neighbour wins, so C takes B's ORIGINAL label although B itself becomes A in the same pass
        self.assertEqual([s["label"] for s in new], ["A", "A", "B", "A"])
