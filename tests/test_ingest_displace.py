"""Unittest-style tests for csdiar.data.ingest_displace on synthetic RTTM strings and VAD sequences.

Stdlib-only (no numpy, no pytest): run with
    python3 -m unittest -v tests.test_ingest_displace
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from csdiar.data.ingest_displace import (
    C_S, FRAME_S, LAYOUT, LD_SUFFIX, SD_SUFFIX, build_turns, control_points, ds_turns, elementary_timeline, main,
    nonspeech_stretches, parse_rttm, pause_bin, pause_ms_at, process_recording, ssdl_points, stratum,
    _speaker_change_times,
)
from collections import Counter


def rttm(rows):
    """rows: (label, onset, dur) -> RTTM lines with an arbitrary TYPE token."""
    return "\n".join(f"SPEAKER file 1 {o:.3f} {d:.3f} <NA> <NA> {lab} <NA> <NA>" for lab, o, d in rows)


def vad_from_speech(spans, dur):
    """Frame probability sequence: 1.0 where speech, 0.0 elsewhere. Frame k covers [0.032k, 0.032(k+1))."""
    n = int(dur / FRAME_S) + 1
    v = [0.0] * n
    for s, e in spans:
        k0 = int(s / FRAME_S)
        k1 = int(e / FRAME_S + 0.999999)
        for k in range(k0, min(k1, n)):
            v[k] = 1.0
    return v


class TestParsing(unittest.TestCase):
    def test_accepts_any_type_skips_comments_and_blanks(self):
        text = "\n".join([
            "; a comment", "# another", "",
            "SPEAKER f 1 0.000 1.000 <NA> <NA> A <NA> <NA>",
            "LANG f 1 1.000 2.000 <NA> <NA> ne <NA> <NA>",
            "garbage line too short",
            "SPEAKER f 1 5.000 0.000 <NA> <NA> B <NA> <NA>",   # non-positive dur dropped
            "SPEAKER f 1 6.000 1.000 <NA> <NA> <NA> <NA> <NA>",  # NA label dropped
        ])
        segs = parse_rttm(text)
        self.assertEqual([(s.onset, s.dur, s.label) for s in segs],
                         [(0.0, 1.0, "A"), (1.0, 2.0, "ne")])

    def test_sorted_by_onset(self):
        segs = parse_rttm(rttm([("A", 3.0, 1.0), ("B", 0.0, 1.0)]))
        self.assertEqual([s.onset for s in segs], [0.0, 3.0])


class TestTimeline(unittest.TestCase):
    def test_clean_overlap_ambiguous_classification(self):
        sd = parse_rttm(rttm([("A", 0.0, 2.0), ("B", 1.0, 2.0)]))   # overlap in [1,2]
        ld = parse_rttm(rttm([("ne", 0.0, 1.0)]))                   # [1,3] has no language
        tl = elementary_timeline(sd, ld)
        by = {(round(iv.start, 3), round(iv.end, 3)): iv for iv in tl}
        self.assertTrue(by[(0.0, 1.0)].is_clean)       # A + ne
        self.assertTrue(by[(1.0, 2.0)].is_overlap)     # A + B
        self.assertTrue(by[(2.0, 3.0)].is_ambiguous)   # B only, no language

    def test_two_languages_is_ambiguous(self):
        sd = parse_rttm(rttm([("A", 0.0, 1.0)]))
        ld = parse_rttm(rttm([("ne", 0.0, 1.0), ("en", 0.0, 1.0)]))
        tl = elementary_timeline(sd, ld)
        self.assertEqual(len(tl), 1)
        self.assertTrue(tl[0].is_ambiguous)


class TestSSDLPoint(unittest.TestCase):
    def _single_turn_switch(self, vad=None):
        # One speaker A, clean ne run [0,2], clean en run [2.4,5]; pause gap [2,2.4].
        sd = parse_rttm(rttm([("A", 0.0, 5.0)]))
        ld = parse_rttm(rttm([("ne", 0.0, 2.0), ("en", 2.4, 2.6)]))
        tl = elementary_timeline(sd, ld)
        turns = build_turns(tl)
        return tl, turns, sd

    def test_clean_switch_gives_one_point(self):
        tl, turns, sd = self._single_turn_switch()
        self.assertEqual(len(turns), 1)
        self.assertEqual([r.lang for r in turns[0].runs], ["ne", "en"])
        excl = Counter()
        pts = ssdl_points(turns, tl, _speaker_change_times(sd), [], excl)
        self.assertEqual(len(pts), 1)
        p = pts[0]
        self.assertEqual((p["lang_left"], p["lang_right"]), ("ne", "en"))
        # No VAD -> t is the gap middle (2.0 + 2.4)/2 = 2.2
        self.assertAlmostEqual(p["t"], 2.2, places=6)
        self.assertEqual(p["pause_bin"], "none")

    def test_stratum_and_elig_p0L1(self):
        # Adjacent runs (no gap) so t sits on the boundary and [t-1,t] / [t,t+1] each lie in one clean run.
        sd = parse_rttm(rttm([("A", 0.0, 5.0)]))
        ld = parse_rttm(rttm([("ne", 0.0, 2.2), ("en", 2.2, 2.8)]))  # ne [0,2.2], en [2.2,5.0]
        tl = elementary_timeline(sd, ld)
        turns = build_turns(tl)
        p = ssdl_points(turns, tl, _speaker_change_times(sd), [], Counter())[0]
        self.assertAlmostEqual(p["t"], 2.2, places=6)
        self.assertEqual(p["stratum"], "ge1")   # shorter run 2.2 s
        self.assertTrue(p["elig_p0L1"])

    def test_elig_false_when_a_run_short(self):
        sd = parse_rttm(rttm([("A", 0.0, 5.0)]))
        ld = parse_rttm(rttm([("ne", 0.0, 0.6), ("en", 0.6, 4.4)]))  # left run 0.6 s < 1 s
        tl = elementary_timeline(sd, ld)
        turns = build_turns(tl)
        p = ssdl_points(turns, tl, _speaker_change_times(sd), [], Counter())[0]
        self.assertEqual(p["stratum"], "0.5to1")
        self.assertFalse(p["elig_p0L1"])

    def test_point_time_from_vad_with_widening(self):
        # Gap [2.0,2.4]; a VAD non-speech stretch sits only in [2.1,2.3]. Widened search [1.9,2.5] (clamped by
        # run middles) finds it; t is its midpoint 2.2. Make the stretch asymmetric to prove VAD (not gap mid).
        sd = parse_rttm(rttm([("A", 0.0, 6.0)]))
        ld = parse_rttm(rttm([("ne", 0.0, 2.0), ("en", 2.5, 3.5)]))  # gap [2.0,2.5], gap-mid would be 2.25
        tl = elementary_timeline(sd, ld)
        turns = build_turns(tl)
        # Non-speech only in [2.0,2.3) (frames 63,64,65,66,67,68 -> ~2.016..2.208). Speech elsewhere.
        vad = vad_from_speech([(0.0, 2.0), (2.3, 3.5)], 3.5)
        stretches = nonspeech_stretches(vad)
        p = ssdl_points(turns, tl, _speaker_change_times(sd), stretches, Counter())[0]
        # midpoint of the clipped non-speech stretch, clearly below the 2.25 gap middle
        self.assertLess(p["t"], 2.25)
        self.assertGreater(p["t"], 2.0)


class TestExclusions(unittest.TestCase):
    def test_overlap_suppresses_point_and_counts(self):
        # A: ne [0,2.2], en [2.2,...]; junction forms at t=2.2. Speaker B overlaps A at [2.5,3.0], an overlap
        # interval inside the exclusion zone [1.2,3.2] -> the point is dropped and counted.
        sd = parse_rttm(rttm([("A", 0.0, 4.0), ("B", 2.5, 0.5)]))
        ld = parse_rttm(rttm([("ne", 0.0, 2.2), ("en", 2.2, 1.8)]))
        tl = elementary_timeline(sd, ld)
        turns = build_turns(tl)
        excl = Counter()
        pts = ssdl_points(turns, tl, _speaker_change_times(sd), [], excl)
        self.assertEqual(pts, [])
        self.assertEqual(excl["overlap_or_ambiguous"], 1)

    def test_ambiguous_language_suppresses_point(self):
        # A speaks [0,5]; language only [0,2] then [2.5,5]; the hole [2,2.5] is ambiguous (0 languages).
        # Force a junction by giving two runs separated by an ambiguous region within 2c.
        sd = parse_rttm(rttm([("A", 0.0, 5.0)]))
        ld = parse_rttm(rttm([("ne", 0.0, 2.0), ("en", 2.1, 2.9)]))  # tiny clean gap -> t near 2.05
        # Add a 2-language ambiguous atom just after onset by overlapping languages at [2.0,2.1].
        ld = parse_rttm(rttm([("ne", 0.0, 2.05), ("en", 2.0, 0.9)]))  # ne&en overlap [2.0,2.05] ambiguous
        tl = elementary_timeline(sd, ld)
        turns = build_turns(tl)
        excl = Counter()
        pts = ssdl_points(turns, tl, _speaker_change_times(sd), [], excl)
        # Either dropped as ambiguous, or no junction formed; the invariant is no surviving point here.
        self.assertEqual(pts, [])

    def test_2c_exclusion_near_speaker_change(self):
        # Clean ne->en switch of A at ~2.2, but A's turn ends and the next speaker change is at 2.5 (<2c).
        sd = parse_rttm(rttm([("A", 0.0, 2.5), ("B", 3.0, 1.0)]))
        ld = parse_rttm(rttm([("ne", 0.0, 2.0), ("en", 2.0, 0.5)]))
        tl = elementary_timeline(sd, ld)
        turns = build_turns(tl)
        excl = Counter()
        pts = ssdl_points(turns, tl, _speaker_change_times(sd), [], excl)
        self.assertEqual(pts, [])
        self.assertEqual(excl["near_speaker_change"], 1)


class TestDSTurns(unittest.TestCase):
    def test_dsdl_vs_dssl_and_gap_rule(self):
        # A ends 2.0 (last run en), B starts 2.5 (first run ne) -> DS-DL, gap 0.5 s <= 2 s.
        sd = parse_rttm(rttm([("A", 0.0, 2.0), ("B", 2.5, 2.0)]))
        ld = parse_rttm(rttm([("en", 0.0, 2.0), ("ne", 2.5, 2.0)]))
        tl = elementary_timeline(sd, ld)
        turns = build_turns(tl)
        ds = ds_turns(turns, tl, [], Counter())
        self.assertEqual([r["kind"] for r in ds], ["dsdl"])
        self.assertAlmostEqual(ds[0]["t"], 2.25, places=6)

        # Same languages -> DS-SL.
        ld2 = parse_rttm(rttm([("ne", 0.0, 2.0), ("ne", 2.5, 2.0)]))
        tl2 = elementary_timeline(sd, ld2)
        ds2 = ds_turns(build_turns(tl2), tl2, [], Counter())
        self.assertEqual([r["kind"] for r in ds2], ["dssl"])

    def test_gap_too_large_excluded(self):
        sd = parse_rttm(rttm([("A", 0.0, 2.0), ("B", 4.5, 2.0)]))  # gap 2.5 s > 2 s
        ld = parse_rttm(rttm([("en", 0.0, 2.0), ("ne", 4.5, 2.0)]))
        tl = elementary_timeline(sd, ld)
        ds = ds_turns(build_turns(tl), tl, [], Counter())
        self.assertEqual(ds, [])

    def test_overlap_across_transition_excluded_and_counted(self):
        # A [0,2.2], B [2.0,4.0] -> overlap [2.0,2.2] across the transition.
        sd = parse_rttm(rttm([("A", 0.0, 2.2), ("B", 2.0, 2.0)]))
        ld = parse_rttm(rttm([("en", 0.0, 2.2), ("ne", 2.2, 1.8)]))
        tl = elementary_timeline(sd, ld)
        excl = Counter()
        ds = ds_turns(build_turns(tl), tl, [], excl)
        self.assertEqual(ds, [])
        self.assertEqual(excl["overlap_or_ambiguous"], 1)


class TestControls(unittest.TestCase):
    def test_controls_respect_edges_and_spacing(self):
        # One long clean run A/ne over [0,10]; grid candidates at 0.5 s steps, >=2c (1.0 s) from edges.
        sd = parse_rttm(rttm([("A", 0.0, 10.0)]))
        ld = parse_rttm(rttm([("ne", 0.0, 10.0)]))
        tl = elementary_timeline(sd, ld)
        turns = build_turns(tl)
        ctrl = control_points(turns, tl, [], reserved=[])
        ts = [r["t"] for r in ctrl]
        self.assertTrue(ts, "expected some controls")
        # all within [1.0, 9.0]
        self.assertGreaterEqual(min(ts), 1.0 - 1e-6)
        self.assertLessEqual(max(ts), 9.0 + 1e-6)
        # pairwise spacing at least 2c
        for i in range(1, len(ts)):
            self.assertGreaterEqual(ts[i] - ts[i - 1], 2 * C_S - 1e-6)

    def test_controls_avoid_reserved_points(self):
        sd = parse_rttm(rttm([("A", 0.0, 10.0)]))
        ld = parse_rttm(rttm([("ne", 0.0, 10.0)]))
        tl = elementary_timeline(sd, ld)
        turns = build_turns(tl)
        ctrl = control_points(turns, tl, [], reserved=[5.0])
        for r in ctrl:
            self.assertGreaterEqual(abs(r["t"] - 5.0), 2 * C_S - 1e-6)


class TestPause(unittest.TestCase):
    def test_pause_measurement_and_bins(self):
        # Non-speech stretch [1.0, 1.4) -> ~0.4 s. Point at t=1.2 overlaps [1.1,1.3] -> pause ~400 ms -> gt300.
        vad = vad_from_speech([(0.0, 1.0), (1.4, 3.0)], 3.0)
        stretches = nonspeech_stretches(vad)
        pms = pause_ms_at(1.2, stretches)
        self.assertGreater(pms, 300.0)
        self.assertEqual(pause_bin(pms), "gt300")

    def test_pause_none_when_no_overlap_within_widening(self):
        vad = vad_from_speech([(0.0, 1.0), (2.0, 3.0)], 3.0)  # non-speech [1.0,2.0]
        stretches = nonspeech_stretches(vad)
        # t far from any non-speech, outside [t-0.1,t+0.1]
        self.assertEqual(pause_ms_at(0.3, stretches), 0.0)
        self.assertEqual(pause_bin(pause_ms_at(0.3, stretches)), "none")

    def test_widening_catches_adjacent_stretch(self):
        # Non-speech frames 30..39 -> [0.96, 1.28). A point at t=0.87 (window [0.77,0.97]) just reaches it via
        # the 0.1 s widening; a point at t=0.50 (window [0.40,0.60]) does not.
        vad = vad_from_speech([(0.0, 0.96), (1.28, 2.0)], 2.0)
        stretches = nonspeech_stretches(vad)
        self.assertGreater(pause_ms_at(0.87, stretches), 0.0)
        self.assertEqual(pause_ms_at(0.50, stretches), 0.0)  # outside widening

    def test_stratum_bins(self):
        self.assertEqual([stratum(x) for x in (0.3, 0.5, 0.9, 1.0, 2.0)],
                         ["lt0.5", "0.5to1", "0.5to1", "ge1", "ge1"])


class TestDeterministicOutput(unittest.TestCase):
    def test_full_recording_order_and_counts(self):
        # A: ne [0,2], en [2.5,5]  (SS-DL inside A's turn)
        # then B starts at 6.0 ne [6,8]  (DS-DL A->B across the 1.0 s gap from A's end at 5.0)
        sd = rttm([("A", 0.0, 5.0), ("B", 6.0, 2.0)])
        ld = rttm([("ne", 0.0, 2.0), ("en", 2.5, 2.5), ("ne", 6.0, 2.0)])
        rows, summary = process_recording("M016", "dev", sd, ld, vad=None)
        kinds = [r["kind"] for r in rows]
        self.assertIn("ssdl", kinds)
        self.assertIn("dsdl", kinds)
        # output strictly ordered by t
        ts = [r["t"] for r in rows]
        self.assertEqual(ts, sorted(ts))
        # determinism: identical re-run yields identical rows
        rows2, _ = process_recording("M016", "dev", sd, ld, vad=None)
        self.assertEqual(rows, rows2)
        self.assertEqual(summary["n_ssdl"], 1)
        self.assertEqual(summary["n_dsdl"], 1)
        self.assertFalse(summary["has_vad"])  # vad=None -> pause "none" everywhere, flagged


class TestReviewRound1(unittest.TestCase):
    """Rules the round-1 code review found untested."""

    def test_long_silence_splits_a_turn_without_inflating_the_first_turns_end(self):
        sd = parse_rttm(rttm([("A", 0.0, 1.0), ("A", 4.0, 1.0)]))     # 3 s silence >= JOIN_GAP_S
        ld = parse_rttm(rttm([("en", 0.0, 1.0), ("en", 4.0, 1.0)]))
        turns = build_turns(elementary_timeline(sd, ld))
        self.assertEqual([(t.start, t.end) for t in turns], [(0.0, 1.0), (4.0, 5.0)])

    def test_short_silence_joins_one_turn(self):
        sd = parse_rttm(rttm([("A", 0.0, 1.0), ("A", 2.5, 1.0)]))     # 1.5 s < JOIN_GAP_S
        ld = parse_rttm(rttm([("en", 0.0, 1.0), ("en", 2.5, 1.0)]))
        turns = build_turns(elementary_timeline(sd, ld))
        self.assertEqual(len(turns), 1)
        self.assertEqual((turns[0].start, turns[0].end), (0.0, 3.5))

    def test_case_variant_labels_are_refused(self):
        sd = rttm([("A", 0.0, 6.0)])
        ld = rttm([("ne", 0.0, 3.0), ("NE", 3.0, 3.0)])
        with self.assertRaises(ValueError):
            process_recording("R1", "dev", sd, ld, None)

    def test_own_silent_junction_gap_keeps_the_point(self):
        # A speaks throughout; the LD track leaves a 0.2 s language-free gap at the switch (A still active).
        sd = rttm([("A", 0.0, 6.0)])
        ld = rttm([("ne", 0.0, 2.9), ("en", 3.1, 2.9)])
        rows, summary = process_recording("R1", "dev", sd, ld, None)
        pts = [r for r in rows if r["kind"] == "ssdl"]
        self.assertEqual(len(pts), 1)
        self.assertAlmostEqual(pts[0]["t"], 3.0, places=3)
        self.assertEqual(summary["languages"], ["en", "ne"])

    def test_speaker_change_across_silence_is_a_change(self):
        sd = parse_rttm(rttm([("A", 0.0, 10.0), ("B", 11.0, 5.0)]))
        times = _speaker_change_times(sd)
        self.assertTrue(any(abs(x - 10.0) < 1e-9 for x in times))
        # an A switch 0.5 s before A's end lies inside 2c of that change and is dropped
        ld = rttm([("ne", 0.0, 9.5), ("en", 9.5, 0.5), ("en", 11.0, 5.0)])
        rows, summary = process_recording("R1", "dev", rttm([("A", 0.0, 10.0), ("B", 11.0, 5.0)]), ld, None)
        self.assertEqual([r for r in rows if r["kind"] == "ssdl"], [])
        self.assertGreaterEqual(sum(summary["excluded"].values()), 1)

    def test_vad_pause_midpoints_become_controls_with_phrase(self):
        # one clean A/en run of 12 s with a 0.4 s VAD pause at [6.0, 6.4]
        sd = rttm([("A", 0.0, 12.0)])
        ld = rttm([("en", 0.0, 12.0)])
        vad = vad_from_speech([(0.0, 6.0), (6.4, 12.0)], 12.0)
        rows, _ = process_recording("R1", "dev", sd, ld, vad)
        ctrl = [r for r in rows if r["kind"] == "ctrl"]
        pause_ctrl = [r for r in ctrl if r["pause_ms"] > 150]
        self.assertTrue(pause_ctrl, "the VAD pause should yield a control")
        self.assertTrue(all(r["phrase"] for r in pause_ctrl))
        self.assertTrue(any(abs(r["t"] - 6.2) < 0.05 for r in pause_ctrl))
        self.assertTrue(all(C_S * 2 - 1e-9 <= r["t"] <= 12.0 - 2 * C_S + 1e-9 for r in ctrl))


class TestMainLayout(unittest.TestCase):
    """The CLI path against the archive layout read from the zips' central directories (2026-10-19)."""

    SD = [("A", 0.0, 5.0), ("B", 6.0, 2.0)]
    LD = [("ne", 0.0, 2.0), ("en", 2.5, 2.5), ("ne", 6.0, 2.0)]   # 1 SS-DL point + 1 DS-DL turn each

    def _tree(self, root, split, sd_ids, ld_ids, ld_rel=None):
        sd_dir = root / LAYOUT[split]["sd"]
        ld_dir = root / (ld_rel or LAYOUT[split]["ld"])
        sd_dir.mkdir(parents=True)
        ld_dir.mkdir(parents=True)
        for i in sd_ids:
            (sd_dir / f"{i}{SD_SUFFIX}").write_text(rttm(self.SD), encoding="utf-8")
        for i in ld_ids:
            (ld_dir / f"{i}{LD_SUFFIX}").write_text(rttm(self.LD), encoding="utf-8")

    def _run(self, root, out, split="dev"):
        with contextlib.redirect_stdout(io.StringIO()):
            return main(["--root", str(root), "--split", split, "--out", str(out)])

    def test_layout_keeps_the_archive_folder_case(self):
        for split in ("dev", "eval"):
            self.assertTrue(LAYOUT[split]["wav"].startswith("Track1_SD_supervised/"))
            self.assertTrue(LAYOUT[split]["sd"].startswith("Track1_SD_supervised/"))
            self.assertTrue(LAYOUT[split]["ld"].startswith("track2_LD_supervised/"))
        self.assertEqual((SD_SUFFIX, LD_SUFFIX), ("_SPEAKER.rttm", "_LANGUAGE.rttm"))

    def test_main_writes_outputs_for_matching_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "root", Path(tmp) / "out"
            self._tree(root, "eval", ["B001", "M002"], ["B001", "M002"])
            self.assertEqual(self._run(root, out, "eval"), 0)
            for kind in ("ssdl", "turns", "ctrl"):
                self.assertTrue((out / f"{kind}_eval.csv.gz").is_file())
            doc = json.loads((out / "summary.json").read_text())
            self.assertEqual(doc["n_recordings"], 2)
            self.assertEqual((doc["totals"]["n_ssdl"], doc["totals"]["n_dsdl"]), (2, 2))
            self.assertEqual(doc["ld_suffix"], LD_SUFFIX)
            self.assertTrue(doc["vad_absent"])

    def test_main_refuses_a_missing_language_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            self._tree(root, "dev", ["B001", "M002"], ["B001"])
            with self.assertRaises(SystemExit) as cm:
                self._run(root, Path(tmp) / "out")
            self.assertIn("speaker only ['M002']", str(cm.exception))
            self.assertFalse((Path(tmp) / "out").exists())

    def test_main_refuses_an_extra_language_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            self._tree(root, "dev", ["B001"], ["B001", "M009"])
            with self.assertRaises(SystemExit) as cm:
                self._run(root, Path(tmp) / "out")
            self.assertIn("language only ['M009']", str(cm.exception))

    def test_main_refuses_a_wrongly_cased_language_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            wrong = LAYOUT["dev"]["ld"].replace("track2_", "Track2_", 1)
            self._tree(root, "dev", ["B001"], ["B001"], ld_rel=wrong)
            if (root / LAYOUT["dev"]["ld"]).is_dir():
                self.skipTest("case-insensitive filesystem")
            with self.assertRaises(SystemExit) as cm:
                self._run(root, Path(tmp) / "out")
            self.assertIn("language label folder not found", str(cm.exception))

    def test_main_refuses_an_empty_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            self._tree(root, "dev", [], [])
            with self.assertRaises(SystemExit) as cm:
                self._run(root, Path(tmp) / "out")
            self.assertIn("no *_SPEAKER.rttm files", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
