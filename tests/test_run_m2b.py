#!/usr/bin/env python3
"""Unit tests for the pure parts of csdiar.pipelines.run_m2b (PREREGISTRATION_M2B.md §3, §4).

These cover cache keys, RTTM writing, duration-weighted centroids, the per-0.1 s-step per-language durations
(change 1, centred LID window), the LID-pair lambda closure, the row-wise embedding transform AND the slot-aware
B1 path (slot_frame_times / slot_languages / b1_transform, change 3), the B2 adapter's exclusive-annotation
inputs (change 2), and the M2 run-splitting adapter at its deployed tau0 = 0 default (change 4) -- WITHOUT
loading any model, reading any audio, or importing torch / pyannote / soundfile. Run under .venv-pyannote4:

  .venv-pyannote4/bin/python -m pytest -q -p no:cacheprovider tests/test_run_m2b.py

numpy is available there (pyannote depends on it); run_m2b's pure helpers use numpy, the m2b_hooks they call are
stdlib-only. The heavy imports in run_m2b are lazy (inside the functions), so importing the module is cheap.
"""
import math
from pathlib import Path

import numpy as np
import pytest

from csdiar.pipelines import run_m2b as R


# --------------------------------------------------------------------------- #
# cache keys
# --------------------------------------------------------------------------- #
class TestCacheKeys:
    def test_recording_id_is_wav_stem(self):
        assert R.recording_id("/a/b/REC_01.wav") == "REC_01"
        assert R.recording_id(Path("x/y/session.flac")) == "session"

    def test_model_tag(self):
        assert R.model_tag("e1") == "e1"
        assert R.model_tag("e2") == "e2"
        assert R.model_tag("e3") == "e3"
        assert R.model_tag("m3:/home/u/data/models/m3/seed0_all.pt") == "m3-seed0_all"

    def test_model_tag_best_pt_uses_run_folder_so_seeds_differ(self):
        # m3_train / run_m3_queue write every checkpoint to <run>/best.pt, so the file stem `best` is shared by
        # all five all-data seeds. model_tag must derive from the run folder (via m3_ckpt_name), NOT the file
        # stem, or the five seeds' stage-1 caches would share one `m3-best` folder and overwrite each other.
        assert R.model_tag("m3:~/data/m3/runs/all_s0_w0.3_none/best.pt") == "m3-all_s0_w0.3_none"
        seeds = [f"m3:~/data/m3/runs/all_s{s}_w0.3_none/best.pt" for s in range(5)]
        tags = {R.model_tag(c) for c in seeds}
        assert len(tags) == 5                                    # five distinct model tags
        # the five stage-1 cache folders are distinct too (one per seed), not all `m3-best`.
        folders = {R.stage1_cache_path("/cache", c, "REC").parent for c in seeds}
        assert len(folders) == 5
        assert R.stage1_cache_path("/cache", seeds[0], "REC") == \
            Path("/cache/stage1/m3-all_s0_w0.3_none/REC.npz")

    def test_model_tag_matches_system_tag_per_seed(self):
        # model_tag and system_tag must name the SAME seed the same way (both via m3_ckpt_name), so a seed's
        # stage-1 cache folder and its output-directory tag never disagree.
        for s in range(5):
            ckpt = f"m3:~/data/m3/runs/all_s{s}_w0.3_none/best.pt"
            assert R.model_tag(ckpt) == "m3-" + R.m3_ckpt_name(ckpt)
            assert R.system_tag("M3", m3_ckpt=ckpt) == "M3_" + R.m3_ckpt_name(ckpt)

    def test_model_tag_rejects_unknown(self):
        with pytest.raises(ValueError):
            R.model_tag("e9")
        with pytest.raises(ValueError):
            R.model_tag("wavlm")

    def test_stage1_cache_path_keyed_by_model_and_rid(self):
        p = R.stage1_cache_path("/cache", "e3", "REC_01")
        assert p == Path("/cache/stage1/e3/REC_01.npz")
        pm = R.stage1_cache_path("/cache", "m3:/x/ck.pt", "REC_01")
        assert pm == Path("/cache/stage1/m3-ck/REC_01.npz")

    def test_lid_cache_path_is_model_independent(self):
        assert R.lid_cache_path("/cache", "REC_01") == Path("/cache/lid/REC_01.npz")

    def test_threshold_tag_two_dp(self):
        assert R.threshold_tag(0.7) == "th0.70"
        assert R.threshold_tag(0.3) == "th0.30"
        assert R.threshold_tag(1.0) == "th1.00"
        # the 0.05 grid should give distinct tags for every grid point
        tags = {R.threshold_tag(round(0.30 + 0.05 * k, 2)) for k in range(15)}
        assert len(tags) == 15

    def test_rttm_paths_layout(self):
        ov, ex = R.rttm_paths("/out", "dev", "B0", 0.7, "REC_01")
        assert ov == Path("/out/dev/B0/th0.70/REC_01.rttm")
        assert ex == Path("/out/dev/B0/th0.70/REC_01.exclusive.rttm")


# --------------------------------------------------------------------------- #
# RTTM writing
# --------------------------------------------------------------------------- #
class TestWriteRTTM:
    def test_writes_nist_columns(self, tmp_path):
        p = tmp_path / "out.rttm"
        R.write_rttm(p, "REC_01", [(0.0, 1.5, "SPEAKER_00"), (1.5, 3.25, "SPEAKER_01")])
        lines = p.read_text().splitlines()
        assert lines[0] == "SPEAKER REC_01 1 0.000 1.500 <NA> <NA> SPEAKER_00 <NA> <NA>"
        assert lines[1] == "SPEAKER REC_01 1 1.500 1.750 <NA> <NA> SPEAKER_01 <NA> <NA>"

    def test_three_decimal_places(self, tmp_path):
        p = tmp_path / "out.rttm"
        R.write_rttm(p, "r", [(0.6665, 7.165, "A")])
        # onset 0.666 (round-half-to-even on .6665 -> .666), duration 6.499 (7.165 - 0.6665 = 6.4985 -> 6.498)
        toks = p.read_text().split()
        assert toks[3] == f"{0.6665:.3f}"
        assert toks[4] == f"{7.165 - 0.6665:.3f}"

    def test_drops_zero_and_negative_duration(self, tmp_path):
        p = tmp_path / "out.rttm"
        R.write_rttm(p, "r", [(1.0, 1.0, "A"), (2.0, 1.5, "B"), (3.0, 4.0, "C")])
        lines = p.read_text().splitlines()
        assert len(lines) == 1
        assert lines[0].split()[7] == "C"

    def test_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "dev" / "B0" / "th0.70" / "r.rttm"
        R.write_rttm(p, "r", [(0.0, 1.0, "A")])
        assert p.exists()

    def test_empty_annotation_writes_empty_file(self, tmp_path):
        p = tmp_path / "out.rttm"
        R.write_rttm(p, "r", [])
        assert p.read_text() == ""


# --------------------------------------------------------------------------- #
# duration-weighted centroids
# --------------------------------------------------------------------------- #
class TestDurationWeightedCentroid:
    def test_unit_norm_output(self):
        embs = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        c = R.duration_weighted_centroid(embs, np.array([1.0, 1.0]))
        assert abs(np.linalg.norm(c) - 1.0) < 1e-9

    def test_weights_by_duration(self):
        # two opposite-ish directions; the heavier one dominates the mean direction
        embs = np.array([[1.0, 0.0], [0.0, 1.0]])
        c = R.duration_weighted_centroid(embs, np.array([9.0, 1.0]))
        # weighted mean = (9,1)/norm -> first component much larger
        assert c[0] > c[1]
        assert abs(np.linalg.norm(c) - 1.0) < 1e-9
        np.testing.assert_allclose(c, R.unit(np.array([9.0, 1.0])), atol=1e-9)

    def test_drops_nan_and_nonpositive_duration(self):
        embs = np.array([[1.0, 0.0], [np.nan, np.nan], [0.0, 1.0]])
        durs = np.array([5.0, 3.0, 0.0])      # row1 NaN, row2 zero dur -> only row0 remains
        c = R.duration_weighted_centroid(embs, durs)
        np.testing.assert_allclose(c, [1.0, 0.0], atol=1e-9)

    def test_raises_when_nothing_usable(self):
        with pytest.raises(ValueError):
            R.duration_weighted_centroid(np.array([[np.nan, np.nan]]), np.array([1.0]))
        with pytest.raises(ValueError):
            R.duration_weighted_centroid(np.array([[1.0, 0.0]]), np.array([0.0]))

    def test_shape_validation(self):
        with pytest.raises(ValueError):
            R.duration_weighted_centroid(np.array([1.0, 0.0]), np.array([1.0]))     # not 2-D
        with pytest.raises(ValueError):
            R.duration_weighted_centroid(np.array([[1.0, 0.0]]), np.array([1.0, 2.0]))  # dur mismatch


# --------------------------------------------------------------------------- #
# per-language durations (lang_dur) -- per-0.1 s step, centred LID window (brief change 1)
# --------------------------------------------------------------------------- #
class TestCentreOffset:
    def test_offset_is_five_for_1s_window_01_hop(self):
        # window k covers [0.1k, 0.1k+1.0]; centre 0.1k+0.5; for that to sit on step edge g*0.1 -> k = g-5
        assert R._centre_offset(1.0, 0.1) == 5

    def test_integer_round_half_up_no_float_xdot5(self):
        # win/(2*hop) exactly x.5 cases must round UP deterministically, not round-half-to-even
        assert R._centre_offset(0.3, 0.1) == 2   # 0.3/0.2 = 1.5 -> 2
        assert R._centre_offset(0.5, 0.1) == 3   # 0.5/0.2 = 2.5 -> 3 (even-rounding would give 2)
        assert R._centre_offset(1.0, 0.1) == 5   # 1.0/0.2 = 5.0 -> 5


class TestLangDur:
    def test_per_step_crediting_with_centred_window(self):
        # pair (0,2) on a 3-language grid. Segment [0.5, 0.8) touches steps g=5,6,7 (0.1 s each);
        # each step credits the window centred on its left edge: k = g - 5. So steps 5,6,7 read windows 0,1,2.
        post = np.zeros((40, 3))
        post[0] = [0.9, 0.0, 0.1]   # step 5 -> lang 0 (0.9 vs 0.1)
        post[1] = [0.1, 0.0, 0.9]   # step 6 -> lang 2
        post[2] = [0.9, 0.0, 0.1]   # step 7 -> lang 0
        out = R.lang_dur_from_segments([(0.5, 0.8)], post, (0, 2))
        # 0.1 s to lang 0 (step 5) + 0.1 to lang 2 (step 6) + 0.1 to lang 0 (step 7)
        assert out["lang_dur"][0] == pytest.approx(0.2, abs=1e-9)
        assert out["lang_dur"][2] == pytest.approx(0.1, abs=1e-9)
        assert out["skipped"] == pytest.approx(0.0, abs=1e-9)

    def test_partial_step_overlap_credits_only_the_overlap(self):
        # segment [0.53, 0.77) overlaps step 5 by 0.07, step 6 by 0.10, step 7 by 0.07
        post = np.zeros((40, 3))
        post[0] = [1.0, 0.0, 0.0]   # step 5 -> lang 0
        post[1] = [1.0, 0.0, 0.0]   # step 6 -> lang 0
        post[2] = [1.0, 0.0, 0.0]   # step 7 -> lang 0
        out = R.lang_dur_from_segments([(0.53, 0.77)], post, (0, 1))
        assert out["lang_dur"][0] == pytest.approx(0.24, abs=1e-9)   # = segment length

    def test_whole_segment_duration_conserved_across_languages_and_skips(self):
        post = np.zeros((40, 3))
        post[0] = [1.0, 0.0, 0.0]   # step 5 -> lang 0
        # steps 6,7 (windows 1,2) left zero -> no pair mass -> skipped seconds
        out = R.lang_dur_from_segments([(0.5, 0.8)], post, (0, 2))
        credited = sum(out["lang_dur"].values())
        assert credited + out["skipped"] == pytest.approx(0.3, abs=1e-9)   # 0.1 credited, 0.2 skipped
        assert out["skipped"] == pytest.approx(0.2, abs=1e-9)

    def test_tie_goes_to_smaller_pair_index(self):
        post = np.zeros((40, 3))
        post[0] = [0.3, 0.0, 0.3]   # step 5, window 0: equal over pair -> smaller index
        out = R.lang_dur_from_segments([(0.5, 0.6)], post, (0, 2))
        assert out["lang_dur"] == {0: pytest.approx(0.1, abs=1e-9)}
        # with the pair reversed the tie still credits the SMALLER index (0), not the first pair entry (2)
        out_rev = R.lang_dur_from_segments([(0.5, 0.6)], post, (2, 0))
        assert out_rev["lang_dur"] == {0: pytest.approx(0.1, abs=1e-9)}

    def test_clamps_window_to_grid_edges(self):
        # a segment near t=0 maps to k = g-5 < 0, clamped to 0; near the end clamps to n-1.
        post = np.zeros((20, 3))
        post[0] = [1.0, 0.0, 0.0]   # used by the clamped-low steps
        post[19] = [0.0, 0.0, 1.0]  # used by the clamped-high steps
        low = R.lang_dur_from_segments([(0.0, 0.2)], post, (0, 2))   # steps 0,1 -> k=-5,-4 -> clamp 0
        assert low["lang_dur"] == {0: pytest.approx(0.2, abs=1e-9)}
        high = R.lang_dur_from_segments([(2.9, 3.0)], post, (0, 2))  # step 29 -> k=24 -> clamp 19
        assert high["lang_dur"] == {2: pytest.approx(0.1, abs=1e-9)}

    def test_skipped_is_seconds_not_count(self):
        post = np.zeros((40, 3))   # no mass anywhere -> whole segment is skipped seconds
        out = R.lang_dur_from_segments([(0.5, 1.5)], post, (0, 2))
        assert out["lang_dur"] == {}
        assert out["skipped"] == pytest.approx(1.0, abs=1e-9)

    def test_rejects_non_distinct_pair(self):
        with pytest.raises(ValueError):
            R.lang_dur_from_segments([(0.0, 1.0)], np.zeros((5, 3)), (1, 1))


# --------------------------------------------------------------------------- #
# lambda closure bound to the recording grid + pair
# --------------------------------------------------------------------------- #
class TestLambdaClosure:
    def test_matches_m2b_hooks_lambda_at(self):
        from csdiar.pipelines.m2b_hooks import lambda_at
        # 30 windows on the 0.1 s grid, pair (0, 1), alternating one-hot every 10 windows
        rows = [[1.0, 0.0] if (k // 10) % 2 == 0 else [0.0, 1.0] for k in range(30)]
        post = np.array(rows, dtype=np.float64)
        lam = R.make_lambda_closure(post, (0, 1))
        for t in (1.0, 1.5, 2.9, 0.5, 3.0):
            assert lam(t) == lambda_at(t, rows, (0, 1))

    def test_one_bit_on_opposite_windows(self):
        rows = [[1.0, 0.0] if (k // 10) % 2 == 0 else [0.0, 1.0] for k in range(30)]
        lam = R.make_lambda_closure(np.array(rows), (0, 1))
        assert lam(1.5) == pytest.approx(1.0, abs=1e-9)

    def test_uses_the_given_pair_indices(self):
        # a 3-language grid; put the (ne,en)-like mass on indices (0, 2), noise on 1
        rows = []
        for k in range(30):
            if (k // 10) % 2 == 0:
                rows.append([1.0, 9.0, 0.0])
            else:
                rows.append([0.0, 9.0, 1.0])
        lam = R.make_lambda_closure(np.array(rows), (0, 2))
        assert lam(1.5) == pytest.approx(1.0, abs=1e-9)   # renormalised over (0,2) -> one-hot opposites


# --------------------------------------------------------------------------- #
# finite-row embedding transform (B1/M1 hook application)
# --------------------------------------------------------------------------- #
class TestApplyTransform:
    def test_identity_when_none(self):
        emb = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        out = R._apply_transform(emb, None)
        np.testing.assert_array_equal(out, emb)

    def test_applies_only_to_finite_rows(self):
        emb = np.ones((1, 2, 3), dtype=np.float32)
        emb[0, 1] = np.nan                       # second (chunk,speaker) slot is NaN
        seen = {}

        def transform(x):                        # x is (n_finite, dim) with the NaN row excluded
            seen["shape"] = x.shape
            return x * 2.0

        out = R._apply_transform(emb, transform)
        assert seen["shape"] == (1, 3)           # only the one finite row was passed
        np.testing.assert_allclose(out[0, 0], [2.0, 2.0, 2.0])
        assert np.isnan(out[0, 1]).all()         # NaN slot preserved

    def test_shape_preserved(self):
        emb = np.random.default_rng(0).standard_normal((3, 4, 5))
        out = R._apply_transform(emb, lambda x: x)
        assert out.shape == (3, 4, 5)


# --------------------------------------------------------------------------- #
# slot-aware language assignment for B1 (brief change 3; no model loaded)
# --------------------------------------------------------------------------- #
def _slot_cache(binarized, seg_start=0.0, seg_step=1.0, rf_start=0.0, rf_step=0.1, rf_duration=0.1):
    """A minimal stage-1-shaped cache with just the fields the slot helpers read.

    `binarized` is (num_chunks, num_frames, num_speakers). Frame f of chunk c has middle time
    seg_start + c*seg_step + rf_start + f*rf_step + rf_duration/2."""
    return {"binarized": np.asarray(binarized, dtype=np.float32),
            "seg_start": np.float64(seg_start), "seg_step": np.float64(seg_step),
            "rf_start": np.float64(rf_start), "rf_step": np.float64(rf_step),
            "rf_duration": np.float64(rf_duration)}


class TestSlotFrameTimes:
    def test_maps_active_frames_to_absolute_middle_time(self):
        # 1 chunk, 3 frames, 2 speakers. rf_step 0.1, rf_duration 0.1 -> frame middles 0.05, 0.15, 0.25.
        bin_ = np.zeros((1, 3, 2), dtype=np.float32)
        bin_[0, 0, 0] = 1.0            # speaker 0 active in frame 0
        bin_[0, 2, 0] = 1.0            # speaker 0 active in frame 2
        bin_[0, 1, 1] = 1.0            # speaker 1 active in frame 1
        t = R.slot_frame_times(_slot_cache(bin_))
        assert t.shape == (1, 2, 3)
        np.testing.assert_allclose(t[0, 0], [0.05, np.nan, 0.25], equal_nan=True, atol=1e-6)
        np.testing.assert_allclose(t[0, 1], [np.nan, 0.15, np.nan], equal_nan=True, atol=1e-6)

    def test_chunk_offset_applied(self):
        bin_ = np.ones((2, 1, 1), dtype=np.float32)      # 2 chunks, 1 frame, 1 speaker
        t = R.slot_frame_times(_slot_cache(bin_, seg_start=0.0, seg_step=5.0))
        # chunk 0 frame middle 0.05 ; chunk 1 frame middle 5.0 + 0.05
        np.testing.assert_allclose(t[0, 0, 0], 0.05, atol=1e-6)
        np.testing.assert_allclose(t[1, 0, 0], 5.05, atol=1e-6)


class TestSlotLanguages:
    def test_argmax_over_active_frames_centred_window(self):
        # 1 chunk, 20 frames, 1 speaker, all active. rf_step 0.1 -> frame middles 0.05..1.95.
        # centred window per frame: k = round(mid/0.1) - 5. Give lang 0 more summed mass over the pair.
        bin_ = np.ones((1, 20, 1), dtype=np.float32)
        n_win = 40
        post = np.zeros((n_win, 3))
        post[:, 0] = 0.6              # lang 0 dominates everywhere over pair (0,2)
        post[:, 2] = 0.4
        langs = R.slot_languages(_slot_cache(bin_), post, (0, 2))
        assert langs.shape == (1, 1)
        assert langs[0, 0] == 0

    def test_inactive_slot_is_minus_one(self):
        bin_ = np.zeros((1, 5, 2), dtype=np.float32)
        bin_[0, :, 0] = 1.0          # only speaker 0 active
        post = np.zeros((20, 2)); post[:, 0] = 1.0
        langs = R.slot_languages(_slot_cache(bin_), post, (0, 1))
        assert langs[0, 0] == 0
        assert langs[0, 1] == -1     # inactive -> -1

    def test_no_pair_mass_is_active_unlabelled_minus_two(self):
        bin_ = np.ones((1, 3, 1), dtype=np.float32)
        post = np.zeros((20, 3))     # mass only on index 1, which is not in pair (0,2)
        post[:, 1] = 1.0
        langs = R.slot_languages(_slot_cache(bin_), post, (0, 2))
        assert langs[0, 0] == -2     # ACTIVE but no pair mass -> -2 (distinct from inactive -1)

    def test_tie_goes_to_smaller_pair_index(self):
        bin_ = np.ones((1, 3, 1), dtype=np.float32)
        post = np.zeros((20, 3))
        post[:, 0] = 0.5; post[:, 2] = 0.5       # equal summed mass over the pair -> smaller index
        langs = R.slot_languages(_slot_cache(bin_), post, (0, 2))
        assert langs[0, 0] == 0
        langs_rev = R.slot_languages(_slot_cache(bin_), post, (2, 0))   # reversed pair, still smaller index
        assert langs_rev[0, 0] == 0


class TestB1Transform:
    def test_centres_each_finite_slot_by_its_language_mean(self):
        means = {0: np.array([1.0, 0.0, 0.0]), 1: np.array([0.0, 1.0, 0.0])}
        slot_langs = np.array([[0, 1]])          # 1 chunk, 2 speakers
        emb = np.zeros((1, 2, 3))
        emb[0, 0] = [3.0, 0.0, 4.0]              # slot (0,0) lang 0: subtract [1,0,0] -> [2,0,4] -> unit
        emb[0, 1] = [0.0, 2.0, 0.0]              # slot (0,1) lang 1: subtract [0,1,0] -> [0,1,0] -> unit
        out = R.b1_transform(means, slot_langs)(emb)
        np.testing.assert_allclose(out[0, 0], R.unit(np.array([2.0, 0.0, 4.0])), atol=1e-9)
        np.testing.assert_allclose(out[0, 1], [0.0, 1.0, 0.0], atol=1e-9)

    def test_leaves_inactive_slots_untouched_even_when_finite(self):
        # lang -1 = INACTIVE (no active frame); clustering drops it. A native embedder (e2) can leave such a slot
        # FINITE, so the skip is keyed on activity (lang -1), not on NaN.
        means = {0: np.zeros(3)}
        slot_langs = np.array([[0, -1]])
        emb = np.ones((1, 2, 3))                 # BOTH slots finite
        emb[0, 0] = [2.0, 0.0, 0.0]
        out = R.b1_transform(means, slot_langs)(emb)
        np.testing.assert_allclose(out[0, 0], [1.0, 0.0, 0.0], atol=1e-9)   # active slot 0 centred -> unit
        np.testing.assert_allclose(out[0, 1], [1.0, 1.0, 1.0], atol=1e-9)   # inactive slot 1 untouched

    def test_leaves_inactive_nan_slots_untouched(self):
        means = {0: np.zeros(3)}
        slot_langs = np.array([[0, -1]])
        emb = np.ones((1, 2, 3)); emb[0, 1] = np.nan
        out = R.b1_transform(means, slot_langs)(emb)
        assert np.isnan(out[0, 1]).all()         # inactive NaN slot preserved

    def test_fails_loudly_on_active_unlabelled_slot_minus_two(self):
        means = {0: np.zeros(3)}
        slot_langs = np.array([[-2]])            # active but no LID pair mass
        emb = np.ones((1, 1, 3))
        with pytest.raises(ValueError):
            R.b1_transform(means, slot_langs)(emb)

    def test_fails_loudly_on_active_slot_with_nonfinite_embedding(self):
        means = {0: np.zeros(3)}
        slot_langs = np.array([[0]])             # active, labelled
        emb = np.full((1, 1, 3), np.nan)         # but embedding non-finite -> should have been dropped
        with pytest.raises(ValueError):
            R.b1_transform(means, slot_langs)(emb)

    def test_zero_means_reproduce_b0_up_to_unit_norm(self):
        # all means zero: e' = unit(e - 0) = unit(e). This is B0 iff AHC-cosine is norm-invariant (checked on the
        # box in acceptance (d)); here we assert the directional identity the transform guarantees.
        means = {0: np.zeros(3), 1: np.zeros(3)}
        slot_langs = np.array([[0, 1]])
        rng = np.random.default_rng(1)
        emb = rng.standard_normal((1, 2, 3))
        out = R.b1_transform(means, slot_langs)(emb)
        for p in range(2):
            np.testing.assert_allclose(out[0, p], R.unit(emb[0, p]), atol=1e-9)
            assert abs(np.linalg.norm(out[0, p]) - 1.0) < 1e-9

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError):
            R.b1_transform({0: np.zeros(3)}, np.array([[0, 0]]))(np.ones((1, 3, 3)))  # 3 speakers vs 2 langs rows


# --------------------------------------------------------------------------- #
# B2 adapter: inputs from the EXCLUSIVE annotation (brief change 2; stub embedder, no model)
# --------------------------------------------------------------------------- #
class _FakeAnnotation:
    """Minimal pyannote-Annotation stand-in: itertracks(yield_label=True), labels(), rename_labels(mapping=)."""

    def __init__(self, segs):
        self._segs = [(float(s), float(e), str(l)) for s, e, l in segs]

    def itertracks(self, yield_label=True):
        from types import SimpleNamespace
        for i, (s, e, l) in enumerate(self._segs):
            yield SimpleNamespace(start=s, end=e), i, l

    def labels(self):
        return sorted({l for _, _, l in self._segs})

    def rename_labels(self, mapping):
        return _FakeAnnotation([(s, e, mapping.get(l, l)) for s, e, l in self._segs])


class TestRunB2MergeExclusiveInputs:
    def test_uses_exclusive_segments_not_overlapping(self):
        # overlapping credits a big chunk of time to A around an overlap; exclusive is the true per-speaker time.
        # Make exclusive segments long enough to embed (>= MIN_SAMPLES/SR = 0.25 s) and keep them 0.1-aligned.
        overlapping = _FakeAnnotation([(0.0, 5.0, "SPEAKER_00"), (0.0, 5.0, "SPEAKER_01"),
                                       (5.0, 6.0, "SPEAKER_00")])
        exclusive = _FakeAnnotation([(0.0, 3.0, "SPEAKER_00"), (3.0, 6.0, "SPEAKER_01")])
        # two near-identical centroids + a cross-language majority so b2_merge considers the pair
        table = {(0.0, 3.0): [1.0, 0.0, 0.0], (3.0, 6.0): [1.0, 0.0, 0.01]}
        emb = _StubEmbedder(table)
        # posterior: steps under SPEAKER_00's exclusive time read lang 0, under SPEAKER_01's read lang 2
        n = 70
        post = np.zeros((n, 3))
        post[:, 0] = 1.0                     # default lang 0 everywhere over pair (0,2)
        # for SPEAKER_01 segment [3.0,6.0): steps 30..59 -> centred windows 25..54 -> make those lang 2
        for g in range(30, 60):
            k = g - 5
            post[k] = [0.0, 0.0, 1.0]
        resolve, skipped = R.run_b2_merge(overlapping, exclusive, emb, post, (0, 2),
                                          threshold=0.9, delta=0.5)
        # both speakers had an exclusive segment, an embedding and language mass -> neither skipped for those
        assert skipped["no_exclusive"] == []
        assert skipped["no_embedding"] == []
        assert skipped["no_lang_mass"] == []
        # cross-language, near-identical centroids, dist ~0 < threshold+delta -> merged to one label
        finals = set(resolve.values())
        assert len(finals) == 1
        # resolve covers every label in either annotation
        assert set(resolve) == {"SPEAKER_00", "SPEAKER_01"}

    def test_speaker_without_exclusive_segment_is_reported_and_kept(self):
        overlapping = _FakeAnnotation([(0.0, 3.0, "SPEAKER_00"), (0.0, 3.0, "SPEAKER_02")])
        exclusive = _FakeAnnotation([(0.0, 3.0, "SPEAKER_00")])   # SPEAKER_02 has no exclusive time
        emb = _StubEmbedder({(0.0, 3.0): [1.0, 0.0]})
        post = np.ones((40, 2))                                  # lang 0 everywhere; pair (0,1)
        resolve, skipped = R.run_b2_merge(overlapping, exclusive, emb, post, (0, 1),
                                          threshold=0.9, delta=0.0)
        assert skipped["no_exclusive"] == ["SPEAKER_02"]
        assert resolve["SPEAKER_02"] == "SPEAKER_02"            # kept its own label
        assert resolve["SPEAKER_00"] == "SPEAKER_00"

    def test_delta_zero_is_identity(self):
        overlapping = _FakeAnnotation([(0.0, 3.0, "A"), (3.0, 6.0, "B")])
        exclusive = _FakeAnnotation([(0.0, 3.0, "A"), (3.0, 6.0, "B")])
        emb = _StubEmbedder({(0.0, 3.0): [1.0, 0.0, 0.0], (3.0, 6.0): [0.0, 0.0, 1.0]})
        n = 70
        post = np.zeros((n, 3)); post[:, 0] = 1.0
        for g in range(30, 60):
            post[g - 5] = [0.0, 0.0, 1.0]
        resolve, _ = R.run_b2_merge(overlapping, exclusive, emb, post, (0, 2), threshold=0.9, delta=0.0)
        assert all(resolve[k] == k for k in resolve)            # delta=0 reproduces B0


# --------------------------------------------------------------------------- #
# M2 run-splitting adapter (with a stub embedder; no model loaded)
# --------------------------------------------------------------------------- #
class _StubEmbedder:
    """Returns a preset unit embedding per (start,end), or None for segments shorter than MIN_SAMPLES-equivalent.

    The map is keyed by (start, end) rounded to 6 dp, mirroring SegmentEmbedder's own cache key."""

    def __init__(self, table):
        self.table = {(round(s, 6), round(e, 6)): v for (s, e), v in table.items()}

    def embed(self, start, end):
        v = self.table.get((round(float(start), 6), round(float(end), 6)))
        if v is None:
            return None
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v]


class TestRunM2Relabel:
    def test_relabels_short_cross_language_segment(self):
        # A (long) | B (short, close to A) | A (long) -> B relabelled to A
        segs = [(0.0, 3.0, "A"), (3.1, 3.5, "B"), (3.6, 6.0, "A")]
        emb = _StubEmbedder({(0.0, 3.0): [1.0, 0.0, 0.0], (3.1, 3.5): [1.0, 0.02, 0.0],
                             (3.6, 6.0): [1.0, 0.0, 0.0]})
        out, recs, skipped = R.run_m2_relabel(segs, emb, lambda t: 0.9, tau0=0.5, alpha=0.0)
        assert [lbl for _, _, lbl in out] == ["A", "A", "A"]
        assert len(recs) == 1 and recs[0]["index"] == 1 and recs[0]["new"] == "A"
        assert skipped == 0

    def test_segment_without_embedding_splits_the_run(self):
        # middle segment has no embedding (None) -> it cannot be relabelled and it breaks the run,
        # so the A|None|A around it are never compared across it.
        segs = [(0.0, 3.0, "A"), (3.1, 3.5, "B"), (3.6, 6.0, "A")]
        emb = _StubEmbedder({(0.0, 3.0): [1.0, 0.0, 0.0], (3.1, 3.5): None,
                             (3.6, 6.0): [1.0, 0.0, 0.0]})
        out, recs, skipped = R.run_m2_relabel(segs, emb, lambda t: 0.9, tau0=0.9, alpha=0.0)
        assert [lbl for _, _, lbl in out] == ["A", "B", "A"]   # nothing relabelled
        assert recs == []
        assert skipped == 1

    def test_global_indices_in_records(self):
        # a no-embedding head, then a relabellable pair -> the record index is the GLOBAL position
        segs = [(0.0, 0.1, "X"), (0.2, 3.0, "A"), (3.1, 3.5, "B"), (3.6, 6.0, "A")]
        emb = _StubEmbedder({(0.0, 0.1): None, (0.2, 3.0): [1.0, 0.0], (3.1, 3.5): [1.0, 0.01],
                             (3.6, 6.0): [1.0, 0.0]})
        out, recs, skipped = R.run_m2_relabel(segs, emb, lambda t: 0.9, tau0=0.5, alpha=0.0)
        assert [lbl for _, _, lbl in out] == ["X", "A", "A", "A"]
        assert len(recs) == 1 and recs[0]["index"] == 2    # the B at global index 2
        assert skipped == 1

    def test_passthrough_when_no_embeddings(self):
        segs = [(0.0, 1.0, "A"), (1.1, 2.0, "B")]
        emb = _StubEmbedder({(0.0, 1.0): None, (1.1, 2.0): None})
        out, recs, skipped = R.run_m2_relabel(segs, emb, lambda t: 1.0, tau0=0.9, alpha=0.0)
        assert out == segs
        assert recs == [] and skipped == 2

    def test_deployed_form_tau0_defaults_to_zero_and_alpha0_is_noop(self):
        # brief change 4: tau0 defaults to 0.0; with alpha=0 the gate distance < 0 + 0*lambda = 0 can never fire,
        # so even adjacent cross-language segments whose embeddings are IDENTICAL (distance 0) are not relabelled.
        segs = [(0.0, 3.0, "A"), (3.1, 3.5, "B"), (3.6, 6.0, "A")]
        emb = _StubEmbedder({(0.0, 3.0): [1.0, 0.0, 0.0], (3.1, 3.5): [1.0, 0.0, 0.0],
                             (3.6, 6.0): [1.0, 0.0, 0.0]})
        out, recs, skipped = R.run_m2_relabel(segs, emb, lambda t: 1.0, alpha=0.0)   # tau0 default 0.0
        assert [lbl for _, _, lbl in out] == ["A", "B", "A"]     # nothing relabelled
        assert recs == []

    def test_explicit_tau0_still_honoured(self):
        # the parameter is still explicit for the grid sweep: a positive tau0 CAN relabel at alpha=0.
        segs = [(0.0, 3.0, "A"), (3.1, 3.5, "B"), (3.6, 6.0, "A")]
        emb = _StubEmbedder({(0.0, 3.0): [1.0, 0.0, 0.0], (3.1, 3.5): [1.0, 0.0, 0.0],
                             (3.6, 6.0): [1.0, 0.0, 0.0]})
        out, recs, _ = R.run_m2_relabel(segs, emb, lambda t: 1.0, alpha=0.0, tau0=0.5)
        assert [lbl for _, _, lbl in out] == ["A", "A", "A"]     # distance 0 < 0.5 -> B relabelled


# --------------------------------------------------------------------------- #
# pair_indices helper
# --------------------------------------------------------------------------- #
class TestPairIndices:
    def test_resolves_codes(self):
        labels = ["en", "hi", "ne", "bn"]
        assert R.pair_indices(labels, "hi", "en") == (1, 0)
        assert R.pair_indices(labels, "ne", "bn") == (2, 3)

    def test_raises_on_missing(self):
        with pytest.raises(ValueError):
            R.pair_indices(["en", "ne"], "en", "zz")


# --------------------------------------------------------------------------- #
# load_stage1 version guard (pure: build a fake .npz, no model)
# --------------------------------------------------------------------------- #
class TestLoadStage1VersionGuard:
    def test_rejects_wrong_version(self, tmp_path):
        p = R.stage1_cache_path(tmp_path, "e3", "REC")
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, version=np.int64(R.CACHE_VERSION - 1), rid="REC")
        with pytest.raises(ValueError):
            R.load_stage1(tmp_path, "e3", "REC")

    def test_accepts_current_version(self, tmp_path):
        p = R.stage1_cache_path(tmp_path, "e3", "REC")
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, version=np.int64(R.CACHE_VERSION), rid="REC", embeddings=np.zeros((1, 1, 4)))
        data = R.load_stage1(tmp_path, "e3", "REC")
        assert int(data["version"]) == R.CACHE_VERSION


# --------------------------------------------------------------------------- #
# system_tag: distinct systems AND parameter values give distinct output paths (brief change 2)
# --------------------------------------------------------------------------- #
class TestSystemTag:
    def test_baselines_are_bare_names(self):
        assert R.system_tag("B0") == "B0"
        assert R.system_tag("B1") == "B1"
        assert R.system_tag("M1") == "M1"

    def test_b2_carries_delta(self):
        assert R.system_tag("B2", delta=0.0) == "B2_d0.000"
        assert R.system_tag("B2", delta=0.25) == "B2_d0.250"
        assert R.system_tag("B2", delta=0.0) != R.system_tag("B2", delta=0.05)

    def test_m2_carries_alpha_and_tau0(self):
        assert R.system_tag("M2", alpha=0.0, tau0=0.0) == "M2_a0.00_t0.00"
        assert R.system_tag("M2", alpha=0.5, tau0=0.0) == "M2_a0.50_t0.00"
        assert R.system_tag("M2", alpha=0.5, tau0=0.0) != R.system_tag("M2", alpha=0.55, tau0=0.0)
        assert R.system_tag("M2", alpha=0.5, tau0=0.0) != R.system_tag("M2", alpha=0.5, tau0=0.1)

    def test_m3_carries_checkpoint_stem(self):
        # Ad-hoc checkpoints whose filename already names the model keep the file stem.
        assert R.system_tag("M3", m3_ckpt="m3:/x/seed0_all.pt") == "M3_seed0_all"
        assert (R.system_tag("M3", m3_ckpt="m3:/a/seed0_all.pt")
                == R.system_tag("M3", m3_ckpt="m3:/b/seed0_all.pt"))   # same stem -> same tag (ad-hoc names only)
        assert (R.system_tag("M3", m3_ckpt="m3:/x/seed0_all.pt")
                != R.system_tag("M3", m3_ckpt="m3:/x/seed1_all.pt"))

    def test_m3_best_pt_uses_run_folder(self):
        # m3_train / run_m3_queue write every checkpoint to <run>/best.pt, so the file stem is "best" for all
        # five seeds. The tag must derive from the run folder so the seeds never share an output directory.
        assert (R.system_tag("M3", m3_ckpt="m3:~/data/m3/runs/all_s0_w0.3_none/best.pt")
                == "M3_all_s0_w0.3_none")
        assert (R.system_tag("M3", m3_ckpt="~/data/m3/runs/all_s0_w0.3_none/best.pt")   # no m3: prefix
                == "M3_all_s0_w0.3_none")
        seeds = [f"m3:~/data/m3/runs/all_s{s}_w0.3_none/best.pt" for s in range(5)]
        tags = {R.system_tag("M3", m3_ckpt=c) for c in seeds}
        assert len(tags) == 5          # best.pt across seeds no longer collides

    def test_m3_ckpt_name_directory_case_keeps_dotted_name(self):
        # Given the RUN FOLDER itself (no best.pt file part), m3_ckpt_name must take the last path component
        # VERBATIM -- not Path.stem, which truncates a dotted folder name (`all_s0_w0.3_none` -> `all_s0_w0`)
        # and would collide two w-values on `all_s0_w0`. A trailing slash is the same case.
        assert R.m3_ckpt_name("~/data/m3/runs/all_s0_w0.3_none") == "all_s0_w0.3_none"
        assert R.m3_ckpt_name("m3:~/data/m3/runs/all_s0_w0.3_none/") == "all_s0_w0.3_none"
        # two w-values that Path.stem would collapse stay distinct.
        assert (R.m3_ckpt_name("~/data/m3/runs/all_s0_w0.3_none")
                != R.m3_ckpt_name("~/data/m3/runs/all_s0_w0.1_none"))
        # a real checkpoint FILE is unaffected: best.pt -> run folder, ad-hoc name -> its stem.
        assert R.m3_ckpt_name("m3:~/data/m3/runs/all_s0_w0.3_none/best.pt") == "all_s0_w0.3_none"
        assert R.m3_ckpt_name("m3:/x/seed0_all.pt") == "seed0_all"
        # and the dotted run folder tags consistently through model_tag / system_tag.
        assert R.model_tag("m3:~/data/m3/runs/all_s0_w0.3_none") == "m3-all_s0_w0.3_none"
        assert R.system_tag("M3", m3_ckpt="~/data/m3/runs/all_s0_w0.3_none") == "M3_all_s0_w0.3_none"

    def test_m3_tag_matches_training_and_freeze_names(self):
        # The DISPLACE tag of each frozen all-data checkpoint must equal m3_<run folder>, exactly the name
        # m3_train uses (confirm_m2b.M3_NAMES / M3_GRIDS and m3_artifacts' ~/data/m3/runs/<name>/best.pt). A
        # drift between how the models are trained/named and how the DISPLACE path names them fails here.
        from csdiar.analysis import confirm_m2b as C
        for name, grid in zip(C.M3_NAMES, C.M3_GRIDS):
            ckpt = f"m3:~/data/m3/runs/{name}/best.pt"
            assert R.system_tag("M3", m3_ckpt=ckpt) == f"M3_{name}"
            assert grid == f"m3_{name}"
        # m3_artifacts' checkpoint paths carry the same run-folder names (no-arg returns only the checkpoints).
        ckpts = [a for a in C.m3_artifacts() if a.endswith("best.pt")]
        assert [R.m3_ckpt_name(c) for c in ckpts] == list(C.M3_NAMES)
        assert C.M3_NAMES == tuple(f"all_s{s}_w0.3_none" for s in range(5))

    def test_m3_artifacts_stage1_caches_match_producer_layout(self):
        # Fails if m3_artifacts(rec)'s per-seed stage-1 cache paths drift from what run_m2b actually writes.
        # confirm_m2b must pin exactly the caches run_m2b produced (<cache_dir>/stage1/m3-<name>/<rid>.npz),
        # NOT a dense-grid or `emb_displace` layout that nothing writes (review m3-naming MAJOR).
        from csdiar.analysis import confirm_m2b as C
        rec = {"paths": {"cache_dir": "/cache"}, "split": {"eval": ["EVA00", "EVA01"]}}
        arts = C.m3_artifacts(rec)
        ckpts = [a for a in arts if a.endswith("best.pt")]
        caches = [a for a in arts if a.endswith(".npz")]
        # checkpoints: one per seed, exactly the run-folder names.
        assert ckpts == [f"~/data/m3/runs/{n}/best.pt" for n in C.M3_NAMES]
        # stage-1 caches: one per (seed, eval rid), each byte-identical to run_m2b.stage1_cache_path's output.
        expected = [str(R.stage1_cache_path("/cache", f"m3:~/data/m3/runs/{n}/best.pt", rid))
                    for n in C.M3_NAMES for rid in ("EVA00", "EVA01")]
        assert sorted(caches) == sorted(expected)
        assert all("/stage1/m3-" in c for c in caches)
        # no fictional layout: nothing under emb_displace or a dense grid is ever listed.
        assert not any("emb_displace" in a or "/emb/" in a or "/dense/" in a for a in arts)
        # the no-record call omits the record-dependent caches (used by the naming test / confirm_h4 parity).
        assert C.m3_artifacts() == ckpts


    def test_uniqueness_across_systems_and_values(self):
        tags = {
            R.system_tag("B0"), R.system_tag("B1"), R.system_tag("M1"),
            R.system_tag("B2", delta=0.0), R.system_tag("B2", delta=0.1),
            R.system_tag("M2", alpha=0.0, tau0=0.0), R.system_tag("M2", alpha=0.5, tau0=0.0),
            R.system_tag("M3", m3_ckpt="m3:/x/seed0_all.pt"),
            R.system_tag("M3", m3_ckpt="m3:/x/seed1_all.pt"),
        }
        assert len(tags) == 9        # every entry distinct

    def test_rejects_unknown_system(self):
        with pytest.raises(ValueError):
            R.system_tag("ZZ")


# --------------------------------------------------------------------------- #
# validate_thresholds (review m5)
# --------------------------------------------------------------------------- #
class TestValidateThresholds:
    def test_accepts_grid(self):
        grid = [round(0.30 + 0.05 * k, 2) for k in range(15)]
        assert R.validate_thresholds(grid) == grid

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            R.validate_thresholds([0.0])
        with pytest.raises(ValueError):
            R.validate_thresholds([2.5])

    def test_rejects_non_finite(self):
        with pytest.raises(ValueError):
            R.validate_thresholds([float("nan")])
        with pytest.raises(ValueError):
            R.validate_thresholds([float("inf")])

    def test_rejects_finer_than_2dp(self):
        with pytest.raises(ValueError):
            R.validate_thresholds([0.7000001])

    def test_rejects_duplicates_after_rounding(self):
        with pytest.raises(ValueError):
            R.validate_thresholds([0.70, 0.70])


# --------------------------------------------------------------------------- #
# CLI refusals: missing required arg AND irrelevant arg (brief change 3 / review m6)
# --------------------------------------------------------------------------- #
class _NS:
    """A minimal argparse.Namespace stand-in for _validate_cluster_args."""
    def __init__(self, **kw):
        base = dict(system="B0", model="e3", pair=None, means=None, v_hat=None,
                    delta=None, alpha=None, tau0=None, wav=None)
        base.update(kw)
        self.__dict__.update(base)


class TestClusterArgValidation:
    def test_b0_clean(self):
        R._validate_cluster_args(_NS(system="B0", model="e3"))

    def test_b1_requires_pair_and_means(self):
        with pytest.raises(SystemExit):
            R._validate_cluster_args(_NS(system="B1", pair="en,hi"))        # no means
        with pytest.raises(SystemExit):
            R._validate_cluster_args(_NS(system="B1", means="m.npz"))       # no pair
        R._validate_cluster_args(_NS(system="B1", pair="en,hi", means="m.npz"))

    def test_m1_requires_v_hat_and_forbids_others(self):
        with pytest.raises(SystemExit):
            R._validate_cluster_args(_NS(system="M1"))                      # no v-hat
        with pytest.raises(SystemExit):
            R._validate_cluster_args(_NS(system="M1", v_hat="v.npy", pair="en,hi"))   # pair irrelevant
        R._validate_cluster_args(_NS(system="M1", v_hat="v.npy"))

    def test_b2_requires_pair_delta_wav(self):
        with pytest.raises(SystemExit):
            R._validate_cluster_args(_NS(system="B2", pair="en,hi", delta=0.1))       # no wav
        with pytest.raises(SystemExit):
            R._validate_cluster_args(_NS(system="B2", pair="en,hi", wav="a.wav"))     # no delta
        R._validate_cluster_args(_NS(system="B2", pair="en,hi", delta=0.1, wav="a.wav"))

    def test_m2_requires_pair_alpha_wav_tau0_optional(self):
        with pytest.raises(SystemExit):
            R._validate_cluster_args(_NS(system="M2", pair="en,hi", alpha=0.5))       # no wav
        R._validate_cluster_args(_NS(system="M2", pair="en,hi", alpha=0.5, wav="a.wav"))
        R._validate_cluster_args(_NS(system="M2", pair="en,hi", alpha=0.5, tau0=0.0, wav="a.wav"))
        with pytest.raises(SystemExit):
            R._validate_cluster_args(_NS(system="M2", pair="en,hi", alpha=0.5, delta=0.1, wav="a.wav"))  # delta irrelevant

    def test_m3_model_rule(self):
        with pytest.raises(SystemExit):
            R._validate_cluster_args(_NS(system="M3", model="e3"))          # M3 needs m3:<ckpt>
        with pytest.raises(SystemExit):
            R._validate_cluster_args(_NS(system="B0", model="m3:/x/ck.pt")) # non-M3 forbids m3 model
        R._validate_cluster_args(_NS(system="M3", model="m3:/x/seed0_all.pt"))

    def test_b2_m2_irrelevant_args_refused(self):
        with pytest.raises(SystemExit):
            R._validate_cluster_args(_NS(system="B2", pair="en,hi", delta=0.1, wav="a.wav", v_hat="v.npy"))
        with pytest.raises(SystemExit):
            R._validate_cluster_args(_NS(system="B0", model="e3", alpha=0.5))

    def test_pair_parse_rejects_bad(self):
        with pytest.raises(SystemExit):
            R._parse_pair_arg("en")
        with pytest.raises(SystemExit):
            R._parse_pair_arg("en,en")
        assert R._parse_pair_arg("en,hi") == ("en", "hi")


# --------------------------------------------------------------------------- #
# pyannote-backed tests: apply_m2_overlapping and run_system_from_cache.
# Use real pyannote.core Annotations (available in .venv-pyannote4) with cluster_from_cache / SegmentEmbedder
# monkeypatched, so no model loads and no audio is read.
# --------------------------------------------------------------------------- #
try:                                        # pyannote.core lives in .venv-pyannote4, not the stdlib test venv
    import pyannote.core as pyannote_core
    Annotation = pyannote_core.Annotation
    Segment = pyannote_core.Segment
    _HAVE_PYANNOTE = True
except ImportError:                         # the pure naming/tag tests above do NOT need it, so only the
    pyannote_core = None                    # pyannote-backed classes below are skipped (see _needs_pyannote)
    Annotation = Segment = None
    _HAVE_PYANNOTE = False

_needs_pyannote = pytest.mark.skipif(not _HAVE_PYANNOTE, reason="pyannote.core not installed (.venv-pyannote4 only)")


def _ann(segs, uri="REC"):
    a = Annotation(uri=uri)
    for i, (s, e, l) in enumerate(segs):
        a[Segment(float(s), float(e)), i] = str(l)
    return a


def _flat(annotation):
    return sorted((round(float(s.start), 4), round(float(s.end), 4), str(l))
                  for s, _, l in annotation.itertracks(yield_label=True))


@_needs_pyannote
class TestApplyM2Overlapping:
    def test_relabels_only_the_source_track_inside_the_span(self):
        # overlapping has A over [0,3] and B over [2,4] (they overlap on [2,3]); M2 relabelled the exclusive
        # B-segment [3,4] -> A. The overlapping B track inside [3,4] becomes A; the B over [2,3] (the overlap
        # partner) stays B, and A is untouched.
        ov = _ann([(0.0, 3.0, "A"), (2.0, 4.0, "B")])
        records = [{"start": 3.0, "end": 4.0, "old": "B", "new": "A"}]
        out = R.apply_m2_overlapping(ov, records)
        flat = _flat(out)
        # A now spans [0,3] + [3,4] -> merged to [0,4]; B remains on [2,3]
        assert (0.0, 4.0, "A") in flat
        assert (2.0, 3.0, "B") in flat
        assert not any(l == "B" and e > 3.0 for _, e, l in flat)   # no B past 3.0

    def test_source_and_target_overlap_in_span(self):
        # the span [3,5] where the source (B) and target (A) BOTH have overlapping tracks: B[3,5] -> A, and A
        # already has a track [3,5]; after relabel the two A pieces merge, and no B remains in the span.
        ov = _ann([(3.0, 5.0, "A"), (3.0, 5.0, "B")])
        records = [{"start": 3.0, "end": 5.0, "old": "B", "new": "A"}]
        out = R.apply_m2_overlapping(ov, records)
        flat = _flat(out)
        assert (3.0, 5.0, "A") in flat
        assert all(l != "B" for _, _, l in flat)       # B fully absorbed into A over the span

    def test_leaves_other_speakers_untouched(self):
        ov = _ann([(0.0, 2.0, "A"), (0.0, 2.0, "C"), (2.0, 3.0, "B")])
        records = [{"start": 2.0, "end": 3.0, "old": "B", "new": "A"}]
        out = R.apply_m2_overlapping(ov, records)
        flat = _flat(out)
        assert (0.0, 2.0, "C") in flat                 # C is an overlap partner, untouched
        assert (0.0, 3.0, "A") in flat or ((0.0, 2.0, "A") in flat and (2.0, 3.0, "A") in flat)

    def test_no_records_is_identity(self):
        ov = _ann([(0.0, 2.0, "A"), (2.0, 4.0, "B")])
        assert _flat(R.apply_m2_overlapping(ov, [])) == _flat(ov)

    def test_two_record_chain_no_cascade(self):
        # m1: a two-record chain. Exclusive M2 produced {old B, new A, [3,4]} and {old C, new B, [4,5]}.
        # The overlapping annotation has a B track over [2,6] and a D overlap partner over [3.5,5.5].
        # Decisions are on the ORIGINAL labels, so there must be NO cascade: the piece of B relabelled to A over
        # [3,4] must NOT then be relabelled again by the (C->B) record. Only C's own tracks become B.
        ov = _ann([(2.0, 6.0, "B"), (3.5, 5.5, "D"), (4.0, 5.0, "C")])
        records = [{"start": 3.0, "end": 4.0, "old": "B", "new": "A"},
                   {"start": 4.0, "end": 5.0, "old": "C", "new": "B"}]
        out = R.apply_m2_overlapping(ov, records)
        flat = _flat(out)
        # B over [2,6]: [2,3] stays B, [3,4] -> A (never re-relabelled to anything by the C->B record), [4,6] B.
        assert (2.0, 3.0, "B") in flat
        assert (3.0, 4.0, "A") in flat
        assert (4.0, 6.0, "B") in flat          # the B->A piece did NOT cascade into the C->B record
        # C over [4,5] -> B; merges with the B piece [4,6] under support() into a single [4,6] B (already above)
        assert all(l != "C" for _, _, l in flat)
        # the overlap partner D is untouched
        assert (3.5, 5.5, "D") in flat
        # out-of-span OLD speech: B before [3,4] and after [4,5] is preserved (the [2,3] B above)
        assert any(l == "B" and s <= 2.0 for s, _, l in flat)

    def test_one_old_relabelled_to_two_new_over_disjoint_spans(self):
        # m1: one OLD label (B) relabelled to two DIFFERENT new labels over disjoint spans: B[1,2] -> A and
        # B[4,5] -> C, with B also carrying out-of-span speech on [2,4] and an E overlap partner over [1.5,4.5].
        ov = _ann([(0.0, 6.0, "B"), (1.5, 4.5, "E")])
        records = [{"start": 1.0, "end": 2.0, "old": "B", "new": "A"},
                   {"start": 4.0, "end": 5.0, "old": "B", "new": "C"}]
        out = R.apply_m2_overlapping(ov, records)
        flat = _flat(out)
        assert (0.0, 1.0, "B") in flat          # before the first span: OLD
        assert (1.0, 2.0, "A") in flat          # first span -> A
        assert (2.0, 4.0, "B") in flat          # out-of-span OLD speech untouched
        assert (4.0, 5.0, "C") in flat          # second span -> C
        assert (5.0, 6.0, "B") in flat          # after the second span: OLD
        assert (1.5, 4.5, "E") in flat          # overlap partner untouched


@_needs_pyannote
class TestRunSystemFromCache:
    """Wire-up tests: monkeypatch cluster_from_cache so no model loads; assert each system's write rules."""

    def _patch_cluster(self, monkeypatch, diar, excl):
        def fake(pipe, cache, threshold, transform=None, slot_transform=None):
            # off-settings must call through with a no-op transform; we just return fixed B0 annotations and
            # record whether a transform was passed (so B1/M1 wiring is exercised).
            fake.calls.append({"transform": transform is not None, "slot_transform": slot_transform is not None})
            return diar.copy(), excl.copy(), {"labels": list(diar.labels()), "num_speakers": len(diar.labels())}
        fake.calls = []
        monkeypatch.setattr(R, "cluster_from_cache", fake)
        return fake

    def test_b0_passes_through(self, monkeypatch):
        diar = _ann([(0.0, 3.0, "SPEAKER_00"), (3.0, 6.0, "SPEAKER_01")])
        excl = _ann([(0.0, 3.0, "SPEAKER_00"), (3.0, 6.0, "SPEAKER_01")])
        self._patch_cluster(monkeypatch, diar, excl)
        d, e, info = R.run_system_from_cache(None, {"embeddings": np.zeros((1, 1, 3))}, 0.7, "B0")
        assert _flat(d) == _flat(diar) and _flat(e) == _flat(excl)

    def test_b2_resolve_applied_to_both_annotations(self, monkeypatch):
        # B2 merges SPEAKER_01 into SPEAKER_00; the resolve must hit BOTH annotations.
        diar = _ann([(0.0, 3.0, "SPEAKER_00"), (3.0, 6.0, "SPEAKER_01")])
        excl = _ann([(0.0, 3.0, "SPEAKER_00"), (3.0, 6.0, "SPEAKER_01")])
        self._patch_cluster(monkeypatch, diar, excl)
        monkeypatch.setattr(R, "run_b2_merge",
                            lambda ov, ex, emb, post, pair, th, delta:
                                ({"SPEAKER_00": "SPEAKER_00", "SPEAKER_01": "SPEAKER_00"}, {}))
        emb = _StubEmbedder({})
        post = np.ones((10, 2))
        d, e, info = R.run_system_from_cache(None, {"embeddings": np.zeros((1, 1, 2))}, 0.7, "B2",
                                             post=post, pair=(0, 1), delta=0.1, embedder=emb)
        assert set(l for _, _, l in _flat(d)) == {"SPEAKER_00"}      # overlapping merged
        assert set(l for _, _, l in _flat(e)) == {"SPEAKER_00"}      # exclusive merged too

    def test_m2_exclusive_relabel_and_overlapping_span(self, monkeypatch):
        # exclusive: A|B|A ; M2 relabels the short B [3,3.5] -> A. The overlapping annotation (which also has a
        # C overlapping [3,3.5]) must get A over that span for the B track and keep C.
        diar = _ann([(0.0, 3.0, "A"), (3.0, 3.5, "B"), (3.0, 3.5, "C"), (3.6, 6.0, "A")])
        excl = _ann([(0.0, 3.0, "A"), (3.0, 3.5, "B"), (3.6, 6.0, "A")])
        self._patch_cluster(monkeypatch, diar, excl)

        def fake_relabel(excl_segs, embedder, lam, alpha, tau0=0.0):
            new = [(s, e, "A" if l == "B" else l) for s, e, l in excl_segs]
            recs = [{"index": 1, "old": "B", "new": "A", "start": 3.0, "end": 3.5,
                     "t": 3.25, "lam": 1.0, "dist": 0.0}]
            return new, recs, 0
        monkeypatch.setattr(R, "run_m2_relabel", fake_relabel)
        emb = _StubEmbedder({})
        post = np.ones((10, 2))
        d, e, info = R.run_system_from_cache(None, {"embeddings": np.zeros((1, 1, 2))}, 0.7, "M2",
                                             post=post, pair=(0, 1), alpha=0.5, embedder=emb)
        # exclusive: B fully relabelled to A, merged -> A over [0,6]-ish (two A pieces + relabelled B)
        assert all(l == "A" for _, _, l in _flat(e))
        # overlapping: the B track over [3,3.5] became A; the C over [3,3.5] stays
        flat_d = _flat(d)
        assert (3.0, 3.5, "C") in flat_d
        assert all(l != "B" for _, _, l in flat_d)
        assert info["m2_relabels"] == 1

    def test_requires_inputs_per_system(self, monkeypatch):
        diar = _ann([(0.0, 3.0, "SPEAKER_00")])
        self._patch_cluster(monkeypatch, diar, diar)
        with pytest.raises(ValueError):
            R.run_system_from_cache(None, {"embeddings": np.zeros((1, 1, 2))}, 0.7, "B1")   # no means/post/pair
        with pytest.raises(ValueError):
            R.run_system_from_cache(None, {"embeddings": np.zeros((1, 1, 2))}, 0.7, "M1")   # no v_hat

    def test_b2_active_delta_merges_cross_language_speakers_differs_from_b0(self, monkeypatch):
        # m2: an ACTIVE B2 operating point whose output must DIFFER from B0, with real run_b2_merge / b2_merge
        # (NOT stubbed) -- only cluster_from_cache is faked so no model loads. Two speakers with near-identical
        # centroids but DIFFERENT majority languages; a positive delta makes their cosine distance < threshold +
        # delta, so b2_merge merges them. B0 has two speakers; active B2 collapses to one.
        diar = _ann([(0.0, 3.0, "SPEAKER_00"), (3.0, 6.0, "SPEAKER_01")])
        excl = _ann([(0.0, 3.0, "SPEAKER_00"), (3.0, 6.0, "SPEAKER_01")])
        self._patch_cluster(monkeypatch, diar, excl)
        # unit centroids with cosine distance ~0.95 (cos ~0.05): BETWEEN threshold (0.9) and threshold + delta,
        # so delta=0 does NOT merge (0.95 !< 0.9) but a big delta does (0.95 < 0.9 + 0.95). Segments >= 0.25 s.
        import math as _m
        b_vec = [0.05, _m.sqrt(1.0 - 0.05 ** 2), 0.0]      # unit, cos with [1,0,0] = 0.05 -> dist 0.95
        emb = _StubEmbedder({(0.0, 3.0): [1.0, 0.0, 0.0], (3.0, 6.0): b_vec})
        # posteriors: SPEAKER_00's exclusive time [0,3) reads lang 0; SPEAKER_01's [3,6) reads lang 2 -> the two
        # speakers have different majority languages, so b2_merge considers the pair.
        n = 70
        post = np.zeros((n, 3)); post[:, 0] = 1.0
        for g in range(30, 60):                       # steps under [3,6): centred window g-5 -> lang 2
            post[g - 5] = [0.0, 0.0, 1.0]
        # B0 (delta small -> dist !< threshold+delta) keeps two speakers; active (big delta) merges to one.
        d0, e0, _ = R.run_system_from_cache(None, {"embeddings": np.zeros((1, 1, 3))}, 0.9, "B2",
                                            post=post, pair=(0, 2), delta=0.0, embedder=emb)
        assert len(set(l for _, _, l in _flat(d0))) == 2     # delta 0 -> B0, two speakers
        assert len(set(l for _, _, l in _flat(e0))) == 2
        d, e, info = R.run_system_from_cache(None, {"embeddings": np.zeros((1, 1, 3))}, 0.9, "B2",
                                             post=post, pair=(0, 2), delta=0.95, embedder=emb)
        assert len(set(l for _, _, l in _flat(d))) == 1      # active delta merged the cross-language pair
        assert len(set(l for _, _, l in _flat(e))) == 1
        assert set(info["b2_resolve"].values()) == set(l for _, _, l in _flat(e))

    def test_m2_active_alpha_relabels_short_segment_differs_from_b0(self, monkeypatch):
        # m2: an ACTIVE M2 operating point whose output must DIFFER from B0, with real run_m2_relabel / m2_relabel
        # and make_lambda_closure (NOT stubbed). Exclusive A|B|A with a SHORT middle B whose embedding is close to
        # its A neighbours; lambda at the boundary is high (opposite one-hot LID windows), so alpha>0 relabels B.
        diar = _ann([(0.0, 3.0, "A"), (3.1, 3.5, "B"), (3.6, 6.0, "A")])
        excl = _ann([(0.0, 3.0, "A"), (3.1, 3.5, "B"), (3.6, 6.0, "A")])
        self._patch_cluster(monkeypatch, diar, excl)
        # B's embedding is near-identical to A's -> small cosine distance.
        emb = _StubEmbedder({(0.0, 3.0): [1.0, 0.0], (3.1, 3.5): [1.0, 0.02], (3.6, 6.0): [1.0, 0.0]})
        # lambda_at uses window pair (k-10, k), k = round(t/hop). The A|B boundary ~3.05 -> k~30, and the B|A
        # boundary ~3.55 -> k~36. Make windows alternate one-hot every 10 so those window pairs are opposite
        # (JSD = 1 bit >= 0.5). Pair (0,1).
        n = 60
        post = np.zeros((n, 2))
        for k in range(n):
            post[k] = [1.0, 0.0] if (k // 10) % 2 == 0 else [0.0, 1.0]
        # alpha=0 (tau0=0) is the deployed no-op -> B0 (B stays B); active alpha relabels the short B to A.
        d0, e0, i0 = R.run_system_from_cache(None, {"embeddings": np.zeros((1, 1, 2))}, 0.7, "M2",
                                             post=post, pair=(0, 1), alpha=0.0, embedder=emb)
        assert [l for _, _, l in _flat(e0)].count("B") == 1       # B0: B untouched
        assert i0["m2_relabels"] == 0
        d, e, info = R.run_system_from_cache(None, {"embeddings": np.zeros((1, 1, 2))}, 0.7, "M2",
                                             post=post, pair=(0, 1), alpha=1.0, tau0=0.0, embedder=emb)
        assert info["m2_relabels"] == 1                           # the short B was relabelled
        assert all(l == "A" for _, _, l in _flat(e))              # exclusive: B -> A, merged to all-A
        assert all(l != "B" for _, _, l in _flat(d))              # overlapping carry too


@_needs_pyannote
class TestAnnotationFromSegments:
    def test_merges_adjacent_same_label(self):
        template = _ann([(0.0, 1.0, "X")])
        out = R.annotation_from_segments(template, [(0.0, 3.0, "A"), (3.0, 6.0, "A"), (6.0, 7.0, "B")])
        flat = _flat(out)
        assert (0.0, 6.0, "A") in flat       # two adjacent A pieces merged
        assert (6.0, 7.0, "B") in flat
        assert out.uri == "REC"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))