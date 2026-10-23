"""Tests for csdiar.data.boundaries on hand-built word timings and VAD frames (no audio involved)."""
from collections import Counter

import numpy as np

from csdiar.data.boundaries import (FRAME_S, Word, clip_points, find_pause, match_family, pause_bin,
                                    region_ok)
from csdiar.data.tags import EN, NE, NEUTRAL, runs, tag_word


def make_words(spec):
    """spec: (text, tag, start, end); start None = unaligned."""
    return [Word(i, t, g, s, e, None if s is None else 0.9) for i, (t, g, s, e) in enumerate(spec)]


def vad_from_speech(spans, dur):
    v = np.zeros(int(np.ceil(dur / FRAME_S)), dtype=np.float32)
    for s, e in spans:
        v[int(s / FRAME_S): int(np.ceil(e / FRAME_S))] = 1.0
    return v


def test_pause_bins():
    assert [pause_bin(x) for x in (0, 32, 149, 150, 300, 301)] == ["none", "lt150", "lt150", "150to300",
                                                                  "150to300", "gt300"]


def test_find_pause_takes_longest_silence_between_word_middles():
    left, right = Word(0, "a", NE, 0.0, 1.0), Word(1, "b", EN, 1.4, 2.0)
    vad = vad_from_speech([(0.0, 1.0), (1.4, 2.0)], 2.0)
    p0, p1 = find_pause(vad, left, right)  # silent frames 32..42
    assert abs(p0 - 32 * FRAME_S) < 1e-9 and abs(p1 - 43 * FRAME_S) < 1e-9
    vad = vad_from_speech([(0.0, 2.0)], 2.0)  # no silence: point falls back to the aligned boundary
    assert find_pause(vad, left, right) == (1.2, 1.2)


def test_region_rejects_other_language_and_unaligned_neighbours():
    w = make_words([("म", NE, 0.0, 0.5), ("त", NE, 0.5, 1.0), ("र", NE, 1.0, 1.5), ("5", NEUTRAL, None, None),
                    ("ho", NE, 1.9, 2.4)])
    run = runs([x.tag for x in w])[0]
    span = (w[run.first].start, w[run.last].end)
    assert region_ok(w, 0.1, 1.4, run, span, 3.0)
    assert not region_ok(w, 1.0, 1.7, run, span, 3.0)  # reaches past "र" where the digit is spoken
    assert not region_ok(w, -0.1, 1.0, run, span, 3.0)  # starts before the clip
    w2 = make_words([("म", NE, 0.0, 0.5), ("1", NEUTRAL, None, None), ("2", NEUTRAL, None, None),
                     ("र", NE, 1.0, 1.5)])
    run2 = runs([x.tag for x in w2])[0]
    assert not region_ok(w2, 0.0, 0.8, run2, (0.0, 1.5), 3.0)  # two unaligned digits in a row
    # A switch window's inner edge at a late pause start (0.3 s after the run's last word) is allowed
    assert not region_ok(w, 0.8, 1.8, run, (0.0, 1.5), 3.0)
    assert region_ok(w[:3], 0.8, 1.8, runs([x.tag for x in w[:3]])[0], (0.0, 1.5), 3.0, check_end=False)


def test_clip_points_switch_and_controls():
    # Nepali run of 4 words (0-1.95 s), 0.35 s gap, English run of 4 words (2.3-4.25 s); clip lasts 4.5 s
    spec = [(f"क{i}", NE, 0.5 * i, 0.5 * i + 0.45) for i in range(4)]
    spec += [(f"w{i}", EN, 2.3 + 0.5 * i, 2.3 + 0.5 * i + 0.45) for i in range(4)]
    w = make_words(spec)
    vad = vad_from_speech([(x.start, x.end) for x in w], 4.5)
    ssdl, ctrl = clip_points(w, vad, 4.5, False, Counter())
    assert len(ssdl) == 1 and ssdl[0]["lang_left"] == NE and ssdl[0]["lang_right"] == EN
    assert ssdl[0]["pause_bin"] == "gt300" and ssdl[0]["pause_ms"] == 320
    assert ssdl[0]["elig_det"] and ssdl[0]["elig_emb1"] and ssdl[0]["elig_emb1.5"]
    assert ssdl[0]["elig_p0L1"] and ssdl[0]["elig_det_w1"]
    assert all(c["lang_left"] == c["lang_right"] for c in ctrl)
    # +-1 s of one run fits only around the middle of the English run (the Nepali run starts at the clip edge)
    emb1 = [c for c in ctrl if c["elig_emb1"]]
    assert len(emb1) == 1 and emb1[0]["lang_left"] == EN and emb1[0]["word"] == 5
    assert any(c["elig_emb0.5"] and c["lang_left"] == NE for c in ctrl)


def test_low_alignment_clip_is_never_eligible():
    spec = [("क", NE, 0.0, 1.2), ("ख", NE, 1.3, 2.4), ("w", EN, 2.6, 3.8), ("x", EN, 3.9, 5.0)]
    w = make_words(spec)
    ssdl, ctrl = clip_points(w, vad_from_speech([(0, 2.4), (2.6, 5.0)], 5.0), 5.0, True, Counter())
    assert ssdl and not any(v for p in ssdl + ctrl for k, v in p.items() if k.startswith("elig_"))


def test_matching_prefers_same_clip_and_never_reuses_controls():
    base = {"pause_bin": "none", "pos_bin": 1, "left_s": 1.2, "right_s": 1.2, "video_id": "v1"}
    ssdl = [dict(base, point_id=f"s{i}", clip_id="c1", elig_det=True) for i in range(2)]
    ctrl = [dict(base, point_id=f"k{i}", clip_id="c1" if i < 2 else "c2", elig_det=True) for i in range(5)]
    ctrl.append(dict(base, point_id="other_video", clip_id="c9", video_id="v2", elig_det=True))
    pairs, n = match_family(ssdl, ctrl, "det", seed=1)
    ids = [p["ctrl_id"] for p in pairs]
    assert len(ids) == len(set(ids)) == 5 and "other_video" not in ids
    assert {p["ctrl_id"] for p in pairs if p["level"] == "clip"} == {"k0", "k1"}
    assert n == Counter({3: 1, 2: 1})


def test_names_are_neutral_and_loans_are_loan():
    assert tag_word("Newton", names=frozenset({"newton"})) == NEUTRAL
    assert tag_word("कम्प्युटर", loans=frozenset({"कम्प्युटर"})) == "loan"
    assert tag_word("कम्प्युटर") == NE


def test_single_letters_are_neutral_only_when_asked():
    assert tag_word("X") == EN and tag_word("X", letters_neutral=True) == NEUTRAL
    assert tag_word("I", letters_neutral=True) == NEUTRAL and tag_word("a", letters_neutral=True) == EN
    assert tag_word("X,", letters_neutral=True) == NEUTRAL and tag_word("is", letters_neutral=True) == EN


def test_detection_switch_region_must_be_free_of_neutral_words():
    # ne word, en word (switch at ~1.0 s), then a digit spoken at ~1.4 s: inside +-0.5 s of the switch
    spec = [("क", NE, 0.0, 0.9), ("go", EN, 1.1, 1.3), ("5", NEUTRAL, None, None), ("ख", NE, 1.6, 2.6)]
    w = make_words(spec)
    vad = vad_from_speech([(0.0, 0.9), (1.1, 1.3), (1.35, 1.55), (1.6, 2.6)], 3.0)
    ssdl, _ = clip_points(w, vad, 3.0, False, Counter())
    first = [p for p in ssdl if p["word"] == 0][0]
    assert first["lang_left"] == NE and not first["elig_det"]
    assert not first["elig_p0L1"]  # the English run lasts 0.2 s, shorter than P0's 1 s window


def test_region_clean_allows_both_languages():
    from csdiar.data.boundaries import region_clean
    w = make_words([("क", NE, 0.0, 0.5), ("go", EN, 0.6, 1.0), ("ख", NE, 1.1, 1.6)])
    assert region_clean(w, 0.2, 1.3, 2.0)
    w2 = make_words([("क", NE, 0.0, 0.5), ("Nepal", NEUTRAL, 0.6, 1.0), ("ख", NE, 1.1, 1.6)])
    assert not region_clean(w2, 0.2, 1.3, 2.0)
