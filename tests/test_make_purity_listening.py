"""Tests for scripts/make_purity_listening.py (stdlib + pytest; no audio)."""
from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "make_purity_listening", Path(__file__).resolve().parents[1] / "scripts/make_purity_listening.py")
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def corpus(n_videos: int = 40, per_video: int = 6) -> tuple[dict, set, list]:
    """Clips v<k>_c<j>; clip 0 of each video is flagged and sampled; speech types alternate en / ne / cs."""
    manifest, flagged, sample = {}, set(), []
    for v in range(n_videos):
        for j in range(per_video):
            cid = f"v{v:02d}_c{j}"
            manifest[cid] = {"clip_id": cid, "video_id": f"v{v:02d}", "speech_type": "en ne cs".split()[j % 3],
                             "duration_s": "5.0", "wav": f"wav/{cid}.wav"}
        flagged.add(f"v{v:02d}_c0")
        sample.append(f"v{v:02d}_c0")
    return manifest, flagged, sample


def test_flagged_pairs_take_an_unflagged_partner_from_the_same_video_and_type():
    manifest, flagged, sample = corpus()
    pairs, skipped = P.make_pairs(manifest, flagged, sample)
    assert skipped == []
    fl = [p for p in pairs if p["kind"] == "flagged"]
    assert sorted(p["a"] if p["flagged_side"] == "A" else p["b"] for p in fl) == sorted(sample)
    for p in fl:
        f, ref = (p["a"], p["b"]) if p["flagged_side"] == "A" else (p["b"], p["a"])
        assert ref not in flagged and manifest[ref]["video_id"] == manifest[f]["video_id"] == p["video_id"]
        assert manifest[ref]["speech_type"] == manifest[f]["speech_type"]        # c3 is the only other "en" clip
    assert set(Counter(p["flagged_side"] for p in fl)) == {"A", "B"}             # both orders occur


def test_controls_are_two_distinct_unflagged_clips_of_one_sampled_video():
    manifest, flagged, sample = corpus()
    pairs, _ = P.make_pairs(manifest, flagged, sample)
    ctl = [p for p in pairs if p["kind"] == "control"]
    assert len(ctl) == P.N_CONTROLS == 30
    for p in ctl:
        assert p["a"] != p["b"] and not {p["a"], p["b"]} & flagged and p["flagged_side"] == "-"
        assert manifest[p["a"]]["video_id"] == manifest[p["b"]]["video_id"] == p["video_id"]
        a_type = manifest[p["a"]]["speech_type"]
        same_left = [c for c, m in manifest.items() if m["video_id"] == p["video_id"] and c not in flagged
                     and c != p["a"] and m["speech_type"] == a_type]
        assert (manifest[p["b"]]["speech_type"] == a_type) == bool(same_left)   # same type whenever one is left
    assert len({p["video_id"] for p in ctl}) == 30                               # 40 videos: one control each


def test_items_are_seeded_shuffled_and_numbered():
    manifest, flagged, sample = corpus()
    a, _ = P.make_pairs(manifest, flagged, sample)
    assert a == P.make_pairs(manifest, flagged, sample)[0]
    assert [p["item"] for p in a] == [f"P{k:03d}" for k in range(1, 71)]
    kinds = [p["kind"] for p in a]
    assert kinds != sorted(kinds) and kinds.count("control") == 30               # controls are mixed in
    assert P.make_pairs(manifest, flagged, sample, seed=P.SEED + 1)[0] != a


def test_a_sampled_clip_without_any_unflagged_partner_is_skipped_and_reported():
    manifest, flagged, sample = corpus(n_videos=3, per_video=2)
    flagged.add("v01_c1")                                                        # v01 has no unflagged clip left
    pairs, skipped = P.make_pairs(manifest, flagged, sample, n_controls=0)
    assert skipped == ["v01_c0"] and len(pairs) == 2


def test_the_page_carries_no_clip_id_and_no_pair_kind():
    items = [{"id": "P001", "a": "QUJD", "b": "REVG", "da": 5.0, "db": 8.0}]
    html = P.page(items)
    data = json.loads(html.split("const ITEMS = ", 1)[1].split(";\n", 1)[0])
    assert data == items and set(data[0]) == {"id", "a", "b", "da", "db"}
    assert "flagged" not in html.lower() and "control" not in html.lower() and "v00_" not in html


def test_window_plays_the_whole_clip_up_to_8_s_else_the_middle():
    assert P.window(5.0) == (0.0, 5.0)
    assert P.window(12.0) == (2.0, 10.0)
