"""Tests for the speaker-purity audit (csdiar/analysis/purity_audit.py) and its clip embedder helpers."""
from __future__ import annotations

import numpy as np

from csdiar.analysis import purity_audit as pa
from csdiar.data import audit_embed as ae


def _rows_and_embs(seed: int = 0):
    """Two channels: ch1 has videos v1, v2 of one speaker (v1 holds one planted odd clip);
    ch2 has videos v3 and v4 of two different speakers."""
    rng = np.random.default_rng(seed)
    dim = 128
    spk = {k: pa.unit(rng.standard_normal(dim)) for k in ("A", "B", "C", "odd")}
    rows, emb = [], {}
    # 30 clips per video and 128 dimensions keep the per-cell spread of mean cosines tight and close to normal,
    # so only the planted anomalies cross the frozen 3.5 robust-SD rule (12 clips in 32-d gave chance flags).
    plan = [("ch1", "v1", "A", 30), ("ch1", "v2", "A", 30), ("ch2", "v3", "B", 30), ("ch2", "v4", "C", 30)]
    for ch, vid, s, n in plan:
        for i in range(n):
            cid = f"{vid}_{i:02d}"
            rows.append({"clip_id": cid, "split": "train", "speaker_id": ch, "video_id": vid,
                         "speech_type": "cs", "cmi": "30.0", "provenance_ok": "1"})
            direction = spk["odd"] if (vid == "v1" and i == 0) else spk[s]
            emb[cid] = pa.unit(direction + 0.075 * rng.standard_normal(dim))
    return rows, emb


def test_planted_odd_clip_is_flagged_and_dropped():
    rows, emb = _rows_and_embs()
    res = pa.run(rows, {"titanet": emb, "campplus": emb})
    assert [f["clip_id"] for f in res["flagged"]] == ["v1_00"]
    assert res["subset"]["dropped_clips"] == ["v1_00"]


def test_two_speaker_channel_videos_split_off_but_single_speaker_channel_kept():
    rows, emb = _rows_and_embs()
    res = pa.run(rows, {"titanet": emb, "campplus": emb})
    assert sorted(v["video_id"] for v in res["split_off_videos"]) == ["v3", "v4"]
    assert res["subset"]["speaker_remap"] == {"v3": "ch2::v3", "v4": "ch2::v4"}
    assert res["subset"]["n_speakers_after"] == 3


def test_flag_requires_both_embeddings():
    rows, emb = _rows_and_embs()
    clean = {k: v.copy() for k, v in emb.items()}
    rng = np.random.default_rng(1)
    clean["v1_00"] = pa.unit(emb["v1_01"] + 0.01 * rng.standard_normal(emb["v1_01"].shape))
    res = pa.run(rows, {"titanet": emb, "campplus": clean})
    assert res["n_flagged"] == 0
    assert res["flagged_single_model"]["titanet"] == 1


def test_cmi_tertiles_cut_within_channel():
    assert pa.cmi_tertiles([1, 2, 3, 4, 5, 6, 7, 8, 9]) == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_zero_mad_cell_flags_nothing():
    rows = [{"clip_id": f"c{i}", "split": "train", "speaker_id": "ch", "video_id": "v", "speech_type": "en",
             "cmi": "0.0", "provenance_ok": "1"} for i in range(8)]
    e = pa.unit(np.ones(8))
    out = pa.flag_cells(rows, {r["clip_id"]: e for r in rows}, pa.strata(rows))
    assert all(c["assessable"] and not c["flag"] for c in out.values())


def test_small_cell_is_not_assessable():
    rows = [{"clip_id": f"c{i}", "split": "train", "speaker_id": "ch", "video_id": "v", "speech_type": "en",
             "cmi": "0.0", "provenance_ok": "1"} for i in range(5)]   # 4 others < MIN_OTHERS
    rng = np.random.default_rng(0)
    out = pa.flag_cells(rows, {r["clip_id"]: pa.unit(rng.standard_normal(4)) for r in rows}, pa.strata(rows))
    assert not any(c["assessable"] for c in out.values())


def test_speech_runs_bridge_short_gaps_and_drop_short_runs():
    f = ae.VAD_FRAME_S
    vad = np.zeros(200, dtype=np.float32)
    vad[10:30] = 0.9            # 0.64 s run
    vad[33:60] = 0.9            # gap of 3 frames (0.096 s) -> bridged with the run above
    vad[100:110] = 0.9          # 0.32 s, isolated -> dropped (< MIN_RUN_S)
    runs = ae.speech_runs(vad, n_samples=int(200 * f * ae.SR))
    assert len(runs) == 1
    a, b = runs[0]
    assert abs(a / ae.SR - 10 * f) < 1e-3 and abs(b / ae.SR - 60 * f) < 1e-3


def test_pool_is_duration_weighted_unit_mean():
    v = np.array([[2.0, 0.0], [0.0, 3.0]])
    out = ae.pool(v, np.array([3.0, 1.0]))
    expect = np.array([0.75, 0.25]) / np.linalg.norm([0.75, 0.25])
    assert np.allclose(out, expect, atol=1e-6) and abs(np.linalg.norm(out) - 1) < 1e-6
