#!/usr/bin/env python3
"""M1 SubspaceProjector: a training-free language-nuisance projection (PLAN.md §5, M1).

The language direction v_lang is the average of within-video differences between duration-matched `ne` and `en`
windows of the SAME speaker (so the difference isolates language, not speaker or channel). An embedding e is
mapped to e' = e - (e . v̂) v̂ and re-normalised, removing its component along the language axis. v_lang is
re-estimated inside every cross-validation fold (never using the fold it is applied to), so it never sees the
test channel's own language direction.

This module is feature-level only: it estimates v_lang from the frozen dense/point embeddings and returns a
projector. It performs NO outcome statistics itself; csdiar.mitigations.eval_h4 applies it and measures the
SS-DL excess (H4) against baselines B0/B1/B2. Real speech only; no TTS.

Secondary (PLAN.md §5): a k-dimensional subspace from the within-speaker cross-language difference vectors,
k in {1,2,4,8,16}, chosen inside the folds; plus a random-direction and top-PCA-direction specificity control.
"""
from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np

SR_SPLITS = ("train", "validation", "test")


def unit(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def read_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def language_windows(points: dict, pe_dir: Path, model: str, L: float, splits) -> dict:
    """Per (speaker, video), collect unit windows tagged `ne` and `en` from switch-point left/right windows.

    A switch point has lang_left != lang_right; its left window carries lang_left, its right lang_right. We use
    only switch points (kind == 'ssdl') so each pair is the same speaker across a single language change, which
    is exactly the within-speaker cross-language contrast v_lang should capture.
    """
    by = defaultdict(lambda: {"ne": [], "en": []})
    for split in splits:
        z = np.load(pe_dir / model / f"{split}_L{L:g}.npz")
        pid_index = {p: i for i, p in enumerate(z["point_id"])}
        for pid, r in points.items():
            if r["kind"] != "ssdl" or pid not in pid_index:
                continue
            i = pid_index[pid]
            key = (r["speaker_id"], r["video_id"])
            for side, lang in (("left", r["lang_left"]), ("right", r["lang_right"])):
                if lang in ("ne", "en"):
                    by[key][lang].append(unit(z[side][i]))
    return by


def estimate_v_lang(points: dict, pe_dir: Path, model: str, L: float, splits, speakers: set | None = None):
    """v_lang = mean over (speaker, video) of [mean(en windows) - mean(ne windows)], over the given speakers.

    Restricting `speakers` to a CV fold's TRAIN channels is how the projector avoids seeing the test channel.
    Returns a unit vector, or None if no (speaker, video) had both languages.
    """
    by = language_windows(points, pe_dir, model, L, splits)
    diffs = []
    for (spk, _vid), d in by.items():
        if speakers is not None and spk not in speakers:
            continue
        if d["ne"] and d["en"]:
            diffs.append(np.mean(d["en"], 0) - np.mean(d["ne"], 0))
    if not diffs:
        return None
    v = np.mean(diffs, 0)
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 1e-9 else None


def project_out(e: np.ndarray, v_hat: np.ndarray) -> np.ndarray:
    """e' = e - (e . v̂) v̂, then re-normalised (PLAN.md §5)."""
    e = e.astype(np.float32)
    comp = (e @ v_hat)[..., None] * v_hat[None, :]
    return unit(e - comp)


def language_mean_centre(e: np.ndarray, lang: str, mu: dict) -> np.ndarray:
    """Baseline B1: subtract the per-language mean embedding (estimated on train), then re-normalise."""
    return unit(e.astype(np.float32) - mu[lang]) if lang in mu else unit(e)
