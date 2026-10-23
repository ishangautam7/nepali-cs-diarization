#!/usr/bin/env python3
"""UMAP of D1 switch-point window embeddings, coloured by speaker and by language (PLAN.md §6; figure
umap_speaker_language_invariance.pdf).

Each SS-DL switch point contributes its two point-anchored L = 1 s windows: the LEFT window (the language before
the switch) and the RIGHT window (the language after). We embed both windows for one model, project all of them
to 2-D with UMAP (McInnes et al. 2018), and draw two panels sharing the layout:
  (left)  coloured by speaker (channel) -- same-speaker points should cluster regardless of language;
  (right) coloured by the window's language side (pre/post switch) -- if the embedding were language-invariant
          the two colours would be interleaved, not separated.
This is a descriptive illustration of H1/H3 (it is not a test). Fit is on a reproducible subsample (SEED).

Usage (box, env-pyannote4 after `source ops/env.sh`):
  .venv-pyannote4/bin/python -m csdiar.figures.umap_fig --boundaries ~/data/d1/boundaries/lex_v1b \\
      --point-emb ~/data/d1/point_emb/lex_v1b --model e3 --out paper/figures/umap_speaker_language_invariance.pdf
"""
from __future__ import annotations

import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 20261018
SPLITS = ("train", "validation", "test")
MAX_POINTS = 4000  # switch points sampled for a legible, reproducible projection


def read_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def unit(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def load(boundaries: Path, pe_dir: Path, model: str):
    """Return (X: 2N x dim unit windows, speaker: 2N, side: 2N in {pre,post})."""
    pts = {}
    for s in SPLITS:
        for r in read_gz(boundaries / f"points_{s}.csv.gz"):
            pts[r["point_id"]] = r
    X, spk, side = [], [], []
    rng = np.random.default_rng(SEED)
    for s in SPLITS:
        z = np.load(pe_dir / model / f"{s}_L1.npz")
        ids = [p for p in z["point_id"] if pts.get(p, {}).get("kind") == "ssdl"]
        for pid, L, R in zip(z["point_id"], unit(z["left"]), unit(z["right"])):
            if pts.get(pid, {}).get("kind") != "ssdl":
                continue
            r = pts[pid]
            # left window = language BEFORE the switch, right = AFTER; lang_left/lang_right are tags
            X.append(L); spk.append(r["speaker_id"]); side.append(r.get("lang_left", "pre"))
            X.append(R); spk.append(r["speaker_id"]); side.append(r.get("lang_right", "post"))
    X, spk, side = np.array(X), np.array(spk), np.array(side)
    if len(X) > 2 * MAX_POINTS:  # sample whole switch points (keep left/right pairs together)
        n_pts = len(X) // 2
        keep = rng.choice(n_pts, MAX_POINTS, replace=False)
        idx = np.sort(np.concatenate([keep * 2, keep * 2 + 1]))
        X, spk, side = X[idx], spk[idx], side[idx]
    return X, spk, side


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--boundaries", type=Path, required=True)
    ap.add_argument("--point-emb", type=Path, required=True)
    ap.add_argument("--model", default="e3")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    X, spk, side = load(args.boundaries, args.point_emb, args.model)
    emb = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=SEED).fit_transform(X)

    fig, (axs, axl) = plt.subplots(1, 2, figsize=(7.0, 3.3), constrained_layout=True)
    # panel 1: colour by speaker (channel)
    speakers = sorted(set(spk))
    cmap = plt.get_cmap("tab20", max(len(speakers), 1))
    for i, s in enumerate(speakers):
        m = spk == s
        axs.scatter(emb[m, 0], emb[m, 1], s=3, color=cmap(i), linewidths=0, alpha=0.6)
    axs.set_title(f"By speaker ({len(speakers)} channels)", fontsize=9)
    # panel 2: colour by language side
    sides = sorted(set(side))
    scol = {s: c for s, c in zip(sides, ("#1f77b4", "#d62728", "#2ca02c", "#9467bd"))}
    for s in sides:
        m = side == s
        axl.scatter(emb[m, 0], emb[m, 1], s=3, color=scol.get(s, "#777"), linewidths=0, alpha=0.6, label=str(s))
    axl.set_title("By language side of the switch", fontsize=9)
    axl.legend(markerscale=3, fontsize=7, loc="best", framealpha=0.8)
    for ax in (axs, axl):
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP-1", fontsize=8); ax.set_ylabel("UMAP-2", fontsize=8)
    fig.suptitle(f"D1 switch-point windows, {args.model.upper()} (n={len(X)} windows)", fontsize=10)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"), dpi=150)
    print(f"wrote {args.out} ({len(X)} windows, {len(speakers)} speakers, sides={sides})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
