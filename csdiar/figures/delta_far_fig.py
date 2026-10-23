#!/usr/bin/env python3
"""Per-channel switch-point excess of the P0 change detector on D1 (figure delta_far_channels.pdf).

Replaces det_ssdl_vs_dsdl.pdf in the paper. That figure (det_fig.py) plots RAW fire rates pooled over all
switch points and all controls of all 20 channels, pilot included; it is point-weighted and not
cell-standardised, so it does not show the estimand of Table II and its operating points visibly disagree
with it. This figure is drawn only from the frozen confirmatory result file, so every number in it is the one
the paper's tables report:

  * one dot per confirmatory channel (17) and embedding: the cell-standardised Delta FAR at the frozen tau on
    the primary p0L1 cell (switch-point fire rate minus the fire rate of its video x pause x position cell's
    controls), dot area growing with the channel's number of p0L1 switch points;
  * a bar at the unweighted mean over channels -- the pre-registered statistic in Table II;
  * a diamond at the point-weighted mean, to show how much the mean depends on channel weighting.

Usage (any machine; needs only matplotlib and the committed results):
  python -m csdiar.figures.delta_far_fig --confirm results/confirm/confirm_m2a.json \\
      --out paper/figures/delta_far_channels.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = ("e1", "e2", "e3")


def channel_rows(primary: dict, model: str) -> list[tuple[int, float]]:
    """(n_ssdl, delta_far_std) per confirmatory channel, in percentage points."""
    return [(c["n_ssdl"], 100 * c["delta_far_std"]) for c in primary[model]["channels"].values()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--confirm", type=Path, required=True, help="results/confirm/confirm_m2a.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = json.loads(args.confirm.read_text())["h2_p0"]["primary"]
    colors = {"e1": "#1f77b4", "e2": "#d62728", "e3": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(3.6, 2.9), constrained_layout=True)
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    summary = {}
    for i, m in enumerate(MODELS):
        rows = channel_rows(primary, m)
        n_tot = sum(n for n, _ in rows)
        unweighted = sum(d for _, d in rows) / len(rows)
        weighted = sum(n * d for n, d in rows) / n_tot
        # the plotted mean must be the reported statistic
        assert abs(unweighted - 100 * primary[m]["delta_far_std"]) < 1e-9, m
        summary[m] = (unweighted, weighted)
        # deterministic horizontal jitter by rank, so overlapping channels stay visible
        for j, (n, d) in enumerate(sorted(rows, key=lambda r: r[1])):
            x = i + (j - (len(rows) - 1) / 2) * 0.022
            ax.scatter(x, d, s=8 + n / 12, color=colors[m], alpha=0.45, edgecolor="none")
        ax.plot([i - 0.28, i + 0.28], [unweighted] * 2, color=colors[m], lw=2.2)
        ax.scatter([i + 0.36], [weighted], marker="D", s=22, color="white", edgecolor=colors[m], zorder=5)
    ax.plot([], [], color="0.3", lw=2.2, label="mean over channels (Table II)")
    ax.scatter([], [], marker="D", s=22, color="white", edgecolor="0.3", label="mean over points")
    ax.set_xticks(range(len(MODELS)), [m.upper() for m in MODELS], fontsize=8)
    ax.set_ylabel(r"$\Delta$FAR at $\tau$ (pp), p0L1 cell", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"), dpi=150)
    print("wrote", args.out, {m: tuple(round(v, 2) for v in s) for m, s in summary.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
