#!/usr/bin/env python3
"""DET-style operating curve for the change detector at SS-DL switch points (PLAN.md §6; figure
det_ssdl_vs_dsdl.pdf).

No longer in the paper (replaced by delta_far_fig.py). Its curves pool RAW fire rates over every switch point and
every det_w1 control of all splits, pilot included: they are point-weighted and not cell-standardised, so its
operating points do not show Table II's Delta FAR (an equal-weight mean of per-channel, cell-standardised values
on the 17 confirmatory channels) and visibly disagree with it. Kept for the DISPLACE hook below.

The paper's headline trade-off is: as the detector's threshold tau sweeps, how many SS-DL switch points does it
wrongly flag as speaker changes (false splits) against how many genuine DIFFERENT-speaker cross-language turns
(DS-DL) it misses? The DS-DL / missed-turn axis needs human-labelled cross-lingual turns, which on this project
come from DISPLACE (DISPLACE-M or DISPLACE 2024); that data is not yet in hand. This script therefore draws what
D1 alone supports now, and accepts a DS-DL scores file to add the real second axis when DISPLACE arrives.

D1 curve (always drawn). Using the frozen P0 peak strengths (csdiar.analysis.pilot: same NMS, same +-c window),
for the primary p0L1 cell:
  x-axis  SS-DL false-split rate  = fraction of SS-DL switch points with peak > tau  (we do NOT want a change here)
  y-axis  control fire rate       = fraction of matched same-speaker same-language controls with peak > tau
The pre-registered operating point tau (10% control fire rate) is marked. A detector that only tracked real
speaker turns would sit on the diagonal; the curve bowing ABOVE it is the excess of false splits at switches.

DISPLACE hook (drawn when --dsdl-scores is given). That file is a .npz with peak strengths at labelled DS-DL
turns ('dsdl') and at non-turn controls ('ctrl'); the second panel then plots the classic DET of
SS-DL false-split rate vs DS-DL miss rate (1 - detection rate) as tau sweeps -- the real diagnostic/mitigation
trade-off. Until then the panel is omitted and the caption says so.

Usage (box, env-pyannote4):
  .venv-pyannote4/bin/python -m csdiar.figures.det_fig --boundaries ~/data/d1/boundaries/lex_v1b \\
      --emb ~/data/d1/emb --pilot ~/data/d1/results/pilot/pilot_v2.json \\
      --out paper/figures/det_ssdl_vs_dsdl.pdf            # [--dsdl-scores displace_peaks.npz]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from csdiar.analysis.pilot import MODELS, p0_curves, peak_strength

SPLITS = ("train", "validation", "test")
HOP = 0.1


def read_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def ssdl_control_peaks(boundaries: Path, emb_dir: Path, model: str):
    """Frozen-cell peak strengths at SS-DL switch points and at their eligible same-speaker controls (p0L1 cell:
    switch points eligible for p0L1, controls eligible for det_w1 -- exactly the primary P0 cell of decision 16)."""
    pts = {}
    for s in SPLITS:
        for r in read_gz(boundaries / f"points_{s}.csv.gz"):
            for k in ("t",):
                r[k] = float(r[k])
            for k in [c for c in r if c.startswith("elig_")]:
                r[k] = r[k] == "True"
            pts[r["point_id"]] = r
    curves = p0_curves(emb_dir, model, SPLITS)
    ssdl, ctrl = [], []
    for pid, r in pts.items():
        d = curves.get(r["clip_id"])
        if d is None:
            continue
        k = int(round(r["t"] / HOP))
        if k >= len(d) or np.isnan(d[k]):
            continue
        peak = peak_strength(d, k)
        if r["kind"] == "ssdl" and r["elig_p0L1"]:
            ssdl.append(peak)
        elif r["kind"] == "ctrl" and r["elig_det_w1"]:
            ctrl.append(peak)
    return np.array(ssdl, np.float32), np.array(ctrl, np.float32)


def rate_above(peaks: np.ndarray, taus: np.ndarray) -> np.ndarray:
    finite = peaks[np.isfinite(peaks)]
    return np.array([(finite > t).mean() if len(finite) else np.nan for t in taus])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--boundaries", type=Path, required=True)
    ap.add_argument("--emb", type=Path, required=True)
    ap.add_argument("--pilot", type=Path, required=True, help="pilot JSON with the frozen tau per model")
    ap.add_argument("--dsdl-scores", type=Path, default=None, help="optional DISPLACE DS-DL peaks .npz")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pilot = json.loads(args.pilot.read_text())
    taus = np.linspace(-0.2, 1.2, 281)
    curves = {m: ssdl_control_peaks(args.boundaries, args.emb, m) for m in MODELS}
    tau_frozen = {m: pilot["h2"][m]["tau"] for m in MODELS}

    have_dsdl = args.dsdl_scores is not None and args.dsdl_scores.exists()
    fig, axes = plt.subplots(1, 2 if have_dsdl else 1, figsize=(7.0 if have_dsdl else 3.6, 3.3),
                             constrained_layout=True, squeeze=False)
    colors = {"e1": "#1f77b4", "e2": "#d62728", "e3": "#2ca02c"}

    ax = axes[0][0]
    for m in MODELS:
        ssdl, ctrl = curves[m]
        x, y = rate_above(ssdl, taus), rate_above(ctrl, taus)
        ax.plot(y, x, color=colors[m], label=f"{m.upper()}")
        # mark the frozen operating point tau (10% control fire rate)
        tf = tau_frozen[m]
        ax.scatter([(ctrl[np.isfinite(ctrl)] > tf).mean()], [(ssdl[np.isfinite(ssdl)] > tf).mean()],
                   color=colors[m], s=28, zorder=5, edgecolor="k", linewidths=0.5)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.6, label="no excess")
    ax.set_xlabel("control fire rate (same-speaker, same-language)", fontsize=8)
    ax.set_ylabel("SS-DL switch-point fire rate", fontsize=8)
    ax.set_title("D1: change-score fire rate at switches vs. controls", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.tick_params(labelsize=7)

    if have_dsdl:
        z = np.load(args.dsdl_scores)
        ax2 = axes[0][1]
        for m in MODELS:
            ssdl, _ = curves[m]
            miss = 1 - rate_above(z[f"dsdl_{m}"] if f"dsdl_{m}" in z else z["dsdl"], taus)
            fs = rate_above(ssdl, taus)
            ax2.plot(miss, fs, color=colors[m], label=m.upper())
        ax2.set_xlabel("DS-DL miss rate (DISPLACE)", fontsize=8)
        ax2.set_ylabel("SS-DL switch-point fire rate", fontsize=8)
        ax2.set_title("Diagnostic trade-off", fontsize=9)
        ax2.legend(fontsize=7)
        ax2.tick_params(labelsize=7)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"), dpi=150)
    print(f"wrote {args.out}; dsdl_panel={have_dsdl}; "
          f"n_ssdl={ {m: int(len(curves[m][0])) for m in MODELS} }, "
          f"n_ctrl={ {m: int(len(curves[m][1])) for m in MODELS} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
