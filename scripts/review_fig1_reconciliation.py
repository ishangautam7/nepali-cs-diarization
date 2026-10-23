#!/usr/bin/env python3
"""EXPLORATORY, post hoc (external review, issue 1): why the old Fig. 1 operating points disagree with Table II.

The old figure (csdiar/figures/det_fig.py) marked, at the frozen tau, the raw fire rate of p0L1 switch points
against the raw fire rate of det_w1 controls, pooled over every point of all 20 D1 channels. Table II's Delta FAR
(confirm_m2a.json h2_p0.primary.<m>.delta_far_std) is the equal-weight mean over the 17 confirmatory channels of
per-channel, cell-standardised values. This script walks from one to the other:

  (a)  pooled raw switch-point fire rate minus pooled raw control fire rate, all 20 channels:
         a_figure  exactly as det_fig.py drew it (points with no NMS peak within +-c leave both denominators);
         a_all     those points kept as non-fires, as the frozen h2_cell counts them;
  (b)  the same two variants on the 17 confirmatory channels (train + test);
  (b2) as b_all, switch points restricted to those with at least one control in their cell;
  (c)  cell-standardised (video x pause bin x position bin, within channel), pooled over points: the mean over
       switch points of [hit - fire rate of the controls in that point's cell];
  (d)  per-channel cell-standardised Delta FAR, equal-weight mean over the included channels (>= MIN_UNITS),
       which must reproduce the reference's delta_far_std to within 1e-6.

The gate runs first. If (d) does not reproduce for every requested model, the script prints the comparison and
exits with status 2, computing and writing nothing else. The reference is Table II's frozen value
(confirm_m2a.json). The frozen E1/E2 inputs were lost with the first GPU box (DEVIATIONS.md entry 6; only E3's
grid embeddings were rebuilt byte-identical), so --rebuilt-models e1 e2 --rebuilt-reference <re-run on the rebuilt
inputs> gates those models against that re-run instead, and the output labels them as rebuilt, with their
difference from Table II. The script only reads: it reuses the frozen P0 code (csdiar.analysis.pilot,
csdiar.analysis.confirm) unchanged.

Usage (repo root):
  python -m scripts.review_fig1_reconciliation --boundaries ~/data/d1/boundaries/lex_v1b --emb ~/data/d1/emb \\
      --pilot results/pilot/pilot_v2.json --confirm results/confirm/confirm_m2a.json \\
      --rebuilt-models e1 e2 --rebuilt-reference results/sensitivity/base/confirm_sens_base.json \\
      --out results/review_followups/fig1_reconciliation.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from csdiar.analysis.confirm import MIN_UNITS, frozen_taus, load_points
from csdiar.analysis.pilot import MODELS, h2, p0_curves, score_points

SPLITS_ALL = ("train", "validation", "test")
SPLITS_CONF = ("train", "test")
TOL = 1e-6


def cell(r: dict) -> tuple:
    return (r["video_id"], r["pause_bin"], r["pos_bin"])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def gate_value(points_conf: dict, curves: dict, tau: float) -> tuple[float, dict]:
    """Step (d) exactly as confirm.run_h2 computes delta_far_std, plus the per-channel values."""
    prim = h2(points_conf, curves, tau=tau)["cells"]["p0L1"]
    chans = {c: v for c, v in prim["channels"].items() if v["auc"] == v["auc"] and v["n_ssdl"] >= MIN_UNITS}
    dfar = [v["delta_far_std"] for v in chans.values() if v["delta_far_std"] is not None]
    return float(np.mean(dfar)), prim["channels"]


def pooled_raw(points: dict, vals: dict, tau: float, keep) -> dict:
    """Pooled raw fire rates of p0L1 switch points and det_w1 controls among the points keep() accepts."""
    s_all, c_all, s_fig, c_fig = [], [], [], []
    for p, v in vals.items():
        r = points[p]
        if not keep(r):
            continue
        if r["kind"] == "ssdl" and r["elig_p0L1"]:
            s_all.append(v["peak"])
        elif r["kind"] == "ctrl" and r["elig_det_w1"]:
            c_all.append(v["peak"])
    s_all, c_all = np.array(s_all, np.float32), np.array(c_all, np.float32)
    s_fig, c_fig = s_all[np.isfinite(s_all)], c_all[np.isfinite(c_all)]
    rate = lambda a: float((a > tau).mean())  # float32 peaks against tau, as det_fig.py compares them
    return {"figure": {"ssdl_rate": rate(s_fig), "ctrl_rate": rate(c_fig), "delta": rate(s_fig) - rate(c_fig),
                       "n_ssdl": int(len(s_fig)), "n_ctrl": int(len(c_fig))},
            "all": {"ssdl_rate": rate(s_all), "ctrl_rate": rate(c_all), "delta": rate(s_all) - rate(c_all),
                    "n_ssdl": int(len(s_all)), "n_ctrl": int(len(c_all))}}


def cell_steps(points: dict, vals: dict, tau: float) -> dict:
    """(b2) and (c) on the given (confirmatory) points, with h2_cell's cell and hit rules."""
    ctrl_hit = defaultdict(lambda: defaultdict(list))
    for p, v in vals.items():
        r = points[p]
        if r["kind"] == "ctrl" and r["elig_det_w1"]:
            ctrl_hit[r["speaker_id"]][cell(r)].append(v["peak"] > tau)
    hits, diffs, used_ctrl_cells = [], [], set()
    for p, v in vals.items():
        r = points[p]
        if r["kind"] != "ssdl" or not r["elig_p0L1"]:
            continue
        ch = ctrl_hit[r["speaker_id"]].get(cell(r))
        if not ch:
            continue
        h = float(v["peak"] > tau)
        hits.append(h)
        diffs.append(h - float(np.mean(ch)))
        used_ctrl_cells.add((r["speaker_id"], cell(r)))
    used = [x for k in used_ctrl_cells for x in ctrl_hit[k[0]][k[1]]]
    all_ctrl = [x for cells in ctrl_hit.values() for xs in cells.values() for x in xs]
    return {"b2": {"ssdl_rate": float(np.mean(hits)), "ctrl_rate": float(np.mean(all_ctrl)),
                   "delta": float(np.mean(hits) - np.mean(all_ctrl)), "n_ssdl": len(hits), "n_ctrl": len(all_ctrl)},
            "c": {"delta": float(np.mean(diffs)), "n_ssdl": len(diffs),
                  "ctrl_rate_raw_in_used_cells": float(np.mean(used)), "n_ctrl_in_used_cells": len(used)}}


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--boundaries", type=Path, required=True)
    ap.add_argument("--emb", type=Path, required=True)
    ap.add_argument("--pilot", type=Path, required=True, help="pilot JSON with the frozen tau")
    ap.add_argument("--confirm", type=Path, required=True, help="frozen confirm_m2a.json (gate reference)")
    ap.add_argument("--also", type=Path, nargs="*", default=[], help="other result files, compared and printed only")
    ap.add_argument("--rebuilt-reference", type=Path, default=None,
                    help="re-run on the rebuilt inputs (results/sensitivity/base/confirm_sens_base.json): the gate "
                         "reference for --rebuilt-models, whose frozen inputs no longer exist")
    ap.add_argument("--rebuilt-models", nargs="*", default=[])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if bool(args.rebuilt_models) != (args.rebuilt_reference is not None):
        raise SystemExit("--rebuilt-reference and --rebuilt-models go together")

    taus = frozen_taus(args.pilot)  # refuses a tau that is not the pre-registered one
    frozen = json.loads(args.confirm.read_text())["h2_p0"]["primary"]
    rebuilt = json.loads(args.rebuilt_reference.read_text())["h2_p0"]["primary"] if args.rebuilt_reference else {}
    also = {str(p): json.loads(p.read_text())["h2_p0"]["primary"] for p in args.also}
    pts_all, _ = load_points(args.boundaries, list(SPLITS_ALL))
    pts_conf = {p: r for p, r in pts_all.items() if r["split"] in SPLITS_CONF}

    curves, gate, failed = {}, {}, []
    for m in args.models:
        curves[m] = p0_curves(args.emb, m, SPLITS_ALL)
        d, chans = gate_value(pts_conf, curves[m], taus[m])
        use_rebuilt = m in args.rebuilt_models
        ref_value = (rebuilt if use_rebuilt else frozen)[m]["delta_far_std"]
        diff = d - ref_value
        gate[m] = {"d": d, "reference": ref_value,
                   "reference_file": str(args.rebuilt_reference if use_rebuilt else args.confirm),
                   "inputs": ("REBUILT after the first box loss (DEVIATIONS.md entry 6); the frozen inputs no longer "
                              "exist, so the gate is the re-run on the rebuilt inputs, not Table II")
                             if use_rebuilt else "frozen (the reference is Table II's value)",
                   "diff": diff, "reproduces": abs(diff) <= TOL,
                   "table_ii_value": frozen[m]["delta_far_std"], "diff_from_table_ii": d - frozen[m]["delta_far_std"],
                   "also": {k: {"value": v[m]["delta_far_std"], "diff": d - v[m]["delta_far_std"]}
                            for k, v in also.items()},
                   "channels": {c: v["delta_far_std"] for c, v in chans.items()}}
        print(f"gate {m}: (d) {100 * d:+.6f} pp, reference {100 * ref_value:+.6f} pp "
              f"({'rebuilt re-run' if use_rebuilt else 'Table II'}), diff {diff:+.3e} -> "
              f"{'reproduces' if gate[m]['reproduces'] else 'DOES NOT REPRODUCE'}; "
              f"Table II {100 * frozen[m]['delta_far_std']:+.6f} pp")
        if not gate[m]["reproduces"]:
            failed.append(m)
    if failed:
        print(f"STOP: step (d) does not reproduce the reference for {failed}; nothing computed or written.")
        return 2

    out = {"label": "EXPLORATORY (post hoc, external review issue 1); not a pre-registered analysis",
           "gate_rule": "step (d) must reproduce its reference within 1e-6; for " + (", ".join(args.rebuilt_models) or "none")
                        + " the reference is the re-run on rebuilt inputs (decision 2026-10-22: option B)",
           "code_commit": git_head(), "tau": {m: taus[m] for m in args.models},
           "inputs": {"boundaries": str(args.boundaries), "emb": str(args.emb), "pilot": str(args.pilot),
                      "reference": str(args.confirm),
                      "sha256": {str(f): sha256(f) for f in sorted(
                          [args.boundaries / f"points_{s}.csv.gz" for s in SPLITS_ALL]
                          + [args.emb / f"{m}_L1" / f"{s}.npz" for m in args.models for s in SPLITS_ALL])}},
           "units": "fractions; *_pp fields are percentage points", "models": {}}
    for m in args.models:
        vals, _ = score_points(pts_all, curves[m])
        a = pooled_raw(pts_all, vals, taus[m], lambda r: True)
        b = pooled_raw(pts_all, vals, taus[m], lambda r: r["split"] in SPLITS_CONF)
        vals_conf = {p: v for p, v in vals.items() if p in pts_conf}
        bc = cell_steps(pts_conf, vals_conf, taus[m])
        d = gate[m]["d"]
        steps = {"a_figure": a["figure"]["delta"], "a_all": a["all"]["delta"], "b_figure": b["figure"]["delta"],
                 "b_all": b["all"]["delta"], "b2": bc["b2"]["delta"], "c": bc["c"]["delta"], "d": d}
        gaps = {"no_peak_points_kept_as_non_fires (a_all - a_figure)": steps["a_all"] - steps["a_figure"],
                "pilot_channels_dropped (b_all - a_all)": steps["b_all"] - steps["a_all"],
                "switch_points_without_cell_controls_dropped (b2 - b_all)": steps["b2"] - steps["b_all"],
                "cell_standardisation (c - b2)": steps["c"] - steps["b2"],
                "equal_channel_weighting (d - c)": d - steps["c"]}
        assert abs(sum(gaps.values()) - (d - steps["a_figure"])) < 1e-9
        out["models"][m] = {"steps": steps, "steps_pp": {k: 100 * v for k, v in steps.items()},
                            "gaps_pp": {k: 100 * v for k, v in gaps.items()},
                            "detail": {"a": a, "b": b, "b2": bc["b2"], "c": bc["c"]}, "gate": gate[m]}
        print(m, {k: round(100 * v, 3) for k, v in steps.items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
