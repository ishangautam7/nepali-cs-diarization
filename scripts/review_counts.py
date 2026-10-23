#!/usr/bin/env python3
"""EXPLORATORY, post hoc (external review, issue 3): exact sample counts behind the H2-P0 cells.

Counts only; computes no outcome. For the 17 confirmatory channels (train + test), per split and in total, and
for the primary cell p0L1 (switch points eligible for p0L1, controls eligible for det_w1) and the pooled cell
det (det / det), with the frozen P0 rules (csdiar.analysis.pilot.score_points and h2_cell, reused unchanged):

  switch points eligible       kind ssdl and elig_<switch family>
  switch points scored         of those, with a defined d(t) at the point; switch points not scored have none
                               (within 1 s of the clip start, where P0 has no left window, or past the last window)
  without cell controls        scored, but no scored control of the control family in the same channel and
                               cell (video x pause bin x position bin); these leave the AUC and Delta FAR
  with cell controls           scored minus without: the switch points the cell's statistics use
  controls scored              kind ctrl, eligible for the control family, with a defined d(t)
  controls in used cells       scored controls whose cell holds at least one switch point with cell controls

Scoring depends only on where d(t) is defined, which is the same for E1-E3; the script checks that and reports
one set. Cross-checks against the frozen confirm_m2a.json: p0L1 scored and without-cell-controls counts
(4,069 and 107, so 3,962 with cell controls) and the pooled det cell's strata sum (120,630).

Usage (repo root):
  python -m scripts.review_counts --boundaries ~/data/d1/boundaries/lex_v1b --emb ~/data/d1/emb \\
      --confirm results/confirm/confirm_m2a.json --out results/review_followups/counts.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from csdiar.analysis.confirm import load_points
from csdiar.analysis.pilot import MODELS, P0_CELLS, p0_curves, score_points

SPLITS = ("train", "test")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def cell_counts(points: dict, vals: dict, s_fam: str, c_fam: str) -> dict:
    cell = lambda r: (r["speaker_id"], r["video_id"], r["pause_bin"], r["pos_bin"])
    n = {s: defaultdict(int) for s in SPLITS}
    ctrl_cells = defaultdict(int)
    for p, r in points.items():
        if r["kind"] == "ctrl" and r[f"elig_{c_fam}"]:
            n[r["split"]]["controls_eligible"] += 1
            if p in vals:
                n[r["split"]]["controls_scored"] += 1
                ctrl_cells[cell(r)] += 1
    used = set()
    for p, r in points.items():
        if r["kind"] != "ssdl" or not r[f"elig_{s_fam}"]:
            continue
        n[r["split"]]["switch_points_eligible"] += 1
        if p not in vals:
            n[r["split"]]["switch_points_not_scored"] += 1
            continue
        n[r["split"]]["switch_points_scored"] += 1
        if ctrl_cells.get(cell(r)):
            n[r["split"]]["switch_points_with_cell_controls"] += 1
            used.add(cell(r))
        else:
            n[r["split"]]["switch_points_without_cell_controls"] += 1
    split_of = {cell(r): r["split"] for r in points.values()}
    for c in used:
        n[split_of[c]]["controls_in_used_cells"] += ctrl_cells[c]
        n[split_of[c]]["cells_used"] += 1
    out = {s: dict(sorted(v.items())) for s, v in n.items()}
    out["total"] = {k: sum(out[s].get(k, 0) for s in SPLITS) for k in sorted({k for v in n.values() for k in v})}
    for s in SPLITS:
        out[s]["channels"] = len({r["speaker_id"] for r in points.values() if r["split"] == s})
    out["total"]["channels"] = len({r["speaker_id"] for r in points.values()})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--boundaries", type=Path, required=True)
    ap.add_argument("--emb", type=Path, required=True)
    ap.add_argument("--confirm", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pts, _ = load_points(args.boundaries, list(SPLITS))
    per_model = {}
    for m in MODELS:
        vals, _ = score_points(pts, p0_curves(args.emb, m, SPLITS))
        per_model[m] = {name: cell_counts(pts, vals, *P0_CELLS[name]) for name in ("p0L1", "det")}
    same = all(per_model[m] == per_model[MODELS[0]] for m in MODELS)
    if not same:
        print("note: counts differ between embeddings; all three are written")
    counts = per_model[MODELS[0]]

    conf = json.loads(args.confirm.read_text())["h2_p0"]
    tot_p, tot_d = counts["p0L1"]["total"], counts["det"]["total"]
    checks = {}
    for m in MODELS:
        fn = conf["primary"][m]["n"]
        strata_sum = sum(s["n"] for s in conf["secondary"][m]["det"]["strata"].values())
        checks[m] = {
            "p0L1_scored": [tot_p["switch_points_scored"], fn["ssdl_scored"]],
            "p0L1_without_cell_controls": [tot_p["switch_points_without_cell_controls"],
                                           fn["ssdl_without_cell_controls"]],
            "p0L1_with_cell_controls": [tot_p["switch_points_with_cell_controls"], 3962],
            "det_with_cell_controls_vs_strata_sum": [tot_d["switch_points_with_cell_controls"], strata_sum, 120630]}
    ok = all(len(set(v)) == 1 for c in checks.values() for v in c.values())

    out = {"label": "EXPLORATORY (post hoc, external review issue 3); counts only, no outcome computed",
           "splits": list(SPLITS), "cells": {k: {"switch_family": P0_CELLS[k][0], "control_family": P0_CELLS[k][1]}
                                             for k in ("p0L1", "det")},
           "inputs": {"boundaries": str(args.boundaries), "emb": str(args.emb),
                      "sha256": {str(args.boundaries / f"points_{s}.csv.gz"): sha256(args.boundaries / f"points_{s}.csv.gz")
                                 for s in SPLITS},
                      "note": "point tables rebuilt after the first box loss (DEVIATIONS.md entry 6); the cross-checks "
                              "below compare them with the frozen M2a counts"},
           "same_for_all_embeddings": same, "counts": counts if same else per_model,
           "cross_checks": {"all_match": ok, "values_[this, frozen(, stated)]": checks}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    for name in ("p0L1", "det"):
        print(name, json.dumps(counts[name]["total"]))
    print("cross-checks all match:", ok)
    print("wrote", args.out)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
