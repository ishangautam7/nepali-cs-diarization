#!/usr/bin/env python3
"""Sensitivity analysis 4 of PREREGISTRATION.md §5: the M2a primary analysis on the purity-verified subset.

The subset is the frozen audit's (csdiar/analysis/purity_audit.py; results/purity/subset.json): flagged clips are
dropped, and each split-off video becomes a cluster of its own (speaker_id "<channel>::<video>"). Everything else
is the frozen M2a analysis, called unchanged from csdiar/analysis/confirm.py on the confirmatory splits (train and
test) with the frozen tau:
- H1 and H2-P0 get the filtered points in memory (matched sets are kept only when the switch point and the control
  both survive);
- H3 reads a filtered copy of the frozen clean windows written to --scratch.
Clusters are therefore the subset's speakers, not the 17 channels, so the sign-flip is Monte Carlo when 2^k
exceeds 2^17. Descriptive (a sensitivity analysis); it changes no primary verdict.
Usage (on the box):
  .venv-pyannote4/bin/python -m csdiar.analysis.purity_sensitivity --scratch /mnt/nvme/tmp/purity_windows \
      --out ~/data/d1/results/purity
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

from csdiar.analysis.confirm import compact, frozen_taus, load_points, run_h1, run_h2, run_h3

SPLITS = ["train", "test"]
SEED = 20261018          # the M2a freeze seed (planning/FREEZE_M2A.json)


def apply_subset(points: dict, matches: list[dict], subset: dict) -> tuple[dict, list[dict], dict]:
    dropped, remap = set(subset["dropped_clips"]), subset["speaker_remap"]
    kept = {}
    for p, r in points.items():
        if r["clip_id"] in dropped:
            continue
        kept[p] = dict(r, speaker_id=remap.get(r["video_id"], r["speaker_id"]))
    m = [x for x in matches if x["ssdl_id"] in kept and x["ctrl_id"] in kept]
    info = {"points_before": len(points), "points_after": len(kept), "matches_before": len(matches),
            "matches_after": len(m), "clusters": len({r["speaker_id"] for r in kept.values()})}
    return kept, m, info


def write_windows(src: Path, dst: Path, subset: dict) -> dict:
    dropped, remap = set(subset["dropped_clips"]), subset["speaker_remap"]
    dst.mkdir(parents=True, exist_ok=True)
    counts = {}
    for s in SPLITS:
        with gzip.open(src / f"windows_{s}.csv.gz", "rt", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        keep = [dict(r, speaker_id=remap.get(r["video_id"], r["speaker_id"])) for r in rows if r["clip_id"] not in dropped]
        with gzip.open(dst / f"windows_{s}.csv.gz", "wt", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(keep)
        counts[s] = {"before": len(rows), "after": len(keep)}
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    home = Path.home() / "data/d1"
    ap.add_argument("--boundaries", type=Path, default=home / "boundaries/lex_v1b")
    ap.add_argument("--point-emb", type=Path, default=home / "point_emb/lex_v1b")
    ap.add_argument("--emb", type=Path, default=home / "emb")
    ap.add_argument("--windows", type=Path, default=home / "windows/lex_v1")
    ap.add_argument("--pilot", type=Path, default=home / "results/pilot/pilot_v2.json")
    ap.add_argument("--subset", type=Path, default=home / "results/purity/subset.json")
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=home / "results/purity")
    args = ap.parse_args()

    out_path = args.out / "sensitivity_m2a_purity.json"
    if out_path.exists():
        raise SystemExit("refusing: the purity sensitivity has already run (one-shot)")
    subset = json.loads(args.subset.read_text())
    taus = frozen_taus(args.pilot)
    rng = np.random.default_rng(SEED)
    points, matches = load_points(args.boundaries, SPLITS)
    pts, m, info = apply_subset(points, matches, subset)
    info["windows"] = write_windows(args.windows, args.scratch, subset)
    chans = sorted({r["speaker_id"] for r in pts.values()})
    block = {"h1": run_h1(pts, m, args.point_emb, SPLITS, rng, chans),
             "h2_p0": run_h2(pts, args.emb, SPLITS, taus, rng, chans),
             "h3": run_h3(args.scratch, args.emb, SPLITS, rng, chans)}
    res = {"analysis": "PREREGISTRATION.md §5 item 4: M2a primary analysis on the purity-verified subset",
           "splits": SPLITS, "tau": taus, "subset": info, "compact": compact(block), **block}
    args.out.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"subset": info, "compact": res["compact"]}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
