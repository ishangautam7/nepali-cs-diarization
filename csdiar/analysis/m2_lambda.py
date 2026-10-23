#!/usr/bin/env python3
"""M2 pre-tuning check (PLAN.md §5, M2): does the LID change score lambda(t) track language, or pauses?

Validation channels only. VoxLingua107 posteriors come from csdiar.models.lid: window k covers [0.1k, 0.1k + 1.0] s
with columns [centre, p_ne, p_en, p_other]. At a point t, lambda(t) = Jensen-Shannon divergence (base 2) between the
renormalised (p_ne, p_en) of the left window [t-1, t] and the right window [t, t+1].
Reported (no tuning):
- AUC of lambda at switch points against cell-matched controls, for the detection family, by stratum and by pause bin;
- the same for the embedding family at L = 1.0 s;
- AUC of lambda for phrase-boundary controls (pause > 150 ms) against other controls (does lambda track pauses?).
DS-SL turns do not exist in D1 clips; that part waits for conversation data.
Output: results/pilot/m2_lambda_<tag>.json and .md
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from csdiar.analysis.pilot import stratified_auc

PILOT_SPLIT = "validation"
HOP = 0.1


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(np.where(a > 0, a * np.log2(np.maximum(a, 1e-12) / np.maximum(b, 1e-12)), 0)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def lam(post: np.ndarray, t: float) -> float | None:
    kl, kr = int(round((t - 1.0) / HOP)), int(round(t / HOP))
    if kl < 0 or kr >= len(post):
        return None
    p = post[kl, 1:3].astype(np.float64)
    q = post[kr, 1:3].astype(np.float64)
    return jsd(p / max(p.sum(), 1e-9), q / max(q.sum(), 1e-9))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--boundaries", type=Path, required=True)
    ap.add_argument("--lid", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()
    lid = np.load(args.lid / f"{PILOT_SPLIT}.npz")
    with gzip.open(args.boundaries / f"points_{PILOT_SPLIT}.csv.gz", "rt", encoding="utf-8") as fh:
        pts = list(csv.DictReader(fh))
    vals = []
    for r in pts:
        v = lam(lid[r["clip_id"]], float(r["t"]))
        if v is not None:
            r["lambda"] = v
            vals.append(r)
    out = {"tag": args.tag, "split": PILOT_SPLIT, "n_scored": len(vals), "families": {}}
    for fam in ("det", "emb1"):
        ok = [r for r in vals if r[f"elig_{fam}"] == "True"]
        cell = lambda r: (r["video_id"], r["pause_bin"], r["pos_bin"])
        neg = defaultdict(list)
        for r in ok:
            if r["kind"] == "ctrl":
                neg[cell(r)].append(r["lambda"])
        res = {}
        for name, sel in [("all", lambda r: True)] + \
                [(f"stratum:{s}", (lambda s: lambda r: r["stratum"] == s)(s)) for s in ("lt0.5", "0.5to1", "ge1")] + \
                [(f"pause:{b}", (lambda b: lambda r: r["pause_bin"] == b)(b)) for b in ("none", "lt150", "150to300", "gt300")]:
            pos = defaultdict(list)
            for r in ok:
                if r["kind"] == "ssdl" and sel(r):
                    pos[cell(r)].append(r["lambda"])
            auc, n = stratified_auc(pos, neg)
            res[name] = {"auc": auc, "n_ssdl": n}
        phrase, other = defaultdict(list), defaultdict(list)
        for r in ok:
            if r["kind"] == "ctrl":
                ((phrase if r["phrase"] == "True" else other)[(r["video_id"], r["pos_bin"])]).append(r["lambda"])
        auc_p, n_p = stratified_auc(phrase, other)
        res["phrase_vs_other_controls"] = {"auc": auc_p, "n_phrase": n_p}
        out["families"][fam] = res
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"m2_lambda_{args.tag}.json").write_text(json.dumps(out, indent=2))
    lines = [f"# M2 lambda(t) pre-tuning check {args.tag} (validation only)", "", "| Family | Comparison | AUC | n |",
             "|---|---|---|---|"]
    for fam, res in out["families"].items():
        for k, v in res.items():
            lines.append(f"| {fam} | {k} | {v['auc']:.3f} | {v.get('n_ssdl', v.get('n_phrase'))} |")
    (args.out / f"m2_lambda_{args.tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
