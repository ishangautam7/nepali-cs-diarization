#!/usr/bin/env python3
"""Report the D4 residual-contamination audit (PREREGISTRATION_H7.md §3), descriptive.

Inputs: the sampler's folder (scripts/d4_contamination_sample.py: <pair>_items.csv, summary.json, instructions.md), the
LLM labeller's <pair>_labels.csv (item,label,gloss,note), and the analyst's notes (JSON: for each pair, the analyst's
label for every item marked for the check, and any item outside the check the analyst would label differently).
Outputs in --out: <pair>_labels.csv (item, split, word, label, gloss, note, analyst_check, analyst_label; transcript
contexts are left out), summary.json and instructions.md copied, and contamination_audit.json with, per pair, the
labels, the share labelled English (LOAN) with a Wilson 95% interval, the analyst's agreement on the checked items,
and, separately and marked as not pre-registered, the share if the analyst's notes outside the check were counted.

Usage: python3 scripts/d4_contamination_report.py --sample <dir> --labels <dir> --analyst <json> --out results/h7/contamination
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

PAIRS = ("hi-en", "bn-en")
LABELS = ("LOAN", "NATIVE", "NAME", "UNSURE")


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def report(items: list[dict], labels: dict[str, dict], notes: dict) -> tuple[list[dict], dict]:
    rows = []
    for r in items:
        lab = labels[r["item"]]
        if lab["label"] not in LABELS:
            raise SystemExit(f"refusing: item {r['item']} has label {lab['label']!r}")
        checked = r["check"] == "1"
        analyst = notes["checked"].get(r["item"], "") if checked else notes["outside"].get(r["item"], "")
        if checked and not analyst:
            raise SystemExit(f"refusing: no analyst label for checked item {r['item']}")
        rows.append({"item": r["item"], "split": r["split"], "word": r["word"], "label": lab["label"],
                     "gloss": lab["gloss"], "note": lab.get("note", ""), "analyst_check": r["check"],
                     "analyst_label": analyst})
    n = len(rows)
    loan = sum(r["label"] == "LOAN" for r in rows)
    chk = [r for r in rows if r["analyst_check"] == "1"]
    extra = sum(1 for r in rows if r["analyst_check"] != "1" and r["analyst_label"] == "LOAN" and r["label"] != "LOAN")
    return rows, {"n": n, "labels": {k: sum(r["label"] == k for r in rows) for k in LABELS},
                  "share_english": loan / n, "share_english_ci95": wilson(loan, n),
                  "english_words": [f'{r["word"]} ({r["gloss"]})' for r in rows if r["label"] == "LOAN"],
                  "analyst_check": {"n": len(chk), "agree": sum(r["analyst_label"] == r["label"] for r in chk)},
                  "outside_the_check_not_preregistered": {
                      "items": [r["item"] for r in rows if r["analyst_check"] != "1" and r["analyst_label"]],
                      "share_english_if_counted": (loan + extra) / n, "ci95": wilson(loan + extra, n)}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--analyst", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    notes = json.loads(args.analyst.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    for f in ("summary.json", "instructions.md"):
        shutil.copyfile(args.sample / f, args.out / f)
    res = {"procedure": "PREREGISTRATION_H7.md §3: 200 random native-tagged tokens per pair (seed 20261018, "
                        "scripts/d4_contamination_sample.py), one LLM-based labeller per pair with instructions.md, "
                        "blind to the check subset; the analyst checks the 50 marked items per pair",
           "pairs": {}}
    for pair in PAIRS:
        with open(args.sample / f"{pair}_items.csv", encoding="utf-8", newline="") as fh:
            items = list(csv.DictReader(fh))
        with open(args.labels / f"{pair}_labels.csv", encoding="utf-8", newline="") as fh:
            labels = {r["item"]: r for r in csv.DictReader(fh)}
        if set(labels) != {r["item"] for r in items}:
            raise SystemExit(f"refusing: the {pair} labels and items do not list the same items")
        rows, res["pairs"][pair] = report(items, labels, notes[pair])
        with open(args.out / f"{pair}_labels.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (args.out / "contamination_audit.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n",
                                                       encoding="utf-8")
    for pair, r in res["pairs"].items():
        print(f'{pair}: {r["labels"]}; English {r["share_english"]:.3f} {r["share_english_ci95"]}; '
              f'check {r["analyst_check"]}; outside the check {r["outside_the_check_not_preregistered"]}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
