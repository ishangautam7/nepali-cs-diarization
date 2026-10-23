#!/usr/bin/env python3
"""EXPLORATORY, post hoc (external review, issue 4): prepare a human-verified evaluation subset for D1.

Labels nothing. It prepares, for two independent bilingual raters:
  * a stratified random sample (fixed seed) of N primary-cell switch points: p0L1 switch points with at least one
    scored det_w1 control in their cell, on the 17 confirmatory channels (train + test). Channels get a share of
    N in proportion to their number of such points (largest remainder, at least one per channel);
  * one matched control per sampled switch point: a scored det_w1 control from the same channel and cell
    (video x pause bin x position bin), drawn at random without replacement;
  * a 4.000 s snippet per item, centred on the boundary (the boundary is at 2.000 s in every file, zero-padded
    where the clip is shorter), 16 kHz mono in the clip's own sample format, written under a blinded item id
    in a shuffled order so that neither the name nor the order shows which items are switches;
  * template.csv, the answer sheet each rater fills in a copy of, and key.csv (kind, point, clip, times,
    padding), which the raters must not see. scripts/score_human_subset.py scores the two answer sheets.

The population follows the frozen P0 rules (csdiar.analysis.pilot.score_points and h2_cell, reused unchanged);
E3's grid embeddings, byte-identical to the frozen inputs, decide which points have a defined d(t). The audio
is CC BY-NC 4.0 corpus audio: share it privately with the raters; audio/ is git-ignored and not redistributed.

Usage (repo root):
  python -m scripts.review_human_subset --boundaries ~/data/d1/boundaries/lex_v1b --emb ~/data/d1/emb \\
      --d1 ~/data/d1 --out results/review_followups/human_subset
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from csdiar.analysis.confirm import load_points
from csdiar.analysis.pilot import p0_curves, score_points

SPLITS = ("train", "test")
SNIPPET_S, CENTRE_S = 4.0, 2.0
TEMPLATE_FIELDS = ["item_id", "audio_file", "same_speaker", "audible_language_switch", "boundary_within_0.1s", "notes"]
KEY_FIELDS = ["item_id", "kind", "point_id", "paired_item", "match_level", "speaker_id", "split", "video_id",
              "clip_id", "t", "lang_left", "lang_right", "left_s", "right_s", "pause_bin", "pos_bin", "stratum",
              "snippet_start_s", "pad_left_s", "pad_right_s"]


def cell(r: dict) -> tuple:
    return (r["speaker_id"], r["video_id"], r["pause_bin"], r["pos_bin"])


def allocate(sizes: dict, n: int) -> dict:
    """Proportional allocation of n over strata by largest remainder, at least one per stratum, capped by size."""
    total = sum(sizes.values())
    if n < len(sizes) or n > total:
        raise SystemExit(f"cannot allocate {n} over {len(sizes)} strata holding {total}")
    quota = {k: n * v / total for k, v in sizes.items()}
    alloc = {k: min(max(1, math.floor(q)), sizes[k]) for k, q in quota.items()}
    frac = sorted(sizes, key=lambda k: (-(quota[k] - math.floor(quota[k])), k))
    while sum(alloc.values()) < n:
        for k in frac:
            if sum(alloc.values()) == n:
                break
            if alloc[k] < sizes[k]:
                alloc[k] += 1
    while sum(alloc.values()) > n:
        for k in reversed(frac):
            if sum(alloc.values()) == n:
                break
            if alloc[k] > 1:
                alloc[k] -= 1
    return alloc


def snippet(wav: Path, t: float) -> tuple[np.ndarray, int, float, float, str]:
    info = sf.info(str(wav))
    if info.samplerate != 16000 or info.channels != 1:
        raise SystemExit(f"{wav}: expected 16 kHz mono")
    x, sr = sf.read(str(wav), dtype="int16" if info.subtype == "PCM_16" else "float32", always_2d=False)
    n = int(round(SNIPPET_S * sr))
    start = int(round((t - CENTRE_S) * sr))
    out = np.zeros(n, dtype=x.dtype)
    lo, hi = max(start, 0), min(start + n, len(x))
    if hi > lo:
        out[lo - start: hi - start] = x[lo:hi]
    return out, sr, max(0, -start) / sr, max(0, start + n - len(x)) / sr, info.subtype


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--boundaries", type=Path, required=True)
    ap.add_argument("--emb", type=Path, required=True, help="grid embeddings folder; only e3_L1 is read")
    ap.add_argument("--d1", type=Path, required=True, help="D1 folder holding manifest.csv and wav/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=150, help="switch points (and as many matched controls)")
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--audio-from-key", action="store_true",
                    help="only re-export audio/ from an existing key.csv (no new draw), e.g. in a fresh clone")
    args = ap.parse_args()
    if args.audio_from_key:
        manifest = {r["clip_id"]: r for r in csv.DictReader(open(args.d1 / "manifest.csv", encoding="utf-8"))}
        (args.out / "audio").mkdir(parents=True, exist_ok=True)
        rows = list(csv.DictReader(open(args.out / "key.csv", encoding="utf-8")))
        for r in rows:
            x, sr, pad_l, pad_r, subtype = snippet(args.d1 / manifest[r["clip_id"]]["wav"], float(r["t"]))
            if (f"{pad_l:.3f}", f"{pad_r:.3f}") != (r["pad_left_s"], r["pad_right_s"]):
                raise SystemExit(f"{r['item_id']}: padding differs from key.csv; the clip audio is not the same")
            sf.write(str(args.out / "audio" / f"{r['item_id']}.wav"), x, sr, subtype=subtype)
        print(f"re-exported {len(rows)} snippets from key.csv")
        return 0
    if (args.out / "key.csv").exists():
        raise SystemExit(f"refusing: {args.out / 'key.csv'} exists; the subset is drawn once")
    rng = np.random.default_rng(args.seed)

    pts, _ = load_points(args.boundaries, list(SPLITS))
    vals, _ = score_points(pts, p0_curves(args.emb, "e3", SPLITS))
    ctrls = defaultdict(list)
    for p, r in pts.items():
        if r["kind"] == "ctrl" and r["elig_det_w1"] and p in vals:
            ctrls[cell(r)].append(p)
    pop = defaultdict(list)
    for p, r in pts.items():
        if r["kind"] == "ssdl" and r["elig_p0L1"] and p in vals and ctrls.get(cell(r)):
            pop[r["speaker_id"]].append(p)
    sizes = {ch: len(v) for ch, v in sorted(pop.items())}
    alloc = allocate(sizes, args.n)

    chosen = []
    for ch in sorted(pop):
        chosen += list(rng.choice(sorted(pop[ch]), size=alloc[ch], replace=False))
    used, pairs = set(), []
    for p in chosen:
        free = sorted(set(ctrls[cell(pts[p])]) - used)
        if not free:
            raise SystemExit(f"no unused control left in the cell of {p}")
        c = str(rng.choice(free))
        used.add(c)
        pairs.append((str(p), c))

    items = [("switch", s, k) for k, (s, _) in enumerate(pairs)] + [("control", c, k) for k, (_, c) in enumerate(pairs)]
    order = rng.permutation(len(items))
    width = len(str(len(items)))
    item_id = {int(j): f"i{rank + 1:0{width}d}" for rank, j in enumerate(order)}
    by_pair = defaultdict(dict)
    for j, (kind, _, k) in enumerate(items):
        by_pair[k][kind] = item_id[j]

    manifest = {r["clip_id"]: r for r in csv.DictReader(open(args.d1 / "manifest.csv", encoding="utf-8"))}
    (args.out / "audio").mkdir(parents=True, exist_ok=True)
    key_rows, template_rows = [], []
    for j in np.argsort([item_id[j] for j in range(len(items))]):
        kind, p, k = items[j]
        r, iid = pts[p], item_id[int(j)]
        x, sr, pad_l, pad_r, subtype = snippet(args.d1 / manifest[r["clip_id"]]["wav"], r["t"])
        sf.write(str(args.out / "audio" / f"{iid}.wav"), x, sr, subtype=subtype)
        row = {f: r[f] for f in KEY_FIELDS if f in r}  # the table's own 'kind' (ssdl/ctrl) is replaced below
        row.update({"item_id": iid, "kind": kind, "point_id": p,
                    "paired_item": by_pair[k]["control" if kind == "switch" else "switch"],
                    "match_level": "same channel and cell", "snippet_start_s": f"{r['t'] - CENTRE_S:.3f}",
                    "pad_left_s": f"{pad_l:.3f}", "pad_right_s": f"{pad_r:.3f}"})
        key_rows.append(row)
        template_rows.append({"item_id": iid, "audio_file": f"audio/{iid}.wav"})
    for name, fields, rows in (("key.csv", KEY_FIELDS, key_rows), ("template.csv", TEMPLATE_FIELDS, template_rows)):
        with open(args.out / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    meta = {"label": "EXPLORATORY (post hoc, external review issue 4); prepared only, nothing labelled",
            "seed": args.seed, "n_switch": len(pairs), "n_control": len(pairs), "code_commit": commit,
            "population": {"definition": "p0L1 switch points with >= 1 scored det_w1 control in their channel x video "
                                         "x pause bin x position bin cell, 17 confirmatory channels",
                           "size": sum(sizes.values()), "per_channel": sizes},
            "allocation_per_channel": alloc, "snippet": {"seconds": SNIPPET_S, "boundary_at_s": CENTRE_S},
            "inputs_sha256": {str(args.boundaries / f"points_{s}.csv.gz"): sha256(args.boundaries / f"points_{s}.csv.gz")
                              for s in SPLITS} | {str(args.d1 / "manifest.csv"): sha256(args.d1 / "manifest.csv")},
            "outputs_sha256": {n: sha256(args.out / n) for n in ("key.csv", "template.csv")},
            "audio_files": len(key_rows)}
    (args.out / "subset.json").write_text(json.dumps(meta, indent=2))
    print(f"population {meta['population']['size']}, sampled {len(pairs)} switch points + {len(pairs)} controls "
          f"over {len(alloc)} channels; padded items: {sum(1 for r in key_rows if r['pad_left_s'] != '0.000' or r['pad_right_s'] != '0.000')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
