#!/usr/bin/env python3
"""Point-anchored window embeddings for the embedding families (PLAN.md §2: cosine drift, pause excluded).

For every switch point and control eligible for emb0.5 / emb1 / emb1.5, embed W_left = [p0 - L, p0] and
W_right = [p1, p1 + L] cut exactly from the clip (no grid approximation). Features, not outcomes: every
split is processed. Output: <out>/<model>/<split>_L<L>.npz with
  point_id (str), left (n, dim) float16, right (n, dim) float16.
Usage (on the box): python -m csdiar.models.point_embed --model e1 --root ~/data/d1 \
    --points ~/data/d1/boundaries/lex_v1 --out ~/data/d1/point_emb/lex_v1
(use .venv-e3/bin/python for e3)
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from csdiar.models.embed import SR, load

EMB_LS = (0.5, 1.0, 1.5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=["e1", "e2", "e3"], required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--points", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["validation", "train", "test"])
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    embed, prov = load(args.model, device)
    manifest = {r["clip_id"]: r for r in csv.DictReader(open(args.root / "manifest.csv", encoding="utf-8"))}
    out_dir = args.out / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model": args.model, **prov, "windows": "left [p0-L, p0], right [p1, p1+L]", "files": {}}
    for split in args.splits:
        with gzip.open(args.points / f"points_{split}.csv.gz", "rt", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for L in EMB_LS:
            out_path = out_dir / f"{split}_L{L:g}.npz"
            if out_path.exists():
                print(f"{split} L={L:g}: exists, skipping", flush=True)
                continue
            sel = [r for r in rows if r[f"elig_emb{L:g}"] == "True"]
            by_clip = defaultdict(list)
            for r in sel:
                by_clip[r["clip_id"]].append(r)
            n = int(round(L * SR))
            ids, segs, embs, t0 = [], [], [], time.time()

            def flush() -> None:
                with torch.inference_mode():
                    for i in range(0, len(segs), args.batch):
                        x = np.stack(segs[i: i + args.batch])
                        embs.append(embed(torch.from_numpy(x).to(device)).astype(np.float16))
                segs.clear()

            for clip_id, pts in by_clip.items():
                wav, sr = sf.read(args.root / manifest[clip_id]["wav"], dtype="float32")
                for r in pts:
                    a = int(round(float(r["p0"]) * SR)) - n
                    b = int(round(float(r["p1"]) * SR))
                    left, right = wav[max(a, 0): a + n], wav[b: b + n]
                    if len(left) != n or len(right) != n:  # rounding at the clip edge: pad by at most one frame
                        left, right = np.pad(left, (n - len(left), 0)), np.pad(right, (0, n - len(right)))
                    ids.append(r["point_id"])
                    segs += [left, right]
                if len(segs) >= 8 * args.batch:
                    flush()
            if segs:
                flush()
            e = np.concatenate(embs) if embs else np.zeros((0, 1), np.float16)
            np.savez(out_path, point_id=np.array(ids), left=e[0::2], right=e[1::2])
            meta["files"][out_path.name] = {"points": len(ids)}
            print(f"{split} L={L:g}: {len(ids)} points in {time.time() - t0:.0f} s", flush=True)
    old = json.loads((out_dir / "meta.json").read_text()) if (out_dir / "meta.json").exists() else {}
    meta["files"] = {**old.get("files", {}), **meta["files"]}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
