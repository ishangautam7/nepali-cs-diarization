#!/usr/bin/env python3
"""Frame-level speech probabilities for every D1 clip with Silero VAD (PLAN.md §2, "Pauses").

Pauses at word boundaries are measured with a VAD because CTC aligners place word ends imprecisely.
Silero VAD (MIT licence) is used because it is not part of any system under test; pyannote's own
segmentation would make pause matching depend on P1. Frame k covers samples [512k, 512k + 512) at 16 kHz,
i.e. [0.032k, 0.032(k + 1)) seconds. The raw probabilities are stored; pause rules live in boundaries.py.

Clips are processed in batches of similar length (zero-padded at the end; padded frames are dropped).
Output: <out>/<split>.npz with one float16 array (n_frames,) per clip_id, plus meta.json.
Usage (on the box, after `source ops/env.sh`):
  .venv-pyannote4/bin/python -m csdiar.data.vad --root ~/data/d1 --out ~/data/d1/vad
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from silero_vad import load_silero_vad

SR = 16000
FRAME = 512


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["validation", "train", "test"])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    model = load_silero_vad().to(args.device)
    rows = list(csv.DictReader(open(args.root / "manifest.csv", encoding="utf-8")))
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"model": "silero-vad", "version": version("silero-vad"), "device": args.device,
            "frame_s": FRAME / SR, "frame_k_covers": "[0.032k, 0.032(k+1)) s", "splits": {}}
    for split in args.splits:
        out_path = args.out / f"{split}.npz"
        if out_path.exists() and not args.limit:
            print(f"{split}: exists, skipping", flush=True)
            continue
        todo = [r for r in rows if r["split"] == split][: args.limit or None]
        todo.sort(key=lambda r: int(r["n_samples"]))
        result, t0 = {}, time.time()
        for b in range(0, len(todo), args.batch):
            chunk = todo[b : b + args.batch]
            wavs = [sf.read(args.root / r["wav"], dtype="float32")[0] for r in chunk]
            n_max = -(-max(len(w) for w in wavs) // FRAME) * FRAME
            x = torch.zeros(len(wavs), n_max)
            for i, w in enumerate(wavs):
                x[i, : len(w)] = torch.from_numpy(w)
            with torch.inference_mode():
                probs = model.audio_forward(x.to(args.device), SR).float().cpu().numpy()
            for r, w, p in zip(chunk, wavs, probs):
                result[r["clip_id"]] = p[: -(-len(w) // FRAME)].astype(np.float16)
        np.savez_compressed(out_path, **result)
        meta["splits"][split] = {"clips": len(todo)}
        print(f"{split}: done {len(todo)} clips in {time.time() - t0:.0f} s", flush=True)
    if not args.limit:
        (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
