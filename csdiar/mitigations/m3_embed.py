#!/usr/bin/env python3
"""Dense L = 1 s grid embeddings of D1 clips with a trained M3 model (PREREGISTRATION_H4.md §4).

Same grid, framing, batching, precision (fp32) and output format as the frozen csdiar/models/embed.py, so P0
(csdiar/analysis/pilot.p0_curves) reads the result exactly as it reads E3's grid: window k of a clip covers
[0.1k, 0.1k + 1.0] s, and <out>/<name>_L1/<split>.npz holds one float16 (n_windows, 256) array per clip_id.
Usage (on the box, .venv-e3):
  .venv-e3/bin/python -m csdiar.mitigations.m3_embed --ckpt ~/data/m3/runs/f0_s0_w0.3_none/best.pt \
      --name m3_f0_s0_w0.3_none --root ~/data/d1 --out ~/data/m3/emb --splits validation train
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from csdiar.mitigations.m3_train import head, load_m3, wavlm_hidden
from csdiar.models.embed import SR, frames

WIN, HOP = 1.0, 0.1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--name", required=True, help="grid name; P0 reads <out>/<name>_L1/")
    ap.add_argument("--root", type=Path, default=Path.home() / "data/d1")
    ap.add_argument("--out", type=Path, default=Path.home() / "data/m3/emb")
    ap.add_argument("--splits", nargs="+", default=["validation", "train"])
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="at most this many clips per split (testing)")
    args = ap.parse_args()

    device = "cuda:0"
    net, prov = load_m3(args.ckpt, device)
    # The grid uses the model's own forward; training and dev scoring used head(wavlm_hidden(.)). Check they agree.
    probe = torch.from_numpy(np.random.default_rng(0).standard_normal((4, SR)).astype(np.float32) * 0.05).to(device)
    with torch.inference_mode():
        a = torch.nn.functional.normalize(net(probe).float())
        b = torch.nn.functional.normalize(head(net, wavlm_hidden(net, probe)).float())
    if float((a * b).sum(1).min()) < 0.9999:
        raise RuntimeError("m3_embed: the model forward and the training forward disagree")
    out_dir = args.out / f"{args.name}_L1"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_sha = hashlib.sha256(args.ckpt.read_bytes()).hexdigest()
    if (out_dir / "meta.json").exists() and json.loads((out_dir / "meta.json").read_text()).get("ckpt_sha256") != ckpt_sha:
        raise SystemExit(f"refusing: {out_dir} was written from a different checkpoint")
    rows = list(csv.DictReader(open(args.root / "manifest.csv", encoding="utf-8")))
    meta = {"model": args.name, **prov, "ckpt_sha256": ckpt_sha,
            "win_s": WIN, "hop_s": HOP, "sample_rate": SR, "window_k_covers": "[k*hop, k*hop + win] seconds",
            "splits": {}}
    for split in args.splits:
        out_path = out_dir / f"{split}.npz"
        if out_path.exists() and not args.limit:
            print(f"{split}: exists, skipping", flush=True)
            continue
        todo = [r for r in rows if r["split"] == split][: args.limit or None]
        result, n_padded, n_windows, t0 = {}, 0, 0, time.time()
        pending: list[tuple[str, np.ndarray]] = []

        def flush() -> None:
            fr = np.concatenate([f for _, f in pending])
            embs = []
            with torch.inference_mode():
                for b in range(0, len(fr), args.batch):
                    embs.append(net(torch.from_numpy(fr[b : b + args.batch]).to(device)).float().cpu().numpy())
            embs = np.concatenate(embs).astype(np.float16)
            i = 0
            for cid, f in pending:
                result[cid] = embs[i : i + len(f)]
                i += len(f)
            pending.clear()

        for k, r in enumerate(todo, 1):
            wav, sr = sf.read(args.root / r["wav"], dtype="float32")
            if sr != SR or wav.ndim != 1:
                raise ValueError(f"{r['wav']}: expected 16 kHz mono")
            fr, padded = frames(wav, WIN, HOP)
            n_padded += padded
            n_windows += len(fr)
            pending.append((r["clip_id"], fr))
            if sum(len(f) for _, f in pending) >= args.batch:
                flush()
            if k % 2000 == 0:
                print(f"{split}: {k}/{len(todo)} clips, {time.time() - t0:.0f} s", flush=True)
        if pending:
            flush()
        if not args.limit:
            np.savez(out_path, **result)
        meta["splits"][split] = {"clips": len(todo), "windows": n_windows, "padded_clips": n_padded}
        print(f"{split}: done {len(todo)} clips, {n_windows} windows in {time.time() - t0:.0f} s", flush=True)
    if not args.limit:
        old = json.loads((out_dir / "meta.json").read_text()) if (out_dir / "meta.json").exists() else {}
        meta["splits"] = {**old.get("splits", {}), **meta["splits"]}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
