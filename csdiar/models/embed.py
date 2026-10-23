#!/usr/bin/env python3
"""Dense sliding-window speaker embeddings for every D1 clip (PLAN.md §5: P0 input, E1 and E2).

Window k of a clip covers [k*hop, k*hop + win] seconds. P0's left window at time t is the window starting at
t - win and its right window the one starting at t, so this grid serves P0 directly. (H1's point-anchored,
pause-excluded windows are cut exactly at the boundaries later; they are not taken from this grid.)
Clips shorter than one window get a single zero-padded window, counted in meta.json.

Models:
  e1  speechbrain/spkrec-ecapa-voxceleb  (ECAPA-TDNN, 192-d)
  e2  WeSpeaker ResNet34-LM, the embedding inside pyannote Community-1 (256-d), from the local model folder
  e3  WavLM-Large + ECAPA-TDNN, UniSpeech SV checkpoint (256-d); run with .venv-e3 (ops/env_e3.sh)

Output: <out>/<model>_L<win>/<split>.npz, one float16 array (n_windows, dim) per clip_id, plus meta.json.
These are features, not study outcomes, so computing them on every channel is allowed (PLAN.md §8).
Usage (on the box, after `source ops/env.sh`):
  .venv-pyannote4/bin/python -m csdiar.models.embed --model e1 --root ~/data/d1 --out ~/data/d1/emb --win 1.0
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SR = 16000
MODELS = Path.home() / "data/models"


def load(model: str, device: str):
    """Return (embed function: float32 tensor (B, T) -> numpy (B, dim), provenance dict)."""
    if model == "e1":
        from speechbrain.inference.speaker import EncoderClassifier
        src = "speechbrain/spkrec-ecapa-voxceleb"
        enc = EncoderClassifier.from_hparams(source=src, savedir=str(MODELS / "spkrec-ecapa-voxceleb"),
                                             run_opts={"device": device})
        ref = Path.home() / "data/hf/hub/models--speechbrain--spkrec-ecapa-voxceleb/refs/main"
        prov = {"source": src, "revision": ref.read_text().strip() if ref.exists() else None}
        return (lambda x: enc.encode_batch(x)[:, 0].float().cpu().numpy()), prov
    if model == "e2":
        from pyannote.audio import Model
        path = MODELS / "pyannote-community-1/embedding"
        net = Model.from_pretrained(str(path)).to(device).eval()
        prov = {"source": str(path), "class": type(net).__name__}
        return (lambda x: net(x[:, None]).float().cpu().numpy()), prov
    if model == "e3":
        from csdiar.models.e3 import load_e3
        net, prov = load_e3(device)
        return (lambda x: net(x).float().cpu().numpy()), prov
    raise ValueError(model)


def frames(wav: np.ndarray, win: float, hop: float) -> tuple[np.ndarray, bool]:
    w, h = int(round(win * SR)), int(round(hop * SR))
    padded = len(wav) < w
    if padded:
        wav = np.pad(wav, (0, w - len(wav)))
    starts = np.arange(0, len(wav) - w + 1, h)
    return np.stack([wav[s : s + w] for s in starts]), padded


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=["e1", "e2", "e3"], required=True)
    ap.add_argument("--root", type=Path, required=True, help="D1 ingest folder with manifest.csv and wav/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["validation", "train", "test"])
    ap.add_argument("--win", type=float, default=1.0, help="window length L in seconds")
    ap.add_argument("--hop", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="embed at most this many clips per split (testing)")
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    embed, prov = load(args.model, device)
    out_dir = args.out / f"{args.model}_L{args.win:g}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.root / "manifest.csv", encoding="utf-8")))
    meta = {"model": args.model, **prov, "win_s": args.win, "hop_s": args.hop, "sample_rate": SR,
            "window_k_covers": "[k*hop, k*hop + win] seconds", "splits": {}}

    for split in args.splits:
        out_path = out_dir / f"{split}.npz"
        if out_path.exists() and not args.limit:
            print(f"{split}: exists, skipping", flush=True)
            continue
        todo = [r for r in rows if r["split"] == split][: args.limit or None]
        result, n_padded, n_windows, t0 = {}, 0, 0, time.time()
        pending: list[tuple[str, np.ndarray]] = []  # windows from several clips, embedded in full batches

        def flush() -> None:
            fr = np.concatenate([f for _, f in pending])
            embs = []
            with torch.inference_mode():
                for b in range(0, len(fr), args.batch):
                    embs.append(embed(torch.from_numpy(fr[b : b + args.batch]).to(device)))
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
            fr, padded = frames(wav, args.win, args.hop)
            n_padded += padded
            n_windows += len(fr)
            pending.append((r["clip_id"], fr))
            if sum(len(f) for _, f in pending) >= args.batch:
                flush()
            if k % 2000 == 0:
                print(f"{split}: {k}/{len(todo)} clips, {time.time() - t0:.0f} s", flush=True)
        if pending:
            flush()
        np.savez(out_path, **result)
        meta["splits"][split] = {"clips": len(todo), "windows": n_windows, "padded_clips": n_padded,
                                 "dim": int(next(iter(result.values())).shape[1]) if result else None}
        print(f"{split}: done {len(todo)} clips, {n_windows} windows in {time.time() - t0:.0f} s", flush=True)
    if not args.limit:
        old = json.loads((out_dir / "meta.json").read_text()) if (out_dir / "meta.json").exists() else {}
        meta["splits"] = {**old.get("splits", {}), **meta["splits"]}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
