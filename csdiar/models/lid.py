#!/usr/bin/env python3
"""VoxLingua107 language-ID posteriors on sliding windows of every D1 clip (PLAN.md §5, M2 input).

Model: speechbrain/lang-id-voxlingua107-ecapa (Valk & Alumäe 2021). For each clip, windows of --win seconds
with a --hop step; for each window we store p(ne) and p(en) renormalised over the two, and the probability
mass of all other languages (high values flag music, noise or a third language). This is preprocessing, not
a study outcome: it feeds M2's language-change score and the script-tag checks.

Output: <out>/<split>.npz with one float16 array per clip_id, shape (n_windows, 4):
        [window_centre_s, p_ne, p_en, p_other].
Usage (on the box, after `source ops/env.sh`):
  .venv-pyannote4/bin/python -m csdiar.models.lid --root ~/data/d1 --out ~/data/d1/lid
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from speechbrain.inference.classifiers import EncoderClassifier

SR = 16000


def windows(wav: np.ndarray, win: float, hop: float) -> tuple[np.ndarray, np.ndarray]:
    w, h = int(win * SR), int(hop * SR)
    if len(wav) < w:
        wav = np.pad(wav, (0, w - len(wav)))
    starts = np.arange(0, len(wav) - w + 1, h)
    frames = np.stack([wav[s : s + w] for s in starts])
    return frames, (starts + w / 2) / SR


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["validation", "train", "test"])
    ap.add_argument("--win", type=float, default=1.0)
    ap.add_argument("--hop", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa",
                                         savedir=str(Path.home() / "data/models/lang-id-voxlingua107-ecapa"),
                                         run_opts={"device": device})
    labels = clf.hparams.label_encoder.ind2lab
    i_ne = next(i for i, lab in labels.items() if lab.split(":")[0].strip() == "ne")
    i_en = next(i for i, lab in labels.items() if lab.split(":")[0].strip() == "en")
    rows = list(csv.DictReader(open(args.root / "manifest.csv", encoding="utf-8")))
    args.out.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        out_path = args.out / f"{split}.npz"
        if out_path.exists():
            print(f"{split}: exists, skipping", flush=True)
            continue
        todo = [r for r in rows if r["split"] == split]
        result, t0 = {}, time.time()
        for k, r in enumerate(todo, 1):
            wav, sr = sf.read(args.root / r["wav"], dtype="float32")
            frames, centres = windows(wav, args.win, args.hop)
            probs = []
            with torch.inference_mode():
                for b in range(0, len(frames), args.batch):
                    x = torch.from_numpy(frames[b : b + args.batch]).to(device)
                    logp = clf.classify_batch(x)[0]  # log-posteriors over 107 languages
                    probs.append(logp.exp().float().cpu().numpy())
            p = np.concatenate(probs)
            pair = p[:, i_ne] + p[:, i_en]
            result[r["clip_id"]] = np.stack([centres, p[:, i_ne] / np.maximum(pair, 1e-9),
                                             p[:, i_en] / np.maximum(pair, 1e-9), 1 - pair], axis=1).astype(np.float16)
            if k % 1000 == 0:
                print(f"{split}: {k}/{len(todo)} clips, {time.time() - t0:.0f} s", flush=True)
        np.savez_compressed(out_path, **result)
        print(f"{split}: done {len(todo)} clips in {time.time() - t0:.0f} s -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
