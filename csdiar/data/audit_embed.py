#!/usr/bin/env python3
"""Clip embeddings for the speaker-purity audit (PREREGISTRATION.md §5 item 4; PLAN.md §4 step 6).

Two embedding families that are not under test:
- titanet:  NeMo TitaNet-L (nvidia/speakerverification_en_titanet_large, 192-d), on the GPU;
- campplus: WeSpeaker CAM++ (Wespeaker/wespeaker-voxceleb-campplus, 512-d), its official ONNX export on the
  CPU, with WeSpeaker's own ONNX front end (Kaldi fbank, 80 bins, 25/10 ms, Hamming, no energy, int16
  scaling, mean normalisation without variance normalisation).

One embedding per clip, "mean over speech". Silero VAD frames with p >= 0.5 (the threshold boundaries.py uses)
are speech; non-speech gaps of at most GAP_BRIDGE_S inside speech are bridged, which gives phrase-like speech
runs. Every run of at least MIN_RUN_S is embedded on its own: runs are never joined to each other, so no audio
is concatenated. The clip embedding is the duration-weighted mean of the L2-normalised run embeddings,
re-normalised. A clip without such a run gets one embedding of the whole clip, and is counted as a fallback.

Output: <out>/<model>/<split>.npz (clip_id -> float32 unit vector) and <out>/<model>/meta.json.
Usage (on the box, after `source ops/env.sh`):
  .venv-audit/bin/python -m csdiar.data.audit_embed --model titanet  --root ~/data/d1 --vad ~/data/d1/vad --out ~/data/d1/audit
  .venv-audit/bin/python -m csdiar.data.audit_embed --model campplus --root ~/data/d1 --vad ~/data/d1/vad --out ~/data/d1/audit
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

SR = 16000
VAD_FRAME_S = 512 / SR          # Silero frame k covers [0.032k, 0.032(k+1)) s (csdiar/data/vad.py)
VAD_THRESHOLD = 0.5             # as in csdiar/data/boundaries.py
GAP_BRIDGE_S = 0.25             # non-speech gaps up to this length inside speech are bridged
MIN_RUN_S = 0.5                 # shortest speech run that is embedded
MODELS = Path.home() / "data/models"
TITANET_PATH = MODELS / "titanet-large/speakerverification_en_titanet_large.nemo"
CAMPP_PATH = MODELS / "wespeaker-campplus/voxceleb_CAM++.onnx"
TITANET_REV = "0dc382f40121a5fbd34db10a2bb04d826c2be6a8"
CAMPP_REV = "acf623ad8ca746e50baa432255cf8fc57c669c45"


def speech_runs(vad: np.ndarray, n_samples: int) -> list[tuple[int, int]]:
    """Sample ranges [a, b) of bridged speech runs that are at least MIN_RUN_S long."""
    speech = np.asarray(vad, dtype=np.float32) >= VAD_THRESHOLD
    runs: list[list[int]] = []
    k, n = 0, len(speech)
    while k < n:
        if not speech[k]:
            k += 1
            continue
        j = k
        while j < n and speech[j]:
            j += 1
        if runs and (k - runs[-1][1]) * VAD_FRAME_S <= GAP_BRIDGE_S:
            runs[-1][1] = j
        else:
            runs.append([k, j])
        k = j
    out = []
    for a, b in runs:
        sa, sb = int(round(a * VAD_FRAME_S * SR)), min(int(round(b * VAD_FRAME_S * SR)), n_samples)
        if (sb - sa) >= MIN_RUN_S * SR:
            out.append((sa, sb))
    return out


def pool(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Duration-weighted mean of L2-normalised vectors, re-normalised."""
    v = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    m = (v * weights[:, None]).sum(axis=0) / weights.sum()
    return (m / max(float(np.linalg.norm(m)), 1e-12)).astype(np.float32)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_titanet(device: str):
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    model = EncDecSpeakerLabelModel.restore_from(str(TITANET_PATH), map_location=device)
    model.preprocessor.featurizer.dither = 0.0      # deterministic features
    return model.eval()


def embed_titanet(model, wavs: list[np.ndarray], device: str, batch: int = 64) -> np.ndarray:
    """Raw TitaNet-L embeddings, one per waveform; padded batches carry their true lengths."""
    import torch
    order = np.argsort([len(w) for w in wavs])
    out = np.zeros((len(wavs), 192), dtype=np.float32)
    for b in range(0, len(order), batch):
        idx = order[b : b + batch]
        lens = [len(wavs[i]) for i in idx]
        x = torch.zeros(len(idx), max(lens))
        for r, i in enumerate(idx):
            x[r, : lens[r]] = torch.from_numpy(wavs[i])
        with torch.inference_mode():
            _, emb = model.forward(input_signal=x.to(device),
                                   input_signal_length=torch.tensor(lens, device=device))
        out[idx] = emb.float().cpu().numpy()
    return out


def load_campplus(threads: int = 4):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(CAMPP_PATH), sess_options=so, providers=["CPUExecutionProvider"])


def campplus_fbank(wav: np.ndarray) -> np.ndarray:
    import torch
    import torchaudio.compliance.kaldi as kaldi
    x = torch.from_numpy(wav).unsqueeze(0) * (1 << 15)
    m = kaldi.fbank(x, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0,
                    sample_frequency=SR, window_type="hamming", use_energy=False)
    return (m - m.mean(dim=0)).numpy()


def embed_campplus(session, wavs: list[np.ndarray]) -> np.ndarray:
    """CAM++ embeddings, one waveform per call: zero padding would distort CMN and statistics pooling."""
    name_in, name_out = session.get_inputs()[0].name, session.get_outputs()[0].name
    return np.stack([session.run([name_out], {name_in: campplus_fbank(w)[None]})[0][0] for w in wavs]
                    ).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=["titanet", "campplus"], required=True)
    ap.add_argument("--root", type=Path, required=True, help="D1 ingest folder with manifest.csv and wav/")
    ap.add_argument("--vad", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["validation", "train", "test"])
    ap.add_argument("--chunk", type=int, default=1500, help="clips read into memory at a time")
    ap.add_argument("--limit", type=int, default=0, help="at most this many clips per split (testing)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.model == "titanet":
        model, path, rev = load_titanet(args.device), TITANET_PATH, TITANET_REV
        embed = lambda ws: embed_titanet(model, ws, args.device)
    else:
        model, path, rev = load_campplus(), CAMPP_PATH, CAMPP_REV
        embed = lambda ws: embed_campplus(model, ws)
    out_dir = args.out / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.root / "manifest.csv", encoding="utf-8")))
    meta = {"model": args.model, "checkpoint": str(path), "revision": rev, "sha256": sha256(path),
            "vad_threshold": VAD_THRESHOLD, "gap_bridge_s": GAP_BRIDGE_S, "min_run_s": MIN_RUN_S,
            "pooling": "duration-weighted mean of L2-normalised speech-run embeddings, re-normalised",
            "limit": args.limit, "splits": {}}
    for split in args.splits:
        out_path = out_dir / f"{split}.npz"
        if out_path.exists() and not args.limit:
            print(f"{split}: exists, skipping", flush=True)
            continue
        vad = np.load(args.vad / f"{split}.npz")
        todo = [r for r in rows if r["split"] == split][: args.limit or None]
        result, fallback, n_runs, speech_s, t0 = {}, [], 0, 0.0, time.time()
        for c in range(0, len(todo), args.chunk):
            chunk = todo[c : c + args.chunk]
            segs, owner, weight = [], [], []
            for r in chunk:
                wav = sf.read(args.root / r["wav"], dtype="float32")[0]
                runs = speech_runs(vad[r["clip_id"]], len(wav))
                if not runs:
                    fallback.append(r["clip_id"])
                    runs = [(0, len(wav))]
                for a, b in runs:
                    segs.append(wav[a:b])
                    owner.append(r["clip_id"])
                    weight.append((b - a) / SR)
            emb = embed(segs)
            owner_arr, weight_arr = np.array(owner), np.array(weight)
            for r in chunk:
                sel = owner_arr == r["clip_id"]
                result[r["clip_id"]] = pool(emb[sel], weight_arr[sel])
            n_runs += len(segs)
            speech_s += float(weight_arr.sum())
            print(f"{split}: {min(c + args.chunk, len(todo))}/{len(todo)} clips, {time.time() - t0:.0f} s",
                  flush=True)
        if not args.limit:
            np.savez_compressed(out_path, **result)
        meta["splits"][split] = {"clips": len(todo), "runs": n_runs, "embedded_speech_h": round(speech_s / 3600, 3),
                                 "fallback_whole_clip": len(fallback), "fallback_clip_ids": fallback}
        print(f"{split}: done {len(todo)} clips, {n_runs} runs, {len(fallback)} fallbacks, "
              f"{time.time() - t0:.0f} s", flush=True)
    if not args.limit:
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    else:
        print(json.dumps({k: v for k, v in meta.items() if k != "splits"}), json.dumps(meta["splits"])[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
