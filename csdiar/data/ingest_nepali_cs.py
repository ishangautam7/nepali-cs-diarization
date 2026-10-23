#!/usr/bin/env python3
"""Ingest D1: write every clip's audio to a WAV file and build a manifest (PLAN.md §4, step 1).

Reads the pinned default-config parquet shards (scripts/download_d1.py). The audio bytes are written as
they are (16 kHz mono WAV, already loudness-normalised by the corpus), so no resampling or editing happens.
Each clip gets a stable ID from its provenance: <video_id>_<start_ms>_<end_ms>. Rows without provenance get
`noprov_<split>_<n>` and `provenance_ok = 0`; analyses drop them (PLAN.md §3, D1 caveats).

Usage:
  python -m csdiar.data.ingest_nepali_cs --shards ~/data/d1_hf/data --out ~/data/d1
Outputs: <out>/wav/<split>/<clip_id>.wav and <out>/manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

SHARD_RE = re.compile(r"^(train|validation|test)-[^/]+\.parquet$")
META = ["speaker_id", "video_id", "start_ms", "end_ms", "duration_s", "speech_type", "cmi", "transcription"]
FIELDS = ["clip_id", "split", "speaker_id", "video_id", "start_ms", "end_ms", "duration_s", "speech_type", "cmi",
          "provenance_ok", "wav", "sample_rate", "n_samples", "transcription"]


def clip_id_for(rec: dict, split: str, counter: dict) -> tuple[str, int]:
    if rec.get("video_id") and rec.get("start_ms") is not None and rec.get("end_ms") is not None:
        return f"{rec['video_id']}_{int(rec['start_ms']):08d}_{int(rec['end_ms']):08d}", 1
    counter[split] = counter.get(split, 0) + 1
    return f"noprov_{split}_{counter[split]:03d}", 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    shards = sorted(p for p in args.shards.glob("*.parquet") if SHARD_RE.match(p.name))
    if not shards or any("augmented" in p.name for p in args.shards.glob("*.parquet")):
        print("expected only default-config shards (train/validation/test)", file=sys.stderr)
        return 1
    rows, seen, noprov = [], set(), {}
    for shard in shards:
        split = SHARD_RE.match(shard.name).group(1)
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=256, columns=META + ["audio"]):
            for rec in batch.to_pylist():
                cid, ok = clip_id_for(rec, split, noprov)
                if cid in seen:
                    raise SystemExit(f"duplicate clip id {cid}")
                seen.add(cid)
                audio = rec["audio"]["bytes"]
                info = sf.info(io.BytesIO(audio))
                if info.samplerate != 16000 or info.channels != 1:
                    raise SystemExit(f"{cid}: unexpected audio format {info.samplerate} Hz, {info.channels} ch")
                wav = args.out / "wav" / split / f"{cid}.wav"
                wav.parent.mkdir(parents=True, exist_ok=True)
                if not wav.exists() or wav.stat().st_size != len(audio):
                    wav.write_bytes(audio)
                rows.append({**{k: rec.get(k) for k in META}, "clip_id": cid, "split": split, "provenance_ok": ok,
                             "wav": str(wav.relative_to(args.out)), "sample_rate": info.samplerate,
                             "n_samples": info.frames})
        print(f"{shard.name}: {len(rows)} clips so far", file=sys.stderr)

    with open(args.out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    hours = sum(r["n_samples"] for r in rows) / 16000 / 3600
    print(f"{len(rows)} clips, {hours:.3f} h, {sum(1 - r['provenance_ok'] for r in rows)} without provenance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
