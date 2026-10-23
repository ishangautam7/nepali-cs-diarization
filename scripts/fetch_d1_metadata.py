#!/usr/bin/env python3
"""Fetch the non-audio columns of saileshbro/nepali-cs-asr (v2, default config) at a pinned revision.

Only the metadata column chunks of each parquet shard are read, over HTTP range requests, so the
audio and video columns are never downloaded. Outputs, in --out:
  clips.csv      one row per clip (split, channel, video, offsets, duration, speech type, CMI, text)
  videos.csv     one row per source video (channel, split, clip count, covered hours, offset range)
  video_ids.txt  the source video IDs, one per line (the clips' source videos)
  summary.json   dataset revision, counts, and provenance problems

Usage:
  python scripts/fetch_d1_metadata.py --out data/d1_meta [--revision <commit sha>]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

REPO = "saileshbro/nepali-cs-asr"
COLUMNS = ["speaker_id", "video_id", "start_ms", "end_ms", "duration_s", "speech_type", "cmi", "transcription"]
# Default config only. The anchored prefix excludes the offline-augmented "augmented-train" shards.
SHARD_RE = re.compile(r"^data/(train|validation|test)-[^/]+\.parquet$")


def read_shards(revision: str) -> list[dict]:
    api = HfApi()
    files = sorted(
        f for f in api.list_repo_files(REPO, repo_type="dataset", revision=revision) if SHARD_RE.match(f)
    )
    if not files:
        raise SystemExit("no default-config parquet shards found")
    fs = HfFileSystem()
    rows: list[dict] = []
    for name in files:
        split = SHARD_RE.match(name).group(1)
        with fs.open(f"datasets/{REPO}@{revision}/{name}", "rb", block_size=1 << 20) as fh:
            pf = pq.ParquetFile(fh)
            cols = [c for c in COLUMNS if c in set(pf.schema_arrow.names)]
            table = pf.read(columns=cols)
        for rec in table.to_pylist():
            rec["split"] = split
            rec["shard"] = name
            rows.append(rec)
        print(f"{name}: {table.num_rows} rows", file=sys.stderr)
    return rows


def summarise(rows: list[dict], revision: str) -> tuple[dict, list[dict]]:
    no_provenance = [r for r in rows if not r.get("video_id") or r.get("start_ms") is None]
    keys = Counter((r["video_id"], r["start_ms"], r["end_ms"]) for r in rows if r.get("video_id"))
    shared_keys = {k: n for k, n in keys.items() if n > 1}

    videos: dict[str, dict] = {}
    video_splits: dict[str, set] = defaultdict(set)
    video_channels: dict[str, set] = defaultdict(set)
    for r in rows:
        vid = r.get("video_id")
        if not vid:
            continue
        v = videos.setdefault(vid, {"video_id": vid, "n_clips": 0, "clip_seconds": 0.0,
                                    "first_start_ms": r["start_ms"], "last_end_ms": r["end_ms"]})
        v["n_clips"] += 1
        v["clip_seconds"] += float(r.get("duration_s") or 0.0)
        v["first_start_ms"] = min(v["first_start_ms"], r["start_ms"])
        v["last_end_ms"] = max(v["last_end_ms"], r["end_ms"])
        video_splits[vid].add(r["split"])
        video_channels[vid].add(r["speaker_id"])
    video_rows = []
    for vid, v in sorted(videos.items()):
        v["speaker_id"] = ";".join(sorted(video_channels[vid]))
        v["split"] = ";".join(sorted(video_splits[vid]))
        v["clip_hours"] = round(v.pop("clip_seconds") / 3600, 4)
        video_rows.append(v)

    def per_split(key):
        out = {}
        for split in ("train", "validation", "test"):
            sub = [r for r in rows if r["split"] == split]
            out[split] = key(sub)
        return out

    summary = {
        "repo": REPO,
        "revision": revision,
        "clips": len(rows),
        "hours": round(sum(float(r.get("duration_s") or 0) for r in rows) / 3600, 3),
        "clips_per_split": per_split(len),
        "hours_per_split": per_split(lambda s: round(sum(float(r.get("duration_s") or 0) for r in s) / 3600, 3)),
        "channels_per_split": per_split(lambda s: len({r["speaker_id"] for r in s})),
        "videos_per_split": per_split(lambda s: len({r["video_id"] for r in s if r.get("video_id")})),
        "channels": len({r["speaker_id"] for r in rows}),
        "videos": len(videos),
        "speech_type": dict(Counter(r.get("speech_type") for r in rows)),
        "rows_without_provenance": [{"split": r["split"], "speaker_id": r["speaker_id"]} for r in no_provenance],
        "shared_provenance_keys": [{"video_id": k[0], "start_ms": k[1], "end_ms": k[2], "rows": n}
                                   for k, n in shared_keys.items()],
        "videos_in_several_splits": [v for v, s in video_splits.items() if len(s) > 1],
        "videos_in_several_channels": [v for v, c in video_channels.items() if len(c) > 1],
    }
    return summary, video_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--revision", default=None, help="dataset commit SHA (default: current main, recorded)")
    args = ap.parse_args()

    revision = args.revision or HfApi().dataset_info(REPO).sha
    rows = read_shards(revision)
    summary, video_rows = summarise(rows, revision)

    args.out.mkdir(parents=True, exist_ok=True)
    clip_fields = ["split", "speaker_id", "video_id", "start_ms", "end_ms", "duration_s", "speech_type", "cmi",
                   "transcription", "shard"]
    with open(args.out / "clips.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=clip_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(args.out / "videos.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["video_id", "speaker_id", "split", "n_clips", "clip_hours",
                                           "first_start_ms", "last_end_ms"])
        w.writeheader()
        w.writerows(video_rows)
    (args.out / "video_ids.txt").write_text("".join(v["video_id"] + "\n" for v in video_rows), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("revision", "clips", "hours", "channels", "videos",
                                               "clips_per_split", "channels_per_split", "videos_per_split",
                                               "speech_type")}, indent=1))
    print(f"rows without provenance: {len(summary['rows_without_provenance'])}; "
          f"shared keys: {len(summary['shared_provenance_keys'])}; "
          f"videos in several splits: {len(summary['videos_in_several_splits'])}; "
          f"videos in several channels: {len(summary['videos_in_several_channels'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
