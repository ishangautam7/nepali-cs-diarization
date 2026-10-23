#!/usr/bin/env python3
"""Download the D1 corpus (saileshbro/nepali-cs-asr, v2 default config) at the pinned revision.

Only the default-config shards are fetched: data/train-*, data/validation-* and data/test-*.
The offline-augmented shards (data/augmented-train-*) are synthetic-adjacent and excluded by design.

Usage:
  python scripts/download_d1.py --out ~/data/d1_hf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "saileshbro/nepali-cs-asr"
REVISION = "dd00ca41dd38774cd6097c372c16939d7f3746df"  # pinned 2026-10-18, see planning/DECISIONS.md
PATTERNS = ["README.md", "data/train-*.parquet", "data/validation-*.parquet", "data/test-*.parquet"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    path = Path(snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=args.out,
                                  allow_patterns=PATTERNS))
    shards = sorted(path.glob("data/*.parquet"))
    if any("augmented" in p.name for p in shards):
        print("augmented shards present; refusing to continue", file=sys.stderr)
        return 1
    total = sum(p.stat().st_size for p in shards)
    print(f"{len(shards)} shards, {total / 1e9:.2f} GB at revision {REVISION} in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
