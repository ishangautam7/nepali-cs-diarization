#!/usr/bin/env python3
"""Write the H7 freeze record, planning/FREEZE_H7.json, which csdiar.analysis.confirm_h7.check_freeze checks.

Two steps, as for M2a (scripts/freeze_m2a.py):
  1. on the box:   python3 scripts/freeze_h7.py hash-features --out ~/run/freeze_h7_features.json
  2. on the host:  python3 scripts/freeze_h7.py write --features freeze_h7_features.json --commit h7-freeze
Step 2 hashes the GUARDED files of confirm_h7.py (read with ast, without importing), checks that each equals its
content in the pinned commit (which must be pushed), and writes the record with the frozen input paths and the
confirmatory command. Only the standard library is used.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAIRS = ("hi-en", "bn-en")
SPLITS = ("train", "test")
MODELS = ("e1", "e2", "e3")
FEATURES = ([f"~/data/d4/{p}/boundaries/lex_d4v1/{k}_{s}.csv.gz" for p in PAIRS for k in ("points", "matches")
             for s in SPLITS]
            + [f"~/data/d4/{p}/point_emb/lex_d4v1/{m}/{s}_L{L}.npz" for p in PAIRS for m in MODELS for s in SPLITS
               for L in ("0.5", "1", "1.5")]
            + [f"~/data/d4/{p}/emb/{m}_L1/{s}.npz" for p in PAIRS for m in MODELS for s in SPLITS]
            + [f"~/data/d4/{p}/manifest.csv" for p in PAIRS] + ["~/data/d4/raw/SHA256SUMS"]
            + [f"~/data/d1/emb/e2_L1/{s}.npz" for s in ("train", "validation", "test")]
            + ["~/data/d1/manifest.csv", "~/data/d1/results/pilot/pilot_v2.json"])
PATHS = {"d4_root": "~/data/d4", "d1_root": "~/data/d1", "pilot": "~/data/d1/results/pilot/pilot_v2.json",
         "out": "~/data/d4/results/h7"}
COMMAND = (".venv-pyannote4/bin/python -m csdiar.analysis.confirm_h7 --d4-root ~/data/d4 --d1-root ~/data/d1 "
           "--pilot ~/data/d1/results/pilot/pilot_v2.json --out ~/data/d4/results/h7")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def guarded() -> list[str]:
    tree = ast.parse((REPO / "csdiar" / "analysis" / "confirm_h7.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "GUARDED" for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise SystemExit("GUARDED not found in confirm_h7.py")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hash-features")
    h.add_argument("--out", type=Path, required=True)
    w = sub.add_parser("write")
    w.add_argument("--features", type=Path, required=True)
    w.add_argument("--commit", default="HEAD")
    args = ap.parse_args()
    if args.cmd == "hash-features":
        rec = {}
        for f in FEATURES:
            p = Path(os.path.expanduser(f))
            rec[f] = {"sha256": sha256(p), "bytes": p.stat().st_size}
            print(f, rec[f]["sha256"][:12], flush=True)
        args.out.expanduser().write_text(json.dumps(rec, indent=2))
        return 0
    head = git("rev-parse", f"{args.commit}^{{commit}}")
    if not git("branch", "-r", "--contains", head):
        raise SystemExit(f"refusing: {head[:10]} is not pushed")
    feats = json.loads(args.features.read_text())
    missing = [f for f in FEATURES if f not in feats]
    if missing:
        raise SystemExit(f"refusing: feature hashes missing for {missing[:3]}")
    code = {}
    for f in guarded():
        committed = subprocess.run(["git", "show", f"{head}:{f}"], cwd=REPO, check=True, capture_output=True).stdout
        if hashlib.sha256(committed).hexdigest() != sha256(REPO / f):
            raise SystemExit(f"refusing: {f} in the working tree differs from {head[:10]}")
        code[f] = sha256(REPO / f)
    rec = {"tag": "h7-freeze", "commit": head, "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "seed": 20261018, "paths": PATHS, "command": COMMAND, "code_sha256": code,
           "feature_sha256": {f: feats[f]["sha256"] for f in FEATURES},
           "feature_bytes": {f: feats[f]["bytes"] for f in FEATURES}}
    (REPO / "planning" / "FREEZE_H7.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(f"wrote planning/FREEZE_H7.json: {len(code)} guarded files, {len(FEATURES)} feature files, commit {head[:10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
