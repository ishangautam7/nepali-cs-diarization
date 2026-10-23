#!/usr/bin/env python3
"""Write the M2a freeze record (PREREGISTRATION.md; csdiar.analysis.confirm.check_freeze).

Two steps, because the features live on the GPU box and the git history on the host:
  1. on the box:   python3 scripts/freeze_m2a.py hash-features --out ~/run/freeze_m2a_features.json
  2. on the host:  python3 scripts/freeze_m2a.py write --features freeze_m2a_features.json
Step 2 hashes every guarded code file (the GUARDED list in csdiar/analysis/confirm.py, read without importing
it) plus PREREGISTRATION.md, checks that each equals its content in the pinned commit (default HEAD; it must be
pushed), and writes planning/FREEZE_M2A.json. Tag m2a-freeze marks the pinned commit; commit the record next and
tag that commit m2a-freeze-record. Only the standard library is used.
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
SPLITS = ("train", "test")
MODELS = ("e1", "e2", "e3")
FEATURES = ([f"~/data/d1/boundaries/lex_v1b/{k}_{s}.csv.gz" for k in ("points", "matches") for s in SPLITS]
            + [f"~/data/d1/point_emb/lex_v1b/{m}/{s}_L{L}.npz" for m in MODELS for s in SPLITS
               for L in ("0.5", "1", "1.5")]
            + [f"~/data/d1/emb/{m}_L1/{s}.npz" for m in MODELS for s in SPLITS]
            + [f"~/data/d1/windows/lex_v1/windows_{s}.csv.gz" for s in SPLITS]
            + [f"~/data/d1/acoustic/lex_v1b/acoustic_{s}.csv.gz" for s in SPLITS]
            + ["~/data/d1/align/train.jsonl", "~/data/d1/results/pilot/pilot_v2.json"])
PATHS = {"boundaries": "~/data/d1/boundaries/lex_v1b", "point_emb": "~/data/d1/point_emb/lex_v1b",
         "emb": "~/data/d1/emb", "windows": "~/data/d1/windows/lex_v1", "align": "~/data/d1/align",
         "pilot": "~/data/d1/results/pilot/pilot_v2.json", "acoustic": "~/data/d1/acoustic/lex_v1b"}
COMMANDS = [
    ".venv-pyannote4/bin/python -m csdiar.analysis.confirm --boundaries ~/data/d1/boundaries/lex_v1b "
    "--point-emb ~/data/d1/point_emb/lex_v1b --emb ~/data/d1/emb --windows ~/data/d1/windows/lex_v1 "
    "--align ~/data/d1/align --pilot ~/data/d1/results/pilot/pilot_v2.json --out ~/data/d1/results/confirm "
    "--splits train test --tag m2a",
    ".venv-pyannote4/bin/python -m csdiar.analysis.attribution --boundaries ~/data/d1/boundaries/lex_v1b "
    "--point-emb ~/data/d1/point_emb/lex_v1b --emb ~/data/d1/emb --acoustic ~/data/d1/acoustic/lex_v1b "
    "--pilot ~/data/d1/results/pilot/pilot_v2.json --confirm ~/data/d1/results/confirm/confirm_m2a.json "
    "--out ~/data/d1/results/confirm --splits train test --tag m2a",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def guarded() -> list[str]:
    tree = ast.parse((REPO / "csdiar" / "analysis" / "confirm.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "GUARDED" for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise SystemExit("GUARDED not found in confirm.py")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hash-features")
    h.add_argument("--out", type=Path, required=True)
    w = sub.add_parser("write")
    w.add_argument("--features", type=Path, required=True)
    w.add_argument("--commit", default="HEAD", help="pushed commit whose code is frozen (default HEAD)")
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
    for f in guarded() + ["planning/PREREGISTRATION.md"]:
        committed = subprocess.run(["git", "show", f"{head}:{f}"], cwd=REPO, check=True, capture_output=True).stdout
        if hashlib.sha256(committed).hexdigest() != sha256(REPO / f):
            raise SystemExit(f"refusing: {f} in the working tree differs from {head[:10]}")
        code[f] = sha256(REPO / f)
    outside = [f for f in FEATURES if not any(f == p or f.startswith(p.rstrip("/") + "/") for p in PATHS.values())]
    if outside:
        raise SystemExit(f"refusing: feature files outside the frozen input paths: {outside[:3]}")
    rec = {"tag": "m2a-freeze", "commit": head, "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
           "record_tag": "m2a-freeze-record",
           "note": "tag m2a-freeze marks the frozen commit below; this file is added in the next commit, tagged "
                   "m2a-freeze-record",
           "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "data_revision": "dd00ca41dd38774cd6097c372c16939d7f3746df", "seed": 20261018,
           "confirmatory_splits": list(SPLITS), "paths": PATHS, "commands": COMMANDS, "code_sha256": code,
           "feature_sha256": {f: feats[f]["sha256"] for f in FEATURES},
           "feature_bytes": {f: feats[f]["bytes"] for f in FEATURES}}
    (REPO / "planning" / "FREEZE_M2A.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(f"wrote planning/FREEZE_M2A.json: {len(code)} code files, {len(FEATURES)} feature files, commit {head[:10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
