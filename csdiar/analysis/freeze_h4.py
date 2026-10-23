#!/usr/bin/env python3
"""Freeze record for the H4 supplement (PREREGISTRATION_H4.md §10). Runs on the box, whose repo mirror holds the
same files as the host commit (the host has no numpy, and the box has no .git).

Subcommands:
  write   planning/FREEZE_H4.json: the host commit (--commit), the seed, the folds, the frozen input paths, the
          commands, the SHA-256 of every guarded code file and of every feature file H4 reads. Copy it to the
          host, check the code hashes there with sha256sum, commit it, and tag that commit h4-freeze-record
          (the frozen commit is h4-freeze).
  add-m3  after training: appends m3_artifacts_sha256, the hashes of the 25 primary M3 checkpoints and their
          dense grids, as an amendment (artifacts created after the freeze by the frozen commands).
  add-m2  after the M2b freeze: appends m2_alpha (--alpha) and the M2b record's commit (--m2b-commit).
  add-stage1  after stage 1: appends the SHA-256 of confirm_h4_stage1.json, which stage 2 reads and checks.
Nothing already recorded may change.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from csdiar.analysis.confirm_h4 import GUARDED, REPO, SEED, m3_artifacts, sha256
from csdiar.mitigations.folds import h4_folds, train_channels

RECORD = REPO / "planning/FREEZE_H4.json"
FEATURES = [
    "~/data/d1/manifest.csv",
    *[f"~/data/d1/boundaries/lex_v1b/{k}_{s}.csv.gz" for k in ("points", "matches") for s in ("validation", "train")],
    "~/data/d1/boundaries/lex_v1b/summary.json",
    "~/data/d1/point_emb/lex_v1b/e3/train_L1.npz",
    *[f"~/data/d1/emb/e3_L1/{s}.npz" for s in ("validation", "train")],
    *[f"~/data/d1/lid/{s}.npz" for s in ("validation", "train")],
    "~/data/d1/windows/lex_v1/windows_train.csv.gz",
    "~/data/d1/windows/lex_v1/windows_validation.csv.gz",
    "~/data/d1/results/purity/subset.json",
    *[f"~/data/d1/audit/{m}/{s}.npz" for m in ("titanet", "campplus") for s in ("validation", "train", "test")],
    "~/data/m3/d1_windows.npz", "~/data/m3/dev_trials.npz", "~/data/m3/d1_offsets.npy", "~/data/m3/d1_audio.int16",
    "~/data/m3/d5_offsets.npy", "~/data/m3/d5_audio.int16", "~/data/d5/manifest.csv",
]
PATHS = {"manifest": "~/data/d1/manifest.csv", "boundaries": "~/data/d1/boundaries/lex_v1b",
         "point_emb": "~/data/d1/point_emb/lex_v1b", "emb": "~/data/d1/emb", "lid": "~/data/d1/lid",
         "windows": "~/data/d1/windows/lex_v1", "m3_emb": "~/data/m3/emb", "out": "~/data/d1/results/h4"}
COMMANDS = {
    "m3_train": ".venv-e3/bin/python -m csdiar.mitigations.m3_train --fold F --seed S --w-dist 0.3",
    "m3_embed": ".venv-e3/bin/python -m csdiar.mitigations.m3_embed --ckpt ~/data/m3/runs/fF_sS_w0.3_none/best.pt "
                "--name m3_fF_sS_w0.3_none --splits validation train",
    "stage1": ".venv-pyannote4/bin/python -m csdiar.analysis.confirm_h4 --stage 1",
    "stage2": ".venv-pyannote4/bin/python -m csdiar.analysis.confirm_h4 --stage 2",
}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def cmd_write(args) -> int:
    if RECORD.exists():
        raise SystemExit("refusing: planning/FREEZE_H4.json exists; amend it instead")
    if len(args.commit) != 40:
        raise SystemExit("refusing: --commit must be the full 40-character host commit")
    feats = {}
    for f in FEATURES:
        p = Path(f).expanduser()
        feats[f] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    rec = {"tag": "h4-freeze", "commit": args.commit, "record_tag": "h4-freeze-record", "date": now(), "seed": SEED,
           "folds": h4_folds(train_channels(Path(PATHS["manifest"]).expanduser())), "paths": PATHS,
           "commands": COMMANDS, "code_sha256": {f: sha256(REPO / f) for f in GUARDED},
           "feature_sha256": {f: v["sha256"] for f, v in feats.items()},
           "feature_bytes": {f: v["bytes"] for f, v in feats.items()}, "amendments": []}
    RECORD.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"wrote {RECORD} for {args.commit}")
    return 0


def cmd_add_m3(_args) -> int:
    rec = json.loads(RECORD.read_text())
    if "m3_artifacts_sha256" in rec:
        raise SystemExit("refusing: M3 artifacts are already recorded")
    rec["m3_artifacts_sha256"] = {f: sha256(Path(f).expanduser()) for f in m3_artifacts()}
    rec["amendments"].append({"date": now(), "what": "added the hashes of the 25 primary M3 checkpoints and their "
                                                     "dense grids, created after the freeze by the frozen commands"})
    RECORD.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"added {len(rec['m3_artifacts_sha256'])} M3 artifact hashes")
    return 0


def cmd_add_m2(args) -> int:
    rec = json.loads(RECORD.read_text())
    if "m2_alpha" in rec:
        raise SystemExit("refusing: M2 alpha is already recorded")
    rec["m2_alpha"] = float(args.alpha)
    rec["m2b_commit"] = args.m2b_commit
    rec["amendments"].append({"date": now(), "what": f"added M2's alpha = {args.alpha} frozen in M2b "
                                                     f"({args.m2b_commit})"})
    RECORD.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"added m2_alpha = {args.alpha}")
    return 0


def cmd_add_stage1(_args) -> int:
    rec = json.loads(RECORD.read_text())
    if "stage1_sha256" in rec:
        raise SystemExit("refusing: the stage-1 result is already recorded")
    s1 = Path(PATHS["out"]).expanduser() / "confirm_h4_stage1.json"
    rec["stage1_sha256"] = sha256(s1)
    rec["amendments"].append({"date": now(), "what": "added the hash of the stage-1 result, which stage 2 reads"})
    RECORD.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"added stage1_sha256 {rec['stage1_sha256']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--commit", required=True)
    sub.add_parser("add-m3")
    sub.add_parser("add-stage1")
    m = sub.add_parser("add-m2")
    m.add_argument("--alpha", type=float, required=True)
    m.add_argument("--m2b-commit", required=True)
    args = ap.parse_args()
    return {"write": cmd_write, "add-m3": cmd_add_m3, "add-m2": cmd_add_m2,
            "add-stage1": cmd_add_stage1}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
