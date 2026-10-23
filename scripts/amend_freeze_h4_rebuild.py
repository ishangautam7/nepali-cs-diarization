"""Record the 2026-10-19 rebuild of H4's pinned inputs in planning/FREEZE_H4.json (DEVIATIONS.md entry 6).

The GPU box holding the files FREEZE_H4.json pins was lost before any H4 outcome existed, and
ops/rebuild_features.sh regenerated them with the frozen code. This script re-pins them, hashing on the box and
writing on the host as the H7 freeze was recorded. The procedure, in order:

  1. hash    on the box: refuses if any H4 one-shot file exists (stage 1 or 2 result or start marker); otherwise
             writes a report with the SHA-256 and size of every file in feature_sha256 and the hash of the
             FREEZE_H4.json it read.
  2. amend   on the host: refuses unless the report is a hash report made against this exact FREEZE_H4.json,
             covers exactly the pinned files with a hash and a size each, lists no one-shot file, and every
             guarded code file still matches code_sha256. It then appends one dated amendment that keeps each old
             hash and size next to the new one and lists the files that reproduced exactly, and updates
             feature_sha256 and feature_bytes. Nothing else changes, and the rebuild cannot be recorded twice.
  3. sync the amended record to the box, then
  4. verify  on the box: refuses unless the record holds the rebuild amendment, no one-shot file exists, and every
             pinned file and guarded code file still matches the record. Only then is the amended record committed.
             This closes the gap between steps 1 and 2, which run on different machines.

Scope: feature files only. The M3 artifacts are pinned afterwards by the guarded `freeze_h4 add-m3`, which had not
run when the box was lost (the record has no m3_artifacts_sha256); `amend` refuses if that key exists, because
artifacts from the lost box could not be re-pinned here. Dry-run outputs (confirm_h4_stage1_dryrun.json) are not
one-shot files and are ignored. This script is deliberately separate from csdiar/analysis/freeze_h4.py, one of the
guarded files. Stdlib only.

Usage:
  box:   python3 scripts/amend_freeze_h4_rebuild.py hash --out ~/logs/freeze_h4_rebuild_report.json
  host:  python3 scripts/amend_freeze_h4_rebuild.py amend --report <copied report> --note "<what differed>"
  box:   python3 scripts/amend_freeze_h4_rebuild.py verify
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / "planning/FREEZE_H4.json"
AMENDMENT_ID = "rebuild-2026-10-19"
DEVIATION = 6
REPORT_WHAT = "FREEZE_H4 rebuild hash report"
# confirm_h4.check_freeze treats these as the one-shot files of each stage (result, then start marker).
ONE_SHOT = tuple(f"confirm_h4_stage{s}.{ext}" for s in (1, 2) for ext in ("json", "started"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def one_shot_files(rec: dict) -> list[str]:
    """The H4 one-shot files present in the frozen output folder (confirm_h4 enforces --out == paths.out)."""
    out_path = rec.get("paths", {}).get("out")
    if not out_path:
        raise SystemExit("refusing: the record has no paths.out, so H4's one-shot files cannot be checked")
    out = Path(out_path).expanduser()
    return [n for n in ONE_SHOT if (out / n).exists()]


def write_atomic(path: Path, text: str) -> None:
    """Write via a temporary file in the same folder and rename it into place, so a crash never truncates."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def check_code(rec: dict) -> None:
    for f, h in rec["code_sha256"].items():
        if sha256(REPO / f) != h:
            raise SystemExit(f"refusing: guarded file {f} changed since the freeze")


def cmd_hash(args) -> int:
    rec = json.loads(args.freeze.read_text())
    present = one_shot_files(rec)
    if present:
        raise SystemExit(f"refusing: H4 has already run or started ({', '.join(present)})")
    files = {}
    for f in sorted(rec["feature_sha256"]):
        p = Path(f).expanduser()
        if not p.is_file():
            raise SystemExit(f"refusing: pinned file {f} is missing")
        files[f] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    report = {"what": REPORT_WHAT, "date": now(), "freeze_sha256": sha256(args.freeze),
              "one_shot_files": present, "files": files}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(args.out, json.dumps(report, indent=1) + "\n")
    print(f"hashed {len(files)} pinned files; no H4 one-shot file present")
    return 0


def cmd_amend(args) -> int:
    rec = json.loads(args.freeze.read_text())
    rep = json.loads(args.report.read_text())
    if rep.get("what") != REPORT_WHAT:
        raise SystemExit("refusing: not a rebuild hash report")
    if rep.get("freeze_sha256") != sha256(args.freeze):
        raise SystemExit("refusing: the report was made against a different FREEZE_H4.json")
    if rep.get("one_shot_files"):
        raise SystemExit("refusing: H4 had already run or started when the report was made")
    files = rep.get("files", {})
    if sorted(files) != sorted(rec["feature_sha256"]):
        raise SystemExit("refusing: the report does not cover exactly the pinned feature files")
    for f, v in files.items():
        if not (isinstance(v, dict) and isinstance(v.get("sha256"), str) and len(v["sha256"]) == 64
                and isinstance(v.get("bytes"), int)):
            raise SystemExit(f"refusing: the report entry for {f} lacks a SHA-256 or a size")
    old_bytes = rec.get("feature_bytes", {})
    if sorted(old_bytes) != sorted(rec["feature_sha256"]):
        raise SystemExit("refusing: feature_bytes does not cover every pinned file")
    if "m3_artifacts_sha256" in rec:
        raise SystemExit("refusing: M3 artifacts are already pinned; this script re-pins feature files only")
    if any(a.get("id") == AMENDMENT_ID for a in rec.get("amendments", [])):
        raise SystemExit("refusing: this rebuild is already recorded")
    check_code(rec)
    if not args.note.strip():
        raise SystemExit("refusing: --note must say what differed and why")
    identical, changed = [], {}
    for f in sorted(rec["feature_sha256"]):
        if files[f]["sha256"] == rec["feature_sha256"][f]:
            identical.append(f)
        else:
            changed[f] = {"old_sha256": rec["feature_sha256"][f], "new_sha256": files[f]["sha256"],
                          "old_bytes": old_bytes[f], "new_bytes": files[f]["bytes"]}
    rec.setdefault("amendments", []).append({
        "id": AMENDMENT_ID, "date": now(), "deviation": DEVIATION,
        "what": ("the GPU box was lost before any H4 outcome existed; ops/rebuild_features.sh regenerated every "
                 "pinned feature file with the frozen code (DEVIATIONS.md entry 6). feature_sha256 and "
                 "feature_bytes now hold the regenerated files; the old values are kept here"),
        "report_date": rep["date"],
        "code_check": f"all {len(rec['code_sha256'])} guarded files match code_sha256",
        "identical": identical, "changed": changed, "note": args.note.strip()})
    for f in rec["feature_sha256"]:                      # update in place: key order is kept
        rec["feature_sha256"][f] = files[f]["sha256"]
        rec["feature_bytes"][f] = files[f]["bytes"]
    write_atomic(args.freeze, json.dumps(rec, indent=1) + "\n")
    print(f"amended {args.freeze.name}: {len(identical)} identical, {len(changed)} changed; sync it to the box and "
          "run `verify` there before committing. The M3 artifacts are pinned later by `freeze_h4 add-m3`, after "
          "the 25 primary models are trained; stage 1 refuses until then.")
    return 0


def cmd_verify(args) -> int:
    rec = json.loads(args.freeze.read_text())
    if not any(a.get("id") == AMENDMENT_ID for a in rec.get("amendments", [])):
        raise SystemExit("refusing: the record does not hold the rebuild amendment")
    present = one_shot_files(rec)
    if present:
        raise SystemExit(f"refusing: H4 has already run or started ({', '.join(present)})")
    for f, h in rec["feature_sha256"].items():
        p = Path(f).expanduser()
        if not p.is_file() or sha256(p) != h:
            raise SystemExit(f"refusing: {f} does not match the amended record")
    check_code(rec)
    print(f"verified: all {len(rec['feature_sha256'])} pinned files and {len(rec['code_sha256'])} guarded code "
          "files match the amended record; no H4 one-shot file present")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hash")
    h.add_argument("--freeze", type=Path, default=RECORD)
    h.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("amend")
    a.add_argument("--freeze", type=Path, default=RECORD)
    a.add_argument("--report", type=Path, required=True)
    a.add_argument("--note", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--freeze", type=Path, default=RECORD)
    args = ap.parse_args(argv)
    return {"hash": cmd_hash, "amend": cmd_amend, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
