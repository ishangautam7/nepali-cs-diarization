#!/usr/bin/env python3
"""Freeze record for the M2b conversation study (PREREGISTRATION_M2B.md §3, §11). Runs on the box, whose repo
mirror holds the same files as the host commit (the host has no numpy, and the box has no .git).

`planning/FREEZE_M2B.json` (tag `m2b-freeze`) is built in the fixed order of §3/§11, one subcommand per step, and
nothing already recorded may change (every `add-*` refuses when its own section is already present, when the
record is missing, when any guarded code file changed since `init`, or once `confirm_m2b` has started):

  init           the code sha256 of every guarded file (§11 list), the sha256 of the dev AND eval feature files
                 (stage-1 caches, LID caches, VAD) and of both reference RTTM sets (bytes hashed, never parsed),
                 the dev/eval split as released (IDs read from the reference file names), the label map, the seed
                 20261018, every input path, the host commit, and the tag `m2b-freeze`.
  add-tuning     hashes the P1-AHC dev-tuning record (`tune_m2b`) AND the P0-gate record (`tune_p0_m2b`) and
                 copies the frozen operating points out of them: each system's clustering threshold, B2's delta,
                 M2's P1-AHC-post-processor alpha (and tau0 = 0), M2's P0-gate alpha and tau0, and the B1/M1 fit
                 files. REFUSES a tuning record whose own code hash (`tune_m2b` / `tune_p0_m2b`) differs from the
                 one `init` froze, so a record built from edited tuning code cannot be adopted.
  add-power      hashes the H5 power result (written by `power_m2b.py`) and copies its `verdict`
                 ("confirmatory" | "descriptive"), which decides whether H5's net-improvement claim is made.
  add-m3         hashes the five all-data M3 checkpoints and their per-seed DISPLACE eval stage-1 caches.
  add-eval-hyps  hashes every expected eval hypothesis file produced by `run_m2b` at the frozen parameters, over
                 the whole clustering-threshold grid, for every P1-AHC system, AND every per-recording native
                 P1/P2 `<rid>.exclusive.rttm`; pins the hypothesis root; REFUSES if any expected file is missing.
                 Pass `--native-absent` to freeze with the native hypotheses ABSENT (their runners do not exist
                 yet): the native RTTMs are then not hashed, `native_absent: true` is recorded, and `confirm_m2b`
                 reports the pipeline H2 as not done and reads nothing from the native dirs.

Usage (on the box, in the repo mirror; .venv-pyannote4):
  .venv-pyannote4/bin/python -m csdiar.analysis.freeze_m2b init --commit <40-char host commit> \
      --root ~/data/displace --label-map planning/displace_label_map.json
  .venv-pyannote4/bin/python -m csdiar.analysis.freeze_m2b add-tuning \
      --tuning ~/data/displace/results/tune_m2b.json --p0-tuning ~/data/displace/results/tune_p0_m2b.json
  .venv-pyannote4/bin/python -m csdiar.analysis.freeze_m2b add-power --power ~/data/displace/results/power_m2b.json
  .venv-pyannote4/bin/python -m csdiar.analysis.freeze_m2b add-m3
  .venv-pyannote4/bin/python -m csdiar.analysis.freeze_m2b add-eval-hyps --hyp-root ~/data/displace/results/run_m2b
Copy the record to the host, check the code hashes there with sha256sum, commit it, and tag the commit
`m2b-freeze` (the record tag is `m2b-freeze-record`).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from csdiar.analysis.confirm_m2b import (
    EVAL_SPLIT, GUARDED, M3_GRIDS, PATH_ARGS, RECORD, REPO, SEED, THRESHOLD_GRID, TUNING_GUARDED,
    confirm_markers_exist, eval_hypothesis_files, expected_native_dirs, expected_native_files,
    m3_artifacts, sha256,
)
from csdiar.data.ingest_displace import LD_SUFFIX, SD_SUFFIX


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ helpers ----------------------------------
def _load() -> dict:
    if not RECORD.exists():
        raise SystemExit(f"refusing: {RECORD} does not exist; run `freeze_m2b init` first")
    return json.loads(RECORD.read_text())


def _guard_amendment(rec: dict, section: str) -> None:
    """Every add-* refuses if the section is already there, if confirm_m2b has started/finished, or if any
    guarded code file changed since init -- so an amendment can never adopt edited code or re-open a done run."""
    if section in rec:
        raise SystemExit(f"refusing: '{section}' is already recorded; nothing already recorded may change")
    started, done = confirm_markers_exist(Path(rec["paths"]["out"]).expanduser())
    if started or done:
        raise SystemExit("refusing: confirm_m2b has already started or finished; the freeze is closed")
    for f, h in rec["code_sha256"].items():
        if sha256(REPO / f) != h:
            raise SystemExit(f"refusing: {f} changed since the freeze (init); the record is invalid")


def _split_from_reference_names(root: Path, split: str, sd_suffix: str = SD_SUFFIX,
                                ld_suffix: str = LD_SUFFIX) -> list[str]:
    """The recording IDs of a split, taken as released from the SD reference FILE NAMES (§3 'the split as
    released (IDs from the reference file names)'). The references are never parsed here -- only their names.
    `sd_suffix`/`ld_suffix` are the init-recorded suffixes (review m2 / brief fix 7)."""
    from csdiar.data.ingest_displace import LAYOUT
    sd_dir = root / LAYOUT[split]["sd"]
    ld_dir = root / LAYOUT[split]["ld"]
    if not sd_dir.is_dir() or not ld_dir.is_dir():
        raise SystemExit(f"refusing: {split} reference folder(s) missing: {sd_dir} / {ld_dir}")
    sd_ids = sorted(p.name[: -len(sd_suffix)] for p in sd_dir.glob(f"*{sd_suffix}"))
    ld_ids = sorted(p.name[: -len(ld_suffix)] for p in ld_dir.glob(f"*{ld_suffix}"))
    if sd_ids != ld_ids:
        raise SystemExit(f"refusing: {split} speaker and language references cover different recordings "
                         f"(speaker only {sorted(set(sd_ids) - set(ld_ids))}, "
                         f"language only {sorted(set(ld_ids) - set(sd_ids))})")
    if not sd_ids:
        raise SystemExit(f"refusing: no *{sd_suffix} references in {sd_dir}")
    return sd_ids


def _reference_hashes(root: Path, split_ids: dict[str, list[str]], sd_suffix: str = SD_SUFFIX,
                      ld_suffix: str = LD_SUFFIX) -> dict:
    """sha256 of every SD and LD reference RTTM, BYTES hashed (never parsed; §3). Keyed by path string."""
    from csdiar.data.ingest_displace import LAYOUT
    out: dict = {}
    for split, ids in split_ids.items():
        sd_dir = root / LAYOUT[split]["sd"]
        ld_dir = root / LAYOUT[split]["ld"]
        for rid in ids:
            for d, suffix in ((sd_dir, sd_suffix), (ld_dir, ld_suffix)):
                p = d / f"{rid}{suffix}"
                out[str(p)] = sha256(p)
    return out


def _wav_extension(wav_dir: Path, split_ids: dict[str, list[str]]) -> str:
    """The single audio-file extension the recordings are stored under in `wav_dir` (review-v2 m5). Every
    recording ID in the dev + eval splits must resolve to exactly one file `wav_dir/<rid><ext>`, and all of them
    must share ONE extension -- a MIX is refused (the in-process eval ingest later looks each recording up by
    the one recorded extension, so an ambiguous set would silently miss files). Returns the extension including
    the leading dot (e.g. '.wav'). Refuses a missing recording or a mixed/empty set."""
    exts: set[str] = set()
    missing: list[str] = []
    for ids in split_ids.values():
        for rid in ids:
            matches = sorted(p for p in wav_dir.glob(f"{rid}.*") if p.is_file())
            if not matches:
                missing.append(str(wav_dir / f"{rid}.*"))
                continue
            rid_exts = {p.suffix for p in matches}
            if len(rid_exts) > 1:
                raise SystemExit(f"refusing: recording {rid} has multiple audio files in {wav_dir} "
                                 f"with different extensions {sorted(rid_exts)}; the freeze needs one")
            exts |= rid_exts
    if missing:
        raise SystemExit(f"refusing: wav files missing, cannot record the extension: {missing[:6]}"
                         f"{' ...' if len(missing) > 6 else ''}")
    if len(exts) != 1:
        raise SystemExit(f"refusing: the recordings mix audio extensions {sorted(exts)} in {wav_dir}; "
                         "the freeze needs exactly one")
    return next(iter(exts))


def _feature_hashes(root: Path, split_ids: dict[str, list[str]], cache_dir: Path, vad_dir: Path) -> dict:
    """sha256 of the dev and eval feature files: the stage-1 E3 caches, the LID caches, and the VAD npz per
    split (§3 'the dev and eval feature files -- stage-1 caches, LID caches, VAD'). Missing files are listed in
    the refusal rather than silently skipped, because an eval hypothesis is produced from the frozen features."""
    from csdiar.pipelines import run_m2b as R
    out: dict = {}
    missing: list[str] = []
    for split, ids in split_ids.items():
        for rid in ids:
            sp = R.stage1_cache_path(cache_dir, "e3", rid)
            lp = R.lid_cache_path(cache_dir, rid)
            for p in (sp, lp):
                if Path(p).exists():
                    out[str(p)] = sha256(Path(p))
                else:
                    missing.append(str(p))
        vp = vad_dir / f"{split}.npz"
        if vp.exists():
            out[str(vp)] = sha256(vp)
        else:
            missing.append(str(vp))
    if missing:
        raise SystemExit(f"refusing: feature files missing, cannot freeze: {missing[:6]}"
                         f"{' ...' if len(missing) > 6 else ''}")
    return out


# ------------------------------------------------------------------ init -------------------------------------
def cmd_init(args) -> int:
    if RECORD.exists():
        raise SystemExit(f"refusing: {RECORD} exists; amend it with the add-* subcommands instead")
    if len(args.commit) != 40:
        raise SystemExit("refusing: --commit must be the full 40-character host commit")
    root = Path(args.root).expanduser()
    label_map_path = Path(args.label_map).expanduser()
    if not label_map_path.exists():
        raise SystemExit(f"refusing: label map {label_map_path} does not exist")
    label_map = json.loads(label_map_path.read_text())
    sd_suffix, ld_suffix = args.sd_suffix, args.ld_suffix
    split_ids = {"dev": _split_from_reference_names(root, "dev", sd_suffix, ld_suffix),
                 EVAL_SPLIT: _split_from_reference_names(root, EVAL_SPLIT, sd_suffix, ld_suffix)}
    cache_dir = Path(args.cache_dir).expanduser()
    vad_dir = Path(args.vad_dir).expanduser()
    wav_dir = Path(args.wav_dir).expanduser()
    wav_ext = _wav_extension(wav_dir, split_ids)       # review-v2 m5: the one WAV extension, refusing a mix
    paths = {"root": str(root), "cache_dir": str(cache_dir), "vad_dir": str(vad_dir),
             "wav_dir": str(wav_dir),
             "units_dev": str(Path(args.units_dev).expanduser()),
             "units_eval": str(Path(args.units_eval).expanduser()),
             "out": str(Path(args.out).expanduser()),
             "label_map": str(label_map_path),
             "confirm_m2a": str(Path(args.confirm_m2a).expanduser())}
    for k in PATH_ARGS:
        if k not in paths:
            raise SystemExit(f"internal error: path {k} not assembled")
    rec = {
        "tag": "m2b-freeze", "record_tag": "m2b-freeze-record", "commit": args.commit, "date": now(),
        "seed": SEED,
        "split": {"dev": split_ids["dev"], "eval": split_ids[EVAL_SPLIT]},
        "label_map": label_map, "label_map_sha256": sha256(label_map_path),
        "paths": paths,
        # review m2 / brief fix 7: the SD/LD reference file suffixes init used, so the in-process eval ingest in
        # confirm_m2b reads with the same suffixes rather than the module defaults.
        "sd_suffix": args.sd_suffix,
        "ld_suffix": args.ld_suffix,
        # review-v2 m5: the WAV extension init found (one for all recordings, no mix), so the in-process eval
        # ingest in confirm_m2b reads <rid><wav_ext> instead of a hard-coded '.wav'.
        "wav_ext": wav_ext,
        "threshold_grid": list(THRESHOLD_GRID),
        "code_sha256": {f: sha256(REPO / f) for f in GUARDED},
        "feature_sha256": _feature_hashes(root, split_ids, cache_dir, vad_dir),
        "reference_sha256": _reference_hashes(root, split_ids, sd_suffix, ld_suffix),
        "confirm_m2a_sha256": sha256(Path(args.confirm_m2a).expanduser()),
        "amendments": [],
    }
    RECORD.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"wrote {RECORD} for {args.commit}: {len(rec['code_sha256'])} code files, "
          f"{len(rec['feature_sha256'])} feature files, {len(rec['reference_sha256'])} reference RTTMs, "
          f"dev {len(split_ids['dev'])} / eval {len(split_ids[EVAL_SPLIT])} recordings, wav ext {wav_ext}")
    return 0


# ------------------------------------------------------------------ add-tuning -------------------------------
def _check_tuning_code(tuning_rec: dict, rec: dict, who: str) -> None:
    """A tuning record carries its own code hashes (tune_m2b `code_sha256`; tune_p0_m2b `code_sha256`). Every
    hash it reports for a guarded file must equal the one `init` froze -- a record built from edited tuning code
    is refused (§3 'refuses a record whose own code hash differs from the frozen one')."""
    tcode = tuning_rec.get("code_sha256", {})
    if not tcode:
        raise SystemExit(f"refusing: {who} tuning record carries no code_sha256 to check against the freeze")
    for f, h in tcode.items():
        frozen = rec["code_sha256"].get(f)
        if frozen is None:
            raise SystemExit(f"refusing: {who} tuning record hashes {f}, which the freeze does not guard")
        if frozen != h:
            raise SystemExit(f"refusing: {who} tuning record's hash of {f} differs from the frozen one "
                             "(built from edited tuning code)")


def cmd_add_tuning(args) -> int:
    rec = _load()
    _guard_amendment(rec, "tuning")
    tuning = json.loads(Path(args.tuning).expanduser().read_text())
    p0 = json.loads(Path(args.p0_tuning).expanduser().read_text())
    _check_tuning_code(tuning, rec, "P1-AHC (tune_m2b)")
    _check_tuning_code(p0, rec, "P0-gate (tune_p0_m2b)")
    op = tuning.get("operating_points", {})
    # per-system clustering thresholds from the P1-AHC tuner's operating points (B0, B1, M1, B2, M3:*, M2)
    thresholds = {name: v.get("threshold") for name, v in op.items() if "threshold" in v}
    not_assessable = {name: v["not_assessable"] for name, v in op.items() if "not_assessable" in v}
    m2 = op.get("M2", {})
    rec["tuning"] = {
        "tune_m2b_sha256": sha256(Path(args.tuning).expanduser()),
        "tune_p0_m2b_sha256": sha256(Path(args.p0_tuning).expanduser()),
        "thresholds": thresholds,
        "b2_delta": tuning.get("fits", {}).get("b2_delta", {}).get("delta"),
        "m2_post_alpha": m2.get("alpha"), "m2_post_tau0": m2.get("tau0"),
        "m2_post_threshold": m2.get("threshold"),
        "m2_assessable": tuning.get("m2_assessable"),
        "p0_gate_alpha": p0.get("alpha"), "p0_gate_tau0": p0.get("tau0"),
        "p0_assessable": p0.get("assessable", p0.get("m2_assessable")),
        "fit_files": {k: v for k, v in tuning.get("fits", {}).items()},
        "not_assessable": not_assessable,
    }
    rec["amendments"].append({"date": now(), "what": "added the dev-tuning operating points (P1-AHC thresholds, "
                              "B2 delta, M2 post-processor alpha/tau0, P0-gate alpha/tau0) from the hashed "
                              "tune_m2b and tune_p0_m2b records"})
    RECORD.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"added tuning: thresholds {thresholds}, m2_post_alpha {m2.get('alpha')}, "
          f"p0_gate_alpha {p0.get('alpha')}")
    return 0


# ------------------------------------------------------------------ add-power --------------------------------
def cmd_add_power(args) -> int:
    rec = _load()
    _guard_amendment(rec, "power")
    power_path = Path(args.power).expanduser()
    power = json.loads(power_path.read_text())
    verdict = power.get("verdict")
    if verdict not in ("confirmatory", "descriptive"):
        raise SystemExit(f"refusing: power result verdict is {verdict!r}, not 'confirmatory'/'descriptive'")
    rec["power"] = {"power_m2b_sha256": sha256(power_path), "path": str(power_path), "verdict": verdict}
    rec["amendments"].append({"date": now(), "what": f"added the H5 power verdict '{verdict}' from the hashed "
                              "power_m2b result"})
    RECORD.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"added power verdict '{verdict}'")
    return 0


# ------------------------------------------------------------------ add-m3 -----------------------------------
def cmd_add_m3(_args) -> int:
    rec = _load()
    _guard_amendment(rec, "m3_artifacts_sha256")
    arts = m3_artifacts(rec)
    rec["m3_artifacts_sha256"] = {f: sha256(Path(f).expanduser()) for f in arts}
    rec["amendments"].append({"date": now(), "what": f"added the hashes of the {len(arts)} M3 artifacts "
                              "(the 5 all-data checkpoints and their per-seed DISPLACE eval stage-1 caches)"})
    RECORD.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"added {len(rec['m3_artifacts_sha256'])} M3 artifact hashes")
    return 0


# ------------------------------------------------------------------ add-eval-hyps ----------------------------
def cmd_add_eval_hyps(args) -> int:
    rec = _load()
    _guard_amendment(rec, "eval_hyps_sha256")
    if "tuning" not in rec:
        raise SystemExit("refusing: add-tuning must run before add-eval-hyps (the frozen thresholds name the "
                         "hypothesis files)")
    hyp_root = Path(args.hyp_root).expanduser()
    files = eval_hypothesis_files(hyp_root, rec)
    missing = [str(p) for p in files if not Path(p).exists()]
    native_absent = bool(getattr(args, "native_absent", False))
    native_files = expected_native_files(hyp_root, rec)          # {'P1': [...], 'P2': [...]}
    if not native_absent:
        # M2 / brief fix 2: hash every per-recording native RTTM, refusing a missing one (not just a missing
        # directory), so a native hypothesis cannot change/regenerate after the freeze undetected.
        for p in ("P1", "P2"):
            missing += [str(f) for f in native_files[p] if not Path(f).exists()]
    if missing:
        raise SystemExit(f"refusing: {len(missing)} expected eval hypothesis file(s) missing: "
                         f"{missing[:6]}{' ...' if len(missing) > 6 else ''}")
    rec["eval_hyps_sha256"] = {str(p): sha256(Path(p)) for p in files}
    # M1 / brief fix 1: pin the hypothesis root the confirm run must read from.
    rec.setdefault("paths", {})["hyp_root"] = str(hyp_root)
    if native_absent:
        rec["native_absent"] = True
        rec["eval_native_dirs"] = [str(d) for d in expected_native_dirs(hyp_root, rec)]
        rec["amendments"].append({"date": now(), "what": f"added the hashes of {len(files)} eval hypothesis "
                                  "RTTMs over the clustering grid; froze with the native P1/P2 hypotheses "
                                  "ABSENT (--native-absent), so confirm_m2b reports the pipeline H2 not done"})
        RECORD.write_text(json.dumps(rec, indent=1) + "\n")
        print(f"added {len(rec['eval_hyps_sha256'])} eval hypothesis hashes; native P1/P2 ABSENT")
        return 0
    native_hashes = {str(f): sha256(Path(f)) for p in ("P1", "P2") for f in native_files[p]}
    rec["native_hyps_sha256"] = native_hashes
    rec["native_absent"] = False
    rec["eval_native_dirs"] = [str(d) for d in expected_native_dirs(hyp_root, rec)]
    rec["amendments"].append({"date": now(), "what": f"added the hashes of {len(files)} eval hypothesis RTTMs "
                              f"over the clustering grid and {len(native_hashes)} native P1/P2 eval RTTMs"})
    RECORD.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"added {len(rec['eval_hyps_sha256'])} eval hypothesis hashes and "
          f"{len(native_hashes)} native P1/P2 hashes")
    return 0


# ------------------------------------------------------------------ main -------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("init")
    i.add_argument("--commit", required=True)
    i.add_argument("--root", required=True, help="DISPLACE archive root (reference RTTMs; §2 layout)")
    i.add_argument("--cache-dir", dest="cache_dir", required=True, help="run_m2b stage-1 + LID caches")
    i.add_argument("--vad-dir", dest="vad_dir", required=True, help="VAD npz per split (<split>.npz)")
    i.add_argument("--wav-dir", dest="wav_dir", required=True, help="recording wavs")
    i.add_argument("--units-dev", dest="units_dev", required=True, help="ingest_displace dev output dir")
    i.add_argument("--units-eval", dest="units_eval", required=True, help="ingest_displace eval output dir")
    i.add_argument("--out", required=True, help="confirm_m2b output dir (markers live here)")
    i.add_argument("--label-map", dest="label_map", required=True, help="LD label -> VoxLingua107 code JSON")
    i.add_argument("--confirm-m2a", dest="confirm_m2a", required=True,
                   help="frozen M2a result (the H2 family reads its P0 D1 p-values; input, hashed)")
    i.add_argument("--sd-suffix", dest="sd_suffix", default=SD_SUFFIX,
                   help=f"SD reference file suffix (default {SD_SUFFIX}); recorded and reused by the eval ingest")
    i.add_argument("--ld-suffix", dest="ld_suffix", default=LD_SUFFIX,
                   help=f"LD reference file suffix (default {LD_SUFFIX}); recorded and reused by the eval ingest")

    t = sub.add_parser("add-tuning")
    t.add_argument("--tuning", required=True, help="tune_m2b dev record")
    t.add_argument("--p0-tuning", dest="p0_tuning", required=True, help="tune_p0_m2b dev record")

    p = sub.add_parser("add-power")
    p.add_argument("--power", required=True, help="power_m2b H5 power result (carries 'verdict')")

    sub.add_parser("add-m3")

    e = sub.add_parser("add-eval-hyps")
    e.add_argument("--hyp-root", dest="hyp_root", required=True, help="run_m2b eval output root")
    e.add_argument("--native-absent", dest="native_absent", action="store_true",
                   help="freeze with the native P1/P2 eval hypotheses ABSENT (their runners do not exist yet); "
                        "confirm_m2b then reports the pipeline H2 as not done and reads nothing from the native "
                        "dirs")

    args = ap.parse_args(argv)
    return {"init": cmd_init, "add-tuning": cmd_add_tuning, "add-power": cmd_add_power,
            "add-m3": cmd_add_m3, "add-eval-hyps": cmd_add_eval_hyps}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
