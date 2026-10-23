#!/usr/bin/env python3
"""H4 on D1: does each mitigation lower the SS-DL excess of P0 + E3 against B0? (PREREGISTRATION_H4.md)

Stage 1 tests M1 and M3 (M2 enters Holm with p = 1 until its alpha is frozen in M2b); stage 2 adds M2 once.
- Folds: the 15 train channels in 5 folds of 3 (csdiar/mitigations/folds.py). Fitting side of fold f = the other
  12 train channels. Calibration = the 3 pilot channels without video zJYS-XNDP2Q (decision 27).
- Systems on the E3 dense grid (L = 1 s): B0 unchanged; B1 language-wise mean centring with LID-predicted window
  languages; B2 the verification-score offset gated by LID-predicted language change; M1 the projector of
  m1_subspace.py; M3 the fold's fine-tuned models (5 seeds, mean of per-seed delta-FAR); M2 the gated curve
  d - alpha * lambda (stage 2).
- tau per system, fold and seed: 10% of the calibration channels' det_w1 controls fire (pilot.calibrate_tau).
- Per scored channel: pilot.h2_cell's cell-standardised delta_far_std in the p0L1 cell. delta_M(c) =
  dFAR_M(c) - dFAR_B0(c). Exact one-sided sign-flip (mean delta < 0), Holm over M1-M3, BCa 98.3% / TOST 96.7%
  for mean delta, smallest effect 0.25 * S_B0 with S_B0 = mean dFAR_B0 held fixed; outcome categories of stats.py.
Without --dry-run the script refuses to run unless planning/FREEZE_H4.json matches every input, and runs once per
stage. --dry-run exercises the code on the calibration channels only and reports structure, not effects.
Usage (on the box, .venv-pyannote4):
  .venv-pyannote4/bin/python -m csdiar.analysis.confirm_h4 --stage 1 --out ~/data/d1/results/h4
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from csdiar.analysis.confirm import load_points, mean_stat
from csdiar.analysis.pilot import calibrate_tau, h2_cell, score_points, unit
from csdiar.analysis.stats import bca_ci, holm, outcome, signflip
from csdiar.mitigations.folds import fitting_channels, h4_folds, train_channels
from csdiar.mitigations.m1_subspace import estimate_v_lang, project_out

REPO = Path(__file__).resolve().parents[2]
FREEZE = REPO / "planning/FREEZE_H4.json"
SEED = 20261018
ALPHA = 0.05
M_FAMILY = 3
MIN_UNITS = 10
SESOI_REL = 0.25
M3_SEEDS = (0, 1, 2, 3, 4)
M3_PRIMARY = "w0.3_none"
LAG = 2.0
B2_PAIRS = 40000
B2_MAX = 0.5
CALIB_EXCLUDE_VIDEO = "zJYS-XNDP2Q"
LAG_K = 10                                   # d[k] compares windows k - 10 and k (1 s apart on the 0.1 s grid)
GUARDED = ("csdiar/analysis/confirm_h4.py", "csdiar/analysis/freeze_h4.py", "csdiar/analysis/confirm.py",
           "csdiar/analysis/pilot.py", "csdiar/analysis/stats.py", "csdiar/analysis/h3.py",
           "csdiar/analysis/purity_audit.py", "csdiar/mitigations/folds.py", "csdiar/mitigations/m1_subspace.py",
           "csdiar/mitigations/m3_train.py", "csdiar/mitigations/m3_embed.py", "csdiar/mitigations/m3_data.py",
           "csdiar/models/e3.py", "csdiar/models/embed.py", "csdiar/data/boundaries.py", "csdiar/data/tags.py",
           "third_party/unispeech/ecapa_tdnn.py", "planning/PREREGISTRATION_H4.md")
PATH_ARGS = ("manifest", "boundaries", "point_emb", "emb", "lid", "windows", "m3_emb", "out")


def m3_artifacts() -> list[str]:
    """The 25 primary M3 checkpoints and their dense grids, which the freeze record must hash before stage 1."""
    out = []
    for f in range(5):
        for s in M3_SEEDS:
            name = f"f{f}_s{s}_{M3_PRIMARY}"
            out.append(f"~/data/m3/runs/{name}/best.pt")
            out += [f"~/data/m3/emb/m3_{name}_L1/{sp}.npz" for sp in ("validation", "train")]
    return out


# ------------------------------------------------------------------ inputs -----------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def load_grid(path: Path, splits, clips: set[str] | None = None) -> dict[str, np.ndarray]:
    """clip_id -> unit float32 (n_windows, dim) from <path>/<split>.npz (the dense L = 1 s grid), optionally only
    for the given clips (npz members are read lazily)."""
    out = {}
    for s in splits:
        with np.load(path / f"{s}.npz") as z:
            keys = z.files if clips is None else [c for c in z.files if c in clips]
            out.update({c: unit(z[c].astype(np.float32)) for c in keys})
    return out


def load_lid(path: Path, splits) -> dict[str, np.ndarray]:
    """clip_id -> renormalised (p_ne, p_en) per grid window, from lid/<split>.npz columns [centre, ne, en, other]."""
    out = {}
    for s in splits:
        with np.load(path / f"{s}.npz") as z:
            for c in z.files:
                p = z[c][:, 1:3].astype(np.float64)
                out[c] = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
    return out


def read_windows(path: Path, split: str) -> list[dict]:
    with gzip.open(path / f"windows_{split}.csv.gz", "rt", encoding="utf-8") as fh:
        return [w for w in csv.DictReader(fh) if w["lang"] in ("ne", "en")]


def curve(e: np.ndarray) -> np.ndarray:
    d = np.full(len(e), np.nan, np.float32)
    if len(e) > LAG_K:
        d[LAG_K:] = 1 - (e[LAG_K:] * e[:-LAG_K]).sum(1)
    return d


def predicted_lang(lid: np.ndarray, n: int) -> np.ndarray:
    """0 = ne, 1 = en per grid window (ne if renormalised p(ne) >= 0.5). Grids must have equal length."""
    if len(lid) != n:
        raise ValueError(f"LID grid has {len(lid)} windows, embeddings {n}")
    return (lid[:, 0] < 0.5).astype(np.int8)


# ------------------------------------------------------------------ systems ----------------------------------
def b1_means(grid: dict, windows: list[dict], fit: set[str]) -> dict[int, np.ndarray]:
    acc = {0: [], 1: []}
    for w in windows:
        if w["speaker_id"] in fit and w["clip_id"] in grid:
            acc[0 if w["lang"] == "ne" else 1].append(grid[w["clip_id"]][int(w["k"])])
    return {l: np.mean(v, 0) for l, v in acc.items()}


def b2_delta(grid: dict, windows: list[dict], fit: set[str]) -> dict:
    """Mean cosine of same-video same-language reference-window pairs minus cross-language pairs, starts >= LAG s
    apart, at most B2_PAIRS pairs per video drawn with the seed; clamped to [0, B2_MAX]."""
    rng = np.random.default_rng(SEED)
    by = defaultdict(list)
    for w in windows:
        if w["speaker_id"] in fit and w["clip_id"] in grid:
            by[w["video_id"]].append(w)
    same, cross = [], []
    for v in sorted(by):
        ws = by[v]
        e = np.stack([grid[w["clip_id"]][int(w["k"])] for w in ws])
        t = np.array([float(w["abs_t0"]) for w in ws])
        lang = np.array([w["lang"] for w in ws])
        a, b = np.triu_indices(len(ws), 1)
        ok = np.abs(t[a] - t[b]) >= LAG
        a, b = a[ok], b[ok]
        if len(a) > B2_PAIRS:
            pick = np.sort(rng.choice(len(a), size=B2_PAIRS, replace=False))
            a, b = a[pick], b[pick]
        cos = (e[a] * e[b]).sum(1)
        same.append(cos[lang[a] == lang[b]]); cross.append(cos[lang[a] != lang[b]])
    s, c = np.concatenate(same), np.concatenate(cross)
    raw = float(s.mean() - c.mean())
    return {"delta": float(min(max(raw, 0.0), B2_MAX)), "raw": raw, "n_same": int(len(s)), "n_cross": int(len(c))}


def jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    kl = lambda a, b: np.where(a > 0, a * np.log2(np.maximum(a, 1e-12) / np.maximum(b, 1e-12)), 0).sum(-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def system_curves(name: str, clips: set[str], grid: dict, lid: dict, fitted: dict) -> dict[str, np.ndarray]:
    """P0 curves of one system on the given clips."""
    out = {}
    for c in clips:
        e = grid.get(c)
        if e is None:
            continue
        if name == "B0":
            out[c] = curve(e)
        elif name == "B1":
            lang = predicted_lang(lid[c], len(e))
            mu = np.where(lang[:, None] == 0, fitted["mu"][0][None, :], fitted["mu"][1][None, :])
            out[c] = curve(unit(e - mu))
        elif name == "B2":
            d = curve(e)
            lang = predicted_lang(lid[c], len(e))
            if len(e) > LAG_K:
                d[LAG_K:] -= fitted["delta"] * (lang[LAG_K:] != lang[:-LAG_K])
            out[c] = d
        elif name == "M1":
            out[c] = curve(project_out(e, fitted["v"]))
        elif name == "M2":
            d = curve(e)
            if len(e) > LAG_K:
                p = lid[c]
                if len(p) != len(e):
                    raise ValueError(f"{c}: LID and embedding grids differ")
                d[LAG_K:] -= fitted["alpha"] * jsd(p[:-LAG_K], p[LAG_K:]).astype(np.float32)
            out[c] = d
        else:
            raise ValueError(name)
    return out


# ------------------------------------------------------------------ scoring ----------------------------------
def cell_stats(points: dict, curves: dict, tau: float) -> dict:
    vals, _ = score_points(points, curves)
    return h2_cell(points, vals, tau, "p0L1", "det_w1")["channels"]


def tau_on(calib: dict, curves: dict) -> float:
    vals, _ = score_points(calib, curves)
    return calibrate_tau(calib, vals)[0]


def test_family(deltas: dict[str, dict[str, float]], s_b0: float, rng, staged: set[str],
                not_assessable: set[str] = frozenset(), carried: dict | None = None) -> dict:
    """deltas: mitigation -> {channel: delta}. Holm over M1-M3. Staged and not-assessable members enter with
    p = 1. `carried` holds stage-1 results whose verdicts are final: their raw p-values enter Holm, and their
    entries are copied unchanged."""
    names = ["M1", "M2", "M3"]
    res = {}
    for n in names:
        if carried and n in carried:
            res[n] = {**carried[n], "carried_from_stage1": True}
            continue
        if n in not_assessable:
            res[n] = {"not_assessable": True}
            continue
        if n in staged or n not in deltas:
            res[n] = {"staged": True}
            continue
        x = [deltas[n][c] for c in sorted(deltas[n])]
        sf = signflip([-v for v in x], rng)
        res[n] = {"n_channels": len(x), "mean_delta": float(np.mean(x)), "relative_reduction":
                  float(-np.mean(x) / s_b0) if s_b0 > 0 else None, "signflip": sf,
                  "ci": bca_ci(x, mean_stat, 1 - ALPHA / M_FAMILY, rng),
                  "tost": bca_ci(x, mean_stat, 1 - 2 * ALPHA / M_FAMILY, rng), "per_channel": deltas[n]}
    p1 = [res[n]["signflip"]["p_one"] if "signflip" in res[n] else 1.0 for n in names]
    p2 = [res[n]["signflip"]["p_two"] if "signflip" in res[n] else 1.0 for n in names]
    a1, a2 = holm(p1), holm(p2)
    for i, n in enumerate(names):
        if "signflip" not in res[n] or res[n].get("carried_from_stage1"):
            continue
        r = res[n]
        r["p_one_holm"], r["p_two_holm"] = a1[i], a2[i]
        if s_b0 > 0:
            r["outcome"] = outcome(a1[i], a2[i], r["mean_delta"], r["tost"]["lo"], r["tost"]["hi"],
                                   SESOI_REL * s_b0, direction=-1)
        else:
            r["outcome"] = {"category": "inconclusive", "reason": "S_B0 <= 0: no excess to reduce"}
    return res


def secondary_vs(deltas_from: dict[str, float], deltas_to: dict[str, float], rng) -> dict:
    common = sorted(set(deltas_from) & set(deltas_to))
    x = [deltas_from[c] - deltas_to[c] for c in common]
    return {"n_channels": len(x), "mean": float(np.mean(x)) if x else None,
            "p_one_unadjusted": signflip([-v for v in x], rng)["p_one"] if x else None}


# ------------------------------------------------------------------ freeze -----------------------------------
def check_freeze(stage: int, args) -> dict:
    if not FREEZE.exists():
        raise SystemExit("refusing: planning/FREEZE_H4.json does not exist")
    rec = json.loads(FREEZE.read_text())
    for f in GUARDED:
        if f not in rec["code_sha256"]:
            raise SystemExit(f"refusing: {f} is not covered by the freeze record")
    for f, h in rec["code_sha256"].items():
        if sha256(REPO / f) != h:
            raise SystemExit(f"refusing: {f} changed since the freeze")
    if stage == 1 and sorted(rec.get("m3_artifacts_sha256", {})) != sorted(m3_artifacts()):
        raise SystemExit("refusing: the record does not hash every primary M3 artifact (freeze_h4 add-m3)")
    for f, h in {**rec["feature_sha256"], **rec.get("m3_artifacts_sha256", {})}.items():
        if sha256(Path(f).expanduser()) != h:
            raise SystemExit(f"refusing: {f} changed since the freeze")
    for k in PATH_ARGS:
        given = Path(getattr(args, k)).expanduser().resolve()
        if given != Path(rec["paths"][k]).expanduser().resolve():
            raise SystemExit(f"refusing: --{k.replace('_', '-')} {given} is not the frozen input {rec['paths'][k]}")
    if rec.get("seed") != SEED:
        raise SystemExit("refusing: the freeze record's seed differs from the code's")
    if stage == 2 and "m2_alpha" not in rec:
        raise SystemExit("refusing: stage 2 needs the M2b-frozen alpha in the freeze record")
    if (args.out / f"confirm_h4_stage{stage}.json").exists() or (args.out / f"confirm_h4_stage{stage}.started").exists():
        raise SystemExit(f"refusing: stage {stage} has already run or started (one-shot)")
    if stage == 2:
        s1 = args.out / "confirm_h4_stage1.json"
        if not s1.exists() or rec.get("stage1_sha256") != sha256(s1):
            raise SystemExit("refusing: stage 2 needs the stage-1 result, hashed in the record (freeze_h4 add-stage1)")
    return rec


# ------------------------------------------------------------------ main -------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    home = Path.home() / "data/d1"
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    ap.add_argument("--manifest", type=Path, default=home / "manifest.csv")
    ap.add_argument("--boundaries", type=Path, default=home / "boundaries/lex_v1b")
    ap.add_argument("--point-emb", type=Path, default=home / "point_emb/lex_v1b")
    ap.add_argument("--emb", type=Path, default=home / "emb")
    ap.add_argument("--lid", type=Path, default=home / "lid")
    ap.add_argument("--windows", type=Path, default=home / "windows/lex_v1")
    ap.add_argument("--m3-emb", type=Path, default=Path.home() / "data/m3/emb")
    ap.add_argument("--out", type=Path, default=home / "results/h4")
    ap.add_argument("--dry-run", action="store_true", help="calibration channels only; structure, not effects")
    ap.add_argument("--dry-m3", default=None, help="dry run: name of one M3 grid to exercise the M3 path")
    args = ap.parse_args()

    rec = None if args.dry_run else check_freeze(args.stage, args)
    if rec is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"confirm_h4_stage{args.stage}.started").write_text(json.dumps({"commit": rec["commit"]}))
    rng = np.random.default_rng(SEED)
    splits = ("validation", "train")
    points, _ = load_points(args.boundaries, list(splits))
    split_of = {r["clip_id"]: r["split"] for r in csv.DictReader(open(args.manifest, encoding="utf-8"))}
    chans = train_channels(args.manifest)
    folds = h4_folds(chans)
    if rec is not None and rec["folds"] != folds:
        raise SystemExit("refusing: fold assignment differs from the freeze record")
    calib = {p: r for p, r in points.items()
             if split_of.get(r["clip_id"]) == "validation" and r["video_id"] != CALIB_EXCLUDE_VIDEO}
    train_pts = {p: r for p, r in points.items() if split_of.get(r["clip_id"]) == "train"}
    grid = load_grid(args.emb / "e3_L1", splits)
    lid = load_lid(args.lid, splits)
    windows = read_windows(args.windows, "train")
    calib_clips = {r["clip_id"] for r in calib.values()}

    if args.dry_run:
        fold_list = [(None, sorted({r["speaker_id"] for r in calib.values()}))]
    else:
        fold_list = list(enumerate(folds))
    # Stage 2 adds M2 only: the stage-1 verdicts of M1 and M3 are final (PREREGISTRATION_H4.md §4).
    systems = ["B0", "B1", "B2", "M1"] if args.stage == 1 else ["B0", "M2"]
    dfar = defaultdict(dict)          # system -> channel -> dFAR (M3: seed mean)
    nunits, extra = {}, defaultdict(dict)
    not_assessable: dict[str, list] = defaultdict(list)
    fitted_log, m3_dropped = {}, {}
    for f, scored in fold_list:
        fit = set(fitting_channels(chans, f)) if f is not None else set(chans)
        scored_pts = calib if f is None else {p: r for p, r in train_pts.items() if r["speaker_id"] in scored}
        clips = calib_clips | {r["clip_id"] for r in scored_pts.values()}
        fitted = {}
        if args.stage == 1:
            b2 = b2_delta(grid, windows, fit)
            fitted = {"mu": b1_means(grid, windows, fit), "delta": b2["delta"],
                      "v": estimate_v_lang(points, args.point_emb, "e3", 1.0, ("train",), speakers=fit)}
            fitted_log[str(f)] = {"b2": b2, "m1_v_norm": float(np.linalg.norm(fitted["v"]))}
        else:
            fitted["alpha"] = float(rec["m2_alpha"]) if rec is not None else 0.0
            fitted_log[str(f)] = {"m2_alpha": fitted["alpha"]}
        for s in systems:
            cv = system_curves(s, clips, grid, lid, fitted)
            tau = tau_on(calib, cv)
            if not np.isfinite(tau):
                not_assessable[s].append({"fold": f, "reason": "tau = -inf"})
                continue
            st = cell_stats(scored_pts, cv, tau)
            for c in scored:
                if c in st:
                    dfar[s][c] = st[c]["delta_far_std"]
                    extra[s][c] = {"tau": tau, "auc": st[c]["auc"], "far_ssdl": st[c]["far_ssdl"],
                                   "far_ctrl_std": st[c]["far_ctrl_std"]}
                    if s == "B0":
                        nunits[c] = st[c]["n_ssdl"]
        if args.stage != 1:
            continue
        m3_names = ([args.dry_m3] if args.dry_m3 else []) if f is None else \
            [f"m3_f{f}_s{sd}_{M3_PRIMARY}" for sd in M3_SEEDS]
        per_seed = defaultdict(list)
        for nm in m3_names:
            gmeta = json.loads((args.m3_emb / f"{nm}_L1" / "meta.json").read_text())
            if f is not None and gmeta.get("m3_meta", {}).get("fold") != f:
                raise SystemExit(f"refusing: grid {nm} was not trained for fold {f}")
            g = load_grid(args.m3_emb / f"{nm}_L1", splits, clips)
            cv = {c: curve(g[c]) for c in clips if c in g}
            tau = tau_on(calib, cv)
            if not np.isfinite(tau):
                not_assessable["M3"].append({"fold": f, "grid": nm, "reason": "tau = -inf"})
                continue
            st = cell_stats(scored_pts, cv, tau)
            for c in scored:
                if c in st and st[c]["delta_far_std"] is not None:
                    per_seed[c].append(st[c]["delta_far_std"])
                    extra["M3_seeds"].setdefault(c, []).append({"grid": nm, "tau": tau, "dfar": st[c]["delta_far_std"]})
        for c in scored:
            v = per_seed.get(c, [])
            if m3_names and len(v) == len(m3_names):
                dfar["M3"][c] = float(np.mean(v))
            elif m3_names:
                m3_dropped[c] = f"{len(v)} of {len(m3_names)} seeds have an estimate"

    if "B0" in not_assessable:
        args.out.mkdir(parents=True, exist_ok=True)
        name = f"confirm_h4_stage{args.stage}{'_dryrun' if args.dry_run else ''}.json"
        (args.out / name).write_text(json.dumps({"stage": args.stage, "not_done": "B0 is not assessable (tau = -inf)",
                                                 "not_assessable": dict(not_assessable)}, indent=1))
        raise SystemExit("B0 is not assessable (tau = -inf): H4 is reported as not done")
    included = sorted(c for c, n in nunits.items() if n >= MIN_UNITS and dfar["B0"].get(c) is not None)
    s_b0 = float(np.mean([dfar["B0"][c] for c in included])) if included else float("nan")
    deltas = {m: {c: dfar[m][c] - dfar["B0"][c] for c in included if dfar[m].get(c) is not None}
              for m in ("M1", "M2", "M3") if m in dfar}
    carried = None
    if args.stage == 2:
        s1 = json.loads((args.out / "confirm_h4_stage1.json").read_text())
        carried = {m: s1["family"][m] for m in ("M1", "M3")}
    staged = {"M2"} if args.stage == 1 else set()
    family_res = test_family(deltas, s_b0, rng, staged, set(not_assessable) & {"M1", "M2", "M3"}, carried)
    secondary = {}
    if args.stage == 1:
        base_d = {b: {c: dfar[b][c] - dfar["B0"][c] for c in included if dfar[b].get(c) is not None}
                  for b in ("B1", "B2")}
        secondary = {f"{b}_vs_B0": secondary_vs(base_d[b], {c: 0.0 for c in base_d[b]}, rng) for b in ("B1", "B2")}
        for m in deltas:
            for b in ("B1", "B2"):
                secondary[f"{m}_vs_{b}"] = secondary_vs(deltas[m], base_d[b], rng)
    res = {"stage": args.stage, "dry_run": args.dry_run, "folds": folds, "included": included,
           "excluded": {c: n for c, n in nunits.items() if c not in included}, "S_B0": s_b0,
           "sesoi": SESOI_REL * s_b0 if s_b0 == s_b0 else None, "family": family_res, "secondary": secondary,
           "not_assessable": dict(not_assessable), "m3_channels_dropped": m3_dropped,
           "dfar": {s: dict(v) for s, v in dfar.items()}, "per_channel_extra": {s: dict(v) for s, v in extra.items()},
           "fitted": fitted_log, "freeze": None if rec is None else {k: rec.get(k) for k in ("tag", "commit", "date")}}
    args.out.mkdir(parents=True, exist_ok=True)
    name = f"confirm_h4_stage{args.stage}{'_dryrun' if args.dry_run else ''}.json"
    (args.out / name).write_text(json.dumps(res, indent=1, default=float))
    if args.dry_run:
        print(json.dumps({"written": str(args.out / name), "systems": sorted(dfar), "channels": included,
                          "fitted_keys": sorted(fitted_log), "not_assessable": sorted(not_assessable),
                          "m3_channels_dropped": len(m3_dropped)}))
    else:
        for n, r in family_res.items():
            label = r.get("outcome", {}).get("category") if "outcome" in r else \
                ("staged" if r.get("staged") else "not assessable")
            print(n, r.get("mean_delta"), r.get("p_one_holm"), label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
