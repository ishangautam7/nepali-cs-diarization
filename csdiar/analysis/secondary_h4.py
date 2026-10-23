#!/usr/bin/env python3
"""H4 secondary analyses on D1 (PREREGISTRATION_H4.md §8): reported, never tested for support.

Runs after stage 1 of csdiar/analysis/confirm_h4.py (the stage-1 result must exist and be hashed in
planning/FREEZE_H4.json), once per scope and part. Every system goes through the frozen P0 code as in confirm_h4:
tau on the calibration channels (10% of their det_w1 controls fire), then pilot.h2_cell's cell-standardised
delta-FAR in the p0L1 cell per scored channel. Channels enter with >= 10 p0L1 switch points that have cell controls
(B0's count, as in confirm_h4). A comparison of S with R reports mean_c [dFAR_S(c) - dFAR_R(c)], the relative
reduction against S_B0 = mean_c dFAR_B0(c), a two-sided 95% BCa interval and unadjusted sign-flip p-values. Where
stage 1 has the same quantity (B0's tau; M1's and every primary M3 grid's tau and per-channel dFAR in the cv scope)
it is recomputed and must agree, or the script stops.

Scopes:
  cv    the 15 train channels in the 5 H4 folds; fitting side = the fold's other 12 train channels.
  test  the 2 test channels (descriptive, PLAN.md §6 E-B (iii)): systems fitted on all 15 train channels, M3 = the
        5 all-data models.
Parts (one output each: <out>/secondary_h4_<scope>_<part>.json):
  lambda   AUC of lambda (M2's change score: base-2 JSD of the renormalised LID posteriors of windows k-10 and k)
           at p0L1 switch points against det_w1 cell controls, and of phrase-boundary against other det_w1
           controls (does lambda track pauses?). No alpha is used.
  m1k      M1 with a k-dimensional subspace: the top-k right singular vectors of the fitting side's per-(speaker,
           video) [mean en - mean ne] switch-point window differences (the units M1 averages), k in {1, 2, 4, 8,
           16} chosen inside each fold by inner cross-validation over the fitting channels (triples of a seeded
           permutation; the H4 folds in the test scope) on mean dFAR, ties to the smaller k. Specificity controls at
           the chosen k: random subspaces (5 draws, dFAR averaged) and the top-k PCA directions of the fitting
           side's switch-point windows. Every fixed k is reported too.
  emb      B1, B2 and M1 with E1 and E2, each against its own B0.
  sens     PREREGISTRATION.md §5 sensitivities of the primary comparison (B0, M1, M3 at their primary taus): the
           scored points restricted to the purity-verified subset (item 4; split-off videos become clusters) and
           to the alignment-score filter (item 1).
  m3       the primary configuration's per-seed spread and operating point; the distillation curve (w_dist 0, 0.1
           and 1 at seed 0, against the seed-0 primary) and the ablations -GRL, -contrastive, -both, -D5 (seed 0).
  speaker  speaker information kept, per embedding (E1, E2, E3 = B0, B1, M1, M1-k, M3): same- and cross-language
           EER on the scored channels' clean windows (H3's trial rules), a linear probe's language accuracy (trained
           on the fitting side's windows, tested on the scored channels') and cross-session speaker-identification
           accuracy over the scored and calibration channels pooled into one classifier (no system is fitted or
           trained on either; each channel's even-ranked videos train the probe, its odd-ranked videos test it),
           both balanced accuracies.
M2's variants (oracle and time-shuffled lambda) and M2 with E1/E2 need alpha (M2b); the GLMM and P1-AHC on clips
are separate scripts.
--dry-run scores the calibration channels only, with systems fitted on train-channel features, and computes no
outcome on a train or test channel (m1k's inner cross-validation is skipped and k = 1 is used).
Usage (on the box, .venv-pyannote4):
  .venv-pyannote4/bin/python -m csdiar.analysis.secondary_h4 --scope cv
  .venv-pyannote4/bin/python -m csdiar.analysis.secondary_h4 --scope test
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from csdiar.analysis import confirm_h4 as h4
from csdiar.analysis.confirm import alignment_filter, load_points, mean_stat
from csdiar.analysis.h3 import eer_mindcf
from csdiar.analysis.pilot import HOP, stratified_auc
from csdiar.analysis.purity_sensitivity import apply_subset
from csdiar.analysis.stats import bca_ci, signflip
from csdiar.mitigations.folds import fitting_channels, h4_folds, train_channels
from csdiar.mitigations.m1_subspace import estimate_v_lang, language_windows, project_out

SEED = h4.SEED
MIN_UNITS = h4.MIN_UNITS
KS = (1, 2, 4, 8, 16)
N_RANDOM = 5
M3_CONFIGS = ("w0_none", "w0.1_none", "w1_none", "w0.3_nogrl", "w0.3_nocon", "w0.3_noboth", "w0.3_nod5")
PARTS = ("lambda", "m1k", "emb", "sens", "m3", "speaker")
LAG_S = 2.0              # H3 trial rule: same-video windows whose starts are >= 2 s apart
TRIAL_PAIRS = 40000      # target pairs drawn per video at most (H3)
NON_DRAWS = 200000       # non-target pair draws per unit (H3)
PROBE_MAX = 20000        # fitting-side windows that train the language probe
KEEP = ("n_ssdl", "auc", "far_ssdl", "far_ctrl_std", "delta_far_std")
RATES = ("auc", "far_ssdl", "far_ctrl_std", "delta_far_std")
_HASHES: dict[str, str] = {}


# ------------------------------------------------------------------ helpers ----------------------------------
def p0_points(points: dict) -> dict:
    """The points P0's p0L1 cell and its tau read: p0L1 switch points and det_w1 controls. Scoring only these gives
    the same tau and p0L1 channel statistics as scoring every detection point (pilot.calibrate_tau reads det_w1
    controls only; pilot.h2_cell's p0L1 cell reads p0L1 switch points and det_w1 controls only)."""
    return {p: r for p, r in points.items()
            if (r["kind"] == "ssdl" and r["elig_p0L1"]) or (r["kind"] == "ctrl" and r["elig_det_w1"])}


def inner_folds(fit: list[str]) -> list[list[str]]:
    """Inner cross-validation triples over a fold's fitting channels: a seeded permutation of the sorted IDs."""
    if not fit or len(fit) % 3:
        raise ValueError(f"inner folds need a multiple of 3 fitting channels, got {len(fit)}")
    order = [sorted(fit)[i] for i in np.random.default_rng(SEED).permutation(len(fit))]
    return [order[i : i + 3] for i in range(0, len(order), 3)]


def choose_k(inner: dict) -> tuple[int, bool]:
    """The k with the lowest mean inner dFAR, ties to the smaller k. With no usable k at all, k = 1 and the
    fallback is flagged in the output."""
    usable = [k for k in KS if inner.get(k) is not None]
    if not usable:
        return KS[0], True
    return min(usable, key=lambda k: (inner[k], k)), False


def session_diffs(by: dict, speakers: set) -> np.ndarray:
    """Per (speaker, video) of the given speakers: mean en-side minus mean ne-side unit window (M1's units)."""
    rows = [np.mean(d["en"], 0) - np.mean(d["ne"], 0) for (s, _v), d in sorted(by.items())
            if s in speakers and d["ne"] and d["en"]]
    return np.stack(rows).astype(np.float64)


def session_windows(by: dict, speakers: set) -> np.ndarray:
    return np.concatenate([np.stack(d["ne"] + d["en"]) for (s, _v), d in sorted(by.items())
                           if s in speakers and (d["ne"] or d["en"])])


def nap_basis(diffs: np.ndarray, k: int) -> np.ndarray:
    """Top-k right singular vectors of the uncentred difference vectors, as (dim, k) orthonormal columns."""
    if len(diffs) < k:
        raise SystemExit(f"refusing: {len(diffs)} difference vectors cannot span k = {k}")
    _, _, vt = np.linalg.svd(np.asarray(diffs, np.float64), full_matrices=False)
    return vt[:k].T.astype(np.float32)


def pca_basis(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x, np.float64)
    _, _, vt = np.linalg.svd(x - x.mean(0), full_matrices=False)
    return vt[:k].T.astype(np.float32)


def random_basis(dim: int, k: int, rng) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((dim, k)))
    return q.astype(np.float32)


def project_basis(e: np.ndarray, u: np.ndarray) -> np.ndarray:
    """e' = unit(e - (e U) U^T): remove the span of U's orthonormal columns, then re-normalise."""
    e = e.astype(np.float32)
    return h4.unit(e - (e @ u) @ u.T)


def basis_curves(clips: set, grid: dict, u: np.ndarray) -> dict:
    return {c: h4.curve(project_basis(grid[c], u)) for c in clips if c in grid}


def lam_at(p: np.ndarray | None, t: float) -> float | None:
    """lambda at a point: JSD of the renormalised (p_ne, p_en) of LID windows k-10 and k, k = round(t / hop)."""
    if p is None:
        return None
    k = int(round(t / HOP))
    if k - h4.LAG_K < 0 or k >= len(p):
        return None
    return float(h4.jsd(p[k - h4.LAG_K], p[k]))


def compare(a: dict, b: dict, chans: list[str], s_b0: float, rng) -> dict:
    """Mean over channels of a(c) - b(c) (negative = a has the smaller excess), with an unadjusted sign-flip."""
    common = [c for c in chans if a.get(c) is not None and b.get(c) is not None]
    x = [a[c] - b[c] for c in common]
    if not x:
        return {"n_channels": 0}
    sf = signflip([-v for v in x], rng)
    return {"n_channels": len(x), "mean_delta": float(np.mean(x)),
            "relative_reduction": float(-np.mean(x) / s_b0) if s_b0 > 0 else None,
            "ci95": bca_ci(x, mean_stat, 0.95, rng), "p_one_unadjusted": sf["p_one"],
            "p_two_unadjusted": sf["p_two"], "per_channel": dict(zip(common, map(float, x)))}


def mean_evals(evs: list[dict]) -> dict:
    """Channel statistics averaged over several evaluations (the random-subspace draws)."""
    chans = {}
    for c in evs[0]["channels"]:
        rows = [e["channels"].get(c) for e in evs]
        if any(r is None for r in rows):
            continue
        chans[c] = {"n_ssdl": rows[0]["n_ssdl"]}
        for k in RATES:
            v = [r[k] for r in rows]
            chans[c][k] = None if any(x is None for x in v) else float(np.mean(v))
    return {"tau": [e["tau"] for e in evs], "channels": chans}


def summarise(table: dict, pairs: list[tuple[str, str]], rng) -> dict:
    """Inclusion, S_B0 and the comparisons, from table[system][channel] = channel statistics."""
    b0 = table.get("B0", {})
    inc = sorted(c for c, s in b0.items() if s["n_ssdl"] >= MIN_UNITS and s["delta_far_std"] is not None)
    s_b0 = float(np.mean([b0[c]["delta_far_std"] for c in inc])) if inc else float("nan")
    col = lambda n: {c: s["delta_far_std"] for c, s in table.get(n, {}).items()}
    means = {}
    for n, t in sorted(table.items()):
        means[n] = {}
        for k in RATES:
            v = [t[c][k] for c in inc if c in t and t[c][k] is not None]
            means[n][k] = float(np.mean(v)) if v else None
    return {"included": inc, "excluded": sorted(set(b0) - set(inc)), "S_B0": s_b0,
            "comparisons": {f"{a}_vs_{b}": compare(col(a), col(b), inc, s_b0, rng)
                            for a, b in pairs if a in table and b in table},
            "mean_over_included": means}


def _json_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, set):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def hash_inputs(paths) -> dict:
    out = {}
    for p in sorted({Path(x).expanduser() for x in paths}):
        if str(p) not in _HASHES:
            _HASHES[str(p)] = h4.sha256(p)
        out[str(p)] = _HASHES[str(p)]
    return out


# ------------------------------------------------------------------ data -------------------------------------
def load_data(args) -> dict:
    splits = ["validation", "train"] + (["test"] if args.scope == "test" and not args.dry_run else [])
    points, _ = load_points(args.boundaries, splits)
    split_of = {r["clip_id"]: r["split"] for r in csv.DictReader(open(args.manifest, encoding="utf-8"))}
    p0 = p0_points(points)
    calib = {p: r for p, r in p0.items()
             if split_of.get(r["clip_id"]) == "validation" and r["video_id"] != h4.CALIB_EXCLUDE_VIDEO}
    chans = train_channels(args.manifest)
    return {"splits": splits, "points": points, "p0": p0, "split_of": split_of, "calib": calib,
            "calib_clips": {r["clip_id"] for r in calib.values()}, "chans": chans, "folds": h4_folds(chans),
            "lid": h4.load_lid(args.lid, splits), "windows_train": h4.read_windows(args.windows, "train")}


def make_units(d: dict, args) -> list[dict]:
    """The scored units: (fold, fitting channels, scored channels, scored points, inner folds, M3 tag)."""
    if args.dry_run:
        return [{"fold": None, "fit": list(d["chans"]), "scored": sorted({r["speaker_id"] for r in d["calib"].values()}),
                 "pts": d["calib"], "inner": None, "m3_tag": None, "split": "validation"}]
    if args.scope == "test":
        pts = {p: r for p, r in d["p0"].items() if d["split_of"].get(r["clip_id"]) == "test"}
        return [{"fold": None, "fit": list(d["chans"]), "scored": sorted({r["speaker_id"] for r in pts.values()}),
                 "pts": pts, "inner": d["folds"], "m3_tag": "all", "split": "test"}]
    out = []
    for f, scored in enumerate(d["folds"]):
        pts = {p: r for p, r in d["p0"].items()
               if d["split_of"].get(r["clip_id"]) == "train" and r["speaker_id"] in scored}
        fit = fitting_channels(d["chans"], f)
        out.append({"fold": f, "fit": fit, "scored": list(scored), "pts": pts, "inner": inner_folds(fit),
                    "m3_tag": f"f{f}", "split": "train"})
    return out


def clips_of(d: dict, pts: dict) -> set[str]:
    return d["calib_clips"] | {r["clip_id"] for r in pts.values()}


def evaluate(d: dict, pts: dict, scored, curves: dict) -> dict:
    """tau on the calibration channels, then the p0L1 cell statistics of the scored channels."""
    tau = h4.tau_on(d["calib"], curves)
    if not np.isfinite(tau):
        return {"tau": None, "not_assessable": "tau = -inf", "channels": {}}
    st = h4.cell_stats(pts, curves, tau)
    return {"tau": tau, "channels": {c: {k: st[c][k] for k in KEEP} for c in scored if c in st}}


def put(table: dict, name: str, ev: dict) -> None:
    for c, s in ev["channels"].items():
        table[name][c] = s


def m3_names(u: dict, dry_m3: str | None) -> dict[str, list[str]]:
    if u["m3_tag"] is None:
        return {"primary": [dry_m3]} if dry_m3 else {}
    out = {"primary": [f"m3_{u['m3_tag']}_s{s}_{h4.M3_PRIMARY}" for s in h4.M3_SEEDS]}
    if u["m3_tag"] != "all":
        out.update({cfg: [f"m3_{u['m3_tag']}_s0_{cfg}"] for cfg in M3_CONFIGS})
    return out


def parse_m3(name: str) -> dict:
    """m3_<fF|all>_s<seed>_w<w_dist>_<ablation> -> its training settings."""
    tag, seed, w, abl = name[3:].split("_")
    return {"fold": None if tag == "all" else int(tag[1:]), "seed": int(seed[1:]), "w_dist": float(w[1:]),
            "ablation": abl}


def load_m3_grid(args, name: str, splits, clips: set, used: list) -> dict | None:
    """The grid's unit windows for the given clips, or None when it is not (fully) written yet. Outside the dry
    run its checkpoint's fold, seed, w_dist and ablation must match the name."""
    path = args.m3_emb / f"{name}_L1"
    files = [path / "meta.json"] + [path / f"{s}.npz" for s in splits]
    if not all(f.exists() for f in files):
        return None
    meta = json.loads(files[0].read_text()).get("m3_meta", {})
    if not args.dry_run:
        for k, v in parse_m3(name).items():
            if k in meta and meta[k] != v:
                raise SystemExit(f"refusing: grid {name} has {k} = {meta[k]}, expected {v}")
    used += files
    return h4.load_grid(path, splits, clips)


def check_stage1(stage1: dict | None, system: str, u: dict, ev: dict, grid: str | None = None) -> None:
    """B0's tau is the same in every fold and scope; in the cv scope M1 and the primary M3 grids are fold-wise
    and stage 1 holds their taus and per-channel dFAR. A recomputed value that differs stops the run."""
    if stage1 is None or ev["tau"] is None:
        return
    ex = stage1["per_channel_extra"]
    if system == "B0":
        refs = {x["tau"] for x in ex["B0"].values()}
    elif u["fold"] is None:
        return
    elif system == "M3":
        rows = [(c, x) for c in u["scored"] for x in ex.get("M3_seeds", {}).get(c, []) if x["grid"] == grid]
        refs = {x["tau"] for _, x in rows}
        for c, x in rows:
            mine = ev["channels"].get(c, {}).get("delta_far_std")
            if mine is None or abs(mine - x["dfar"]) > 1e-12:
                raise SystemExit(f"refusing: {grid} dFAR on {c} is {mine} here, {x['dfar']} in stage 1")
    else:
        refs = {ex[system][c]["tau"] for c in u["scored"] if c in ex.get(system, {})}
    if u["fold"] is not None and system in ("B0", "M1"):
        for c, ref in stage1["dfar"].get(system, {}).items():
            if c in u["scored"]:
                mine = ev["channels"].get(c, {}).get("delta_far_std")
                if (ref is None) != (mine is None) or (ref is not None and abs(ref - mine) > 1e-12):
                    raise SystemExit(f"refusing: {system} dFAR on {c} is {mine} here, {ref} in stage 1")
    if len(refs) != 1 or abs(next(iter(refs)) - ev["tau"]) > 1e-12:
        raise SystemExit(f"refusing: {system} {grid or ''} tau {ev['tau']} does not reproduce stage 1 {sorted(refs)}")


# ------------------------------------------------------------------ parts ------------------------------------
def part_lambda(d: dict, us: list[dict], args, rng, ctx: dict) -> dict:
    codes = {r.get("phrase") for u in us for r in u["pts"].values() if r["kind"] == "ctrl"}
    if not codes <= {"True", "False"}:
        raise SystemExit(f"refusing: the points' phrase column holds {sorted(map(str, codes))[:5]}, not True/False")
    per = {}
    for u in us:
        pos, neg = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
        phr, oth = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
        for r in u["pts"].values():
            v = lam_at(d["lid"].get(r["clip_id"]), r["t"])
            if v is None:
                continue
            ch = r["speaker_id"]
            if r["kind"] == "ssdl":
                pos[ch][(r["video_id"], r["pause_bin"], r["pos_bin"])].append(v)
            else:
                neg[ch][(r["video_id"], r["pause_bin"], r["pos_bin"])].append(v)
                (phr if r["phrase"] == "True" else oth)[ch][(r["video_id"], r["pos_bin"])].append(v)
        for ch in u["scored"]:
            auc, n = stratified_auc(pos[ch], neg[ch])
            auc_p, n_p = stratified_auc(phr[ch], oth[ch])
            per[ch] = {"auc": auc, "n_ssdl": n, "auc_phrase_vs_other": auc_p, "n_phrase": n_p}
    inc = sorted(c for c, v in per.items() if v["n_ssdl"] >= MIN_UNITS)

    def summ(key: str) -> dict:
        x = [per[c][key] for c in inc if np.isfinite(per[c][key])]
        if not x:
            return {"n_channels": 0}
        sf = signflip([v - 0.5 for v in x], rng)
        return {"n_channels": len(x), "mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1)) if len(x) > 1 else None,
                "p_one_unadjusted": sf["p_one"], "p_two_unadjusted": sf["p_two"]}

    return {"per_channel": per, "included": inc, "summary": {k: summ(k) for k in ("auc", "auc_phrase_vs_other")}}


def part_m1k(d: dict, us: list[dict], args, rng, ctx: dict) -> dict:
    grid, stage1 = ctx["e3"], ctx["stage1"]
    by = language_windows(d["points"], args.point_emb, "e3", 1.0, ("train",))
    table, units = defaultdict(dict), []
    for u in us:
        fit, clips = set(u["fit"]), clips_of(d, u["pts"])
        ev = {"B0": evaluate(d, u["pts"], u["scored"], h4.system_curves("B0", clips, grid, d["lid"], {}))}
        v = estimate_v_lang(d["points"], args.point_emb, "e3", 1.0, ("train",), speakers=fit)
        ev["M1"] = evaluate(d, u["pts"], u["scored"], h4.system_curves("M1", clips, grid, d["lid"], {"v": v}))
        for s in ("B0", "M1"):
            check_stage1(stage1, s, u, ev[s])
        inner = None
        if u["inner"] is not None:
            vals = defaultdict(list)
            for held in u["inner"]:
                diffs = session_diffs(by, fit - set(held))
                ipts = {p: r for p, r in d["p0"].items()
                        if d["split_of"].get(r["clip_id"]) == "train" and r["speaker_id"] in held}
                iclips = clips_of(d, ipts)
                for k in KS:
                    e_k = evaluate(d, ipts, held, basis_curves(iclips, grid, nap_basis(diffs, k)))
                    vals[k] += [s["delta_far_std"] for s in e_k["channels"].values()
                                if s["n_ssdl"] >= MIN_UNITS and s["delta_far_std"] is not None]
            inner = {k: (float(np.mean(vals[k])) if vals[k] else None) for k in KS}
            k_star, k_fallback = choose_k(inner)
        else:
            k_star, k_fallback = KS[0], False
        diffs = session_diffs(by, fit)
        for k in KS:
            ev[f"M1k{k}"] = evaluate(d, u["pts"], u["scored"], basis_curves(clips, grid, nap_basis(diffs, k)))
        ev["M1k"] = ev[f"M1k{k_star}"]
        rr = np.random.default_rng([SEED, 0 if u["fold"] is None else u["fold"] + 1])
        ev["random"] = mean_evals([evaluate(d, u["pts"], u["scored"],
                                            basis_curves(clips, grid, random_basis(diffs.shape[1], k_star, rr)))
                                   for _ in range(N_RANDOM)])
        ev["pca"] = evaluate(d, u["pts"], u["scored"], basis_curves(clips, grid, pca_basis(session_windows(by, fit), k_star)))
        for name, e in ev.items():
            put(table, name, e)
        units.append({"fold": u["fold"], "scored": u["scored"], "k_star": k_star, "k_fallback": k_fallback,
                      "inner_mean_dfar": inner, "n_session_diffs": int(len(diffs)),
                      "tau": {n: e["tau"] for n, e in ev.items()}})
    pairs = [("M1", "B0"), ("M1k", "B0"), ("random", "B0"), ("pca", "B0"), ("M1k", "M1"), ("M1k", "random"),
             ("M1k", "pca")] + [(f"M1k{k}", "B0") for k in KS]
    return {"units": units, "summary": summarise(table, pairs, rng), "table": table}


def part_emb(d: dict, us: list[dict], args, rng, ctx: dict) -> dict:
    out = {}
    for model in ("e1", "e2"):
        grid = h4.load_grid(args.emb / f"{model}_L1", d["splits"])
        ctx["used"] += [args.emb / f"{model}_L1" / f"{s}.npz" for s in d["splits"]]
        ctx["used"].append(args.point_emb / model / "train_L1.npz")
        table, units = defaultdict(dict), []
        for u in us:
            fit, clips = set(u["fit"]), clips_of(d, u["pts"])
            b2 = h4.b2_delta(grid, d["windows_train"], fit)
            fitted = {"mu": h4.b1_means(grid, d["windows_train"], fit), "delta": b2["delta"],
                      "v": estimate_v_lang(d["points"], args.point_emb, model, 1.0, ("train",), speakers=fit)}
            taus = {}
            for s in ("B0", "B1", "B2", "M1"):
                e = evaluate(d, u["pts"], u["scored"], h4.system_curves(s, clips, grid, d["lid"], fitted))
                put(table, s, e)
                taus[s] = e["tau"]
            units.append({"fold": u["fold"], "b2": b2, "m1_v_norm": float(np.linalg.norm(fitted["v"])), "tau": taus})
        pairs = [("B1", "B0"), ("B2", "B0"), ("M1", "B0"), ("M1", "B1"), ("M1", "B2")]
        out[model] = {"units": units, "summary": summarise(table, pairs, rng), "table": table}
        del grid
    return out


def part_sens(d: dict, us: list[dict], args, rng, ctx: dict) -> dict:
    grid, stage1 = ctx["e3"], ctx["stage1"]
    subset = json.loads(args.purity.read_text())
    scored_pts = {p: r for u in us for p, r in u["pts"].items()}
    pur, _, pur_info = apply_subset(scored_pts, [], subset)
    aln, _, thr = alignment_filter(scored_pts, [], args.align)
    ctx["used"] += [args.purity, args.align / "train.jsonl"]
    filters = {"purity": pur, "alignment": aln}
    tables = {f: defaultdict(dict) for f in filters}
    seeds = {f: defaultdict(list) for f in filters}
    counts = {len(m3_names(u, args.dry_m3).get("primary", [])) for u in us}
    if len(counts) != 1:
        raise SystemExit(f"refusing: folds have different numbers of primary M3 grids {sorted(counts)}")
    n_grids, missing = counts.pop(), []

    def score(u: dict, name: str, curves: dict, grid_name: str | None = None) -> None:
        ev = evaluate(d, u["pts"], u["scored"], curves)          # the primary tau and statistics
        check_stage1(stage1, name, u, ev, grid_name)
        for f, fp in filters.items():
            sub = {p: fp[p] for p in u["pts"] if p in fp}
            st = h4.cell_stats(sub, curves, ev["tau"]) if ev["tau"] is not None else {}
            for c, s in st.items():
                if name == "M3":
                    if s["delta_far_std"] is not None:
                        seeds[f][c].append(s["delta_far_std"])
                else:
                    tables[f][name][c] = {k: s[k] for k in KEEP}

    for u in us:
        fit, clips = set(u["fit"]), clips_of(d, u["pts"])
        v = estimate_v_lang(d["points"], args.point_emb, "e3", 1.0, ("train",), speakers=fit)
        score(u, "B0", h4.system_curves("B0", clips, grid, d["lid"], {}))
        score(u, "M1", h4.system_curves("M1", clips, grid, d["lid"], {"v": v}))
        prim = m3_names(u, args.dry_m3).get("primary", [])
        for nm in prim:
            g = load_m3_grid(args, nm, d["splits"], clips, ctx["used"])
            if g is None:
                missing.append(nm)
                continue
            score(u, "M3", {c: h4.curve(g[c]) for c in clips if c in g}, nm)
            del g
    res = {"purity_subset": pur_info, "alignment_threshold": thr, "m3_missing": missing, "filters": {}}
    for f in filters:
        t = tables[f]
        for c, v in seeds[f].items():
            if n_grids and len(v) == n_grids and c in t["B0"]:
                t["M3"][c] = {"n_ssdl": t["B0"][c]["n_ssdl"], "delta_far_std": float(np.mean(v)),
                              **{k: None for k in ("auc", "far_ssdl", "far_ctrl_std")}}
        res["filters"][f] = summarise(t, [("M1", "B0"), ("M3", "B0")], rng)
    return res


def part_m3(d: dict, us: list[dict], args, rng, ctx: dict) -> dict:
    grid, stage1 = ctx["e3"], ctx["stage1"]
    table, units, missing = defaultdict(dict), [], []
    for u in us:
        clips = clips_of(d, u["pts"])
        b0 = evaluate(d, u["pts"], u["scored"], h4.system_curves("B0", clips, grid, d["lid"], {}))
        check_stage1(stage1, "B0", u, b0)
        put(table, "B0", b0)
        taus = {"B0": b0["tau"]}
        for cfg, names in m3_names(u, args.dry_m3).items():
            for nm in names:
                g = load_m3_grid(args, nm, d["splits"], clips, ctx["used"])
                if g is None:
                    missing.append(nm)
                    continue
                ev = evaluate(d, u["pts"], u["scored"], {c: h4.curve(g[c]) for c in clips if c in g})
                del g
                if cfg == "primary":
                    check_stage1(stage1, "M3", u, ev, nm)
                    key = f"M3_s{0 if args.dry_run else parse_m3(nm)['seed']}"
                else:
                    key = f"M3_{cfg}"
                put(table, key, ev)
                taus[nm] = ev["tau"]
        units.append({"fold": u["fold"], "tau": taus})
    seed_keys = [f"M3_s{s}" for s in h4.M3_SEEDS if f"M3_s{s}" in table]
    n_seeds = 1 if args.dry_run else len(h4.M3_SEEDS)
    for c in table["B0"]:
        rows = [table[k][c] for k in seed_keys if c in table[k]]
        if seed_keys and len(rows) == n_seeds and all(r["delta_far_std"] is not None for r in rows):
            table["M3"][c] = {"n_ssdl": rows[0]["n_ssdl"],
                              **{k: (None if any(r[k] is None for r in rows) else float(np.mean([r[k] for r in rows])))
                                 for k in RATES}}
    pairs = [("M3", "B0")] + [(k, "B0") for k in seed_keys] + \
            [p for cfg in M3_CONFIGS for p in ((f"M3_{cfg}", "B0"), (f"M3_{cfg}", "M3_s0"))]
    summ = summarise(table, pairs, rng)
    inc = summ["included"]
    reps = [summ["comparisons"][f"{k}_vs_B0"]["mean_delta"] for k in seed_keys
            if summ["comparisons"].get(f"{k}_vs_B0", {}).get("n_channels")]
    sds = [float(np.std([table[k][c]["delta_far_std"] for k in seed_keys], ddof=1)) for c in inc
           if len(seed_keys) > 1 and all(c in table[k] and table[k][c]["delta_far_std"] is not None for k in seed_keys)]
    summ["seed_spread"] = {"replicate_mean_delta": reps,
                           "sd_of_replicates": float(np.std(reps, ddof=1)) if len(reps) > 1 else None,
                           "mean_per_channel_seed_sd": float(np.mean(sds)) if sds else None}
    return {"units": units, "missing_grids": missing, "summary": summ, "table": table}


def window_rows(g: dict, wins: list[dict]) -> np.ndarray:
    return np.stack([g[w["clip_id"]][int(w["k"])] for w in wins]).astype(np.float32)


def draw_trials(wins: list[dict], rng) -> dict:
    """H3's trial rules on one unit's windows: targets are same-video pairs whose starts are >= 2 s apart (at most
    TRIAL_PAIRS draws per video); non-targets pair windows of different channels (NON_DRAWS draws). Each is split
    into same-language and cross-language trials."""
    lang = np.array([w["lang"] for w in wins])
    t = np.array([float(w["abs_t0"]) for w in wins])
    chan = np.array([w["speaker_id"] for w in wins])
    by = defaultdict(list)
    for i, w in enumerate(wins):
        by[w["video_id"]].append(i)
    ta, tb = [np.zeros(0, int)], [np.zeros(0, int)]
    for v in sorted(by):
        idx = np.array(by[v])
        if len(idx) < 2:
            continue
        n = min(TRIAL_PAIRS, len(idx) * (len(idx) - 1) // 2)
        a, b = rng.choice(idx, n), rng.choice(idx, n)
        keep = (a != b) & (np.abs(t[a] - t[b]) >= LAG_S)
        ta.append(a[keep]); tb.append(b[keep])
    a, b = np.concatenate(ta), np.concatenate(tb)
    na, nb = rng.integers(0, len(wins), NON_DRAWS), rng.integers(0, len(wins), NON_DRAWS)
    keep = chan[na] != chan[nb]
    na, nb = na[keep], nb[keep]
    return {"tgt": (a, b, lang[a] != lang[b]), "non": (na, nb, lang[na] != lang[nb])}


def trial_eers(e: np.ndarray, tr: dict) -> dict:
    (a, b, xa), (na, nb, xn) = tr["tgt"], tr["non"]
    st, sn = (e[a] * e[b]).sum(1), (e[na] * e[nb]).sum(1)
    out = {}
    for cond, mt, mn in (("mono", ~xa, ~xn), ("cross", xa, xn)):
        out[cond] = eer_mindcf(st[mt], sn[mn])[0] if mt.any() and mn.any() else None
        out[f"n_{cond}"] = [int(mt.sum()), int(mn.sum())]
    return out


def lang_probe(x_tr, y_tr, x_te, y_te) -> float | None:
    """Balanced accuracy of a standardised logistic-regression probe for the window's language."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
        return None
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(x_tr, y_tr)
    return float(balanced_accuracy_score(y_te, clf.predict(x_te)))


def speaker_probe(x, spk, vid) -> float | None:
    """Cross-session speaker identification among the scored channels: each channel's even-ranked videos train
    the probe and its odd-ranked videos test it; channels with one video are left out."""
    spk, vid = np.asarray(spk), np.asarray(vid)
    tr, te = np.zeros(len(x), bool), np.zeros(len(x), bool)
    for c in sorted(set(spk)):
        vids = sorted(set(vid[spk == c]))
        if len(vids) < 2:
            continue
        for i, v in enumerate(vids):
            (tr if i % 2 == 0 else te)[(spk == c) & (vid == v)] = True
    if len(set(spk[tr])) < 2:
        return None
    return lang_probe(x[tr], spk[tr], x[te], spk[te])


def part_speaker(d: dict, us: list[dict], args, rng, ctx: dict) -> dict:
    grid = ctx["e3"]
    by = language_windows(d["points"], args.point_emb, "e3", 1.0, ("train",))
    k_stars = [x["k_star"] for x in ctx["m1k"]["units"]] if ctx.get("m1k") else None
    split = us[0]["split"]
    scored_w = h4.read_windows(args.windows, split)
    calib_w = [] if args.dry_run else [w for w in h4.read_windows(args.windows, "validation")
                                       if w["video_id"] != h4.CALIB_EXCLUDE_VIDEO]
    ctx["used"] += [args.windows / f"windows_{s}.csv.gz" for s in {split, "validation"}]
    prep = []
    for i, u in enumerate(us):
        ws = [w for w in scored_w if w["speaker_id"] in set(u["scored"]) and w["video_id"] != h4.CALIB_EXCLUDE_VIDEO
              and w["abs_t0"] != ""]
        wf = [w for w in d["windows_train"] if w["speaker_id"] in set(u["fit"])]
        if len(wf) > PROBE_MAX:
            wf = [wf[j] for j in np.sort(rng.choice(len(wf), PROBE_MAX, replace=False))]
        fit = set(u["fit"])
        fitted = {"mu": h4.b1_means(grid, d["windows_train"], fit),
                  "v": estimate_v_lang(d["points"], args.point_emb, "e3", 1.0, ("train",), speakers=fit),
                  "U": nap_basis(session_diffs(by, fit), k_stars[i]) if k_stars else None}
        prep.append({"ws": ws, "wf": wf, "wsid": ws + calib_w, "trials": draw_trials(ws, rng), "fitted": fitted,
                     "clips": {w["clip_id"] for w in ws + wf + calib_w}})

    def b1(g: dict, mu: dict):
        def emb(wins: list[dict]) -> np.ndarray:
            lang = np.array([h4.predicted_lang(d["lid"][w["clip_id"]], len(g[w["clip_id"]]))[int(w["k"])]
                             for w in wins])
            return h4.unit(window_rows(g, wins) - np.where(lang[:, None] == 0, mu[0][None, :], mu[1][None, :]))
        return emb

    results = defaultdict(list)

    def measure(name: str, i: int, emb) -> None:
        """emb maps a window list to the system's unit embeddings of those windows."""
        p = prep[i]
        es, ef, esid = emb(p["ws"]), emb(p["wf"]), emb(p["wsid"])
        results[name].append({"fold": us[i]["fold"], "eer": trial_eers(es, p["trials"]),
                              "lang_acc": lang_probe(ef, [w["lang"] for w in p["wf"]], es, [w["lang"] for w in p["ws"]]),
                              "spk_acc": speaker_probe(esid, [w["speaker_id"] for w in p["wsid"]],
                                                       [w["video_id"] for w in p["wsid"]]),
                              "n_speaker_classes": len({w["speaker_id"] for w in p["wsid"]})})

    raw = lambda g: (lambda wins: window_rows(g, wins))
    for i, p in enumerate(prep):
        f = p["fitted"]
        measure("E3", i, raw(grid))
        measure("B1", i, b1(grid, f["mu"]))
        measure("M1", i, lambda wins, v=f["v"]: project_out(window_rows(grid, wins), v))
        if f["U"] is not None:
            measure("M1k", i, lambda wins, U=f["U"]: project_basis(window_rows(grid, wins), U))
    for model in ("e1", "e2"):
        g = h4.load_grid(args.emb / f"{model}_L1", d["splits"])
        ctx["used"] += [args.emb / f"{model}_L1" / f"{s}.npz" for s in d["splits"]]
        for i in range(len(prep)):
            measure(model.upper(), i, raw(g))
        del g
    missing = []
    for i, u in enumerate(us):
        for cfg, names in m3_names(u, args.dry_m3).items():
            for nm in names:
                g = load_m3_grid(args, nm, d["splits"], prep[i]["clips"], ctx["used"])
                if g is None:
                    missing.append(nm)
                    continue
                key = f"M3_s{0 if args.dry_run else parse_m3(nm)['seed']}" if cfg == "primary" else f"M3_{cfg}"
                measure(key, i, raw(g))
                del g

    def mean_of(rows: list[dict], get) -> float | None:
        v = [get(r) for r in rows]
        v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else None

    summary = {n: {"n_units": len(rows), "eer_mono": mean_of(rows, lambda r: r["eer"]["mono"]),
                   "eer_cross": mean_of(rows, lambda r: r["eer"]["cross"]),
                   "lang_probe_bacc": mean_of(rows, lambda r: r["lang_acc"]),
                   "speaker_id_bacc": mean_of(rows, lambda r: r["spk_acc"])} for n, rows in sorted(results.items())}
    return {"k_star": k_stars, "missing_grids": missing, "summary": summary, "per_unit": dict(results),
            "n_windows": [{"scored": len(p["ws"]), "fit_probe": len(p["wf"]), "speaker_id": len(p["wsid"])}
                          for p in prep]}


PART_FN = {"lambda": part_lambda, "m1k": part_m1k, "emb": part_emb, "sens": part_sens, "m3": part_m3,
           "speaker": part_speaker}


# ------------------------------------------------------------------ main -------------------------------------
def check_ready(args) -> tuple[dict, str]:
    rec = json.loads(h4.FREEZE.read_text())
    s1 = args.out / "confirm_h4_stage1.json"
    if not s1.exists() or rec.get("stage1_sha256") != h4.sha256(s1):
        raise SystemExit("refusing: the secondary analyses run after stage 1, whose result must be hashed in "
                         "planning/FREEZE_H4.json (freeze_h4 add-stage1)")
    return json.loads(s1.read_text()), rec["stage1_sha256"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    home = Path.home() / "data/d1"
    ap.add_argument("--scope", choices=("cv", "test"), default="cv")
    ap.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    ap.add_argument("--manifest", type=Path, default=home / "manifest.csv")
    ap.add_argument("--boundaries", type=Path, default=home / "boundaries/lex_v1b")
    ap.add_argument("--point-emb", type=Path, default=home / "point_emb/lex_v1b")
    ap.add_argument("--emb", type=Path, default=home / "emb")
    ap.add_argument("--lid", type=Path, default=home / "lid")
    ap.add_argument("--windows", type=Path, default=home / "windows/lex_v1")
    ap.add_argument("--align", type=Path, default=home / "align")
    ap.add_argument("--purity", type=Path, default=home / "results/purity/subset.json")
    ap.add_argument("--m3-emb", type=Path, default=Path.home() / "data/m3/emb")
    ap.add_argument("--out", type=Path, default=None, help="default results/h4 (results/h4_dryrun with --dry-run)")
    ap.add_argument("--dry-run", action="store_true", help="calibration channels only; structure, not effects")
    ap.add_argument("--dry-m3", default=None, help="dry run: one M3 grid name to exercise the M3 paths")
    args = ap.parse_args()
    args.out = (args.out or home / ("results/h4_dryrun" if args.dry_run else "results/h4")).expanduser()
    label = "dryrun" if args.dry_run else args.scope
    stage1, s1_sha = (None, None) if args.dry_run else check_ready(args)
    for part in args.parts:
        if not args.dry_run and (args.out / f"secondary_h4_{label}_{part}.json").exists():
            raise SystemExit(f"refusing: part {part} ({label}) has already run (one-shot)")

    d = load_data(args)
    us = make_units(d, args)
    ctx = {"stage1": stage1, "e3": None, "m1k": None, "used": []}
    base_inputs = [args.manifest, args.point_emb / "e3" / "train_L1.npz", args.windows / "windows_train.csv.gz",
                   *[args.boundaries / f"{k}_{s}.csv.gz" for k in ("points",) for s in d["splits"]],
                   *[args.lid / f"{s}.npz" for s in d["splits"]]]
    if set(args.parts) - {"lambda", "emb"}:
        ctx["e3"] = h4.load_grid(args.emb / "e3_L1", d["splits"])
        base_inputs += [args.emb / "e3_L1" / f"{s}.npz" for s in d["splits"]]
    if "speaker" in args.parts and "m1k" not in args.parts:
        prior = args.out / f"secondary_h4_{label}_m1k.json"
        ctx["m1k"] = json.loads(prior.read_text())["result"] if prior.exists() else None
    for part in [p for p in PARTS if p in args.parts]:
        rng = np.random.default_rng(SEED)
        ctx["used"] = []
        res = PART_FN[part](d, us, args, rng, ctx)
        if part == "m1k":
            ctx["m1k"] = res
        doc = {"part": part, "scope": args.scope, "dry_run": args.dry_run,
               "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "code_sha256": h4.sha256(Path(__file__)), "stage1_sha256": s1_sha,
               "units": [{"fold": u["fold"], "scored": u["scored"], "fit": u["fit"]} for u in us],
               "inputs_sha256": hash_inputs(base_inputs + ctx["used"]), "result": res}
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / f"secondary_h4_{label}_{part}.json"
        path.write_text(json.dumps(doc, indent=1, default=_json_default))
        print(f"{part}: wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
