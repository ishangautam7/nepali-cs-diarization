#!/usr/bin/env python3
"""Confirmatory D1 analyses: H1, H2 (P0) and H3 (PLAN.md §7; PREREGISTRATION.md; decisions 13-17).

Confirmatory channels are the 17 non-pilot D1 channels (train + test). The module refuses any split other than
validation unless planning/FREEZE_M2A.json exists and every code and feature file it lists still has the
recorded SHA-256, so confirmatory outcomes can only come from the frozen analysis. With --splits validation it is
a dry run of the same code (3 channels, never significant) for checking.

Per hypothesis and embedding (E1-E3). A channel enters a primary test only with at least MIN_UNITS = 10 units
(H1 matched sets; H2-P0 switch points with cell controls; H3 needs one lag bin with >= 20 cross and >= 20 mono
trials); excluded channels are listed with their counts, and the sign-flip runs over the k included channels.
- H1: per-channel mean paired delta-cos (L = 1.0 s, 1:3 matched sets). Exact one-sided sign-flip over channels;
  Holm over m = 3. Estimate dz = unweighted mean of channel means / pooled within-channel SD of per-set
  delta-cos, sqrt(sum_c (n_c - 1) s_c^2 / sum_c (n_c - 1)), recomputed in every bootstrap resample. BCa CI at
  1 - 0.05/3; TOST interval at 1 - 2(0.05/3); SESOI dz = 0.2 (point estimate). Secondary: plain dz (mean / SD
  of all per-set delta-cos, PLAN.md §2's definition) and L = 0.5 / 1.5 s.
- H2-P0: per-channel cell-standardised AUC of d(t) in the primary cell p0L1 (det_w1 controls), frozen tau.
  Sign-flip on AUC - 0.5; Holm over m = 5 (P1, P2 enter as p = 1 until the long-form runs exist, which can only
  make the adjustment more conservative). BCa CI at 1 - 0.05/5; TOST at 1 - 2(0.05/5); SESOI AUC = 0.55.
  Secondary: pooled det, det_w1, strata, cell-standardised delta-FAR.
- H3: per-channel same-session lag-matched shift (cross - mono) in target-score SD units; predicted negative.
  Sign-flip; Holm over m = 3; BCa CI at 1 - 0.05/3; TOST at 1 - 2(0.05/3); SESOI 0.2 target SD.
  Secondary: cross-session shift, EER and minDCF.
- Sensitivity (pre-registered): points whose boundary words have an alignment score below the 5th percentile
  of train word scores are removed and H1 / P0 are recomputed (descriptive); and the whole primary analysis is
  repeated without the 2 test channels (15 train channels; descriptive), because the 2-channel test split on its
  own is descriptive only.
Outcome categories follow csdiar.analysis.stats.outcome.
Output: <out>/confirm_<tag>.json and .md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from csdiar.analysis.h3 import compute_model, load_windows
from csdiar.analysis.pilot import MODELS, h1, h2, p0_curves, read_gz
from csdiar.analysis.stats import bca_ci, holm, outcome, signflip

REPO = Path(__file__).resolve().parents[2]
FREEZE = REPO / "planning" / "FREEZE_M2A.json"
GUARDED = ["csdiar/analysis/confirm.py", "csdiar/analysis/stats.py", "csdiar/analysis/pilot.py",
           "csdiar/analysis/h3.py", "csdiar/analysis/attribution.py", "csdiar/analysis/glmm.R",
           "csdiar/data/boundaries.py", "csdiar/data/windows.py", "csdiar/data/tags.py", "csdiar/data/acoustic.py"]
ALPHA = 0.05
MIN_UNITS = 10  # a channel enters a primary test only with at least this many units (PREREGISTRATION.md §4)
TAU_PREREG = {"e1": 0.917, "e2": 0.863, "e3": 0.929}
ALIGN_THRESHOLD_PREREG = 0.093  # 5th percentile of aligned train word probabilities (PREREGISTRATION.md §5)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def check_freeze(splits: list[str], paths: dict | None = None) -> dict:
    """Dry run on validation; otherwise every hash in the freeze record and every input path must match it."""
    if set(splits) <= {"validation"}:
        return {"mode": "dry run on the pilot channels; not confirmatory"}
    if not FREEZE.exists():
        raise SystemExit("refusing: confirmatory splits requested but planning/FREEZE_M2A.json does not exist")
    rec = json.loads(FREEZE.read_text())
    for f in GUARDED:
        if f not in rec["code_sha256"]:
            raise SystemExit(f"refusing: {f} is not covered by the freeze record")
    for f, h in rec["code_sha256"].items():
        if sha256(REPO / f) != h:
            raise SystemExit(f"refusing: {f} changed since the freeze")
    for f, h in rec.get("feature_sha256", {}).items():
        if sha256(Path(f).expanduser()) != h:
            raise SystemExit(f"refusing: feature file {f} changed since the freeze")
    for k, v in (paths or {}).items():
        want = rec.get("paths", {}).get(k)
        if want is None or Path(v).expanduser().resolve() != Path(want).expanduser().resolve():
            raise SystemExit(f"refusing: --{k.replace('_', '-')} {v} is not the frozen input {want}")
    return {"mode": "confirmatory", "freeze": {k: rec[k] for k in ("tag", "commit", "date")}}


def frozen_taus(pilot_json: Path) -> dict:
    taus = {m: json.loads(pilot_json.read_text())["h2"][m]["tau"] for m in MODELS}
    for m, t in taus.items():
        if round(t, 3) != TAU_PREREG[m]:
            raise SystemExit(f"refusing: tau for {m} is {t}, not the pre-registered {TAU_PREREG[m]}")
    return taus


def pooled_sd(clusters: list[np.ndarray]) -> float:
    num = sum((len(c) - 1) * np.var(c, ddof=1) for c in clusters if len(c) > 1)
    den = sum(len(c) - 1 for c in clusters if len(c) > 1)
    return math.sqrt(num / den) if den else float("nan")


def dz_stat(clusters: list[np.ndarray]) -> float:
    return float(np.mean([c.mean() for c in clusters]) / pooled_sd(clusters))


def mean_stat(clusters: list[float]) -> float:
    return float(np.mean(clusters))


def family(results: dict, est_key: str, tost_key: str, sesoi: float, direction: int, m_total: int) -> None:
    """Holm-adjust the family in place (missing members enter as p = 1) and attach the outcome category."""
    names = list(results)
    p_one = [results[n]["signflip"]["p_one"] for n in names] + [1.0] * (m_total - len(names))
    p_two = [results[n]["signflip"]["p_two"] for n in names] + [1.0] * (m_total - len(names))
    a1, a2 = holm(p_one), holm(p_two)
    for i, n in enumerate(names):
        r = results[n]
        r["p_one_holm"], r["p_two_holm"] = a1[i], a2[i]
        r["outcome"] = outcome(a1[i], a2[i], r[est_key], r[tost_key]["lo"], r[tost_key]["hi"], sesoi, direction)


def included(values: dict, minimum: int = MIN_UNITS) -> tuple[dict, dict]:
    """Split per-channel unit lists into (included arrays, excluded counts) by the minimum-units rule."""
    return ({c: np.asarray(v) for c, v in values.items() if len(v) >= minimum},
            {c: len(v) for c, v in values.items() if len(v) < minimum})


def plain_dz(clusters: list[np.ndarray]) -> float:
    allv = np.concatenate(clusters)
    return float(allv.mean() / allv.std(ddof=1))


def run_h1(points, matches, pe_dir, splits, rng, channels: list | None = None) -> dict:
    res, sens = {}, {}
    for m in MODELS:
        r = h1(points, matches, pe_dir, m, 1.0, tuple(splits))
        values = r.pop("_values")
        inc, exc = included(values)
        exc.update({c: 0 for c in channels or [] if c not in values})
        clusters = list(inc.values())
        sf = signflip([c.mean() for c in clusters], rng)
        res[m] = {"n_channels": len(clusters), "excluded_channels": exc, "n_sets": int(sum(len(c) for c in clusters)),
                  "n_unmatched": r["n_ssdl_unmatched"],
                  "delta_cos_mean_of_channels": float(np.mean([c.mean() for c in clusters])), "dz": dz_stat(clusters),
                  "dz_plain": plain_dz(clusters),
                  "delta_cos_in_target_sd": float(np.mean([c.mean() for c in clusters]) / r["target_score_sd"]),
                  "signflip": sf,
                  "dz_ci": bca_ci(clusters, dz_stat, 1 - ALPHA / 3, rng),
                  "dz_tost": bca_ci(clusters, dz_stat, 1 - 2 * ALPHA / 3, rng),
                  "channels": r["channels"]}
        for L in (0.5, 1.5):
            rl = h1(points, matches, pe_dir, m, L, tuple(splits))
            cl = list(included(rl.pop("_values"))[0].values())
            sens[f"{m}_L{L:g}"] = {"dz": dz_stat(cl), "dz_ci": bca_ci(cl, dz_stat, 1 - ALPHA / 3, rng),
                                   "n_sets": int(sum(len(c) for c in cl)), "n_channels": len(cl)}
    family(res, "dz", "dz_tost", 0.2, +1, 3)
    return {"primary": res, "sensitivity_window_length": sens}


def run_h2(points, emb_dir, splits, taus, rng, channels: list | None = None) -> dict:
    res, secondary = {}, {}
    for m in MODELS:
        r = h2(points, p0_curves(emb_dir, m, tuple(splits)), tau=taus[m])
        prim = r["cells"]["p0L1"]
        chans = {c: v for c, v in prim["channels"].items() if v["auc"] == v["auc"] and v["n_ssdl"] >= MIN_UNITS}
        exc = {c: v["n_ssdl"] for c, v in prim["channels"].items() if c not in chans}
        exc.update({c: 0 for c in channels or [] if c not in prim["channels"]})
        aucs = [v["auc"] for v in chans.values()]
        sf = signflip([a - 0.5 for a in aucs], rng)
        ci = bca_ci(aucs, mean_stat, 1 - ALPHA / 5, rng)
        tost = bca_ci(aucs, mean_stat, 1 - 2 * ALPHA / 5, rng)
        dfar = [v["delta_far_std"] for v in chans.values() if v["delta_far_std"] is not None]
        res[m] = {"tau": taus[m], "n_channels": len(aucs), "excluded_channels": exc, "auc": float(np.mean(aucs)),
                  "auc_minus_half": float(np.mean(aucs)) - 0.5, "signflip": sf, "auc_ci": ci,
                  "tost": {"lo": tost["lo"] - 0.5, "hi": tost["hi"] - 0.5, "level": tost["level"]},
                  "delta_far_std": float(np.mean(dfar)), "delta_far_signflip": signflip(dfar, rng),
                  "n": prim["n"], "dropped_edge": r["dropped_edge"], "channels": prim["channels"]}
        secondary[m] = {name: {"auc": cell["auc_mean_of_channels"],
                               "auc_ci": bca_ci([c["auc"] for c in cell["channels"].values() if c["auc"] == c["auc"]],
                                                mean_stat, 1 - ALPHA / 5, rng),
                               "delta_far_std": cell["delta_far_std_mean_of_channels"], "strata": cell["strata"]}
                        for name, cell in r["cells"].items() if name != "p0L1"}
        secondary[m]["p0L1_strata"] = prim["strata"]
    family(res, "auc_minus_half", "tost", 0.05, +1, 5)
    return {"primary": res, "secondary": secondary}


def run_h3(windows_dir, emb_dir, splits, rng, channels: list | None = None) -> dict:
    wins = load_windows(windows_dir, tuple(splits))
    n_win = Counter(w["speaker_id"] for w in wins)
    all_ch = sorted(set(n_win) | set(channels or []))
    res = {}
    for m in MODELS:
        r = compute_model(wins, emb_dir, m, rng)
        sh = [v / r["target_score_sd"] for v in r["shift_per_channel"].values()]
        sf = signflip([-s for s in sh], rng)  # predicted negative
        sf_out = {"p_one": sf["p_one"], "p_two": sf["p_two"], "exact": sf["exact"], "n": sf["n"]}
        res[m] = {"n_channels": len(sh), "shift_in_target_sd": float(np.mean(sh)), "shift_raw": r["shift_mean"],
                  "excluded_channels": {c: {"windows": n_win.get(c, 0), "reason": "no lag bin met the trial rule"}
                                        for c in all_ch if c not in r["shift_per_channel"]},
                  "target_score_sd": r["target_score_sd"], "signflip": sf_out,
                  "ci": bca_ci(sh, mean_stat, 1 - ALPHA / 3, rng), "tost": bca_ci(sh, mean_stat, 1 - 2 * ALPHA / 3, rng),
                  "cross_session_shift": r["cross_session_shift"], "eer": r["eer"], "min_dcf": r["min_dcf"],
                  "n_trials": r["n_trials"], "shift_per_channel": r["shift_per_channel"]}
    family(res, "shift_in_target_sd", "tost", 0.2, -1, 3)
    return {"primary": res, "n_windows": len(wins)}


def alignment_filter(points: dict, matches: list[dict], align_dir: Path) -> tuple[dict, list[dict], float]:
    scores = []
    for line in open(align_dir / "train.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        scores += [w["score"] for w in rec["words"] if w.get("aligned")]
    thr = float(np.percentile(scores, 5))
    if round(thr, 3) != ALIGN_THRESHOLD_PREREG:
        raise SystemExit(f"refusing: alignment threshold {thr:.4f} is not the pre-registered {ALIGN_THRESHOLD_PREREG}")
    ok = lambda r: min(float(r["score_left"] or 0), float(r["score_right"] or 0)) >= thr
    kept = {p: r for p, r in points.items() if ok(r)}
    kept_m = [m for m in matches if m["ssdl_id"] in kept and m["ctrl_id"] in kept]
    return kept, kept_m, thr


def load_points(bdir: Path, splits: list[str]) -> tuple[dict, list[dict]]:
    pts, matches = {}, []
    for s in splits:
        for r in read_gz(bdir / f"points_{s}.csv.gz"):
            for k in ("t", "p0", "p1", "left_s", "right_s"):
                r[k] = float(r[k])
            for k in [k for k in r if k.startswith("elig_")] + ["low_align"]:
                r[k] = r[k] == "True"
            pts[r["point_id"]] = r
        matches += read_gz(bdir / f"matches_{s}.csv.gz")
    return pts, matches


def primary_block(pts: dict, matches: list[dict], splits: list[str], args, taus: dict, rng) -> dict:
    chans = sorted({r["speaker_id"] for r in pts.values()})
    return {"h1": run_h1(pts, matches, args.point_emb, splits, rng, chans),
            "h2_p0": run_h2(pts, args.emb, splits, taus, rng, chans),
            "h3": run_h3(args.windows, args.emb, splits, rng, chans)}


def compact(block: dict) -> dict:
    keys = {"h1": ("dz", "dz_ci"), "h2_p0": ("auc", "auc_ci"), "h3": ("shift_in_target_sd", "ci")}
    return {h: {m: {"estimate": r[keys[h][0]], "ci": r[keys[h][1]], "p_one_holm": r["p_one_holm"],
                    "outcome": r["outcome"]["category"], "n_channels": r["n_channels"]}
                for m, r in block[h]["primary"].items()} for h in keys}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--boundaries", type=Path, required=True)
    ap.add_argument("--point-emb", type=Path, required=True)
    ap.add_argument("--emb", type=Path, required=True)
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--align", type=Path, required=True)
    ap.add_argument("--pilot", type=Path, required=True, help="pilot JSON that holds the frozen tau")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--tag", default="m2a")
    ap.add_argument("--seed", type=int, default=20261018)
    args = ap.parse_args()
    mode = check_freeze(args.splits, {k: getattr(args, k) for k in
                                      ("boundaries", "point_emb", "emb", "windows", "align", "pilot")})
    rng = np.random.default_rng(args.seed)
    taus = frozen_taus(args.pilot)
    pts, matches = load_points(args.boundaries, args.splits)
    need = {"split", "score_left", "score_right", "pause_bin", "pos_bin", "speaker_id", "video_id", "clip_id"}
    if pts and not need <= set(next(iter(pts.values()))):
        raise SystemExit(f"refusing: point tables lack {sorted(need - set(next(iter(pts.values()))))}")
    if any(r["low_align"] and any(v for k, v in r.items() if k.startswith("elig_")) for r in pts.values()):
        raise SystemExit("refusing: a low-alignment point is marked eligible")
    out = {"tag": args.tag, "splits": args.splits, "seed": args.seed, **mode, "tau": taus,
           **primary_block(pts, matches, args.splits, args, taus, rng)}
    args.out.mkdir(parents=True, exist_ok=True)
    write_outputs(out, args)  # primary results first, so a later failure cannot lose them
    fp, fm, thr = alignment_filter(pts, matches, args.align)
    out["sensitivity_alignment"] = {"word_score_threshold": thr, "points_kept": len(fp), "points_all": len(pts),
                                    "h1": {m: {"dz": v["dz"], "dz_ci": v["dz_ci"]}
                                           for m, v in run_h1(fp, fm, args.point_emb, args.splits, rng)["primary"].items()},
                                    "h2_p0": {m: {"auc": v["auc"], "auc_ci": v["auc_ci"]}
                                              for m, v in run_h2(fp, args.emb, args.splits, taus, rng)["primary"].items()}}
    if "test" in args.splits and len(args.splits) > 1:
        # Pre-registered sensitivity: the confirmatory analysis without the 2 test channels (descriptive).
        sub = [s for s in args.splits if s != "test"]
        sp = {p: r for p, r in pts.items() if r["split"] in sub}
        sm = [m for m in matches if m["ssdl_id"] in sp and m["ctrl_id"] in sp]
        out["sensitivity_without_test_channels"] = {"splits": sub, **compact(primary_block(sp, sm, sub, args, taus, rng))}
    write_outputs(out, args)
    return 0


def write_outputs(out: dict, args) -> None:
    (args.out / f"confirm_{args.tag}.json").write_text(json.dumps(out, indent=2, default=float))
    lines = [f"# D1 outcome table ({args.tag}; {out['mode']})", "",
             "| H | Model | Channels | Estimate | CI | p (one-sided, Holm) | Outcome |", "|---|---|---|---|---|---|---|"]
    for h, key, est, ci_key, fmt in (("H1", "h1", "dz", "dz_ci", "dz {:+.3f}"),
                                     ("H2-P0", "h2_p0", "auc", "auc_ci", "AUC {:.3f}"),
                                     ("H3", "h3", "shift_in_target_sd", "ci", "shift {:+.3f} SD")):
        for m, r in out[key]["primary"].items():
            ci = r[ci_key]
            lines.append(f"| {h} | {m} | {r['n_channels']} | {fmt.format(r[est])} | [{ci['lo']:.3f}, {ci['hi']:.3f}] | "
                         f"{r['p_one_holm']:.4f} | {r['outcome']['category']} |")
    (args.out / f"confirm_{args.tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    sys.exit(main())
