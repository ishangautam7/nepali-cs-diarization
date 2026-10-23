#!/usr/bin/env python3
"""Attribution analyses for H1 and H2-P0 (PLAN.md §1 claim grading; PREREGISTRATION.md §6; decision 18).

An effect that is significant at switch points (switch-associated) is called language-attributable for an
embedding only if all of the following hold. This is an intersection-union rule: each condition is tested at
alpha = .05 and requiring all of them keeps the error rate at alpha, so no further adjustment is made.
 (a) the primary outcome is significant in the predicted direction (Holm; csdiar.analysis.confirm).
 (b) Covariate adjustment, tested at the channel level: within each channel, y is regressed by least squares
     on the condition and the five acoustic-change covariates (|right - left| of f0, energy, rate, tilt and snr;
     csdiar.data.acoustic) after demeaning within sets, which absorbs set-level differences including the pause
     and position bins. The per-channel condition coefficients get an exact one-sided sign-flip over channels
     (p < .05). H1: y = window cosine over the L = 1.0 s matched sets (emb windows; set = matched set; predicted:
     switch points lower). H2-P0: y = the false-alarm indicator (an NMS peak of d(t) above the frozen tau within
     +-c) over the p0L1 switch points and the det_w1 controls of their cells (p0 windows; set = the video x pause
     bin x position bin cell; predicted: switch points higher). The lme4 mixed models of PLAN.md §7
     (csdiar/analysis/glmm.R) are fitted on the same rows and reported as a descriptive model-based summary; they
     do not enter the grade, because a Wald z without the random slope is not a valid level-alpha test here.
 (c) Phrase-boundary contrast: with only points whose pause is > 150 ms, the effect is still significant (exact
     one-sided sign-flip over channels, p < .05). H1: matched sets whose switch point has a pause > 150 ms;
     exact matching on pause bin makes their controls phrase-boundary controls. H2-P0: the cell-standardised
     AUC within the 150-300 ms and > 300 ms pause cells.
 (d) H3 is significant in the predicted direction for the same embedding (Holm; csdiar.analysis.confirm).
Grades: language-attributable (a-d all hold), switch-associated (a holds), not shown (a fails). Each grade is
reported with the primary outcome category, e.g. "language-attributable (nonzero but trivial)". A criterion that
cannot be assessed (fewer than MIN_CHANNELS = 5 channels, where an exact one-sided sign-flip cannot reach
p < .05) counts as not met and is flagged. All analyses use the channels the primary test used; a channel enters
(b) with at least 10 switch points and a phrase-boundary contrast with at least MIN_PHRASE_UNITS = 5 units.
Descriptive: per-channel FAR of the phrase-boundary controls (det_w1 controls with a pause > 150 ms), of the
other det_w1 controls and of the p0L1 switch points with cell controls.
Refuses non-validation splits without a matching freeze record (csdiar.analysis.confirm.check_freeze).
Output: <out>/attribution_<tag>.json and .md; model inputs and R outputs in <out>/attribution_<tag>_models/.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np

from csdiar.analysis.confirm import check_freeze, frozen_taus, load_points
from csdiar.analysis.pilot import MODELS, p0_curves, score_points, stratified_auc, unit
from csdiar.analysis.stats import signflip

PAUSE_BINS = {"none", "lt150", "150to300", "gt300"}
PHRASE_BINS = ("150to300", "gt300")
MIN_UNITS = 10  # switch points a channel needs to enter criterion (b), as for the primary tests
MIN_PHRASE_UNITS = 5  # a channel enters a phrase-boundary contrast only with at least this many units
MIN_CHANNELS = 5  # below this an exact one-sided sign-flip cannot reach p < .05 (2^-4 = .0625): not assessable
MEASURES = ("f0", "energy", "rate", "tilt", "snr")
NA_COV = {f"d_{m}": None for m in MEASURES}
COLUMNS = ["y", "cond", "speaker", "video", "clip", "set", "pause_bin", "pos_bin"] + [f"d_{m}" for m in MEASURES]
ALPHA = 0.05
GLMM_R = Path(__file__).resolve().with_name("glmm.R")
_N = NormalDist()


def load_acoustic(adir: Path, splits: list[str], cfg: str) -> dict[str, dict]:
    out = {}
    for s in splits:
        with gzip.open(adir / f"acoustic_{s}.csv.gz", "rt", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                d = {}
                for m in MEASURES:
                    lv, rv = r[f"{cfg}_{m}_l"], r[f"{cfg}_{m}_r"]
                    d[f"d_{m}"] = abs(float(rv) - float(lv)) if lv != "" and rv != "" else None
                out[r["point_id"]] = d
    return out


def base_row(r: dict, y: float, cond: int, set_id: str, cov: dict) -> dict:
    return {"y": y, "cond": cond, "speaker": r["speaker_id"], "video": r["video_id"] or r["clip_id"],
            "clip": r["clip_id"], "set": set_id, "pause_bin": r["pause_bin"], "pos_bin": r["pos_bin"], **cov}


def h1_inputs(points: dict, matches: list[dict], pe_dir: Path, model: str, splits: list[str], cov: dict,
              exclude: set, rng: np.random.Generator) -> tuple[list[dict], dict]:
    """Rows for the H1 mixed model and the H1 phrase-boundary contrast, on the channels of the primary test."""
    cos = {}
    for s in splits:
        z = np.load(pe_dir / model / f"{s}_L1.npz")
        cos.update(zip(z["point_id"], (unit(z["left"]) * unit(z["right"])).sum(1)))
    sets = defaultdict(list)
    for m in matches:
        if m["family"] == "emb1" and points[m["ssdl_id"]]["speaker_id"] not in exclude:
            sets[m["ssdl_id"]].append(m["ctrl_id"])
    rows, phrase = [], defaultdict(list)
    for s, cs in sets.items():
        for p, cond in [(s, 1)] + [(c, 0) for c in cs]:
            y = cos.get(p)
            rows.append(base_row(points[p], None if y is None else float(y), cond, s, cov.get(p, NA_COV)))
        if points[s]["pause_bin"] in PHRASE_BINS and s in cos and all(c in cos for c in cs):
            phrase[points[s]["speaker_id"]].append(float(np.mean([cos[c] for c in cs]) - cos[s]))
    inc = {c: v for c, v in phrase.items() if len(v) >= MIN_PHRASE_UNITS}
    means = [float(np.mean(v)) for v in inc.values()]
    contrast = {"statistic": "per-channel mean delta-cos in sets with a pause > 150 ms (predicted > 0)",
                "n_sets": sum(len(v) for v in inc.values()), "n_channels": len(means),
                "excluded_channels": {c: len(v) for c, v in phrase.items() if c not in inc},
                "mean_of_channels": float(np.mean(means)) if means else None,
                "channels": {c: {"n": len(v), "mean": float(np.mean(v))} for c, v in inc.items()},
                "signflip": signflip(means, rng)}
    return rows, contrast


def p0_inputs(points: dict, emb_dir: Path, model: str, splits: list[str], tau: float, cov: dict,
              exclude: set, rng: np.random.Generator) -> tuple[list[dict], dict, dict]:
    """Rows for the P0 GLMM, the P0 phrase-boundary contrast and descriptive FARs, on the primary test's channels."""
    vals, _ = score_points(points, p0_curves(emb_dir, model, tuple(splits)))
    cell = lambda r: (r["speaker_id"], r["video_id"], r["pause_bin"], r["pos_bin"])  # as pilot.h2_cell, per channel
    ctrl = defaultdict(list)
    for p in vals:
        r = points[p]
        if r["kind"] == "ctrl" and r["elig_det_w1"] and r["speaker_id"] not in exclude:
            ctrl[cell(r)].append(p)
    sw = [p for p in vals if points[p]["kind"] == "ssdl" and points[p]["elig_p0L1"] and ctrl.get(cell(points[p]))]
    used = sorted({cell(points[p]) for p in sw})
    sid = lambda k: "|".join(map(str, k))
    rows = [base_row(points[p], float(vals[p]["peak"] > tau), 1, sid(cell(points[p])), cov.get(p, NA_COV)) for p in sw]
    rows += [base_row(points[p], float(vals[p]["peak"] > tau), 0, sid(k), cov.get(p, NA_COV))
             for k in used for p in ctrl[k]]
    pos, neg = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for p in sw:
        k = cell(points[p])
        if k[2] in PHRASE_BINS:
            pos[k[0]][k[1:]].append(vals[p]["d"])
    for k, ps in ctrl.items():
        neg[k[0]][k[1:]] += [vals[p]["d"] for p in ps]
    aucs, exc = {}, {}
    for chan in sorted(pos):
        auc, n = stratified_auc(pos[chan], neg[chan])
        if auc == auc and n >= MIN_PHRASE_UNITS:
            aucs[chan] = {"auc": auc, "n_ssdl": n}
        else:
            exc[chan] = n
    a = [v["auc"] - 0.5 for v in aucs.values()]
    contrast = {"statistic": "per-channel cell-standardised AUC within the pause > 150 ms cells (predicted > 0.5)",
                "n_channels": len(a), "excluded_channels": exc,
                "auc_mean_of_channels": float(np.mean(a)) + 0.5 if a else None,
                "channels": aucs, "signflip": signflip(a, rng)}
    far = defaultdict(lambda: {"ssdl": [], "phrase_ctrl": [], "other_ctrl": []})
    for p in sw:
        far[points[p]["speaker_id"]]["ssdl"].append(vals[p]["peak"] > tau)
    for k, ps in ctrl.items():
        for p in ps:
            far[k[0]]["phrase_ctrl" if k[2] in PHRASE_BINS else "other_ctrl"].append(vals[p]["peak"] > tau)
    desc = {g: float(np.mean([np.mean(f[g]) for f in far.values() if f[g]])) for g in ("ssdl", "phrase_ctrl", "other_ctrl")}
    desc["note"] = "mean over channels of the per-channel FAR at the frozen tau"
    return rows, contrast, desc


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow(["NA" if r[c] is None else r[c] for c in COLUMNS])


def adjusted_effects(rows: list[dict], direction: int, rng: np.random.Generator) -> dict:
    """Criterion (b): per-channel covariate-adjusted condition effect, then an exact sign-flip over channels.

    Within each channel, complete-case rows are demeaned within their set (H1: matched set; P0: cell), which
    absorbs set-level differences including pause and position bins, and y is regressed by least squares on the
    condition and the five covariates. A set counts only with at least one switch point and one control left. A
    channel enters with at least MIN_UNITS switch points in such sets. The sign-flip is one-sided in the predicted
    direction and is assessable only with at least MIN_CHANNELS channels.
    """
    by_chan = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["y"] is not None and all(r[f"d_{m}"] is not None for m in MEASURES):
            by_chan[r["speaker"]][r["set"]].append(r)
    chans, exc = {}, {}
    for chan, sets in sorted(by_chan.items()):
        blocks = [s for s in sets.values() if any(r["cond"] == 1 for r in s) and any(r["cond"] == 0 for r in s)]
        n_sw = sum(r["cond"] == 1 for s in blocks for r in s)
        if n_sw < MIN_UNITS:
            exc[chan] = n_sw
            continue
        y, X = [], []
        for s in blocks:
            ys = np.array([r["y"] for r in s], dtype=np.float64)
            xs = np.array([[r["cond"]] + [r[f"d_{m}"] for m in MEASURES] for r in s], dtype=np.float64)
            y.append(ys - ys.mean())
            X.append(xs - xs.mean(0))
        beta = np.linalg.lstsq(np.vstack(X), np.concatenate(y), rcond=None)[0]
        chans[chan] = {"beta_cond": float(beta[0]), "n_switch": int(n_sw), "n_sets": len(blocks)}
    b = [direction * v["beta_cond"] for v in chans.values()]
    return {"statistic": "per-channel covariate-adjusted condition coefficient (within-set least squares); "
                         "sign-flip on direction x coefficient", "direction": direction,
            "n_channels": len(b), "excluded_channels": exc, "assessable": len(b) >= MIN_CHANNELS,
            "mean_coefficient": float(np.mean([v["beta_cond"] for v in chans.values()])) if chans else None,
            "channels": chans, "signflip": signflip(b, rng)}


def wald(res: dict, direction: int) -> dict:
    z = (res.get("cond") or {}).get("z")
    if res.get("failed") or z is None:
        return {"p_one": float("nan"), "p_two": float("nan"), "direction": direction}
    return {"p_one": _N.cdf(-direction * z), "p_two": 2 * (1 - _N.cdf(abs(z))), "direction": direction}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--boundaries", type=Path, required=True)
    ap.add_argument("--point-emb", type=Path, required=True)
    ap.add_argument("--emb", type=Path, required=True)
    ap.add_argument("--acoustic", type=Path, required=True)
    ap.add_argument("--pilot", type=Path, required=True, help="pilot JSON that holds the frozen tau")
    ap.add_argument("--confirm", type=Path, required=True, help="confirm_<tag>.json of the same splits")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--tag", default="m2a")
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--rscript", default=str(Path.home() / "opt" / "r-glmm" / "bin" / "Rscript"))
    args = ap.parse_args()
    mode = check_freeze(args.splits, {k: getattr(args, k) for k in ("boundaries", "point_emb", "emb", "acoustic", "pilot")})
    conf = json.loads(args.confirm.read_text())
    if conf["splits"] != args.splits:
        raise SystemExit(f"refusing: {args.confirm} covers {conf['splits']}, not {args.splits}")
    if conf.get("mode") != mode["mode"] or conf.get("freeze") != mode.get("freeze"):
        raise SystemExit(f"refusing: {args.confirm} was not produced under the same freeze and mode")
    rng = np.random.default_rng(args.seed)
    taus = frozen_taus(args.pilot)
    pts, matches = load_points(args.boundaries, args.splits)
    labels = {r["pause_bin"] for r in pts.values()}
    if not labels <= PAUSE_BINS or not set(PHRASE_BINS) <= labels:
        raise SystemExit(f"refusing: unexpected pause-bin labels {sorted(labels)}")
    if any(r["low_align"] and any(v for k, v in r.items() if k.startswith("elig_")) for r in pts.values()):
        raise SystemExit("refusing: a low-alignment point is marked eligible")
    n_no_video = sum(1 for r in pts.values() if not r["video_id"])  # 3 test clips without provenance (DEVIATIONS 1)
    cov = {cfg: load_acoustic(args.acoustic, args.splits, cfg) for cfg in ("emb", "p0")}
    mdir = args.out / f"attribution_{args.tag}_models"
    mdir.mkdir(parents=True, exist_ok=True)
    out = {"tag": args.tag, "splits": args.splits, "seed": args.seed, **mode, "tau": taus,
           "points_without_video_id": n_no_video, "models": {}}
    jobs = []
    for m in MODELS:
        ex_h1 = set(conf["h1"]["primary"][m].get("excluded_channels", {}))
        ex_p0 = set(conf["h2_p0"]["primary"][m].get("excluded_channels", {}))
        h1_rows, h1_con = h1_inputs(pts, matches, args.point_emb, m, args.splits, cov["emb"], ex_h1, rng)
        p0_rows, p0_con, p0_far = p0_inputs(pts, args.emb, m, args.splits, taus[m], cov["p0"], ex_p0, rng)
        for name, rows in (("h1", h1_rows), ("h2_p0", p0_rows)):
            if not rows or not any(r["cond"] == 1 for r in rows):
                raise SystemExit(f"refusing: no rows or no switch points for {name} {m}")
        out["models"][m] = {"h1": {"phrase_contrast": h1_con, "adjusted": adjusted_effects(h1_rows, -1, rng)},
                            "h2_p0": {"phrase_contrast": p0_con, "far": p0_far,
                                      "adjusted": adjusted_effects(p0_rows, +1, rng)}}
        for h, rows, fam in (("h1", h1_rows, "gaussian"), ("h2_p0", p0_rows, "binomial")):
            write_csv(rows, mdir / f"{h}_{m}.csv")
            jobs.append((m, h, subprocess.Popen([args.rscript, str(GLMM_R), str(mdir / f"{h}_{m}.csv"),
                                                 str(mdir / f"{h}_{m}.json"), fam],
                                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)))
    for m, h, proc in jobs:  # descriptive model-based summary; a failure is recorded, not fatal
        log = proc.communicate()[0]
        (mdir / f"{h}_{m}.log").write_text(log)
        jp = mdir / f"{h}_{m}.json"
        res = json.loads(jp.read_text()) if proc.returncode == 0 and jp.exists() else {"failed": True,
                                                                                        "log_tail": log[-2000:]}
        out["models"][m][h]["mixed_model"] = {**res, **wald(res, -1 if h == "h1" else +1)}
    lines = [f"# Attribution ({args.tag}; {out['mode']})", "",
             "| H | Model | (a) primary, Holm p [category] | (b) adjusted, channels, p | "
             "(c) pause > 150 ms only: channels, p | (d) H3, Holm p | Grade | lme4 (descriptive): coef "
             "[no covariates], structure |", "|---|---|---|---|---|---|---|---|"]
    est = lambda d, k: "NA" if d.get(k) is None else f"{d[k]['estimate']:+.4f}"
    for m in MODELS:
        h3p = conf["h3"]["primary"][m]["p_one_holm"]
        for h in ("h1", "h2_p0"):
            r = out["models"][m][h]
            prim = conf[h]["primary"][m]
            pa, cat = prim["p_one_holm"], prim["outcome"]["category"]
            adj, pcon, mm = r["adjusted"], r["phrase_contrast"], r["mixed_model"]
            pb, pc = adj["signflip"]["p_one"], pcon["signflip"]["p_one"]
            assessable = {"b": adj["assessable"], "c": pcon["n_channels"] >= MIN_CHANNELS}
            flags = {"a": pa < ALPHA, "b": assessable["b"] and pb < ALPHA, "c": assessable["c"] and pc < ALPHA,
                     "d": h3p < ALPHA}
            grade = ("language-attributable" if all(flags.values()) else
                     "switch-associated" if flags["a"] else "not shown")
            r.update(criteria=flags, assessable=assessable, grade=grade, primary_category=cat,
                     label=f"{grade} ({cat})")
            structure = mm.get("structure", "failed") + ("" if mm.get("accepted", True) else ", not accepted")
            lines.append(f"| {h} | {m} | {pa:.4f} [{cat}] | {adj['n_channels']}, {pb:.4f} | "
                         f"{pcon['n_channels']}, {pc:.4f} | {h3p:.4f} | {grade} | "
                         f"{est(mm, 'cond')} [{est(mm, 'cond_without_covariates')}], {structure} |")
    (args.out / f"attribution_{args.tag}.json").write_text(json.dumps(out, indent=2, default=float))
    (args.out / f"attribution_{args.tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
