#!/usr/bin/env python3
"""Pilot analysis on the 3 D1 validation channels (PLAN.md §6 "Pilot"; baselines only).

Pre-registration hygiene: before the M2a freeze, outcome statistics may only be computed on the
validation channels. This module therefore reads only the validation split, and fails if asked for any other.

Computed per embedding (E1-E3):
- H1 (embedding family): cos(W_left, W_right) at each switch point and at its 1:3 matched controls (point-anchored,
  pause-excluded windows). Delta-cos = mean(control) - switch. Reported per channel; dz at the point level
  (grand mean of channel means / pooled within-channel SD); and the mean in target-score SD units, where the
  target-score SD is the SD of control cosines (same-speaker, same-lag trials).
- H2 / P0 (detection family, L = 1.0 s grid): d(t) = 1 - cos(left, right) on the dense 0.1 s grid. AUC of
  d at switch points against controls, standardised by cell (video x pause bin x position bin), per channel.
  Peaks: local maxima of d within +-0.5 s (non-maximum suppression). tau is set so that 10% of controls have a
  peak above tau within +-c (c = 0.5 s). FAR at that tau for switch points and controls, by stratum.
- Variance components and a power simulation for the 17 confirmatory channels at the smallest effects of
  interest (H1 dz = 0.2; H2 AUC = 0.55), with the confirmatory channels' own point counts (counts only).
Output: results/pilot/pilot_<tag>.json and .md
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HOP = 0.1
C = 0.5
NMS = 0.5
MODELS = ("e1", "e2", "e3")
PILOT_SPLIT = "validation"


def read_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def unit(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def stratified_auc(pos: dict, neg: dict) -> tuple[float, int]:
    """Cell-standardised AUC: per cell P(pos > neg) + 0.5 P(tie), weighted by the cell's positives."""
    num = den = 0.0
    for cell, ps in pos.items():
        ns = neg.get(cell)
        if not ns:
            continue
        p, n = np.asarray(ps), np.sort(np.asarray(ns))
        lo, hi = np.searchsorted(n, p, "left"), np.searchsorted(n, p, "right")
        num += float(((lo + 0.5 * (hi - lo)) / len(n)).sum())
        den += len(p)
    return (num / den if den else float("nan")), int(den)


def h1(points: dict, matches: list[dict], pe_dir: Path, model: str, L: float,
       splits: tuple[str, ...] = (PILOT_SPLIT,)) -> dict:
    cos = {}
    for split in splits:
        z = np.load(pe_dir / model / f"{split}_L{L:g}.npz")
        cos.update(zip(z["point_id"], (unit(z["left"]) * unit(z["right"])).sum(1)))
    fam = f"emb{L:g}"
    sets = defaultdict(list)
    for m in matches:
        if m["family"] == fam:
            sets[m["ssdl_id"]].append(m["ctrl_id"])
    per_chan = defaultdict(list)
    for s, cs in sets.items():
        per_chan[points[s]["speaker_id"]].append(float(np.mean([cos[c] for c in cs]) - cos[s]))
    ctrl_cos = [cos[p] for p, r in points.items() if r["kind"] == "ctrl" and r[f"elig_{fam}"] and p in cos]
    chans = {c: {"n": len(v), "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))} for c, v in per_chan.items()}
    means = np.array([v["mean"] for v in chans.values()])
    within = math.sqrt(sum((v["n"] - 1) * v["sd"] ** 2 for v in chans.values()) /
                       max(sum(v["n"] - 1 for v in chans.values()), 1))
    tsd = float(np.std(ctrl_cos, ddof=1))
    return {"channels": chans, "grand_mean": float(means.mean()), "between_channel_sd": float(means.std(ddof=1)),
            "within_channel_sd": within, "dz_point": float(means.mean() / within), "target_score_sd": tsd,
            "mean_in_target_sd": float(means.mean() / tsd), "n_sets": len(sets),
            "n_ssdl_unmatched": sum(1 for p, r in points.items() if r["kind"] == "ssdl" and r[f"elig_{fam}"]
                                    and p not in sets),
            "_values": {c: v for c, v in per_chan.items()}}


def p0_curves(emb_dir: Path, model: str, splits: tuple[str, ...] = (PILOT_SPLIT,)) -> dict[str, np.ndarray]:
    lag = int(round(1.0 / HOP))
    out = {}
    for split in splits:
        z = np.load(emb_dir / f"{model}_L1" / f"{split}.npz")
        for clip in z.keys():
            e = unit(z[clip])
            d = np.full(len(e), np.nan, np.float32)
            if len(e) > lag:
                d[lag:] = 1 - (e[lag:] * e[:-lag]).sum(1)  # d[k]: boundary at t = 0.1k, windows k-10 and k
            out[clip] = d
    return out


def peak_strength(d: np.ndarray, k: int) -> float:
    """Largest NMS peak value of d within +-C of grid index k (-inf if none)."""
    w, r = int(round(NMS / HOP)), int(round(C / HOP))
    best = -np.inf
    for j in range(max(k - r, 0), min(k + r, len(d) - 1) + 1):
        if np.isnan(d[j]):
            continue
        nb = d[max(j - w, 0): j + w + 1]
        if d[j] >= np.nanmax(nb):
            best = max(best, float(d[j]))
    return best


P0_CELLS = {  # name: (switch-point family, control family); decision 16
    "p0L1": ("p0L1", "det_w1"),      # primary: both runs >= 1 s, P0's +-1 s field clean on both sides
    "det": ("det", "det"),           # secondary: all switch points, +-0.5 s clean
    "det_w1": ("det_w1", "det_w1"),  # sensitivity: all switch points, +-1 s clean
}
TAU_CONTROLS = "det_w1"  # tau: 10% of these controls have a peak above tau within +-c


def score_points(points: dict, curves: dict) -> tuple[dict, dict]:
    """d(t) at the point and the strongest NMS peak within +-c, for every point with a defined d(t)."""
    vals, dropped = {}, defaultdict(int)
    for pid, r in points.items():
        if not any(r[f"elig_{f}"] for f in ("det", "det_w1", "p0L1")):
            continue
        d = curves.get(r["clip_id"])
        k = int(round(r["t"] / HOP))
        if d is None or k >= len(d) or np.isnan(d[k]):
            dropped[r["kind"]] += 1  # within 1 s of a clip edge: P0 has no left window
            continue
        vals[pid] = {"d": float(d[k]), "peak": peak_strength(d, k)}
    return vals, dict(dropped)


def calibrate_tau(points: dict, vals: dict) -> tuple[float, float]:
    peaks = np.array([v["peak"] for p, v in vals.items()
                      if points[p]["kind"] == "ctrl" and points[p][f"elig_{TAU_CONTROLS}"]])
    finite = np.isfinite(peaks)
    share = float(finite.mean())
    tau = float(np.quantile(peaks[finite], 1 - 0.10 / share)) if share > 0.10 else float("-inf")
    return tau, share


def h2_cell(points: dict, vals: dict, tau: float, s_fam: str, c_fam: str) -> dict:
    cell = lambda r: (r["video_id"], r["pause_bin"], r["pos_bin"])
    ctrl_d, ctrl_hit = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for p, v in vals.items():
        r = points[p]
        if r["kind"] == "ctrl" and r[f"elig_{c_fam}"]:
            ctrl_d[r["speaker_id"]][cell(r)].append(v["d"])
            ctrl_hit[r["speaker_id"]][cell(r)].append(v["peak"] > tau)
    out = {"channels": {}, "strata": defaultdict(dict), "n": defaultdict(int)}
    strata_acc = defaultdict(lambda: {"pos": defaultdict(lambda: defaultdict(list)), "hits": [], "ctrl_rate": []})
    for chan in sorted({points[p]["speaker_id"] for p in vals}):
        pos, hits, rates = defaultdict(list), [], []
        for p, v in vals.items():
            r = points[p]
            if r["speaker_id"] != chan or r["kind"] != "ssdl" or not r[f"elig_{s_fam}"]:
                continue
            out["n"]["ssdl_scored"] += 1
            c = cell(r)
            if not ctrl_hit[chan].get(c):
                out["n"]["ssdl_without_cell_controls"] += 1
                continue
            pos[c].append(v["d"])
            h, rate = float(v["peak"] > tau), float(np.mean(ctrl_hit[chan][c]))
            hits.append(h); rates.append(rate)
            s = strata_acc[r["stratum"]]
            s["pos"][chan][c].append(v["d"]); s["hits"].append(h); s["ctrl_rate"].append(rate)
        auc, n = stratified_auc(pos, ctrl_d[chan])
        out["channels"][chan] = {"auc": auc, "n_ssdl": n, "far_ssdl": float(np.mean(hits)) if hits else None,
                                 "far_ctrl_std": float(np.mean(rates)) if rates else None,
                                 "delta_far_std": float(np.mean(hits) - np.mean(rates)) if hits else None}
    for st, s in strata_acc.items():
        aucs = [stratified_auc(s["pos"][ch], ctrl_d[ch])[0] for ch in s["pos"]]
        out["strata"][st] = {"n": len(s["hits"]), "auc_mean_of_channels": float(np.nanmean(aucs)),
                             "far_ssdl": float(np.mean(s["hits"])), "far_ctrl_std": float(np.mean(s["ctrl_rate"])),
                             "delta_far_std": float(np.mean(s["hits"]) - np.mean(s["ctrl_rate"]))}
    aucs = [c["auc"] for c in out["channels"].values() if c["auc"] is not None and not np.isnan(c["auc"])]
    out["auc_mean_of_channels"] = float(np.mean(aucs)) if aucs else float("nan")
    out["auc_between_channel_sd"] = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else float("nan")
    dfs = [c["delta_far_std"] for c in out["channels"].values() if c["delta_far_std"] is not None]
    out["delta_far_std_mean_of_channels"] = float(np.mean(dfs)) if dfs else float("nan")
    out["n_channels_with_estimate"] = len(dfs)
    out["strata"], out["n"] = dict(out["strata"]), dict(out["n"])
    return out


def h2(points: dict, curves: dict, tau: float | None = None) -> dict:
    """P0 statistics. With tau=None (pilot) tau is calibrated here; the confirmatory run passes the frozen tau."""
    vals, dropped = score_points(points, curves)
    share = None
    if tau is None:
        tau, share = calibrate_tau(points, vals)
    return {"tau": tau, "tau_controls": TAU_CONTROLS, "ctrl_with_any_peak": share, "dropped_edge": dropped,
            "cells": {name: h2_cell(points, vals, tau, s, c) for name, (s, c) in P0_CELLS.items()}}


_SIGNS: dict[int, np.ndarray] = {}


def signflip_p(x: np.ndarray, n_mc: int, rng) -> float:
    """One-sided sign-flip p-value for mean(x) > 0 (exact if 2^len(x) <= n_mc)."""
    n = len(x)
    if 2 ** n <= n_mc:
        if n not in _SIGNS:
            _SIGNS[n] = (((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1).astype(np.float32)
        signs = _SIGNS[n]
    else:
        signs = rng.choice([-1.0, 1.0], size=(n_mc, n)).astype(np.float32)
    flipped = signs @ x.astype(np.float32) / n
    return float((flipped >= x.mean() - 1e-7).mean())


def power_h1(within_sd: float, between_sd: float, counts: list[int], dz: float, alpha: float, rng,
             n_sim: int = 2000) -> dict:
    hits_p = hits_both = 0
    delta = dz * within_sd
    for _ in range(n_sim):
        means = np.array([delta + rng.normal(0, between_sd) + rng.normal(0, within_sd / math.sqrt(max(n, 1)))
                          for n in counts])
        p = signflip_p(means, 2 ** 17, rng)
        est = means.mean() / within_sd
        hits_p += p < alpha
        hits_both += (p < alpha) and est >= 0.2
    return {"dz_true": dz, "power_p": hits_p / n_sim, "power_p_and_sesoi": hits_both / n_sim}


def power_h2(between_sd: float, se_by_n: dict, counts: list[int], auc: float, alpha: float, rng,
             n_sim: int = 2000) -> dict:
    hits_p = hits_both = 0
    for _ in range(n_sim):
        a = np.array([auc - 0.5 + rng.normal(0, between_sd) + rng.normal(0, se_by_n(n)) for n in counts])
        p = signflip_p(a, 2 ** 17, rng)
        hits_p += p < alpha
        hits_both += (p < alpha) and (a.mean() + 0.5) >= 0.55
    return {"auc_true": auc, "power_p": hits_p / n_sim, "power_p_and_sesoi": hits_both / n_sim}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--boundaries", type=Path, required=True)
    ap.add_argument("--point-emb", type=Path, required=True)
    ap.add_argument("--emb", type=Path, required=True, help="dense grid embeddings folder")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--seed", type=int, default=20261018)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    pts = {}
    for r in read_gz(args.boundaries / f"points_{PILOT_SPLIT}.csv.gz"):
        if r["split"] != PILOT_SPLIT:
            raise SystemExit("pilot refuses non-validation points")
        for k in ("t", "p0", "p1", "left_s", "right_s"):
            r[k] = float(r[k])
        for k in [k for k in r if k.startswith("elig_")] + ["low_align"]:
            r[k] = r[k] == "True"
        pts[r["point_id"]] = r
    matches = read_gz(args.boundaries / f"matches_{PILOT_SPLIT}.csv.gz")
    summary = json.loads((args.boundaries / "summary.json").read_text())
    # Counts per confirmatory channel (counts only, no outcomes; train + test): matched sets for H1 (emb1), and
    # switch points of the primary P0 cell that have at least one control in their cell.
    conf_counts = {"emb1_sets": defaultdict(int), "p0L1": defaultdict(int)}
    for split in ("train", "test"):
        prow = {r["point_id"]: r for r in read_gz(args.boundaries / f"points_{split}.csv.gz")}
        for sid in {m["ssdl_id"] for m in read_gz(args.boundaries / f"matches_{split}.csv.gz") if m["family"] == "emb1"}:
            conf_counts["emb1_sets"][prow[sid]["speaker_id"]] += 1
        cells = {(r["video_id"], r["pause_bin"], r["pos_bin"]) for r in prow.values()
                 if r["kind"] == "ctrl" and r["elig_det_w1"] == "True"}
        for r in prow.values():
            if r["kind"] == "ssdl" and r["elig_p0L1"] == "True" and (r["video_id"], r["pause_bin"], r["pos_bin"]) in cells:
                conf_counts["p0L1"][r["speaker_id"]] += 1
    out = {"tag": args.tag, "split": PILOT_SPLIT, "seed": args.seed, "boundaries": str(args.boundaries),
           "lexicon": {"loans": summary["loans"], "names": summary["names"]},
           "confirmatory_channels": {f: len(v) for f, v in conf_counts.items()},
           "confirmatory_counts_per_channel": {f: sorted(v.values()) for f, v in conf_counts.items()},
           "h1": {}, "h2": {}, "power": {}}
    alpha_h1, alpha_h2 = 0.05 / 3, 0.05 / 5  # Bonferroni over each family's m (H1: m = 3; H2: m = 5)
    # With 3 pilot channels the between-channel SD is poorly estimated: use its upper 80% confidence bound
    # (chi-square, 2 df: factor sqrt(2 / 0.446) = 2.12) in the power simulation.
    inflate = math.sqrt(2 / 0.446)
    for m in MODELS:
        out["h1"][m] = {f"L{L:g}": h1(pts, matches, args.point_emb, m, L) for L in (0.5, 1.0, 1.5)}
        out["h2"][m] = h2(pts, p0_curves(args.emb, m))
        p1 = out["h1"][m]["L1"]
        between = p1["between_channel_sd"] * inflate
        p1["variance_components"] = {"within_sd": p1["within_channel_sd"], "between_sd": p1["between_channel_sd"],
                                     "between_sd_upper80": between,
                                     "icc": p1["between_channel_sd"] ** 2 / (p1["between_channel_sd"] ** 2
                                                                             + p1["within_channel_sd"] ** 2)}
        counts = sorted(conf_counts["emb1_sets"].values())
        out["power"][m] = {"h1": [power_h1(p1["within_channel_sd"], between, counts, dz, alpha_h1, rng)
                                  for dz in (0.2, 0.3)]}
        prim = out["h2"][m]["cells"]["p0L1"]
        # within-channel sampling SD of an AUC from n switch points (Hanley-McNeil, AUC near 0.55); ignores the
        # dependence of points within a clip, so it is optimistic
        se = lambda n: math.sqrt(0.55 * 0.45 / max(n, 1) * (1 + 1 / 3)) if n else 1.0
        counts_p0 = sorted(conf_counts["p0L1"].values())
        out["power"][m]["h2_p0L1"] = [power_h2(max(prim["auc_between_channel_sd"] * inflate, 0.01), se, counts_p0,
                                               a, alpha_h2, rng) for a in (0.55, 0.60)]
    args.out.mkdir(parents=True, exist_ok=True)
    for m in MODELS:
        for v in out["h1"][m].values():
            v.pop("_values", None)
    (args.out / f"pilot_{args.tag}.json").write_text(json.dumps(out, indent=2))
    lines = [f"# Pilot {args.tag} (validation channels only, baselines)", "",
             "| Model | H1 Δcos (L=1) | dz | sets | P0 AUC p0L1 (primary) | P0 AUC det (pooled) | τ | ΔFAR p0L1 "
             "(cell-std) | power H1 dz=0.2 | power P0 AUC=0.55 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        a, b, pw = out["h1"][m]["L1"], out["h2"][m], out["power"][m]
        prim, pooled = b["cells"]["p0L1"], b["cells"]["det"]
        lines.append(f"| {m} | {a['grand_mean']:+.4f} | {a['dz_point']:+.3f} | {a['n_sets']} | "
                     f"{prim['auc_mean_of_channels']:.3f} | {pooled['auc_mean_of_channels']:.3f} | {b['tau']:.3f} | "
                     f"{prim['delta_far_std_mean_of_channels']:+.3f} | {pw['h1'][0]['power_p']:.2f} | "
                     f"{pw['h2_p0L1'][0]['power_p']:.2f} |")
    (args.out / f"pilot_{args.tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
