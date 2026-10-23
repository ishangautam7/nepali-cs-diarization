#!/usr/bin/env python3
"""H4 pilot: does a mitigation lower the SS-DL false-split excess (PLAN.md §7, H4)?

Pre-registration hygiene: this PILOT runs on the 3 D1 VALIDATION channels only (like csdiar.analysis.pilot), to
size H4 and fix its operating choices before any confirmatory freeze. It computes, per channel and per system,
the P0 detector's cell-standardised SS-DL excess on the primary p0L1 cell:
  - B0  baseline: the unmodified E-embedding (the system H2-P0 already used).
  - B1  per-language mean centring (csdiar.mitigations.m1_subspace.language_mean_centre).
  - M1  SubspaceProjector: project out v_lang (csdiar.mitigations.m1_subspace).
The language direction v_lang (M1) and the per-language means (B1) are estimated LEAVE-ONE-CHANNEL-OUT across the
pilot channels, the pilot analogue of the confirmatory nested CV, so a channel's own language statistics never
enter its own score. H4's pre-registered support rule (Holm p < .05 AND >= 25% relative reduction in delta-FAR
vs B0) is a CONFIRMATORY test and is NOT applied here; the pilot only reports the per-channel delta-FAR and the
mean relative change, to check the sign and size before freezing.

Real speech only. No outcome statistics on train/test here.
Usage (box): .venv-pyannote4/bin/python -m csdiar.mitigations.eval_h4 --root ~/data/d1 --boundaries ~/data/d1/boundaries/lex_v1b --pilot ~/data/d1/results/pilot/pilot_v2.json --out ~/data/d1/results/h4_pilot
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from csdiar.analysis.pilot import HOP, MODELS, peak_strength, unit
from csdiar.mitigations.m1_subspace import estimate_v_lang, language_mean_centre, language_windows, project_out

SPLIT = "validation"
LAG = int(round(1.0 / HOP))


def read_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def lang_means(points: dict, pe_dir: Path, model: str, speakers: set) -> dict:
    """Per-language mean embedding (B1), estimated over the given speakers' switch-point windows."""
    by = language_windows(points, pe_dir, model, 1.0, (SPLIT,))
    acc = {"ne": [], "en": []}
    for (spk, _v), d in by.items():
        if spk in speakers:
            acc["ne"] += d["ne"]; acc["en"] += d["en"]
    return {k: np.mean(v, 0).astype(np.float32) for k, v in acc.items() if v}


def p0_curves_transformed(emb_dir: Path, model: str, transform) -> dict:
    """d(t) = 1 - cos(left, right) on the dense grid, after applying `transform` to every window embedding."""
    z = np.load(emb_dir / f"{model}_L1" / f"{SPLIT}.npz")
    out = {}
    for clip in z.keys():
        e = transform(z[clip].astype(np.float32))
        d = np.full(len(e), np.nan, np.float32)
        if len(e) > LAG:
            d[LAG:] = 1 - (e[LAG:] * e[:-LAG]).sum(1)
        out[clip] = d
    return out


def delta_far(points: dict, curves: dict, tau: float) -> dict:
    """Per-channel cell-standardised delta-FAR on the p0L1 cell (switch eligible p0L1, controls eligible det_w1)."""
    cell = lambda r: (r["video_id"], r["pause_bin"], r["pos_bin"])
    ctrl_hit = defaultdict(lambda: defaultdict(list))
    for pid, r in points.items():
        if r["kind"] == "ctrl" and r["elig_det_w1"]:
            d = curves.get(r["clip_id"]); k = int(round(r["t"] / HOP))
            if d is None or k >= len(d) or np.isnan(d[k]):
                continue
            ctrl_hit[r["speaker_id"]][cell(r)].append(peak_strength(d, k) > tau)
    out = {}
    for chan in sorted({r["speaker_id"] for r in points.values()}):
        hits, rates = [], []
        for pid, r in points.items():
            if r["speaker_id"] != chan or r["kind"] != "ssdl" or not r["elig_p0L1"]:
                continue
            d = curves.get(r["clip_id"]); k = int(round(r["t"] / HOP))
            if d is None or k >= len(d) or np.isnan(d[k]) or not ctrl_hit[chan].get(cell(r)):
                continue
            hits.append(float(peak_strength(d, k) > tau)); rates.append(float(np.mean(ctrl_hit[chan][cell(r)])))
        if hits:
            out[chan] = float(np.mean(hits) - np.mean(rates))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--boundaries", type=Path, required=True)
    ap.add_argument("--pilot", type=Path, required=True, help="pilot JSON with the frozen tau per model")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    pe_dir, emb_dir = args.root / "point_emb" / "lex_v1b", args.root / "emb"
    pts = {}
    for r in read_gz(args.boundaries / f"points_{SPLIT}.csv.gz"):
        for k in ("t",):
            r[k] = float(r[k])
        for k in [c for c in r if c.startswith("elig_")]:
            r[k] = r[k] == "True"
        pts[r["point_id"]] = r
    taus = {m: json.loads(args.pilot.read_text())["h2"][m]["tau"] for m in MODELS}
    channels = sorted({r["speaker_id"] for r in pts.values()})
    out = {"split": SPLIT, "systems": {}, "channels": channels}
    for m in MODELS:
        tau = taus[m]
        res = {"B0": {}, "B1": {}, "M1": {}}
        # leave-one-channel-out estimation of v_lang (M1) and language means (B1)
        v_by_chan, mu_by_chan = {}, {}
        for held in channels:
            others = set(channels) - {held}
            v_by_chan[held] = estimate_v_lang(pts, pe_dir, m, 1.0, (SPLIT,), speakers=others)
            mu_by_chan[held] = lang_means(pts, pe_dir, m, others)
        res["B0"] = delta_far(pts, p0_curves_transformed(emb_dir, m, lambda e: unit(e)), tau)
        # B1/M1 need a per-window language tag; the dense grid has no tags, so we transform with the
        # held-out-channel stats but apply a SINGLE channel's stats to its own clips (clip->channel via manifest)
        clip_chan = {r["clip_id"]: r["speaker_id"] for r in pts.values()}
        for name, estim in (("B1", mu_by_chan), ("M1", v_by_chan)):
            # transform depends on the clip's channel; build per-channel curves and merge
            merged = {}
            for held in channels:
                st = estim[held]
                if name == "M1":
                    tf = (lambda e, v=st: project_out(e, v)) if st is not None else (lambda e: unit(e))
                else:
                    # B1 without per-window language: centre by the mean of both language means (language-agnostic
                    # fallback for the dense grid) -- a conservative stand-in, flagged in the pilot writeup
                    mu = None if not st else 0.5 * (st.get("ne", 0) + st.get("en", 0))
                    tf = (lambda e, mu=mu: unit(e - mu)) if mu is not None else (lambda e: unit(e))
                cur = p0_curves_transformed(emb_dir, m, tf)
                merged.update({c: cur[c] for c in cur if clip_chan.get(c) == held})
            res[name] = delta_far(pts, merged, tau)
        # relative change vs B0 per channel
        rel = {}
        for chan in res["B0"]:
            b0 = res["B0"][chan]
            rel[chan] = {name: (None if abs(b0) < 1e-9 else (res[name].get(chan, float("nan")) - b0) / abs(b0))
                         for name in ("B1", "M1")}
        out["systems"][m] = {"tau": tau, "delta_far": res,
                             "mean_delta_far": {n: float(np.mean(list(res[n].values()))) for n in res if res[n]},
                             "rel_change_vs_B0": rel}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "h4_pilot.json").write_text(json.dumps(out, indent=2))
    for m in MODELS:
        s = out["systems"][m]["mean_delta_far"]
        print(f"{m}: mean dFAR B0={s.get('B0'):.4f} B1={s.get('B1'):.4f} M1={s.get('M1'):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
