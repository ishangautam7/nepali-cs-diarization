#!/usr/bin/env python3
"""Leave-one-channel-out sensitivity of the D1 main effects (post hoc, exploratory; review follow-up).

Added on 2026-10-22 after an external review; not pre-registered. It reads the frozen confirmatory result (tag
m2a-freeze, results/confirm/confirm_m2a.json) and never changes it, and the frozen verdicts stand whatever it shows.
For each main effect it gives the full-sample estimate and the 17 estimates with one confirmatory channel left out,
channel-weighted (each channel counts equally, as in the frozen analysis) and switch-weighted (each switch point counts
equally), with the exact one-sided sign-flip p over the remaining 16 channels (unadjusted). Every leave-one-out fit is
reported, and no other exclusion rule is tried.

Part A, exact (from the frozen per-channel statistics; no input is re-read):
  H1     channel-weighted: d_z = mean of the channel mean Delta-cos / pooled within-channel SD (frozen 'dz');
         switch-weighted: mean / SD over all matched sets, one set per switch point (frozen 'dz_plain').
  H2-P0  channel-weighted: mean of the per-channel cell-standardised p0L1 AUCs (frozen 'auc');
         switch-weighted: their n_ssdl-weighted mean, which equals the cell-standardised AUC pooled over all switch
         points (stratified_auc's numerator and denominator add up over channels).
  DFAR   the p0L1 operating-point excess at the frozen tau: channel mean (frozen 'delta_far_std') and n_ssdl-weighted.
  H3     cross- minus mono-language target-score shift in target-score SD units, channel mean (frozen
         'shift_in_target_sd'). Per-channel trial counts are not saved, so H3 has no switch-weighted form here, and a
         leave-one-out fit keeps the full-sample target-score SD as its unit.
Part B, rebuilt inputs: the pooled det and det_w1 AUCs (Table II) need per-channel values the frozen run did not save.
  The repository's own score_points and h2_cell run at the frozen tau on the boundary tables and grid embeddings
  rebuilt after the first box loss (DEVIATIONS.md entry 6; of these, only the E3 grid embeddings match FREEZE_M2A).
  The full-sample difference from the frozen values is reported, and every Part B number is labelled 'rebuilt inputs'.
Shared speaker: video zJYS-XNDP2Q of pilot channel UCeq9rTo9W93NfJNP9n1eRGw is the nearest cross-channel match of four
  videos of confirmatory channel UCOC6C8_U9PsAs3tQodDDmGw (results/purity/purity_audit.json). Reported: every effect
  without that channel (Part A, exact; Part B) and, from the rebuilt inputs, the detection effects without only those
  four videos.

Usage (repository root; Part A needs only numpy, Part B the analysis environment):
  PYTHONPATH=. python scripts/review_channel_exclusion.py --confirm results/confirm/confirm_m2a.json \\
      --purity results/purity/purity_audit.json \\
      --boundaries ../nepdiar-box-backup/data/d1/boundaries/lex_v1b --emb ../nepdiar-box-backup/data/d1/emb \\
      --out results/review_followups/channel_exclusion.json --md results/review_followups/channel_exclusion.md
  Without --boundaries and --emb only Part A runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from csdiar.analysis.stats import signflip

MODELS = ("e1", "e2", "e3")
MIN_UNITS = 10  # csdiar.analysis.confirm.MIN_UNITS: a channel enters a primary test with at least this many units
SPLITS_CONF = ("train", "test")
LABEL = ("POST HOC, EXPLORATORY: leave-one-channel-out sensitivity added on 2026-10-22 after an external review; "
         "not pre-registered; the frozen m2a-freeze results and verdicts are unchanged")
REPO = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(0)  # signflip's Monte Carlo fallback only; with at most 17 channels every test is exact
TOL = 1e-12
PILOT_CHANNEL, PILOT_VIDEO = "UCeq9rTo9W93NfJNP9n1eRGw", "zJYS-XNDP2Q"
CONF_CHANNEL = "UCOC6C8_U9PsAs3tQodDDmGw"

# effect: (null value, predicted direction, smallest effect of interest or None, description)
EFFECTS = {
    "h1_dz": (0.0, +1, 0.2, "H1 d_z (same-speaker similarity drop across a switch)"),
    "h2p0_auc": (0.5, +1, 0.55, "H2-P0 AUC (change score, primary cell p0L1)"),
    "dfar_p0L1": (0.0, +1, None, "Delta FAR at tau, p0L1 (operating-point excess)"),
    "h3_shift_sd": (0.0, -1, -0.2, "H3 shift (cross- minus mono-language target score, SD units)"),
    "det_auc": (0.5, +1, None, "pooled det AUC (all switch points; the pooled null)"),
    "det_w1_auc": (0.5, +1, None, "det_w1 AUC (all switch points, +-1 s controls)"),
}


def rel(p) -> str:
    """Path relative to the repository, or to its parent folder for inputs kept beside it (never a home path)."""
    p = Path(p).resolve()
    for base, prefix in ((REPO, ""), (REPO.parent, "../")):
        try:
            return prefix + str(p.relative_to(base))
        except ValueError:
            pass
    return p.name


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def h1_estimates(ch: dict) -> dict:
    """d_z both ways from per-channel {n, mean, sd} of the per-set Delta-cos values (confirm.py's definitions)."""
    n = np.array([v["n"] for v in ch.values()], float)
    m = np.array([v["mean"] for v in ch.values()], float)
    s = np.array([v["sd"] for v in ch.values()], float)
    ss_within = float(((n - 1) * s ** 2).sum())
    within = math.sqrt(ss_within / float((n - 1).sum()))
    grand = float((n * m).sum() / n.sum())
    total_sd = math.sqrt((ss_within + float((n * (m - grand) ** 2).sum())) / (n.sum() - 1))
    return {"channel_weighted": float(m.mean() / within), "switch_weighted": grand / total_sd,
            "p_one": signflip(m, RNG)["p_one"], "n_channels": len(ch), "n_units": int(n.sum())}


def auc_estimates(ch: dict) -> dict:
    """Channel mean of per-channel cell-standardised AUCs, and their n_ssdl-weighted mean (= pooled over points)."""
    a = np.array([v["auc"] for v in ch.values()], float)
    n = np.array([v["n_ssdl"] for v in ch.values()], float)
    return {"channel_weighted": float(a.mean()), "switch_weighted": float((n * a).sum() / n.sum()),
            "p_one": signflip(a - 0.5, RNG)["p_one"], "n_channels": len(ch), "n_units": int(n.sum())}


def dfar_estimates(ch: dict) -> dict:
    d = np.array([v["delta_far_std"] for v in ch.values()], float)
    n = np.array([v["n_ssdl"] for v in ch.values()], float)
    return {"channel_weighted": float(d.mean()), "switch_weighted": float((n * d).sum() / n.sum()),
            "p_one": signflip(d, RNG)["p_one"], "n_channels": len(ch), "n_units": int(n.sum())}


def h3_estimates(ch: dict, target_sd: float) -> dict:
    sh = np.array(list(ch.values()), float) / target_sd
    return {"channel_weighted": float(sh.mean()), "switch_weighted": None,
            "p_one": signflip(-sh, RNG)["p_one"], "n_channels": len(ch), "n_units": None}


def loco(channels: dict, estimate) -> dict:
    """The full-sample estimate and one estimate per left-out channel."""
    return {"full": estimate(channels),
            "leave_one_out": {c: estimate({k: v for k, v in channels.items() if k != c}) for c in sorted(channels)}}


def summarise(res: dict, effect: str, units: dict) -> dict:
    """Range of the leave-one-out estimates, direction changes and SESOI crossings, relative to the full sample."""
    null, _, sesoi, _ = EFFECTS[effect]
    out = {}
    for w in ("channel_weighted", "switch_weighted"):
        full = res["full"][w]
        if full is None:
            continue
        fits = {c: f[w] for c, f in res["leave_one_out"].items()}
        lo, hi = min(fits, key=fits.get), max(fits, key=fits.get)
        side = np.sign(full - null)
        s = {"full": full, "min": fits[lo], "min_without": lo, "max": fits[hi], "max_without": hi,
             "direction_changes": sorted(c for c, v in fits.items() if np.sign(v - null) != side)}
        if sesoi is not None:
            meets = (lambda v: v >= sesoi) if sesoi > null else (lambda v: v <= sesoi)
            s["full_meets_sesoi"] = bool(meets(full))
            s["sesoi_crossings"] = sorted(c for c, v in fits.items() if meets(v) != meets(full))
        out[w] = s
    ps = {c: f["p_one"] for c, f in res["leave_one_out"].items()}
    worst = max(ps, key=ps.get)
    out["p_one_full"] = res["full"]["p_one"]
    out["p_one_max"], out["p_one_max_without"] = ps[worst], worst
    out["n_fits_p_one_ge_0.05"] = int(sum(p >= 0.05 for p in ps.values()))
    out["units_of_left_out_channel"] = {c: units.get(c) for c in sorted(ps)}
    return out


def part_a(confirm: dict) -> tuple[dict, dict]:
    res, repro = {}, {}
    for m in MODELS:
        h1 = confirm["h1"]["primary"][m]
        h2 = confirm["h2_p0"]["primary"][m]
        h3 = confirm["h3"]["primary"][m]
        h1c = {c: v for c, v in h1["channels"].items() if v["n"] >= MIN_UNITS}
        h2c = {c: v for c, v in h2["channels"].items() if v["auc"] == v["auc"] and v["n_ssdl"] >= MIN_UNITS}
        dfc = {c: v for c, v in h2c.items() if v["delta_far_std"] is not None}
        tsd = h3["target_score_sd"]
        r = {"h1_dz": loco(h1c, h1_estimates), "h2p0_auc": loco(h2c, auc_estimates),
             "dfar_p0L1": loco(dfc, dfar_estimates),
             "h3_shift_sd": loco(h3["shift_per_channel"], lambda c, t=tsd: h3_estimates(c, t))}
        units = {"h1_dz": {c: v["n"] for c, v in h1c.items()}, "h2p0_auc": {c: v["n_ssdl"] for c, v in h2c.items()},
                 "dfar_p0L1": {c: v["n_ssdl"] for c, v in dfc.items()}, "h3_shift_sd": {}}
        res[m] = {k: dict(v, summary=summarise(v, k, units[k])) for k, v in r.items()}
        checks = {
            "h1_dz": (r["h1_dz"]["full"]["channel_weighted"], h1["dz"]),
            "h1_dz_plain": (r["h1_dz"]["full"]["switch_weighted"], h1["dz_plain"]),
            "h1_p_one": (r["h1_dz"]["full"]["p_one"], h1["signflip"]["p_one"]),
            "h2p0_auc": (r["h2p0_auc"]["full"]["channel_weighted"], h2["auc"]),
            "h2p0_p_one": (r["h2p0_auc"]["full"]["p_one"], h2["signflip"]["p_one"]),
            "dfar_p0L1": (r["dfar_p0L1"]["full"]["channel_weighted"], h2["delta_far_std"]),
            "dfar_p_one": (r["dfar_p0L1"]["full"]["p_one"], h2["delta_far_signflip"]["p_one"]),
            "h3_shift_sd": (r["h3_shift_sd"]["full"]["channel_weighted"], h3["shift_in_target_sd"]),
            "h3_p_one": (r["h3_shift_sd"]["full"]["p_one"], h3["signflip"]["p_one"]),
        }
        repro[m] = {k: {"recomputed": a, "frozen": b, "diff": a - b, "reproduces": abs(a - b) <= TOL}
                    for k, (a, b) in checks.items()}
    return res, repro


def cell_channels(cell: dict, primary: bool) -> dict:
    """Channels as the frozen run used them: the primary cell needs MIN_UNITS; secondary cells take every AUC."""
    return {c: v for c, v in cell["channels"].items()
            if v["auc"] is not None and v["auc"] == v["auc"] and (not primary or v["n_ssdl"] >= MIN_UNITS)}


def part_b(confirm: dict, bdir: Path, emb: Path, videos: list[str]) -> dict:
    from csdiar.analysis.confirm import load_points
    from csdiar.analysis.pilot import P0_CELLS, h2_cell, p0_curves, score_points

    pts, _ = load_points(bdir, list(SPLITS_CONF))
    pts_v = {p: r for p, r in pts.items() if r["video_id"] not in videos}
    conf_videos = sorted({r["video_id"] for r in pts.values() if r["speaker_id"] == CONF_CHANNEL})
    out = {"confirmatory_channel_videos_in_points": conf_videos,
           "points_removed_with_the_four_videos": len(pts) - len(pts_v), "models": {}}
    for m in MODELS:
        tau = confirm["tau"][m]
        vals, _ = score_points(pts, p0_curves(emb, m, SPLITS_CONF))
        vals_v = {p: v for p, v in vals.items() if p in pts_v}
        full = {n: h2_cell(pts, vals, tau, s, c) for n, (s, c) in P0_CELLS.items()}
        less = {n: h2_cell(pts_v, vals_v, tau, s, c) for n, (s, c) in P0_CELLS.items()}
        p0 = cell_channels(full["p0L1"], True)
        p0_v = cell_channels(less["p0L1"], True)
        r = {"h2p0_auc": loco(p0, auc_estimates),
             "dfar_p0L1": loco({c: v for c, v in p0.items() if v["delta_far_std"] is not None}, dfar_estimates),
             "det_auc": loco(cell_channels(full["det"], False), auc_estimates),
             "det_w1_auc": loco(cell_channels(full["det_w1"], False), auc_estimates)}
        units = {}
        for k, cell in (("h2p0_auc", "p0L1"), ("dfar_p0L1", "p0L1"), ("det_auc", "det"), ("det_w1_auc", "det_w1")):
            units[k] = {c: v["n_ssdl"] for c, v in full[cell]["channels"].items()}
        frozen = {"h2p0_auc": confirm["h2_p0"]["primary"][m]["auc"],
                  "dfar_p0L1": confirm["h2_p0"]["primary"][m]["delta_far_std"],
                  "det_auc": confirm["h2_p0"]["secondary"][m]["det"]["auc"],
                  "det_w1_auc": confirm["h2_p0"]["secondary"][m]["det_w1"]["auc"]}
        without_videos = {
            "h2p0_auc": auc_estimates(p0_v),
            "dfar_p0L1": dfar_estimates({c: v for c, v in p0_v.items() if v["delta_far_std"] is not None}),
            "det_auc": auc_estimates(cell_channels(less["det"], False)),
            "det_w1_auc": auc_estimates(cell_channels(less["det_w1"], False))}
        out["models"][m] = {
            "label": "rebuilt inputs",
            "effects": {k: dict(v, summary=summarise(v, k, units[k])) for k, v in r.items()},
            "full_sample_vs_frozen": {k: {"rebuilt": r[k]["full"]["channel_weighted"], "frozen": frozen[k],
                                          "diff": r[k]["full"]["channel_weighted"] - frozen[k]} for k in r},
            "without_the_four_videos": without_videos,
            "confirmatory_channel_units": {k: units[k].get(CONF_CHANNEL) for k in r}}
    return out


def shared_speaker(purity: dict) -> dict:
    pairs = [p for p in purity["cross_channel_closest_pairs"]
             if p["video_b"] == PILOT_VIDEO and p["channel_b"] == PILOT_CHANNEL and p["channel_a"] == CONF_CHANNEL]
    split = next((v for v in purity["split_off_videos"] if v["video_id"] == PILOT_VIDEO), None)
    return {"pilot_channel": PILOT_CHANNEL, "pilot_video": PILOT_VIDEO, "confirmatory_channel": CONF_CHANNEL,
            "confirmatory_videos": sorted({p["video_a"] for p in pairs}),
            "evidence_pairs": pairs, "pilot_video_split_off": split,
            "note": ("cos = video-centroid cosine under the two audit embeddings (TitaNet-L, CAM++); "
                     "percentile_in_split_half_null is in percent of same-video split-half cosines below it. These "
                     "are the closest cross-channel pairs in the corpus, and the pilot video's best cosine to its own "
                     "channel is lower, so it was split off; descriptive only, not a speaker verification.")}


def fmt(v, effect: str) -> str:
    if v is None:
        return "n/a"
    return f"{100 * v:+.2f} pp" if effect == "dfar_p0L1" else f"{v:.3f}"


def table(block: dict, title: str) -> list[str]:
    lines = [f"### {title}", "",
             "| Effect | Model | Weighting | Full | Leave-one-out min (without) | max (without) | Direction changes "
             "| SESOI crossings | max p (without) |", "|---|---|---|---|---|---|---|---|---|"]
    for effect in EFFECTS:
        for m in MODELS:
            if effect not in block.get(m, {}):
                continue
            s = block[m][effect]["summary"]
            for w in ("channel_weighted", "switch_weighted"):
                if w not in s:
                    continue
                x = s[w]
                cross = "n/a" if "sesoi_crossings" not in x else (", ".join(x["sesoi_crossings"]) or "none")
                p = f"{s['p_one_max']:.2g} ({s['p_one_max_without']})" if w == "channel_weighted" else ""
                lines.append(f"| {effect} | {m.upper()} | {w.split('_')[0]} | {fmt(x['full'], effect)} | "
                             f"{fmt(x['min'], effect)} ({x['min_without']}) | {fmt(x['max'], effect)} "
                             f"({x['max_without']}) | {', '.join(x['direction_changes']) or 'none'} | {cross} | {p} |")
    return lines + [""]


def markdown(res: dict) -> str:
    a, b, sh = res["part_a"], res.get("part_b"), res["shared_speaker"]
    out = [f"# Leave-one-channel-out sensitivity (D1)", "", f"**{LABEL}.**", "",
           "Every confirmatory channel is left out in turn (17 fits per effect and model). Channel-weighted = each "
           "channel equal, as frozen; switch-weighted = each switch point equal. p = exact one-sided sign-flip over "
           "the remaining 16 channels, unadjusted. 'Direction changes' lists left-out channels whose fit is on the "
           "other side of the null from the full-sample estimate; 'SESOI crossings' those whose fit is on the other "
           "side of the smallest effect of interest. Channel IDs are the dataset's speaker_id.", "",
           "Reproduction check (Part A): every full-sample estimate and sign-flip p recomputed from the frozen "
           f"per-channel statistics equals the frozen value (max |diff| {res['reproduction']['part_a_max_abs_diff']:.1e}).",
           ""]
    out += table(a, "Part A: exact, from the frozen per-channel statistics")
    if b:
        out += ["Part B, rebuilt inputs: full-sample difference from the frozen value", "",
                "| Model | " + " | ".join(EFFECTS_B) + " |", "|---|" + "---|" * len(EFFECTS_B)]
        for m in MODELS:
            d = b["models"][m]["full_sample_vs_frozen"]
            out.append(f"| {m.upper()} | " + " | ".join(f"{fmt(d[k]['rebuilt'], k)} vs {fmt(d[k]['frozen'], k)}"
                                                       for k in EFFECTS_B) + " |")
        out += [""]
        out += table({m: b["models"][m]["effects"] for m in MODELS}, "Part B: rebuilt inputs")
    out += ["## Possible shared speaker (pilot and confirmatory channel)", "",
            f"Pilot channel `{sh['pilot_channel']}`, video `{sh['pilot_video']}`; confirmatory channel "
            f"`{sh['confirmatory_channel']}`, videos {', '.join('`%s`' % v for v in sh['confirmatory_videos'])}.", ""]
    for p in sh["evidence_pairs"]:
        out.append(f"- `{p['video_a']}` vs `{p['video_b']}`: cos TitaNet {p['cos']['titanet']:.3f}, "
                   f"CAM++ {p['cos']['campplus']:.3f}")
    if sh["pilot_video_split_off"]:
        t = sh["pilot_video_split_off"]
        out.append(f"- The pilot video's best cosine to its own channel: TitaNet {t['titanet']['best_cos_to_channel']:.3f}, "
                   f"CAM++ {t['campplus']['best_cos_to_channel']:.3f} (split off by the audit).")
    out += ["", "Without the confirmatory channel (Part A, exact; channel-weighted):", "",
            "| Model | H1 d_z | H2-P0 AUC | Delta FAR | H3 shift |", "|---|---|---|---|---|"]
    for m in MODELS:
        f = {k: a[m][k]["leave_one_out"][CONF_CHANNEL]["channel_weighted"] for k in ("h1_dz", "h2p0_auc", "dfar_p0L1",
                                                                                  "h3_shift_sd")}
        out.append(f"| {m.upper()} | {f['h1_dz']:.3f} | {f['h2p0_auc']:.3f} | {100 * f['dfar_p0L1']:+.2f} pp | "
                   f"{f['h3_shift_sd']:.3f} |")
    if b:
        out += ["", "Without only the four matching videos (Part B, rebuilt inputs; channel-weighted):", "",
                "| Model | H2-P0 AUC | Delta FAR | det AUC | det_w1 AUC |", "|---|---|---|---|---|"]
        for m in MODELS:
            w = b["models"][m]["without_the_four_videos"]
            out.append(f"| {m.upper()} | {w['h2p0_auc']['channel_weighted']:.3f} | "
                       f"{100 * w['dfar_p0L1']['channel_weighted']:+.2f} pp | {w['det_auc']['channel_weighted']:.3f} | "
                       f"{w['det_w1_auc']['channel_weighted']:.3f} |")
    return "\n".join(out) + "\n"


EFFECTS_B = ("h2p0_auc", "dfar_p0L1", "det_auc", "det_w1_auc")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--confirm", type=Path, required=True, help="frozen results/confirm/confirm_m2a.json")
    ap.add_argument("--purity", type=Path, required=True, help="results/purity/purity_audit.json")
    ap.add_argument("--boundaries", type=Path, help="rebuilt boundary tables (Part B)")
    ap.add_argument("--emb", type=Path, help="rebuilt grid embeddings (Part B)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--md", type=Path, required=True)
    args = ap.parse_args()
    if (args.boundaries is None) != (args.emb is None):
        raise SystemExit("--boundaries and --emb go together")

    confirm = json.loads(args.confirm.read_text())
    purity = json.loads(args.purity.read_text())
    a, repro_a = part_a(confirm)
    bad = [(m, k) for m, r in repro_a.items() for k, v in r.items() if not v["reproduces"]]
    if bad:
        raise SystemExit(f"Part A does not reproduce the frozen values: {bad}")
    sh = shared_speaker(purity)
    inputs = {rel(args.confirm): sha256(args.confirm), rel(args.purity): sha256(args.purity)}
    b = None
    if args.boundaries:
        b = part_b(confirm, args.boundaries, args.emb, sh["confirmatory_videos"])
        for s in SPLITS_CONF:
            f = args.boundaries / f"points_{s}.csv.gz"
            inputs[rel(f)] = sha256(f)
            for m in MODELS:
                f = args.emb / f"{m}_L1" / f"{s}.npz"
                inputs[rel(f)] = sha256(f)
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    res = {"label": LABEL, "code_commit_base": git,
           "command": "PYTHONPATH=. python " + " ".join([rel(sys.argv[0])] + [
               rel(x) if Path(x).exists() else x for x in sys.argv[1:]]),
           "environment": {"python": platform.python_version(), "numpy": np.__version__},
           "inputs_sha256": inputs,
           "definitions": {k: v[3] for k, v in EFFECTS.items()},
           "reproduction": {"part_a": repro_a,
                            "part_a_max_abs_diff": max(abs(v["diff"]) for r in repro_a.values() for v in r.values())},
           "part_a": a, "part_b": b, "shared_speaker": sh}
    args.out.write_text(json.dumps(res, indent=1, sort_keys=False) + "\n")
    args.md.write_text(markdown(res))
    for m in MODELS:
        for k in ("h1_dz", "h2p0_auc", "dfar_p0L1", "h3_shift_sd"):
            s = a[m][k]["summary"]["channel_weighted"]
            print(f"{m} {k}: full {s['full']:.4f}, LOCO {s['min']:.4f}..{s['max']:.4f}, "
                  f"direction changes {len(s['direction_changes'])}, "
                  f"SESOI crossings {len(s.get('sesoi_crossings', []))}, max p {a[m][k]['summary']['p_one_max']:.3g}")
    print(f"wrote {rel(args.out)} and {rel(args.md)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
