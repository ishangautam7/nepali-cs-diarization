#!/usr/bin/env python3
"""H7: replication of the D1 SS-DL diagnostic on D4 (MUCS 2021 Hindi-English and Bengali-English).

Pre-registered in planning/PREREGISTRATION_H7.md (frozen as planning/FREEZE_H7.json, tag h7-freeze). The tests
are D1's, computed by the same frozen functions (csdiar.analysis.pilot.h1 / h2, csdiar.analysis.stats), for each
pair (hi-en, bn-en) and embedding (E1-E3):
- H1: per-speaker mean paired delta-cos over the L = 1.0 s matched sets; dz = mean of speaker means / pooled
  within-speaker SD; predicted > 0.
- H2-P0: per-speaker cell-standardised AUC of d(t) in the p0L1 cell against det_w1 controls; predicted > 0.5.
Clusters are MUCS speaker IDs. A speaker enters a test with at least MIN_UNITS = 5 units (H1 matched sets;
H2-P0 switch points with cell controls); excluded speakers are listed with their counts. One-sided sign-flip over
speakers, with 2^17 seeded random sign patterns when 2^k exceeds that (csdiar.analysis.stats.signflip). Holm over
the 12 tests (2 hypotheses x 3 embeddings x 2 pairs). BCa intervals at 1 - 0.05/12, TOST intervals at
1 - 2(0.05/12). Replicated = Holm one-sided p < .05 in the predicted direction; D1's outcome categories (smallest
effects dz 0.2, AUC 0.55) are reported as well.
Descriptive: plain dz, H1 at L = 0.5 / 1.5 s, P0 delta-FAR at D1's frozen tau, pooled det and det_w1 AUCs with
strata, per-direction counts.
Sensitivity (pre-registered): the same tests with speakers grouped into voice clusters (VOICE_* below).
Refuses to run unless every hash and path in planning/FREEZE_H7.json matches.
Output: <out>/confirm_h7.json and .md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from csdiar.analysis.confirm import dz_stat, frozen_taus, included, load_points, mean_stat, plain_dz, sha256
from csdiar.analysis.pilot import MODELS, h1, h2, p0_curves, unit
from csdiar.analysis.stats import bca_ci, holm, outcome, signflip

REPO = Path(__file__).resolve().parents[2]
FREEZE = REPO / "planning" / "FREEZE_H7.json"
GUARDED = ["csdiar/analysis/confirm_h7.py", "csdiar/analysis/confirm.py", "csdiar/analysis/stats.py",
           "csdiar/analysis/pilot.py", "csdiar/data/boundaries.py", "csdiar/data/boundaries_d4.py",
           "csdiar/data/tags.py", "csdiar/data/tags_d4.py", "planning/lexicon/d4/hi-en_loans.txt",
           "planning/lexicon/d4/hi-en_names.txt", "planning/lexicon/d4/bn-en_loans.txt",
           "planning/lexicon/d4/bn-en_names.txt", "planning/PREREGISTRATION_H7.md"]
PAIRS = ("hi-en", "bn-en")
SPLITS = ["train", "test"]
ALPHA = 0.05
M_FAMILY = 12
MIN_UNITS = 5
VOICE_MODEL = "e2"  # embedding used to group recordings into voice clusters (sensitivity analysis)
VOICE_D1_PERCENTILE = 99.0  # merge threshold: this percentile of D1 cosines between different channels' videos
VOICE_THETA_FROZEN = 0.5137  # its value from the frozen D1 features (checked at run time, 4 decimals)
SEED = 20261018
N_MC = 2 ** 17


def signflip_h7(x, rng) -> dict:
    """csdiar.analysis.stats.signflip; when it samples (k > 17), p = (b + 1) / (N_MC + 1), where b counts the
    sampled sign patterns at least as extreme as the observed one (one-sided: mean >= observed mean; two-sided:
    |mean| >= |observed mean|)."""
    r = signflip(x, rng, n_mc=N_MC)
    if not r["exact"]:
        r = {**r, "p_one": (round(r["p_one"] * N_MC) + 1) / (N_MC + 1),
             "p_two": (round(r["p_two"] * N_MC) + 1) / (N_MC + 1), "monte_carlo_patterns": N_MC}
    return r


def check_freeze(root: Path, d1_root: Path, pilot: Path, out: Path) -> dict:
    if not FREEZE.exists():
        raise SystemExit("refusing: planning/FREEZE_H7.json does not exist")
    rec = json.loads(FREEZE.read_text())
    for f in GUARDED:
        if rec["code_sha256"].get(f) is None:
            raise SystemExit(f"refusing: {f} is not covered by the freeze record")
    for f, h in rec["code_sha256"].items():
        if sha256(REPO / f) != h:
            raise SystemExit(f"refusing: {f} changed since the freeze")
    for f, h in rec["feature_sha256"].items():
        if sha256(Path(f).expanduser()) != h:
            raise SystemExit(f"refusing: feature file {f} changed since the freeze")
    for k, v in (("d4_root", root), ("d1_root", d1_root), ("pilot", pilot), ("out", out)):
        if Path(v).expanduser().resolve() != Path(rec["paths"][k]).expanduser().resolve():
            raise SystemExit(f"refusing: --{k.replace('_', '-')} {v} is not the frozen input {rec['paths'][k]}")
    if rec.get("seed") != SEED:
        raise SystemExit("refusing: the freeze record's seed differs from the code's")
    return {"mode": "confirmatory", "freeze": {k: rec[k] for k in ("tag", "commit", "date")}}


def h1_test(points, matches, pe_dir, m, L, rng, chans) -> dict:
    r = h1(points, matches, pe_dir, m, L, tuple(SPLITS))
    values = r.pop("_values")
    inc, exc = included(values, MIN_UNITS)
    exc.update({c: 0 for c in chans if c not in values})
    clusters = list(inc.values())
    out = {"n_clusters": len(clusters), "excluded": exc, "n_sets": int(sum(len(c) for c in clusters)),
           "delta_cos_mean_of_clusters": float(np.mean([c.mean() for c in clusters])), "dz": dz_stat(clusters),
           "dz_plain": plain_dz(clusters), "signflip": signflip_h7([c.mean() for c in clusters], rng),
           "dz_ci": bca_ci(clusters, dz_stat, 1 - ALPHA / M_FAMILY, rng),
           "dz_tost": bca_ci(clusters, dz_stat, 1 - 2 * ALPHA / M_FAMILY, rng)}
    return out


def h2_test(points, curves, tau, rng, chans) -> tuple[dict, dict]:
    r = h2(points, curves, tau=tau)
    prim = r["cells"]["p0L1"]
    inc = {c: v for c, v in prim["channels"].items() if v["auc"] == v["auc"] and v["n_ssdl"] >= MIN_UNITS}
    exc = {c: v["n_ssdl"] for c, v in prim["channels"].items() if c not in inc}
    exc.update({c: 0 for c in chans if c not in prim["channels"]})
    aucs = [v["auc"] for v in inc.values()]
    tost = bca_ci(aucs, mean_stat, 1 - 2 * ALPHA / M_FAMILY, rng)
    dfar = [v["delta_far_std"] for v in inc.values() if v["delta_far_std"] is not None]
    out = {"n_clusters": len(aucs), "excluded": exc, "auc": float(np.mean(aucs)),
           "auc_minus_half": float(np.mean(aucs)) - 0.5, "signflip": signflip_h7([a - 0.5 for a in aucs], rng),
           "auc_ci": bca_ci(aucs, mean_stat, 1 - ALPHA / M_FAMILY, rng),
           "tost": {"lo": tost["lo"] - 0.5, "hi": tost["hi"] - 0.5, "level": tost["level"]},
           "delta_far_std_at_d1_tau": float(np.mean(dfar)) if dfar else None, "n": prim["n"],
           "dropped_edge": r["dropped_edge"]}
    secondary = {name: {"auc": cell["auc_mean_of_channels"], "strata": cell["strata"],
                        "auc_ci": bca_ci([c["auc"] for c in cell["channels"].values() if c["auc"] == c["auc"]],
                                         mean_stat, 1 - ALPHA / M_FAMILY, rng)}
                 for name, cell in r["cells"].items() if name != "p0L1"}
    return out, secondary


def video_centroids(emb_dir: Path, manifest: list[dict], splits: list[str]) -> dict:
    """Unit-length mean of the clip means of VOICE_MODEL dense L = 1 s windows, per (speaker, video)."""
    zs = {s: np.load(emb_dir / f"{VOICE_MODEL}_L1" / f"{s}.npz") for s in splits}
    by = defaultdict(list)
    for r in manifest:
        if r["video_id"] and r["split"] in zs:
            by[(r["speaker_id"], r["video_id"])].append(r)
    out = {}
    for key, clips in sorted(by.items()):
        means = [unit(zs[c["split"]][c["clip_id"]].astype(np.float32)).mean(0) for c in clips
                 if c["clip_id"] in zs[c["split"]] and len(zs[c["split"]][c["clip_id"]])]
        if means:
            out[key] = unit(np.mean(means, 0))
    return out


def d1_voice_threshold(d1_root: Path) -> tuple[float, dict]:
    """theta = the VOICE_D1_PERCENTILE percentile of cosines between video centroids of different D1 channels."""
    manifest = list(csv.DictReader(open(d1_root / "manifest.csv", encoding="utf-8")))
    cents = video_centroids(d1_root / "emb", manifest, ["train", "validation", "test"])
    keys = sorted(cents)
    X = np.stack([cents[k] for k in keys])
    S = X @ X.T
    ch = np.array([k[0] for k in keys])
    iu = np.triu_indices(len(keys), 1)
    same = ch[iu[0]] == ch[iu[1]]
    diff = S[iu][~same]
    theta = float(np.percentile(diff, VOICE_D1_PERCENTILE))
    return theta, {"theta": theta, "d1_videos": len(keys),
                   "d1_same_channel_pairs_above_theta": float((S[iu][same] > theta).mean())}


def voice_clusters(emb_dir: Path, manifest: list[dict], theta: float) -> tuple[dict, dict]:
    """Average-linkage clustering of a pair's recording centroids, cut at cosine distance 1 - theta."""
    from scipy.cluster.hierarchy import fcluster, linkage
    cents = video_centroids(emb_dir, manifest, SPLITS)
    keys = sorted(cents)
    lab = fcluster(linkage(np.stack([cents[k] for k in keys]), method="average", metric="cosine"),
                   t=1 - theta, criterion="distance")
    rec_to_cluster = {k[1]: f"v{int(l)}" for k, l in zip(keys, lab)}
    return rec_to_cluster, {"n_recordings": len(keys), "n_clusters": int(len(set(lab))),
                            "largest_cluster": int(np.bincount(lab).max()),
                            "recordings_in_merged_clusters": int(sum(c for c in np.bincount(lab) if c > 1))}


def run_pair(pair, root, taus, rng, cluster_of=None) -> dict:
    bdir, pe_dir, emb_dir = root / pair / "boundaries" / "lex_d4v1", root / pair / "point_emb" / "lex_d4v1", root / pair / "emb"
    pts, matches = load_points(bdir, SPLITS)
    if cluster_of is not None:  # sensitivity: replace speaker IDs by voice clusters
        for r in pts.values():
            r["speaker_id"] = cluster_of.get(r["video_id"], r["speaker_id"])
    chans = sorted({r["speaker_id"] for r in pts.values()})
    out = {"n_points": len(pts), "n_clusters_total": len(chans), "models": {}}
    for m in MODELS:
        res = {"h1": h1_test(pts, matches, pe_dir, m, 1.0, rng, chans)}
        res["h2_p0"], res["secondary"] = h2_test(pts, p0_curves(emb_dir, m, tuple(SPLITS)), taus[m], rng, chans)
        if cluster_of is None:
            res["h1_window_length"] = {f"L{L:g}": {k: h1_test(pts, matches, pe_dir, m, L, rng, chans)[k]
                                                   for k in ("dz", "dz_ci", "n_sets", "n_clusters")} for L in (0.5, 1.5)}
        out["models"][m] = res
    return out


def grade(block: dict) -> dict:
    """Holm over the 12 tests of one block (both pairs); attach each test's replication flag and D1 category, and
    return the verdict per hypothesis and pair: 'replicated' when all three embeddings replicate, 'partly
    replicated' when one or two do, 'not replicated' when none does; plus how many met D1's smallest effect."""
    keys = [(p, m, h) for p in PAIRS for m in MODELS for h in ("h1", "h2_p0")]
    res = [block[p]["models"][m][h] for p, m, h in keys]
    a1 = holm([r["signflip"]["p_one"] for r in res])
    a2 = holm([r["signflip"]["p_two"] for r in res])
    for (p, m, h), r, q1, q2 in zip(keys, res, a1, a2):
        est, lo, hi, sesoi = ((r["dz"], r["dz_tost"]["lo"], r["dz_tost"]["hi"], 0.2) if h == "h1" else
                              (r["auc_minus_half"], r["tost"]["lo"], r["tost"]["hi"], 0.05))
        r["p_one_holm"], r["p_two_holm"] = float(q1), float(q2)
        r["outcome"] = outcome(q1, q2, est, lo, hi, sesoi, +1)
        r["replicated"] = bool(q1 < ALPHA)
    verdicts = {}
    for p in PAIRS:
        for h in ("h1", "h2_p0"):
            rs = [block[p]["models"][m][h] for m in MODELS]
            n = int(sum(r["replicated"] for r in rs))
            verdicts[f"{p}/{h}"] = {"verdict": "replicated" if n == 3 else ("partly replicated" if n else "not replicated"),
                                    "n_replicated": n, "n_supported_at_d1_sesoi":
                                        int(sum(r["outcome"]["category"] == "supported" for r in rs))}
    block["verdicts"] = verdicts
    return verdicts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--d4-root", type=Path, required=True, help="folder with hi-en/ and bn-en/")
    ap.add_argument("--d1-root", type=Path, required=True, help="D1 ingest folder (voice-threshold calibration)")
    ap.add_argument("--pilot", type=Path, required=True, help="D1 pilot JSON with the frozen tau")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261018)
    args = ap.parse_args()
    mode = check_freeze(args.d4_root, args.d1_root, args.pilot, args.out)
    if args.seed != SEED:
        raise SystemExit(f"refusing: the frozen seed is {SEED}")
    if (args.out / "confirm_h7.json").exists():
        raise SystemExit(f"refusing: {args.out / 'confirm_h7.json'} exists; H7 is run once")
    taus = frozen_taus(args.pilot)
    # Sensitivity inputs first, so a failed check cannot leave a partial confirmatory output behind.
    info, clusters = {}, {}
    theta, info["d1_calibration"] = d1_voice_threshold(args.d1_root)
    if round(theta, 4) != VOICE_THETA_FROZEN:
        raise SystemExit(f"refusing: voice threshold {theta:.4f} differs from the frozen {VOICE_THETA_FROZEN}")
    for p in PAIRS:
        manifest = list(csv.DictReader(open(args.d4_root / p / "manifest.csv", encoding="utf-8")))
        cluster_of, info[p] = voice_clusters(args.d4_root / p / "emb", manifest, theta)
        # a recording without any E2 window (none expected) stays a cluster of its own
        missing = sorted({r["video_id"] for r in manifest} - set(cluster_of))
        cluster_of.update({v: f"single:{v}" for v in missing})
        info[p]["recordings_without_centroid"] = len(missing)
        clusters[p] = cluster_of
    rng = np.random.default_rng(args.seed)
    out = {"seed": args.seed, **mode, "tau_d1": taus, "min_units": MIN_UNITS, "m_family": M_FAMILY,
           "primary": {p: run_pair(p, args.d4_root, taus, rng) for p in PAIRS}}
    grade(out["primary"])
    args.out.mkdir(parents=True, exist_ok=True)
    write(out, args.out)
    sens = {p: run_pair(p, args.d4_root, taus, rng, clusters[p]) for p in PAIRS}
    grade(sens)
    out["sensitivity_voice_clusters"] = {"clustering": info, **sens}
    write(out, args.out)
    return 0


def _json_default(o):
    """Serialize NumPy scalars by their native Python type (so a NumPy bool/int is not widened to a float)."""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def write(out: dict, d: Path) -> None:
    (d / "confirm_h7.json").write_text(json.dumps(out, indent=2, default=_json_default))
    lines = [f"# H7: D1 diagnostic replicated on D4 ({out['mode']})", "",
             "| Pair | H | Model | Speakers | Estimate (CI) | Holm p (one-sided, m = 12) | Replicated | D1 category |",
             "|---|---|---|---|---|---|---|---|"]
    for p in PAIRS:
        for m in MODELS:
            for h, est, ck, fmt in (("h1", "dz", "dz_ci", "dz {:.3f}"), ("h2_p0", "auc", "auc_ci", "AUC {:.3f}")):
                r = out["primary"][p]["models"][m][h]
                ci = r[ck]
                lines.append(f"| {p} | {'H1' if h == 'h1' else 'H2-P0'} | {m.upper()} | {r['n_clusters']} | "
                             f"{fmt.format(r[est])} ({ci['lo']:.3f}-{ci['hi']:.3f}) | "
                             f"{r.get('p_one_holm', float('nan')):.1e} | {r.get('replicated')} | "
                             f"{r.get('outcome', {}).get('category')} |")
    lines += ["", "| Pair | Hypothesis | Verdict (embeddings replicated / meeting D1's smallest effect) |", "|---|---|---|"]
    for key, v in out["primary"].get("verdicts", {}).items():
        p, h = key.split("/")
        lines.append(f"| {p} | {'H1' if h == 'h1' else 'H2-P0'} | {v['verdict']} ({v['n_replicated']}/3; "
                     f"{v['n_supported_at_d1_sesoi']}/3) |")
    (d / "confirm_h7.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    sys.exit(main())
