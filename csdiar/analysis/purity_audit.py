#!/usr/bin/env python3
"""Speaker-purity audit of D1 (PREREGISTRATION.md §5 item 4; PLAN.md §4 step 6). Rules as frozen:

- Embeddings: one per clip from two families not under test, TitaNet-L and CAM++ (csdiar/data/audit_embed.py).
- Flagged clip: under BOTH embeddings, the clip's mean cosine to the other clips of its video in the same
  language-mix stratum is more than 3.5 robust SDs (1.4826 x MAD) below that stratum's median. The stratum is
  speech_type x the channel's CMI tertile, and the clip needs at least 5 other clips in it.
- Split-off video: under BOTH embeddings, the video's centroid has a cosine below the 1st percentile of
  within-video split-half centroid cosines (pooled over all videos) to EVERY other video centroid of its channel.
- The subset: flagged clips are dropped, and split-off videos become speaker clusters of their own.
- Listening to flagged clips (capped at 200) is descriptive; this script writes the stratified sample.

Operational readings, fixed here before the audit runs (logged in planning/DEVIATIONS.md with this file's hash):
- "that stratum's median" is the median, and the MAD, of the per-clip mean cosines within the same video and
  stratum, i.e. over the set the comparison is made in. A cell whose MAD is 0 flags nothing.
- The CMI tertiles are cut at the 1/3 and 2/3 quantiles of the clip CMI values of the channel (all clips):
  tertile 0 if cmi <= q1, 1 if q1 < cmi <= q2, else 2.
- A centroid is the re-normalised mean of the unit clip embeddings, over all clips of the video.
- The split-half null uses every video with at least 4 clips, 20 random halvings per video (seed 0), separately
  for each embedding. A channel with a single video has no split-off test.
- Same teacher on two channels (PLAN §4 step 6c) is descriptive only: the closest cross-channel video pairs,
  with their place in the split-half null. It changes nothing in the subset.
Output: <out>/purity_audit.json, purity_audit.md, subset.json and listening_sample.csv.
Usage (on the box):
  .venv-pyannote4/bin/python -m csdiar.analysis.purity_audit --root ~/data/d1 --emb ~/data/d1/audit --out ~/data/d1/results/purity
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MODELS = ("titanet", "campplus")
SPLITS = ("validation", "train", "test")
Z_FLAG = 3.5
MAD_SCALE = 1.4826
MIN_OTHERS = 5
SPLIT_HALF_PCT = 1.0
SPLIT_HALF_MIN_CLIPS = 4
SPLIT_HALF_REPEATS = 20
SEED = 0
LISTEN_CAP = 200


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def centroid(vectors: np.ndarray) -> np.ndarray:
    return unit(unit(vectors).mean(axis=0))


def cmi_tertiles(cmis: list[float]) -> list[int]:
    a = np.asarray(cmis, dtype=float)
    q1, q2 = np.quantile(a, [1 / 3, 2 / 3])
    return [0 if c <= q1 else (1 if c <= q2 else 2) for c in a]


def strata(rows: list[dict]) -> dict[str, tuple[str, int]]:
    """clip_id -> (speech_type, channel CMI tertile)."""
    by_channel: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_channel[r["speaker_id"]].append(r)
    out = {}
    for rs in by_channel.values():
        for r, t in zip(rs, cmi_tertiles([float(r["cmi"]) for r in rs])):
            out[r["clip_id"]] = (r["speech_type"], t)
    return out


def flag_cells(rows: list[dict], emb: dict[str, np.ndarray], stratum: dict[str, tuple[str, int]]) -> dict:
    """Per clip: mean cosine to the other clips of its (video, stratum) cell, the cell median and robust SD,
    and whether it lies more than Z_FLAG robust SDs below the median. Clips in cells with fewer than
    MIN_OTHERS other clips are 'not assessable'."""
    cells: dict[tuple, list[str]] = defaultdict(list)
    for r in rows:
        if r["clip_id"] in emb:
            cells[(r["video_id"], *stratum[r["clip_id"]])].append(r["clip_id"])
    out = {}
    for key, ids in cells.items():
        if len(ids) - 1 < MIN_OTHERS:
            for i in ids:
                out[i] = {"assessable": False}
            continue
        v = unit(np.stack([emb[i] for i in ids]))
        sim = v @ v.T
        np.fill_diagonal(sim, np.nan)
        m = np.nanmean(sim, axis=1)
        med = float(np.median(m))
        rsd = MAD_SCALE * float(np.median(np.abs(m - med)))
        for i, mi in zip(ids, m):
            z = (float(mi) - med) / rsd if rsd > 0 else 0.0
            out[i] = {"assessable": True, "mean_cos": round(float(mi), 5), "cell_median": round(med, 5),
                      "robust_sd": round(rsd, 6), "z": round(z, 3), "flag": bool(rsd > 0 and z < -Z_FLAG)}
    return out


def split_half_null(by_video: dict[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    vals = []
    for vid in sorted(by_video):
        x = by_video[vid]
        if len(x) < SPLIT_HALF_MIN_CLIPS:
            continue
        for _ in range(SPLIT_HALF_REPEATS):
            p = rng.permutation(len(x))
            h = len(x) // 2
            vals.append(float(centroid(x[p[:h]]) @ centroid(x[p[h:]])))
    return np.asarray(vals)


def video_tests(rows: list[dict], emb: dict[str, np.ndarray]) -> dict:
    """Video centroids, the split-half null and each video's best cosine to another video of its channel."""
    by_video: dict[str, list[np.ndarray]] = defaultdict(list)
    channel_of: dict[str, str] = {}
    for r in rows:
        if r["clip_id"] in emb:
            by_video[r["video_id"]].append(emb[r["clip_id"]])
            channel_of[r["video_id"]] = r["speaker_id"]
    arrays = {v: np.stack(x) for v, x in by_video.items()}
    null = split_half_null(arrays, np.random.default_rng(SEED))
    thr = float(np.percentile(null, SPLIT_HALF_PCT))
    cents = {v: centroid(x) for v, x in arrays.items()}
    videos_of: dict[str, list[str]] = defaultdict(list)
    for v, ch in channel_of.items():
        videos_of[ch].append(v)
    per_video = {}
    for v, c in cents.items():
        others = [u for u in videos_of[channel_of[v]] if u != v]
        best = max((float(c @ cents[u]) for u in others), default=None)
        per_video[v] = {"channel": channel_of[v], "n_clips": len(arrays[v]), "n_other_videos": len(others),
                        "best_cos_to_channel": None if best is None else round(best, 5),
                        "below_null_p1": bool(best is not None and best < thr)}
    return {"null_n": int(len(null)), "null_p1": round(thr, 5),
            "null_quantiles": {q: round(float(np.percentile(null, q)), 5) for q in (1, 5, 50)},
            "videos": per_video, "centroids": cents, "null": null}


def cross_channel(tests: dict[str, dict], top: int = 10) -> list[dict]:
    """Closest cross-channel video pairs by the smaller of the two embeddings' cosines (descriptive)."""
    vids = sorted(tests[MODELS[0]]["centroids"])
    ch = {v: tests[MODELS[0]]["videos"][v]["channel"] for v in vids}
    pairs = []
    for i, a in enumerate(vids):
        for b in vids[i + 1 :]:
            if ch[a] == ch[b]:
                continue
            cs = {m: float(tests[m]["centroids"][a] @ tests[m]["centroids"][b]) for m in MODELS}
            pairs.append((min(cs.values()), a, b, cs))
    pairs.sort(reverse=True)
    out = []
    for _, a, b, cs in pairs[:top]:
        pct = {m: round(float((tests[m]["null"] < cs[m]).mean() * 100), 2) for m in MODELS}
        out.append({"video_a": a, "channel_a": ch[a], "video_b": b, "channel_b": ch[b],
                    "cos": {m: round(cs[m], 5) for m in MODELS}, "percentile_in_split_half_null": pct})
    return out


def listening_sample(flagged: list[dict], cap: int = LISTEN_CAP) -> list[dict]:
    """Stratified (by speech_type x tertile) round-robin sample of flagged clips, most extreme first."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for f in sorted(flagged, key=lambda f: f["z_max"]):
        groups[(f["speech_type"], f["cmi_tertile"])].append(f)
    out, keys = [], sorted(groups)
    while len(out) < cap and any(groups[k] for k in keys):
        for k in keys:
            if groups[k] and len(out) < cap:
                out.append(groups[k].pop(0))
    return out


def run(rows: list[dict], embs: dict[str, dict[str, np.ndarray]]) -> dict:
    common = set.intersection(*(set(e) for e in embs.values()))
    rows = [r for r in rows if r["clip_id"] in common]
    stratum = strata(rows)
    cells = {m: flag_cells(rows, embs[m], stratum) for m in MODELS}
    tests = {m: video_tests(rows, embs[m]) for m in MODELS}
    flagged, assessable = [], 0
    rate: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        cid = r["clip_id"]
        ok = all(cells[m][cid]["assessable"] for m in MODELS)
        key = f"{r['speech_type']}|t{stratum[cid][1]}"
        rate[key]["clips"] += 1
        if not ok:
            continue
        assessable += 1
        rate[key]["assessable"] += 1
        if all(cells[m][cid]["flag"] for m in MODELS):
            rate[key]["flagged"] += 1
            flagged.append({"clip_id": cid, "split": r["split"], "channel": r["speaker_id"],
                            "video_id": r["video_id"], "speech_type": r["speech_type"],
                            "cmi_tertile": stratum[cid][1],
                            "z": {m: cells[m][cid]["z"] for m in MODELS},
                            "z_max": max(cells[m][cid]["z"] for m in MODELS)})
    split_off = sorted(v for v in tests[MODELS[0]]["videos"]
                       if all(tests[m]["videos"][v]["below_null_p1"] for m in MODELS))
    dropped = {f["clip_id"] for f in flagged}
    channel_of = {r["video_id"]: r["speaker_id"] for r in rows}
    remap = {v: f"{channel_of[v]}::{v}" for v in split_off}
    kept = [r for r in rows if r["clip_id"] not in dropped]
    speakers = {remap.get(r["video_id"], r["speaker_id"]) for r in kept}
    by_split = {s: {"clips": sum(r["split"] == s for r in rows),
                    "flagged": sum(f["split"] == s for f in flagged),
                    "kept": sum(r["split"] == s for r in kept),
                    "speakers_after": len({remap.get(r["video_id"], r["speaker_id"]) for r in kept if r["split"] == s})}
                for s in SPLITS}
    return {
        "rules": {"z_flag": Z_FLAG, "mad_scale": MAD_SCALE, "min_others": MIN_OTHERS,
                  "split_half_pct": SPLIT_HALF_PCT, "split_half_min_clips": SPLIT_HALF_MIN_CLIPS,
                  "split_half_repeats": SPLIT_HALF_REPEATS, "seed": SEED, "models": list(MODELS)},
        "n_clips": len(rows), "n_assessable_both": assessable, "n_flagged": len(flagged),
        "flag_rate_by_language_mix": {k: dict(v) | {"rate": round(v["flagged"] / v["assessable"], 4)
                                                    if v["assessable"] else None} for k, v in sorted(rate.items())},
        "flagged_single_model": {m: sum(1 for c in cells[m].values() if c.get("flag")) for m in MODELS},
        "split_half_null": {m: {k: tests[m][k] for k in ("null_n", "null_p1", "null_quantiles")} for m in MODELS},
        "split_off_videos": [{"video_id": v, **{m: tests[m]["videos"][v] for m in MODELS}} for v in split_off],
        "videos_below_null_single_model": {m: sum(t["below_null_p1"] for t in tests[m]["videos"].values())
                                           for m in MODELS},
        "cross_channel_closest_pairs": cross_channel(tests),
        "subset": {"by_split": by_split, "n_speakers_after": len(speakers),
                   "dropped_clips": sorted(dropped), "speaker_remap": remap},
        "flagged": sorted(flagged, key=lambda f: f["z_max"]),
        "listening_sample": [f["clip_id"] for f in listening_sample(flagged)],
    }


def write_md(res: dict) -> str:
    lines = ["# D1 speaker-purity audit (PREREGISTRATION §5 item 4)", "",
             f"Clips with both embeddings: {res['n_clips']}; assessable under both: {res['n_assessable_both']}; "
             f"flagged under both: {res['n_flagged']} (single-model flags: {res['flagged_single_model']}).", "",
             "| language mix | clips | assessable | flagged | rate |", "|---|---|---|---|---|"]
    for k, v in res["flag_rate_by_language_mix"].items():
        lines.append(f"| {k} | {v.get('clips', 0)} | {v.get('assessable', 0)} | {v.get('flagged', 0)} | {v['rate']} |")
    lines += ["", "Split-half null (1st percentile): " + ", ".join(
        f"{m} {res['split_half_null'][m]['null_p1']} (n={res['split_half_null'][m]['null_n']})" for m in MODELS),
        f"Split-off videos (both embeddings): {len(res['split_off_videos'])} "
        f"(single-model: {res['videos_below_null_single_model']}).", "",
        "| split | clips | flagged | kept | speakers after |", "|---|---|---|---|---|"]
    for s, v in res["subset"]["by_split"].items():
        lines.append(f"| {s} | {v['clips']} | {v['flagged']} | {v['kept']} | {v['speakers_after']} |")
    lines += ["", "Closest cross-channel video pairs (descriptive; percentile within the split-half null):"]
    for p in res["cross_channel_closest_pairs"][:5]:
        lines.append(f"- {p['video_a']} ({p['channel_a']}) vs {p['video_b']} ({p['channel_b']}): "
                     f"cos {p['cos']}, percentile {p['percentile_in_split_half_null']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--emb", type=Path, required=True, help="folder with <model>/<split>.npz from audit_embed")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(args.root / "manifest.csv", encoding="utf-8"))
            if r.get("provenance_ok", "1") == "1" and r["video_id"]]
    embs = {}
    for m in MODELS:
        embs[m] = {}
        for s in SPLITS:
            with np.load(args.emb / m / f"{s}.npz") as z:
                embs[m].update({k: z[k].astype(np.float64) for k in z.files})
    res = run(rows, embs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "purity_audit.json").write_text(json.dumps(res, indent=1))
    (args.out / "purity_audit.md").write_text(write_md(res))
    (args.out / "subset.json").write_text(json.dumps(res["subset"], indent=1))
    with open(args.out / "listening_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip_id"])
        w.writerows([c] for c in res["listening_sample"])
    print(write_md(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
