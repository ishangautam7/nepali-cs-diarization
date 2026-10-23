#!/usr/bin/env python3
"""H3 pilot: cross-language score shift on lag-matched same-speaker trials (PLAN.md §2, §7; decision 14).

Validation channels only (pre-registration hygiene). Units are the clean monolingual 1 s windows from
csdiar.data.windows, scored with the dense-grid embeddings (E1-E3).
- Same-session target trials: two windows from the same video, starts >= 2 s apart (non-adjacent). Lag bins
  [2,4), [4,8), [8,16), [16,64), >= 64 s.
- Per channel and lag bin: shift = mean(cross) - mean of the two monolingual means (ne-ne, en-en; equal weight, so
  that language-specific score levels do not masquerade as a cross-language effect). The channel shift is the
  mean over bins weighted by cross trials. The target-score SD is the SD of all same-session monolingual scores.
- Cross-session (same channel, different video) shift, reported separately.
- Secondary: EER and minDCF (P_target = 0.05), cross vs mono. Nontargets pair windows of different channels with
  the same language pattern as the condition.
At most MAX_PAIRS random pairs are drawn per video, so cost stays bounded.
Output: results/pilot/h3_<tag>.json and .md
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PILOT_SPLIT = "validation"
MODELS = ("e1", "e2", "e3")
BINS = (2.0, 4.0, 8.0, 16.0, 64.0, float("inf"))
MAX_PAIRS = 40000


def unit(x):
    x = x.astype(np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def eer_mindcf(tgt: np.ndarray, non: np.ndarray, p_target: float = 0.05) -> tuple[float, float]:
    s = np.concatenate([tgt, non])
    y = np.concatenate([np.ones(len(tgt)), np.zeros(len(non))])
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y) / len(tgt)
    fp = np.cumsum(1 - y) / len(non)
    fnr, fpr = 1 - tp, fp
    i = int(np.argmin(np.abs(fnr - fpr)))
    dcf = p_target * fnr + (1 - p_target) * fpr
    return float((fnr[i] + fpr[i]) / 2), float(dcf.min() / min(p_target, 1 - p_target))


def load_windows(windows_dir: Path, splits: tuple[str, ...]) -> list[dict]:
    wins = []
    for split in splits:
        with gzip.open(windows_dir / f"windows_{split}.csv.gz", "rt", encoding="utf-8") as fh:
            wins += [dict(r, split=split) for r in csv.DictReader(fh) if r["abs_t0"] != ""]
    return wins


def compute_model(wins: list[dict], emb_dir: Path, m: str, rng: np.random.Generator) -> dict:
    """H3 statistics for one embedding over the given windows (see module doc)."""
    zs = {s: np.load(emb_dir / f"{m}_L1" / f"{s}.npz") for s in {w["split"] for w in wins}}
    E = unit(np.stack([zs[w["split"]][w["clip_id"]][int(w["k"])] for w in wins]))
    lang = np.array([w["lang"] for w in wins])
    t = np.array([float(w["abs_t0"]) for w in wins])
    chan = np.array([w["speaker_id"] for w in wins])
    vid = np.array([w["video_id"] for w in wins])
    by_video = defaultdict(list)
    for i, v in enumerate(vid):
        by_video[v].append(i)
    acc = defaultdict(lambda: defaultdict(list))  # (channel, bin) -> condition -> scores
    mono_all, cross_tgt, mono_tgt = [], [], []
    for v, idx in by_video.items():
        idx = np.array(idx)
        if len(idx) < 2:
            continue
        n_pairs = min(MAX_PAIRS, len(idx) * (len(idx) - 1) // 2)
        a, b = rng.choice(idx, n_pairs), rng.choice(idx, n_pairs)
        keep = a != b
        a, b = a[keep], b[keep]
        lag = np.abs(t[a] - t[b])
        keep = lag >= BINS[0]
        a, b, lag = a[keep], b[keep], lag[keep]
        s = (E[a] * E[b]).sum(1)
        bins = np.searchsorted(BINS, lag, side="right") - 1
        for ai, bi, si, bn in zip(a, b, s, bins):
            la, lb = lang[ai], lang[bi]
            cond = "cross" if la != lb else f"mono_{la}"
            acc[(chan[ai], int(bn))][cond].append(float(si))
            if cond == "cross":
                cross_tgt.append(si)
            else:
                mono_tgt.append(si); mono_all.append(si)
    tsd = float(np.std(mono_all, ddof=1))
    chans = defaultdict(lambda: [0.0, 0.0])
    bin_table = {}
    for (c, bn), conds in acc.items():
        monos = [np.mean(conds[k]) for k in ("mono_ne", "mono_en") if len(conds[k]) >= 20]
        if len(conds["cross"]) < 20 or not monos:
            continue
        shift = float(np.mean(conds["cross"]) - np.mean(monos))
        w = len(conds["cross"])
        chans[c][0] += shift * w
        chans[c][1] += w
        bin_table[f"{c[-4:]}:{BINS[bn]:g}"] = {"shift": shift, "n_cross": w,
                                               "n_mono_ne": len(conds["mono_ne"]), "n_mono_en": len(conds["mono_en"])}
    per_chan = {c: v[0] / v[1] for c, v in chans.items() if v[1]}
    # cross-session (same channel, different video)
    xs = defaultdict(lambda: defaultdict(list))
    for c in sorted(set(chan)):
        idx = np.where(chan == c)[0]
        a, b = rng.choice(idx, MAX_PAIRS), rng.choice(idx, MAX_PAIRS)
        keep = vid[a] != vid[b]
        a, b = a[keep], b[keep]
        s = (E[a] * E[b]).sum(1)
        for ai, bi, si in zip(a, b, s):
            xs[c]["cross" if lang[ai] != lang[bi] else f"mono_{lang[ai]}"].append(float(si))
    xs_shift = {c: float(np.mean(d["cross"]) - np.mean([np.mean(d[k]) for k in ("mono_ne", "mono_en") if d[k]]))
                for c, d in xs.items() if d["cross"]}
    # nontargets: windows of different channels, same language pattern
    n = 200000
    a, b = rng.integers(0, len(wins), n), rng.integers(0, len(wins), n)
    keep = chan[a] != chan[b]
    a, b = a[keep], b[keep]
    s = (E[a] * E[b]).sum(1)
    cross_non, mono_non = s[lang[a] != lang[b]], s[lang[a] == lang[b]]
    eer_c, dcf_c = eer_mindcf(np.array(cross_tgt), cross_non)
    eer_m, dcf_m = eer_mindcf(np.array(mono_tgt), mono_non)
    vals = np.array(list(per_chan.values()))
    return {"shift_per_channel": per_chan, "shift_mean": float(vals.mean()),
            "shift_in_target_sd": float(vals.mean() / tsd), "target_score_sd": tsd,
            "between_channel_sd": float(vals.std(ddof=1)) if len(vals) > 1 else None,
            "cross_session_shift": xs_shift, "bins": bin_table,
            "eer": {"cross": eer_c, "mono": eer_m}, "min_dcf": {"cross": dcf_c, "mono": dcf_m},
            "n_trials": {"cross_tgt": len(cross_tgt), "mono_tgt": len(mono_tgt),
                         "cross_non": int(len(cross_non)), "mono_non": int(len(mono_non))}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--emb", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--seed", type=int, default=20261018)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    wins = load_windows(args.windows, (PILOT_SPLIT,))
    out = {"tag": args.tag, "split": PILOT_SPLIT, "n_windows": len(wins),
           "n_by_lang": {l: sum(w["lang"] == l for w in wins) for l in ("ne", "en")}, "models": {}}
    for m in MODELS:
        out["models"][m] = compute_model(wins, args.emb, m, rng)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"h3_{args.tag}.json").write_text(json.dumps(out, indent=2))
    lines = [f"# H3 pilot {args.tag} (validation channels only)", "",
             "| Model | shift (cross − mono) | in target SD | per channel | cross-session | EER cross / mono | "
             "minDCF cross / mono |", "|---|---|---|---|---|---|---|"]
    for m, r in out["models"].items():
        pc = " ".join(f"{c[-4:]}:{v:+.3f}" for c, v in r["shift_per_channel"].items())
        xs_ = " ".join(f"{c[-4:]}:{v:+.3f}" for c, v in r["cross_session_shift"].items())
        lines.append(f"| {m} | {r['shift_mean']:+.4f} | {r['shift_in_target_sd']:+.3f} | {pc} | {xs_} | "
                     f"{r['eer']['cross']:.3f} / {r['eer']['mono']:.3f} | {r['min_dcf']['cross']:.3f} / "
                     f"{r['min_dcf']['mono']:.3f} |")
    (args.out / f"h3_{args.tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
