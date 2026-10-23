#!/usr/bin/env python3
"""EXPLORATORY, post hoc (external review, issue 2): score full diarization pipelines on the evaluation set.

For each system with hypotheses (scripts/review_pipeline_run.py) and each boundary type of
scripts/review_pipeline_set.py:
  change at a boundary   the hypothesis changes speaker within +-0.5 s of it: on a 10 ms grid, the most recent
                         non-empty set of active hypothesis labels becomes a different non-empty set (silence in
                         between does not count; a second speaker joining does);
  false change rate      share of WITHIN-speaker boundaries (switch, control, join_ss_same) with a change;
  miss rate              share of CROSS-speaker joins (join_xs_same, join_xs_cross) without a change;
  split / merge          secondary, cluster level: the majority hypothesis label differs between the 2 s before
                         and after a natural boundary, or between the two parts of a join (split for one speaker,
                         merge = same label for two speakers);
  DER per file           pyannote.metrics DiarizationErrorRate(collar=0.5, i.e. +-0.25 s, overlap scored) against
                         reference.rttm (aligned-word speech, the channel as the speaker), with its components.
Channel-level contrasts, as in csdiar/analysis: per-channel rates (equal weight over the channels that have both
types), the exact sign-flip test (csdiar.analysis.stats.signflip; p_one for a positive difference) and a BCa 95%
interval over channels (csdiar.analysis.stats.bca_ci). Clip joins are ARTIFICIAL boundaries.

Usage (repo root, an env with pyannote.metrics 4.2.0):
  python -m scripts.review_pipeline_score --set results/review_followups/pipeline/set.json --systems p1 p2 \\
      --out results/review_followups/pipeline_exploratory
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from csdiar.analysis.stats import bca_ci, signflip

STEP, WINDOW, SIDE = 0.01, 0.5, 2.0
WITHIN = ("switch", "control", "join_ss_same")
CROSS = ("join_xs_same", "join_xs_cross")
CONTRASTS = {  # name: (type A, type B, measure); difference = A - B per channel
    "false_change: switch - control": ("switch", "control", "change"),
    "false_change: switch - join_ss_same": ("switch", "join_ss_same", "change"),
    "false_change: join_ss_same - control": ("join_ss_same", "control", "change"),
    "miss: join_xs_cross - join_xs_same": ("join_xs_cross", "join_xs_same", "miss"),
    "split: switch - control": ("switch", "control", "split"),
}


def read_rttm(lines) -> dict[str, list[tuple[float, float, str]]]:
    out = defaultdict(list)
    for line in lines:
        f = line.split()
        if len(f) >= 8 and f[0] == "SPEAKER":
            out[f[1]].append((float(f[3]), float(f[3]) + float(f[4]), f[7]))
    return out


def change_times(segs: list[tuple[float, float, str]], dur: float) -> np.ndarray:
    labels = sorted({s[2] for s in segs})
    n = int(np.ceil(dur / STEP)) + 1
    act = np.zeros((len(labels), n), bool)
    for s, e, lab in segs:
        act[labels.index(lab), int(round(s / STEP)): int(round(e / STEP))] = True
    times, prev = [], None
    for k in range(n):
        cur = frozenset(np.flatnonzero(act[:, k]).tolist())
        if not cur:
            continue
        if prev is not None and cur != prev:
            times.append(k * STEP)
        prev = cur
    return np.array(times)


def majority(segs, lo: float, hi: float) -> str | None:
    dur = defaultdict(float)
    for s, e, lab in segs:
        o = min(e, hi) - max(s, lo)
        if o > 0:
            dur[lab] += o
    return max(dur, key=lambda k: (dur[k], k)) if dur else None


def score_system(files: dict, hyp: dict, ref: dict) -> tuple[list[dict], dict]:
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate
    rows, der = [], {}
    metric = DiarizationErrorRate(collar=0.5, skip_overlap=False)
    for fid, f in sorted(files.items()):
        segs, dur = hyp.get(fid, []), f["duration"]
        ch = change_times(segs, dur)
        for b in f["boundaries"]:
            t = b["t"]
            changed = bool(len(ch) and np.any(np.abs(ch - t) <= WINDOW + 1e-9))
            if f["kind"] == "natural":
                a, c = majority(segs, max(0.0, t - SIDE), t), majority(segs, t, min(dur, t + SIDE))
            else:
                p0, p1 = f["parts"]
                a, c = majority(segs, 0.0, p1["offset"]), majority(segs, p1["offset"], dur)
            split = None if a is None or c is None else a != c
            rows.append({"file_id": fid, "type": b["type"], "channel": b["channel"], "change": changed,
                         "miss": (not changed) if b["type"] in CROSS else None,
                         "split": split if b["type"] not in CROSS else None,
                         "merge": (not split if split is not None else None) if b["type"] in CROSS else None})
        r, h = Annotation(uri=fid), Annotation(uri=fid)
        for k, (s, e, lab) in enumerate(ref.get(fid, [])):
            r[Segment(s, e), k] = lab
        for k, (s, e, lab) in enumerate(segs):
            h[Segment(s, e), k] = lab
        d = metric(r, h, uem=Timeline([Segment(0, dur)]), detailed=True)
        tot = d["total"]
        der[fid] = {"kind": f["kind"], "der": d["diarization error rate"],
                    **{k: (d[k] / tot if tot else None) for k in ("missed detection", "false alarm", "confusion")},
                    "n_hyp_speakers": len({s[2] for s in segs})}
    return rows, der


def summarise(rows: list[dict], der: dict, rng) -> dict:
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m in ("change", "miss", "split", "merge"):
            if r[m] is not None:
                by[(r["type"], m)][r["channel"]].append(float(r[m]))
    rates = {}
    for (typ, m), chans in sorted(by.items()):
        allv = [v for vs in chans.values() for v in vs]
        means = [float(np.mean(v)) for v in chans.values()]
        rates.setdefault(typ, {})[m] = {"pooled": float(np.mean(allv)), "n": len(allv),
                                        "channel_mean": float(np.mean(means)), "n_channels": len(means)}
    contrasts = {}
    for name, (a, b, m) in CONTRASTS.items():
        ca, cb = by.get((a, m), {}), by.get((b, m), {})
        chans = sorted(set(ca) & set(cb))
        diff = [float(np.mean(ca[c]) - np.mean(cb[c])) for c in chans]
        contrasts[name] = {"n_channels": len(diff), "mean_difference": float(np.mean(diff)) if diff else None,
                           "signflip": signflip(diff, rng),
                           "ci95": bca_ci(diff, lambda xs: float(np.mean(xs)), 0.95, rng),
                           "per_channel": dict(zip(chans, diff))}
    ders = defaultdict(list)
    for v in der.values():
        ders[v["kind"]].append(v)
    der_summary = {k: {"files": len(v), "mean_der": float(np.mean([x["der"] for x in v])),
                       "median_der": float(np.median([x["der"] for x in v])),
                       "mean_confusion_share": float(np.nanmean([x["confusion"] if x["confusion"] is not None
                                                                 else np.nan for x in v])),
                       "share_files_with_>1_hyp_speaker": float(np.mean([x["n_hyp_speakers"] > 1 for x in v]))}
                   for k, v in sorted(ders.items())}
    return {"rates": rates, "contrasts": contrasts, "der": der_summary}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", type=Path, required=True)
    ap.add_argument("--systems", nargs="+", default=["p1", "p2"])
    ap.add_argument("--out", type=Path, required=True, help="output prefix; writes <out>.json and <out>.md")
    ap.add_argument("--seed", type=int, default=20261023)
    args = ap.parse_args()
    base = args.set.parent
    s = json.loads(args.set.read_text())
    files = s["files"]
    ref = read_rttm(open(base / "reference.rttm"))
    out = {"label": "EXPLORATORY (post hoc, external review issue 2). Clip joins are ARTIFICIAL boundaries; a D1 "
                    "channel stands for one speaker (unverified, see issue 4).",
           "window_s": WINDOW, "der_collar_s": 0.5, "boundary_counts": s["boundary_counts"],
           "audio_hours": s["audio_hours"], "systems": {}}
    for system in args.systems:
        timing = base / f"timing_{system}.jsonl"
        if not timing.exists():
            out["systems"][system] = {"status": "not run"}
            continue
        t = [json.loads(line) for line in open(timing)]
        done = {x["file_id"] for x in t}
        sub = {k: v for k, v in files.items() if k in done}
        hyp = read_rttm(line for fid in sorted(done) for line in open(base / f"hyp_{system}" / f"{fid}.rttm"))
        rows, der = score_system(sub, hyp, ref)
        rng = np.random.default_rng(args.seed)
        run = json.loads((base / f"run_{system}.json").read_text()) if (base / f"run_{system}.json").exists() else {}
        out["systems"][system] = {
            "status": "complete" if len(done) == len(files) else f"partial: {len(done)} of {len(files)} files",
            "pins": run.get("pins"), "device": run.get("device"), "torch": run.get("torch"),
            "rtf": sum(x["seconds"] for x in t) / sum(x["audio_s"] for x in t),
            **summarise(rows, der, rng), "der_per_file": der, "boundaries": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(out, indent=1))

    pct = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"
    lines = ["# Full-pipeline check on D1 (EXPLORATORY, post hoc)", "",
             "Clip joins are **artificial** boundaries: two separately recorded segments are concatenated, so a join also "
             "carries a recording discontinuity. A D1 channel stands for one speaker, which is unverified (issue 4). "
             f"Change window ±{WINDOW} s; DER with a 0.5 s collar (±0.25 s), overlap scored.", ""]
    for system, r in out["systems"].items():
        lines += [f"## {system.upper()} ({r['status']})", ""]
        if "rates" not in r:
            continue
        lines += [f"Device {r.get('device') or 'unknown'} (torch {r.get('torch') or '?'}); real-time factor {r['rtf']:.3f}.", "",
                  "| Boundary | n | Change rate, pooled | Change rate, channel mean | Miss rate (cross-speaker) | Split / merge |",
                  "|---|---|---|---|---|---|"]
        for typ in WITHIN + CROSS:
            x = r["rates"].get(typ)
            if not x:
                continue
            sm = x.get("split") or x.get("merge")
            lines.append(f"| {typ} | {x['change']['n']} | {pct(x['change']['pooled'])} | {pct(x['change']['channel_mean'])} | "
                         f"{pct(x['miss']['pooled']) if 'miss' in x else '-'} | {pct(sm['pooled']) if sm else 'n/a'} |")
        lines += ["", "| Contrast (A - B, per channel) | Channels | Mean | 95% BCa | Sign-flip p (one-sided, two-sided) |",
                  "|---|---|---|---|---|"]
        for name, c in r["contrasts"].items():
            ci = c["ci95"]
            lines.append(f"| {name} | {c['n_channels']} | {pct(c['mean_difference'])} | {pct(ci['lo'])} to {pct(ci['hi'])} | "
                         f"{c['signflip']['p_one']:.4f}, {c['signflip']['p_two']:.4f} |")
        lines += ["", "| Files | n | Mean DER | Median DER | Confusion share | Files with >1 speaker found |", "|---|---|---|---|---|---|"]
        for kind, d in r["der"].items():
            lines.append(f"| {kind} | {d['files']} | {pct(d['mean_der'])} | {pct(d['median_der'])} | "
                         f"{pct(d['mean_confusion_share'])} | {pct(d['share_files_with_>1_hyp_speaker'])} |")
        lines.append("")
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
