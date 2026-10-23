#!/usr/bin/env python3
"""EXPLORATORY, post hoc (external review, issue 2): the evaluation set for a full-pipeline check on D1.

The paper tested speaker embeddings and the P0 change score, not a full diarization pipeline. This builds the
files on which full pipelines (scripts/review_pipeline_run.py) are scored (scripts/review_pipeline_score.py).
Boundary types, all on the 17 confirmatory channels (train + test):

  switch          NATURAL within-speaker language switch: a p0L1 switch point with cell controls (the paper's
                  primary cell), scored inside its own recording clip;
  control         NATURAL within-speaker same-language word boundary: one det_w1 control from the switch point's
                  cell (channel x video x pause bin x position bin), as in the paper;
  join_ss_same    ARTIFICIAL boundary, same speaker: two consecutive clips of the same video joined end to start,
                  the same language on both sides of the join;
  join_xs_same    ARTIFICIAL boundary, different speakers: a clip of this channel joined to a clip of another
                  channel, the same language on both sides;
  join_xs_cross   ARTIFICIAL boundary, different speakers, a language change at the join.

Clip joins are artificial boundaries: two separately recorded segments are concatenated (10 ms fade-out and
fade-in, no gap), so a join also carries a recording discontinuity that natural boundaries do not. Every join
side must end (or start) with a run of at least 1 s in its boundary language, mirroring p0L1's "both runs >= 1 s".
N switch points are drawn per channel in proportion to its p0L1 count (largest remainder, at least one); each
join type gets the same per-channel numbers (fewer where a channel runs out, recorded). A clip is used once.

The reference (reference.rttm) marks speech as the union of aligned words (gaps under 0.5 s merged; a part
without aligned words counts as all speech), labelled with the channel: D1 has no speaker labels, so a channel
stands for one speaker, which is the very assumption issue 4 questions.

Usage (repo root):
  python -m scripts.review_pipeline_set --boundaries ~/data/d1/boundaries/lex_v1b --emb ~/data/d1/emb \\
      --align ~/data/d1/align --d1 ~/data/d1 --out results/review_followups/pipeline
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from csdiar.analysis.confirm import load_points
from csdiar.analysis.pilot import p0_curves, score_points
from scripts.review_human_subset import allocate, cell

SPLITS = ("train", "test")
SR = 16000
FADE = int(0.010 * SR)
MIN_RUN = 1.0
MERGE_GAP = 0.5
LANGS = ("ne", "en")


def edge_runs(words: list[dict]) -> tuple[tuple[str, float] | None, tuple[str, float] | None]:
    """(language, seconds) of the initial and the final run of language-tagged aligned words."""
    w = [x for x in words if x.get("aligned") and x.get("tag") in LANGS]
    if not w:
        return None, None
    i = 0
    while i + 1 < len(w) and w[i + 1]["tag"] == w[0]["tag"]:
        i += 1
    j = len(w) - 1
    while j - 1 >= 0 and w[j - 1]["tag"] == w[-1]["tag"]:
        j -= 1
    return (w[0]["tag"], w[i]["end"] - w[0]["start"]), (w[-1]["tag"], w[-1]["end"] - w[j]["start"])


def speech_regions(words: list[dict], dur: float) -> list[tuple[float, float]]:
    iv = sorted((max(0.0, x["start"]), min(dur, x["end"])) for x in words if x.get("aligned"))
    out = []
    for s, e in iv:
        if out and s - out[-1][1] < MERGE_GAP:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        elif e > s:
            out.append((s, e))
    return out or [(0.0, dur)]


def read_wav(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != SR or x.ndim != 1:
        raise SystemExit(f"{path}: expected 16 kHz mono")
    return x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--boundaries", type=Path, required=True)
    ap.add_argument("--emb", type=Path, required=True, help="grid embeddings folder; only e3_L1 is read")
    ap.add_argument("--align", type=Path, required=True)
    ap.add_argument("--d1", type=Path, required=True, help="D1 folder with manifest.csv and wav/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=300, help="natural switch points; each other type gets as many")
    ap.add_argument("--seed", type=int, default=20261023)
    args = ap.parse_args()
    if (args.out / "set.json").exists():
        raise SystemExit(f"refusing: {args.out / 'set.json'} exists; the set is drawn once")
    rng = np.random.default_rng(args.seed)

    manifest = {r["clip_id"]: r for r in csv.DictReader(open(args.d1 / "manifest.csv", encoding="utf-8"))
                if r["split"] in SPLITS and r["provenance_ok"] == "1"}
    words = {}
    for s in SPLITS:
        for line in open(args.align / f"{s}.jsonl", encoding="utf-8"):
            rec = json.loads(line)
            if rec.get("status") == "ok" and rec["clip_id"] in manifest:
                words[rec["clip_id"]] = rec["words"]

    # natural boundaries: the paper's primary-cell population, as in scripts/review_human_subset.py
    pts, _ = load_points(args.boundaries, list(SPLITS))
    vals, _ = score_points(pts, p0_curves(args.emb, "e3", SPLITS))
    ctrls = defaultdict(list)
    for p, r in pts.items():
        if r["kind"] == "ctrl" and r["elig_det_w1"] and p in vals:
            ctrls[cell(r)].append(p)
    pop = defaultdict(list)
    for p, r in pts.items():
        if r["kind"] == "ssdl" and r["elig_p0L1"] and p in vals and ctrls.get(cell(r)):
            pop[r["speaker_id"]].append(p)
    alloc = allocate({ch: len(v) for ch, v in sorted(pop.items())}, args.n)
    natural, used_ctrl = [], set()
    for ch in sorted(pop):
        for s in rng.choice(sorted(pop[ch]), size=alloc[ch], replace=False):
            free = sorted(set(ctrls[cell(pts[s])]) - used_ctrl)
            c = str(rng.choice(free))
            used_ctrl.add(c)
            natural += [("switch", str(s)), ("control", c)]
    used_clips = {pts[p]["clip_id"] for _, p in natural}

    files = {}
    for kind, p in natural:
        r = pts[p]
        f = files.setdefault(f"nat_{r['clip_id']}", {"kind": "natural", "channel": r["speaker_id"],
                                                    "parts": [{"clip_id": r["clip_id"], "offset": 0.0}],
                                                    "boundaries": []})
        f["boundaries"].append({"type": kind, "t": r["t"], "point_id": p, "lang_left": r["lang_left"],
                                "lang_right": r["lang_right"], "channel": r["speaker_id"]})

    # clip edges for the joins
    edges = {c: edge_runs(w) for c, w in words.items()}
    ok_end = lambda c: edges[c][1] is not None and edges[c][1][1] >= MIN_RUN
    ok_start = lambda c: edges[c][0] is not None and edges[c][0][1] >= MIN_RUN
    by_chan = defaultdict(list)
    for c in sorted(words):
        if c not in used_clips:
            by_chan[manifest[c]["speaker_id"]].append(c)
    by_video = defaultdict(list)
    for c in sorted(words, key=lambda c: (manifest[c]["video_id"], int(manifest[c]["start_ms"]))):
        by_video[manifest[c]["video_id"]].append(c)
    used = set(used_clips)
    shortfall = defaultdict(dict)

    def add_join(kind: str, a: str, b: str) -> None:
        la, lb = edges[a][1][0], edges[b][0][0]
        files[f"{kind}_{a}__{b}"] = {"kind": kind, "channel": manifest[a]["speaker_id"],
                                     "parts": [{"clip_id": a, "offset": 0.0}, {"clip_id": b, "offset": None}],
                                     "boundaries": [{"type": kind, "t": None, "lang_left": la, "lang_right": lb,
                                                     "channel": manifest[a]["speaker_id"]}]}
        used.update((a, b))

    for ch in sorted(alloc):
        k = alloc[ch]
        # same speaker: consecutive clips of one video, same language across the join
        cands = [(a, b) for v in sorted({manifest[c]["video_id"] for c in by_chan[ch]})
                 for a, b in zip(by_video[v], by_video[v][1:])
                 if a not in used_clips and b not in used_clips and ok_end(a) and ok_start(b)
                 and edges[a][1][0] == edges[b][0][0]]
        for j in rng.permutation(len(cands)):
            if k == 0:
                break
            a, b = cands[int(j)]
            if a in used or b in used:
                continue
            add_join("join_ss_same", a, b)
            k -= 1
        if k:
            shortfall[ch]["join_ss_same"] = k
        # different speakers: a clip of this channel, then a clip of a random other channel
        others = [o for o in sorted(by_chan) if o != ch]
        for kind, same in (("join_xs_same", True), ("join_xs_cross", False)):
            k = alloc[ch]
            starts = [c for c in by_chan[ch] if ok_end(c)]
            for ia in rng.permutation(len(starts)):
                if k == 0:
                    break
                a = starts[int(ia)]
                if a in used:
                    continue
                la = edges[a][1][0]
                for _ in range(50):
                    o = others[int(rng.integers(len(others)))]
                    pool = [c for c in by_chan[o] if c not in used and ok_start(c) and (edges[c][0][0] == la) == same]
                    if pool:
                        add_join(kind, a, pool[int(rng.integers(len(pool)))])
                        k -= 1
                        break
            if k:
                shortfall[ch][kind] = k

    # audio for the joins, durations, boundary times and the reference
    (args.out / "audio").mkdir(parents=True, exist_ok=True)
    ref_lines = []
    for fid in sorted(files):
        f = files[fid]
        xs, t0 = [], 0.0
        for part in f["parts"]:
            m = manifest[part["clip_id"]]
            part["offset"] = t0
            part["duration"] = int(m["n_samples"]) / SR
            part["speaker"] = m["speaker_id"]
            part["wav"] = m["wav"]
            if f["kind"] != "natural":
                x = read_wav(args.d1 / m["wav"])
                if len(x) != int(m["n_samples"]):
                    raise SystemExit(f"{part['clip_id']}: length differs from the manifest")
                x = x.copy()
                ramp = np.linspace(0.0, 1.0, FADE, dtype=np.float32)
                if xs:
                    x[:FADE] *= ramp
                if part is not f["parts"][-1]:
                    x[-FADE:] *= ramp[::-1]
                xs.append(x)
            for s, e in speech_regions(words.get(part["clip_id"], []), part["duration"]):
                ref_lines.append(f"SPEAKER {fid} 1 {t0 + s:.3f} {e - s:.3f} <NA> <NA> {m['speaker_id']} <NA> <NA>")
            t0 += part["duration"]
        f["duration"] = t0
        if f["kind"] != "natural":
            f["boundaries"][0]["t"] = f["parts"][1]["offset"]
            sf.write(str(args.out / "audio" / f"{fid}.wav"), np.concatenate(xs), SR, subtype="PCM_16")
            f["audio"] = f"audio/{fid}.wav"
        else:
            f["audio"] = None  # the D1 clip itself: <d1>/<manifest wav>

    counts = defaultdict(int)
    for f in files.values():
        for b in f["boundaries"]:
            counts[b["type"]] += 1
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "reference.rttm").write_text("\n".join(ref_lines) + "\n")
    out = {"label": "EXPLORATORY (post hoc, external review issue 2); clip joins are ARTIFICIAL boundaries",
           "seed": args.seed, "n": args.n, "allocation_per_channel": alloc, "shortfall": dict(shortfall),
           "boundary_counts": dict(counts), "files": files,
           "audio_hours": round(sum(f["duration"] for f in files.values()) / 3600, 3),
           "inputs_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             [args.boundaries / f"points_{s}.csv.gz" for s in SPLITS]
                             + [args.align / f"{s}.jsonl" for s in SPLITS] + [args.d1 / "manifest.csv"]}}
    (args.out / "set.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("boundary_counts", "audio_hours", "shortfall")}), f"files {len(files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
