#!/usr/bin/env python3
"""Clean monolingual windows on the dense L = 1.0 s grid, for H3's lag-matched trials (PLAN.md §2, decision 14).

A grid window k covers [0.1k, 0.1k + 1.0] s. It is clean when:
- every word it touches is of one language (ne or en) from one run;
- no neutral or loan word lies between them (an unaligned word next to the touched span counts as inside);
- at least 2 words are touched;
- at least 60% of its VAD frames are speech.
Low-alignment clips (decision 10) are skipped. Within a clip, clean windows are thinned greedily in time order so
that chosen windows start at least 1.0 s apart (no overlap). These are definitions and features, not outcomes, so
every split is processed.
Output: <out>/windows_<split>.csv.gz with clip_id, video_id, speaker_id, k, t0, abs_t0 (seconds from the video
start), lang, n_words, speech_share.
Usage (on the box): python -m csdiar.data.windows --root ~/data/d1 --align ~/data/d1/align --vad ~/data/d1/vad \
    --out ~/data/d1/windows/lex_v1 --loans planning/lexicon/loans.txt --names planning/lexicon/names.txt
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

from csdiar.data.boundaries import (FRAME_S, LOW_ALIGN_PERCENTILE, VAD_THRESHOLD, clip_score, load_words,
                                    read_list)
from csdiar.data.tags import EN, NE

L = 1.0
HOP = 0.1
SR = 16000
MIN_WORDS = 2
MIN_SPEECH = 0.6


def clean_windows(words, vad: np.ndarray, n_samples: int) -> list[dict]:
    w, h = int(L * SR), int(HOP * SR)
    n = (n_samples - w) // h + 1 if n_samples >= w else 1
    out, last = [], -1e9
    for k in range(n):
        r0, r1 = k * HOP, k * HOP + L
        if r0 < last + L - 1e-9:
            continue
        touched = [x.idx for x in words if x.aligned and x.end > r0 and x.start < r1]
        if len(touched) < MIN_WORDS:
            continue
        a, b = min(touched), max(touched)
        edge_end, edge_start = words[b].end, words[a].start
        while b + 1 < len(words) and not words[b + 1].aligned and r1 > edge_end:
            b += 1
        while a > 0 and not words[a - 1].aligned and r0 < edge_start:
            a -= 1
        langs = {words[i].tag for i in range(a, b + 1)}
        if len(langs) != 1 or not langs <= {NE, EN}:
            continue
        k0, k1 = int(r0 / FRAME_S), min(int(np.ceil(r1 / FRAME_S)), len(vad))
        share = float((vad[k0:k1] >= VAD_THRESHOLD).mean()) if k1 > k0 else 0.0
        if share < MIN_SPEECH:
            continue
        out.append({"k": k, "t0": round(r0, 3), "lang": langs.pop(), "n_words": b - a + 1,
                    "speech_share": round(share, 3)})
        last = r0
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--align", type=Path, required=True)
    ap.add_argument("--vad", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--loans", type=Path)
    ap.add_argument("--names", type=Path)
    ap.add_argument("--splits", nargs="+", default=["validation", "train", "test"])
    args = ap.parse_args()
    loans, _ = read_list(args.loans)
    names, _ = read_list(args.names)
    manifest = {r["clip_id"]: r for r in csv.DictReader(open(args.root / "manifest.csv", encoding="utf-8"))}
    load = lambda s: {json.loads(l)["clip_id"]: json.loads(l) for l in open(args.align / f"{s}.jsonl", encoding="utf-8")}
    train_scores = [x for x in (clip_score(r) for r in load("train").values()) if x is not None]
    threshold = float(np.percentile(train_scores, LOW_ALIGN_PERCENTILE))
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in args.splits:
        vad = np.load(args.vad / f"{split}.npz")
        rows = []
        for clip_id, rec in load(split).items():
            score = clip_score(rec)
            if rec["status"] != "ok" or score is None or score < threshold:
                continue
            m = manifest[clip_id]
            words = load_words(rec, loans, names)
            offset = int(m["start_ms"]) / 1000 if m["start_ms"] else None  # no provenance: no video time
            for wnd in clean_windows(words, vad[clip_id].astype(np.float32), int(m["n_samples"])):
                rows.append({"clip_id": clip_id, "video_id": m["video_id"], "speaker_id": m["speaker_id"],
                             "abs_t0": "" if offset is None else round(offset + wnd["t0"], 3), **wnd})
        with gzip.open(args.out / f"windows_{split}.csv.gz", "wt", encoding="utf-8", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=["clip_id", "video_id", "speaker_id", "k", "t0", "abs_t0", "lang",
                                                "n_words", "speech_share"])
            wr.writeheader()
            wr.writerows(rows)
        summary[split] = {"windows": len(rows), "ne": sum(r["lang"] == NE for r in rows),
                          "en": sum(r["lang"] == EN for r in rows)}
        print(split, summary[split], flush=True)
    (args.out / "summary.json").write_text(json.dumps({"threshold": threshold, "splits": summary}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
