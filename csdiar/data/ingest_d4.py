#!/usr/bin/env python3
"""Ingest D4 (MUCS 2021 sub-task 2, OpenSLR 104) for H7 (PLAN.md §3, D4).

Each archive holds continuous spoken-tutorial recordings (<split>/<rec>.wav) and Kaldi files
(<split>/transcripts/{wav.scp,segments,text,utt2spk}). Every published segment is cut out of its continuous
recording as one clip (cutting regions out of one real recording is allowed by PLAN.md §3; nothing is joined,
mixed or modified). The output folder has the same layout as D1's ingest, so align, VAD, embedding and boundary
jobs run on it unchanged:
  <out>/manifest.csv   clip_id, split, speaker_id, video_id (= recording), start_ms, end_ms, duration_s,
                       speech_type, cmi, provenance_ok, wav, sample_rate, n_samples, transcription
  <out>/wav/<split>/<clip_id>.wav   16 kHz mono PCM-16
  <out>/ingest.json    counts, skipped segments and their reasons, archive SHA-256 (from SHA256SUMS)
speech_type is from script only (Latin only -> en; native script only -> the pair's native code; both -> cs);
cmi is left empty (the corpus has none). Segments that are empty, shorter than 0.5 s, run past the recording's
end by more than 50 ms, or come from a recording that is missing or not 16 kHz are skipped and counted.
Extraction is checked file by file against the archive's member sizes, and any missing or partial file is
re-extracted, so an interrupted run is repaired by re-running.
Usage (on the box):
  .venv-pyannote4/bin/python -m csdiar.data.ingest_d4 --raw ~/data/d4/raw --pair Hindi-English --out ~/data/d4/hi-en
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tarfile
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
NATIVE = {"Hindi-English": ("hi", "DEVANAGARI"), "Bengali-English": ("bn", "BENGALI")}
MIN_DUR = 0.5
END_TOL = 0.05


def script_type(text: str, native_script: str, native_code: str) -> str:
    latin = native = False
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            latin |= name.startswith("LATIN")
            native |= name.startswith(native_script)
    return "cs" if latin and native else ("en" if latin else (native_code if native else "none"))


def read_kaldi(path: Path, n_fields: int | None = None) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1 if n_fields is None else n_fields - 1)
        if parts:
            out[parts[0]] = parts[1] if len(parts) > 1 else ""
    return out


def ensure_extracted(arc: Path, src: Path) -> dict:
    """Extract arc into src, re-extracting any member whose file is missing or has the wrong size."""
    with tarfile.open(arc) as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
    bad = [m.name for m in members if not (src / m.name).exists() or (src / m.name).stat().st_size != m.size]
    if bad:
        with tarfile.open(arc) as tf:
            todo = [m for m in tf.getmembers() if m.name in set(bad)]
            tf.extractall(src, members=todo, filter="data")
    still = [m.name for m in members if (src / m.name).stat().st_size != m.size]
    if still:
        raise RuntimeError(f"{arc.name}: {len(still)} members still differ after extraction, e.g. {still[:3]}")
    return {"members": len(members), "re_extracted": len(bad)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", type=Path, required=True, help="folder with the OpenSLR 104 archives")
    ap.add_argument("--pair", choices=sorted(NATIVE), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    args = ap.parse_args()
    code, script = NATIVE[args.pair]
    src = args.out / "src"
    sums = dict(line.split()[::-1] for line in (args.raw / "SHA256SUMS").read_text().splitlines() if line.strip())
    report = {"pair": args.pair, "archives": {}, "splits": {}}
    rows = []
    for split in args.splits:
        arc = args.raw / f"{args.pair}_{split}.tar.gz"
        report["archives"][arc.name] = {"sha256": sums.get(f"./{arc.name}"), **ensure_extracted(arc, src)}
        tdir = src / split / "transcripts"
        scp = {k: v.split()[-1] for k, v in read_kaldi(tdir / "wav.scp").items()}
        text, utt2spk = read_kaldi(tdir / "text"), read_kaldi(tdir / "utt2spk")
        segs = [line.split() for line in (tdir / "segments").read_text(encoding="utf-8").splitlines() if line.strip()]
        segs.sort(key=lambda s: (s[1], float(s[2])))
        skipped = Counter()
        (args.out / "wav" / split).mkdir(parents=True, exist_ok=True)
        cache = {}
        for utt, rec, t0, t1 in segs:
            t0, t1 = float(t0), float(t1)
            txt = " ".join(text.get(utt, "").split())
            if not txt:
                skipped["empty_transcript"] += 1
                continue
            if t1 - t0 < MIN_DUR:
                skipped["shorter_than_0.5s"] += 1
                continue
            if rec not in cache:
                p = src / split / scp.get(rec, f"{rec}.wav")
                if not p.exists():
                    cache = {rec: None}
                else:
                    x, sr = sf.read(p, dtype="int16", always_2d=True)
                    cache = {rec: (x[:, 0], sr, x.shape[1])}
            if cache[rec] is None:
                skipped["recording_missing"] += 1
                continue
            x, sr, n_ch = cache[rec]
            if sr != SR:
                skipped["not_16k"] += 1
                continue
            if t1 > len(x) / SR + END_TOL:
                skipped["past_recording_end"] += 1
                continue
            clip = x[int(round(t0 * SR)): min(int(round(t1 * SR)), len(x))]
            rel = Path("wav") / split / f"{utt}.wav"
            sf.write(args.out / rel, clip, SR, subtype="PCM_16")
            rows.append({"clip_id": utt, "split": split, "speaker_id": utt2spk.get(utt, ""), "video_id": rec,
                         "start_ms": int(round(t0 * 1000)), "end_ms": int(round(t1 * 1000)),
                         "duration_s": round(len(clip) / SR, 3), "speech_type": script_type(txt, script, code),
                         "cmi": "", "provenance_ok": 1, "wav": str(rel), "sample_rate": SR,
                         "n_samples": len(clip), "transcription": txt})
            if n_ch != 1:
                skipped["multichannel_first_channel_used"] += 1
        sp = [r for r in rows if r["split"] == split]
        report["splits"][split] = {"segments": len(segs), "clips": len(sp), "skipped": dict(skipped),
                                   "hours": round(sum(r["duration_s"] for r in sp) / 3600, 3),
                                   "speakers": len({r["speaker_id"] for r in sp}),
                                   "recordings": len({r["video_id"] for r in sp}),
                                   "speech_type": dict(Counter(r["speech_type"] for r in sp))}
        print(split, json.dumps(report["splits"][split]), flush=True)
    with open(args.out / "manifest.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    spk_recs = Counter()
    for spk, rec in {(r["speaker_id"], r["video_id"]) for r in rows}:
        spk_recs[spk] += 1
    report["recordings_per_speaker"] = dict(Counter(spk_recs.values()))
    report["speakers_in_both_splits"] = len({r["speaker_id"] for r in rows if r["split"] == "train"}
                                            & {r["speaker_id"] for r in rows if r["split"] == "test"})
    (args.out / "ingest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("recordings_per_speaker", "speakers_in_both_splits")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
