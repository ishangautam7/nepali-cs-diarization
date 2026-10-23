#!/usr/bin/env python3
"""Training data for M3, the language-adversarial fine-tuner (PREREGISTRATION_H4.md §7).

Subcommands (run on the box):
  d5     The Nepali speaker pool D5 from IndicVoices Nepali: utterances of at least 2.0 s; speakers with at least
         20 of them; 50 per speaker drawn with seed 20261018 (all when fewer). Decoded to 16 kHz PCM-16 WAV under
         --out with manifest.csv. Needs pyarrow (.venv-data).
  d1     The D1 window table for every non-test clip that is not low-alignment: window starts on the 0.1 s grid,
         lengths {1.0, 1.25, 1.5, 1.75, 2.0} s, and a language label per window. A window is `ne` or `en` when
         that language covers at least 80% of the aligned ne+en word time inside it (lex_v1b tags; other tags
         ignored), `mixed` when both occur and neither reaches 80%, and unlabelled when it holds less than 0.4 s of
         aligned ne/en word time. Also the speaker label of every clip (channel, or the video under the
         speaker-level split rule of decision 27) and the audit's flagged clips. Writes d1_windows.npz.
  dev    Dev trials on the calibration channels (the pilot channels without video zJYS-XNDP2Q): 20,000
         cross-language same-video target pairs of frozen clean windows (windows/lex_v1) with starts at least
         2 s apart, and 20,000 non-target pairs of clean windows from different channels, drawn once with seed
         20261018. Writes dev_trials.npz.
  audio  int16 memmaps of the D1 clips in d1_windows.npz and of the D5 utterances, shared by training processes.
These are features and training inputs, not outcomes: nothing here touches a P0 statistic.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 20261018
SR = 16000
HOP = 0.1
LENGTHS = (1.0, 1.25, 1.5, 1.75, 2.0)
LABELS = ("ne", "en", "mixed")
PURE = 0.8
MIN_LANG_S = 0.4
D5_MIN_DUR = 2.0
D5_MIN_UTTS = 20
D5_PER_SPEAKER = 50
DEV_PAIRS = 20000
DEV_MIN_LAG = 2.0
SHARED_TEACHER_VIDEO = "zJYS-XNDP2Q"   # decision 27: very likely the teacher of scored channel UCOC6C8_...


# ---------------------------------------------------------------- D5 ----------------------------------------
def select_d5(speakers: np.ndarray, durations: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Indices of the chosen utterances: eligible = duration >= D5_MIN_DUR; speakers with >= D5_MIN_UTTS eligible;
    D5_PER_SPEAKER drawn without replacement per speaker (all when fewer), speakers in sorted order."""
    by: dict[str, list[int]] = defaultdict(list)
    for i, (s, d) in enumerate(zip(speakers, durations)):
        if d >= D5_MIN_DUR:
            by[s].append(i)
    out = []
    for s in sorted(by):
        idx = np.array(by[s])
        if len(idx) < D5_MIN_UTTS:
            continue
        out.extend(sorted(rng.choice(idx, size=min(D5_PER_SPEAKER, len(idx)), replace=False).tolist()))
    return np.array(sorted(out), dtype=np.int64)


def cmd_d5(args) -> int:
    import pyarrow.parquet as pq
    import soundfile as sf
    files = sorted(Path(args.src).glob("*.parquet"))
    spk, dur, loc = [], [], []
    for fi, f in enumerate(files):
        t = pq.read_table(f, columns=["speaker_id", "duration", "lang"])
        for ri, (s, d, lg) in enumerate(zip(t["speaker_id"].to_pylist(), t["duration"].to_pylist(),
                                            t["lang"].to_pylist())):
            if lg == "ne":
                spk.append(s); dur.append(d); loc.append((fi, ri))
    chosen = select_d5(np.array(spk), np.array(dur), np.random.default_rng(SEED))
    want: dict[int, set[int]] = defaultdict(set)
    for i in chosen:
        want[loc[i][0]].add(loc[i][1])
    out = Path(args.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    rows = []
    for fi, f in enumerate(files):
        if fi not in want:
            continue
        t = pq.read_table(f, columns=["audio_filepath", "speaker_id", "duration", "scenario", "gender"])
        for ri in sorted(want[fi]):
            rec = {c: t[c][ri].as_py() for c in ("audio_filepath", "speaker_id", "duration", "scenario", "gender")}
            wav, sr = sf.read(io.BytesIO(rec["audio_filepath"]["bytes"]), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != SR:
                raise ValueError(f"{f.name} row {ri}: sample rate {sr}")
            rel = Path("wav") / rec["speaker_id"] / f"{f.stem}_{ri:05d}.wav"
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            sf.write(out / rel, wav, SR, subtype="PCM_16")
            rows.append({"speaker_id": rec["speaker_id"], "wav": str(rel), "duration": round(len(wav) / SR, 3),
                         "scenario": rec["scenario"], "gender": rec["gender"]})
        print(f"d5: file {fi + 1}/{len(files)}, {len(rows)} utterances", flush=True)
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    meta = {"source": str(args.src), "files": len(files), "seed": SEED, "min_dur_s": D5_MIN_DUR,
            "min_utts": D5_MIN_UTTS, "per_speaker": D5_PER_SPEAKER, "utterances": len(rows),
            "speakers": len({r["speaker_id"] for r in rows}), "hours": round(sum(r["duration"] for r in rows) / 3600, 2)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))
    return 0


# ---------------------------------------------------------------- D1 windows --------------------------------
def window_labels(starts: np.ndarray, ends: np.ndarray, langs: np.ndarray, n_grid: int, length: float
                  ) -> np.ndarray:
    """Label per grid start k (window [0.1k, 0.1k + length]): 0 ne, 1 en, 2 mixed, -1 unlabelled.
    starts/ends/langs describe aligned ne (0) / en (1) words only."""
    t0 = np.arange(n_grid) * HOP
    t1 = t0 + length
    if len(starts) == 0:
        return np.full(n_grid, -1, np.int8)
    ov = np.clip(np.minimum(t1[:, None], ends[None, :]) - np.maximum(t0[:, None], starts[None, :]), 0, None)
    ne = (ov * (langs == 0)[None, :]).sum(1)
    en = (ov * (langs == 1)[None, :]).sum(1)
    tot = ne + en
    lab = np.full(n_grid, -1, np.int8)
    ok = tot >= MIN_LANG_S
    lab[ok & (ne >= PURE * tot)] = 0
    lab[ok & (en >= PURE * tot)] = 1
    lab[ok & (lab == -1) & (ne > 0) & (en > 0)] = 2
    return lab


def video_speakers(audit_emb: Path, rows: list[dict]) -> dict[str, str]:
    """Decision 27's speaker-level rule: a video is its own speaker when its best within-channel centroid cosine
    is below the 99.9th percentile of cross-channel video-centroid cosines, under both audit embeddings."""
    from csdiar.analysis.purity_audit import centroid
    channel_of = {r["video_id"]: r["speaker_id"] for r in rows if r["video_id"]}
    below = {}
    for m in ("titanet", "campplus"):
        emb = {}
        for s in ("validation", "train", "test"):
            with np.load(audit_emb / m / f"{s}.npz") as z:
                emb.update({k: z[k] for k in z.files})
        by = defaultdict(list)
        for r in rows:
            if r["video_id"] and r["clip_id"] in emb:
                by[r["video_id"]].append(emb[r["clip_id"]])
        vids = sorted(by)
        c = np.stack([centroid(np.stack(by[v])) for v in vids])
        sim = c @ c.T
        same = np.array([[channel_of[a] == channel_of[b] for b in vids] for a in vids])
        iu = np.triu_indices(len(vids), 1)
        thr = float(np.percentile(sim[iu][~same[iu]], 99.9))
        for i, v in enumerate(vids):
            mates = [j for j in range(len(vids)) if j != i and same[i, j]]
            best = max((sim[i, j] for j in mates), default=None)
            below.setdefault(v, []).append(best is not None and best < thr)
    return {v: f"{channel_of[v]}::{v}" if all(b) and len(b) == 2 else channel_of[v] for v, b in below.items()}


def cmd_d1(args) -> int:
    from csdiar.data.boundaries import clip_score, load_words, read_list
    rows_all = [r for r in csv.DictReader(open(args.root / "manifest.csv", encoding="utf-8")) if r["video_id"]]
    rows = [r for r in rows_all if r["split"] != "test"]
    loans, _ = read_list(args.loans)
    names, _ = read_list(args.names)
    summary = json.loads((args.boundaries / "summary.json").read_text())
    low = summary["low_align_threshold"]
    spk_of_video = video_speakers(args.audit_emb, rows_all)   # the reference distribution uses every video
    flagged = set(json.loads((args.purity / "subset.json").read_text())["dropped_clips"])
    recs = {}
    for split in ("validation", "train"):
        with open(args.align / f"{split}.jsonl", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                recs[rec["clip_id"]] = rec
    clip_ids, clip_split, clip_spk, clip_video, clip_flag = [], [], [], [], []
    w_clip, w_k, w_len, w_lab = [], [], [], []
    n_low = 0
    for r in rows:
        rec = recs.get(r["clip_id"])
        score = clip_score(rec) if rec else None
        if score is None or score < low:
            n_low += 1
            continue
        words = load_words(rec, loans, names)
        al = [w for w in words if w.start is not None and w.tag in ("ne", "en")]
        st = np.array([w.start for w in al], float)
        en = np.array([w.end for w in al], float)
        lg = np.array([0 if w.tag == "ne" else 1 for w in al], np.int8)
        ci = len(clip_ids)
        clip_ids.append(r["clip_id"]); clip_split.append(r["split"]); clip_video.append(r["video_id"])
        clip_spk.append(spk_of_video.get(r["video_id"], r["speaker_id"])); clip_flag.append(r["clip_id"] in flagged)
        dur = float(r["duration_s"])
        for li, L in enumerate(LENGTHS):
            n_grid = int(np.floor((dur - L) / HOP + 1e-6)) + 1
            if n_grid <= 0:
                continue
            lab = window_labels(st, en, lg, n_grid, L)
            keep = np.nonzero(lab >= 0)[0]
            if len(keep) and keep.max() >= np.iinfo(np.int16).max:
                raise ValueError(f"{r['clip_id']}: window start {keep.max()} does not fit int16")
            w_clip.append(np.full(len(keep), ci, np.int32)); w_k.append(keep.astype(np.int16))
            w_len.append(np.full(len(keep), li, np.int8)); w_lab.append(lab[keep])
    out = args.out / "d1_windows.npz"
    np.savez_compressed(out, clip_id=np.array(clip_ids), clip_split=np.array(clip_split),
                        clip_speaker=np.array(clip_spk), clip_video=np.array(clip_video),
                        clip_flagged=np.array(clip_flag), win_clip=np.concatenate(w_clip),
                        win_k=np.concatenate(w_k), win_len=np.concatenate(w_len), win_label=np.concatenate(w_lab),
                        lengths=np.array(LENGTHS), labels=np.array(LABELS))
    lab_all, len_all = np.concatenate(w_lab), np.concatenate(w_len)
    meta = {"clips": len(clip_ids), "low_alignment_skipped": n_low, "flagged": int(sum(clip_flag)),
            "speakers_split_off": sorted({s for s in clip_spk if "::" in s}),
            "windows_by_length_label": {f"{LENGTHS[li]}": {LABELS[j]: int(((len_all == li) & (lab_all == j)).sum())
                                                         for j in range(3)} for li in range(len(LENGTHS))}}
    (args.out / "d1_windows_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=1))
    return 0


# ---------------------------------------------------------------- dev trials --------------------------------
def dev_pairs(win: list[dict], rng: np.random.Generator, n: int = DEV_PAIRS) -> tuple[np.ndarray, np.ndarray]:
    """(targets, nontargets): index pairs (i < j) into `win`.
    Targets: same video, different language, |abs_t0 difference| >= DEV_MIN_LAG; all qualifying pairs are
    enumerated and n are drawn without replacement. Nontargets: different channels; n distinct pairs drawn by
    rejection sampling of uniform random pairs."""
    tgt = []
    by_video: dict[str, list[int]] = defaultdict(list)
    for i, w in enumerate(win):
        by_video[w["video_id"]].append(i)
    for v in sorted(by_video):
        idx = np.array(by_video[v])
        lang = np.array([win[i]["lang"] for i in idx])
        t = np.array([float(win[i]["abs_t0"]) for i in idx])
        a, b = np.triu_indices(len(idx), 1)
        ok = (lang[a] != lang[b]) & (np.abs(t[a] - t[b]) >= DEV_MIN_LAG)
        tgt.append(np.stack([idx[a[ok]], idx[b[ok]]], 1))
    tgt = np.concatenate(tgt) if tgt else np.zeros((0, 2), int)
    tgt = tgt[np.sort(rng.choice(len(tgt), size=min(n, len(tgt)), replace=False))]
    spk = np.array([w["speaker_id"] for w in win])
    seen, non = set(), []
    while len(non) < n:
        i, j = rng.integers(0, len(win), 2)
        if spk[i] == spk[j]:
            continue
        key = (min(i, j), max(i, j))
        if key not in seen:
            seen.add(key)
            non.append(key)
    return tgt, np.array(sorted(non))


def cmd_dev(args) -> int:
    with gzip.open(args.windows / "windows_validation.csv.gz", "rt", encoding="utf-8") as fh:
        win = [w for w in csv.DictReader(fh) if w["video_id"] != SHARED_TEACHER_VIDEO and w["lang"] in ("ne", "en")]
    tgt, non = dev_pairs(win, np.random.default_rng(SEED))
    np.savez_compressed(args.out / "dev_trials.npz", clip_id=np.array([w["clip_id"] for w in win]),
                        k=np.array([int(w["k"]) for w in win], np.int32),
                        speaker=np.array([w["speaker_id"] for w in win]), targets=tgt, nontargets=non)
    meta = {"windows": len(win), "targets": int(len(tgt)), "nontargets": int(len(non)),
            "excluded_video": SHARED_TEACHER_VIDEO, "channels": sorted({w["speaker_id"] for w in win})}
    (args.out / "dev_trials_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))
    return 0


def cmd_audio(args) -> int:
    """Concatenate the audio M3 reads into int16 memmaps that every training process shares through the page
    cache: D1 clips from d1_windows.npz, and the D5 utterances. Writes <name>_audio.int16 and <name>_offsets.npy
    (start sample of each item; the last entry is the total)."""
    import soundfile as sf
    z = np.load(args.out / "d1_windows.npz")
    man = {r["clip_id"]: r["wav"] for r in csv.DictReader(open(args.root / "manifest.csv", encoding="utf-8"))}
    d1 = [args.root / man[c] for c in z["clip_id"]]
    d5_rows = list(csv.DictReader(open(args.d5 / "manifest.csv", encoding="utf-8"))) if (args.d5 / "manifest.csv").exists() else []
    d5 = [args.d5 / r["wav"] for r in d5_rows]
    for name, paths in (("d1", d1), ("d5", d5)):
        if name not in args.which:
            continue
        if not paths:
            print(f"{name}: no items, skipped", flush=True)
            continue
        if (args.out / f"{name}_audio.int16").exists():
            # Running training processes memmap this file; rewriting it would corrupt their reads.
            print(f"{name}: {name}_audio.int16 exists, not rewritten", flush=True)
            continue
        offs = [0]
        tmp = args.out / f"{name}_audio.int16.partial"
        with open(tmp, "wb") as fh:
            for i, p in enumerate(paths):
                wav, sr = sf.read(p, dtype="int16")
                if sr != SR or wav.ndim != 1:
                    raise ValueError(f"{p}: expected 16 kHz mono")
                fh.write(wav.tobytes())
                offs.append(offs[-1] + len(wav))
                if (i + 1) % 5000 == 0:
                    print(f"{name}: {i + 1}/{len(paths)}", flush=True)
        np.save(args.out / f"{name}_offsets.npy", np.array(offs, np.int64))
        tmp.rename(args.out / f"{name}_audio.int16")
        print(f"{name}: {len(paths)} items, {offs[-1] / SR / 3600:.1f} h", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("d5")
    a.add_argument("--src", type=Path, default=Path.home() / "data/indicvoices_nepali/nepali")
    a.add_argument("--out", type=Path, default=Path.home() / "data/d5")
    e = sub.add_parser("audio")
    e.add_argument("--root", type=Path, default=Path.home() / "data/d1")
    e.add_argument("--d5", type=Path, default=Path.home() / "data/d5")
    e.add_argument("--out", type=Path, default=Path.home() / "data/m3")
    e.add_argument("--which", nargs="+", choices=["d1", "d5"], default=["d1", "d5"])
    b = sub.add_parser("d1")
    b.add_argument("--root", type=Path, default=Path.home() / "data/d1")
    b.add_argument("--align", type=Path, default=Path.home() / "data/d1/align")
    b.add_argument("--boundaries", type=Path, default=Path.home() / "data/d1/boundaries/lex_v1b")
    b.add_argument("--audit-emb", type=Path, default=Path.home() / "data/d1/audit")
    b.add_argument("--purity", type=Path, default=Path.home() / "data/d1/results/purity")
    b.add_argument("--loans", type=Path, default=Path("planning/lexicon/loans.txt"))
    b.add_argument("--names", type=Path, default=Path("planning/lexicon/names.txt"))
    b.add_argument("--out", type=Path, default=Path.home() / "data/m3")
    c = sub.add_parser("dev")
    c.add_argument("--windows", type=Path, default=Path.home() / "data/d1/windows/lex_v1")
    c.add_argument("--out", type=Path, default=Path.home() / "data/m3")
    args = ap.parse_args()
    if getattr(args, "out", None):
        args.out.mkdir(parents=True, exist_ok=True)
    return {"d5": cmd_d5, "d1": cmd_d1, "dev": cmd_dev, "audio": cmd_audio}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
