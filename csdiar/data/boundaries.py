#!/usr/bin/env python3
"""Switch points, control points and their matching in D1 clips (PLAN.md §2; clarifications in decision 10).

Inputs: the word alignment (csdiar.data.align), Silero VAD frames (csdiar.data.vad) and the manifest.
Words are re-tagged here with the curated loan lexicon and name list.

- Pause at a word boundary: the longest run of VAD non-speech frames (p < 0.5) whose centres lie in the aligned
  gap widened by 0.1 s on each side, never past the words' middles. Bins: none, < 150 ms, 150-300 ms, > 300 ms.
  Point time t = the pause midpoint, or the aligned boundary (middle of the gap between the words) when there is
  no pause.
- SS-DL switch point: consecutive runs A, B of different language with no loan and no neutral word between
  them, both boundary words aligned. Windows W_left = [p0 - L, p0] and W_right = [p1, p1 + L].
- Embedding family at L: both runs >= L; windows inside the clip, outer edge at most 0.1 s outside the run
  (alignment tolerance) and touching only words of that run's language (so no neutral or loan word).
- Detection families: switch points whose region t +- W touches no neutral or loan word (both languages are
  allowed), W = 0.5 s ("det", the collar c) or 1.0 s ("det_w1"). Stratum = the shorter run: < 0.5, 0.5-1.0,
  >= 1.0 s.
- Primary P0 cell "p0L1" (decision 16): both runs >= 1.0 s and P0's own windows [t - 1, t] and [t, t + 1] each
  inside one run and clean. Its controls are the det_w1 controls.
- Within-language control: a boundary between two consecutive language words of one run whose region
  [p0 - W, p1 + W] satisfies the same containment and same-language rules: W = L for an embedding family,
  W = 0.5 s (the collar c) for the detection family "det" and W = 1.0 s for its sensitivity family "det_w1".
  Phrase-boundary controls are those with a pause over 150 ms.
- Matching: up to 3 controls per switch point, exact on pause bin and position bin (2 s bins from the clip
  start, last bin >= 10 s); embedding families also on context length (shorter same-language side < 2L or
  >= 2L). Same clip first, else the same video (one recording); never across videos. Without replacement
  within a family, in seeded random order. Detection-family strata are reported, not matched. Because
  short-run switch points far outnumber clean controls, the detection families are compared primarily by
  cell standardisation (every eligible control in the same video x pause bin x position bin cell, with
  reuse); the 1:3 matches are kept as a sensitivity analysis (decision 10).
- Clips whose mean word alignment score is below the 5th percentile of train clips are flagged low_align
  and excluded from every family (kept in the files for sensitivity analyses).

These are definitions and counts, not outcomes, so they may be computed on every split.
Usage (on the box): python -m csdiar.data.boundaries --root ~/data/d1 --align ~/data/d1/align \
    --vad ~/data/d1/vad --out ~/data/d1/boundaries/<tag> [--loans FILE] [--names FILE]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from csdiar.data.tags import EN, LOAN, NE, NEUTRAL, Run, loan_key, runs, switches, tag_word

FRAME_S = 0.032
VAD_THRESHOLD = 0.5
TOL_S = 0.1
DET_HALF_S = {"det": 0.5, "det_w1": 1.0}  # detection-family region: the collar c, and +-1 s
P0_FAMILY, P0_L = "p0L1", 1.0  # primary P0 cell (decision 16): both runs >= L and P0's own windows clean
EMB_LS = (0.5, 1.0, 1.5)
FAMILIES = tuple(DET_HALF_S) + (P0_FAMILY,) + tuple(f"emb{L:g}" for L in EMB_LS)
POS_BIN_S = 2.0
POS_LAST_BIN = 5
MAX_CONTROLS = 3
LOW_ALIGN_PERCENTILE = 5.0
SPLITS = ("validation", "train", "test")


@dataclass(frozen=True)
class Word:
    idx: int
    text: str
    tag: str
    start: float | None = None
    end: float | None = None
    score: float | None = None

    @property
    def aligned(self) -> bool:
        return self.start is not None and self.end is not None


def pause_bin(ms: float) -> str:
    if ms <= 0:
        return "none"
    if ms < 150:
        return "lt150"
    if ms <= 300:
        return "150to300"
    return "gt300"


def stratum(shorter_s: float) -> str:
    if shorter_s < 0.5:
        return "lt0.5"
    if shorter_s < 1.0:
        return "0.5to1"
    return "ge1"


def pos_bin(t: float) -> int:
    return min(int(t // POS_BIN_S), POS_LAST_BIN)


def find_pause(vad: np.ndarray, left: Word, right: Word) -> tuple[float, float]:
    """(p0, p1) of the longest VAD non-speech run in the aligned gap between two words.

    The search region is the aligned gap widened by TOL_S on each side (alignment tolerance), but never past the
    middles of the two words, so a stop closure deep inside a word cannot win (decision 16). A frame k covers
    [0.032k, 0.032(k+1)) and counts when its centre lies in the region. Ties go to the run closest to the aligned
    boundary. With no non-speech frame, p0 == p1 == boundary.
    """
    boundary = (left.end + right.start) / 2
    lo = max((left.start + left.end) / 2, left.end - TOL_S)
    hi = min((right.start + right.end) / 2, right.start + TOL_S)
    k_lo = max(int(np.ceil(lo / FRAME_S - 0.5)), 0)
    k_hi = min(int(np.floor(hi / FRAME_S - 0.5)), len(vad) - 1)
    best = None
    k = k_lo
    while k <= k_hi:
        if vad[k] < VAD_THRESHOLD:
            j = k
            while j + 1 <= k_hi and vad[j + 1] < VAD_THRESHOLD:
                j += 1
            p0, p1 = k * FRAME_S, (j + 1) * FRAME_S
            rank = (j - k + 1, -abs((p0 + p1) / 2 - boundary))
            if best is None or rank > best[0]:
                best = (rank, p0, p1)
            k = j + 1
        else:
            k += 1
    return (boundary, boundary) if best is None else (best[1], best[2])


def run_span(words: Sequence[Word], run: Run) -> tuple[float, float] | None:
    first, last = words[run.first], words[run.last]
    return (first.start, last.end) if first.aligned and last.aligned else None


def region_ok(words: Sequence[Word], r0: float, r1: float, run: Run, span: tuple[float, float],
              clip_dur: float, check_start: bool = True, check_end: bool = True) -> bool:
    """[r0, r1] lies in the clip, touches only that run's language words, and its outer edges are at most
    TOL_S outside the run. A switch window's inner edge sits at the pause, so only its outer edge is checked.

    Unaligned words lying between touched words count as touched, so a digit inside the region fails it.
    """
    if r0 < 0 or r1 > clip_dur:
        return False
    if (check_start and r0 < span[0] - TOL_S) or (check_end and r1 > span[1] + TOL_S):
        return False
    touched = [w.idx for w in words if w.aligned and w.end > r0 and w.start < r1]
    if not touched:
        return False
    a, b = min(touched), max(touched)
    # An unaligned word (usually a digit) next to the touched span may be spoken inside the region.
    edge_end, edge_start = words[b].end, words[a].start
    while b + 1 < len(words) and not words[b + 1].aligned and r1 > edge_end:
        b += 1
    while a > 0 and not words[a - 1].aligned and r0 < edge_start:
        a -= 1
    if a < run.first or b > run.last:
        return False
    return all(words[i].tag == run.lang for i in range(a, b + 1))


def region_clean(words: Sequence[Word], r0: float, r1: float, clip_dur: float) -> bool:
    """[r0, r1] lies in the clip and touches no neutral or loan word; ne and en words are both allowed.

    The detection-family counterpart of region_ok for switch points: their region necessarily spans two
    languages, but it must be as free of neutral and loan words as a control's region (decision 16).
    """
    if r0 < 0 or r1 > clip_dur:
        return False
    touched = [w.idx for w in words if w.aligned and w.end > r0 and w.start < r1]
    if not touched:
        return False
    a, b = min(touched), max(touched)
    edge_end, edge_start = words[b].end, words[a].start
    while b + 1 < len(words) and not words[b + 1].aligned and r1 > edge_end:
        b += 1
    while a > 0 and not words[a - 1].aligned and r0 < edge_start:
        a -= 1
    return all(words[i].tag in (NE, EN) for i in range(a, b + 1))


def clip_points(words: Sequence[Word], vad: np.ndarray, clip_dur: float, low_align: bool,
                excluded: Counter) -> tuple[list[dict], list[dict]]:
    """(switch points, control points) of one clip, each a dict with eligibility per family."""
    tags = [w.tag for w in words]
    ssdl, ctrl = [], []
    for sw in switches(tags):
        a, b = sw.left, sw.right
        if sw.neutral_between:
            excluded["neutral_between"] += 1
            continue
        lw, rw = words[a.last], words[b.first]
        sa, sb = run_span(words, a), run_span(words, b)
        if not (lw.aligned and rw.aligned) or sa is None or sb is None:
            excluded["unaligned"] += 1
            continue
        p0, p1 = find_pause(vad, lw, rw)
        t = (p0 + p1) / 2
        dur_a, dur_b = sa[1] - sa[0], sb[1] - sb[0]
        elig = {fam: not low_align and region_clean(words, t - half, t + half, clip_dur)
                for fam, half in DET_HALF_S.items()}
        # Primary P0 cell: both runs at least as long as P0's window, and P0's own windows clean on each side.
        elig[P0_FAMILY] = (not low_align and dur_a >= P0_L and dur_b >= P0_L
                           and region_ok(words, t - P0_L, t, a, sa, clip_dur, check_end=False)
                           and region_ok(words, t, t + P0_L, b, sb, clip_dur, check_start=False))
        for L in EMB_LS:
            elig[f"emb{L:g}"] = (not low_align and dur_a >= L and dur_b >= L
                                 and region_ok(words, p0 - L, p0, a, sa, clip_dur, check_end=False)
                                 and region_ok(words, p1, p1 + L, b, sb, clip_dur, check_start=False))
        ssdl.append(_point("ssdl", a.last, p0, p1, a.lang, b.lang, dur_a, dur_b, lw, rw, elig))
    for run in runs(tags):
        span = run_span(words, run)
        if span is None:
            continue
        for j in range(run.first, run.last):
            lw, rw = words[j], words[j + 1]
            if lw.tag != run.lang or rw.tag != run.lang or not (lw.aligned and rw.aligned):
                continue
            p0, p1 = find_pause(vad, lw, rw)
            elig = {}
            for fam in FAMILIES:
                if fam == P0_FAMILY:
                    continue
                half = DET_HALF_S[fam] if fam in DET_HALF_S else float(fam[3:])
                elig[fam] = not low_align and region_ok(words, p0 - half, p1 + half, run, span, clip_dur)
            elig[P0_FAMILY] = elig["det_w1"]  # P0's +-1 s field is clean within one run
            if any(elig.values()):
                ctrl.append(_point("ctrl", j, p0, p1, run.lang, run.lang, p0 - span[0], span[1] - p1,
                                   lw, rw, elig))
    return ssdl, ctrl


def _point(kind, word_idx, p0, p1, lang_l, lang_r, left_s, right_s, lw, rw, elig) -> dict:
    t = (p0 + p1) / 2
    pause_ms = round((p1 - p0) * 1000)
    shorter = min(left_s, right_s)
    return {"kind": kind, "word": word_idx, "t": round(t, 3), "p0": round(p0, 3), "p1": round(p1, 3),
            "pause_ms": pause_ms, "pause_bin": pause_bin(pause_ms), "pos_bin": pos_bin(t),
            "lang_left": lang_l, "lang_right": lang_r, "left_s": round(left_s, 3), "right_s": round(right_s, 3),
            "stratum": stratum(shorter), "phrase": kind == "ctrl" and pause_ms > 150,
            "score_left": lw.score, "score_right": rw.score, **{f"elig_{f}": elig[f] for f in FAMILIES}}


def match_family(ssdl: list[dict], ctrl: list[dict], fam: str, seed: int) -> tuple[list[dict], Counter]:
    """Assign up to MAX_CONTROLS controls to each eligible switch point of one family (see module doc)."""
    rng = random.Random(f"{seed}:{fam}")
    emb = fam.startswith("emb")
    two_l = 2 * float(fam[3:]) if emb else 0.0

    def key(p: dict) -> tuple:
        # Embedding families also match on context length: the shorter same-language side (run length for a
        # switch point, distance to the run edge for a control) under or over 2L. The detection family's
        # strata (< 0.5 s etc.) cannot exist for a clean control, so they are used for reporting only.
        k = (p["video_id"] or p["clip_id"], p["pause_bin"], p["pos_bin"])
        return k + (min(p["left_s"], p["right_s"]) >= two_l,) if emb else k

    pool, pool_clip = defaultdict(list), defaultdict(list)
    for c in ctrl:
        if c[f"elig_{fam}"]:
            pool[key(c)].append(c["point_id"])
            pool_clip[(key(c), c["clip_id"])].append(c["point_id"])
    queues = {}
    for store in (pool, pool_clip):
        for k, ids in store.items():
            rng.shuffle(ids)
            queues[(store is pool_clip, k)] = deque(ids)
    order = [s for s in ssdl if s[f"elig_{fam}"]]
    rng.shuffle(order)
    used, pairs, n_matched = set(), [], Counter()
    for s in order:
        k, chosen = key(s), []
        for level, q in (("clip", queues.get((True, (k, s["clip_id"])))), ("video", queues.get((False, k)))):
            while q and len(chosen) < MAX_CONTROLS:
                c = q.popleft()
                if c not in used:
                    used.add(c)
                    chosen.append((c, level))
        for rank, (c, level) in enumerate(chosen, 1):
            pairs.append({"family": fam, "ssdl_id": s["point_id"], "ctrl_id": c, "level": level, "rank": rank})
        n_matched[len(chosen)] += 1
    return pairs, n_matched


def read_list(path: Path | None) -> tuple[frozenset[str], str | None]:
    if path is None:
        return frozenset(), None
    raw = path.read_bytes()
    items = {loan_key(line.split("#")[0]) for line in raw.decode("utf-8").splitlines()}
    return frozenset(i for i in items if i), hashlib.sha256(raw).hexdigest()


def load_words(rec: dict, loans: frozenset[str], names: frozenset[str]) -> list[Word]:
    out = []
    for i, w in enumerate(rec["words"]):
        al = w.get("aligned", False)
        out.append(Word(i, w["w"], tag_word(w["w"], loans, names, letters_neutral=True), w["start"] if al else None,
                        w["end"] if al else None, w.get("score") if al else None))
    return out


def clip_score(rec: dict) -> float | None:
    s = [w["score"] for w in rec["words"] if w.get("aligned")]
    return float(np.mean(s)) if rec["status"] == "ok" and s else None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_bytes(b"")
        return
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--align", type=Path, required=True)
    ap.add_argument("--vad", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--loans", type=Path)
    ap.add_argument("--names", type=Path)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--seed", type=int, default=20261018)
    args = ap.parse_args()

    loans, loans_sha = read_list(args.loans)
    names, names_sha = read_list(args.names)
    manifest = {r["clip_id"]: r for r in csv.DictReader(open(args.root / "manifest.csv", encoding="utf-8"))}
    align = {s: {json.loads(l)["clip_id"]: json.loads(l) for l in open(args.align / f"{s}.jsonl", encoding="utf-8")}
             for s in set(args.splits) | {"train"}}
    train_scores = [x for x in (clip_score(r) for r in align["train"].values()) if x is not None]
    threshold = float(np.percentile(train_scores, LOW_ALIGN_PERCENTILE))
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"seed": args.seed, "low_align_threshold": round(threshold, 4),
               "loans": {"n": len(loans), "sha256": loans_sha}, "names": {"n": len(names), "sha256": names_sha},
               "params": {"frame_s": FRAME_S, "vad_threshold": VAD_THRESHOLD, "tol_s": TOL_S,
                          "det_half_s": DET_HALF_S, "emb_L": EMB_LS, "pos_bin_s": POS_BIN_S,
                          "max_controls": MAX_CONTROLS}, "splits": {}}
    for split in args.splits:
        vad = np.load(args.vad / f"{split}.npz")
        all_ssdl, all_ctrl, excluded, n_low = [], [], Counter(), 0
        for clip_id, rec in align[split].items():
            m = manifest[clip_id]
            score = clip_score(rec)
            low = score is None or score < threshold
            n_low += low
            if rec["status"] != "ok":
                continue
            words = load_words(rec, loans, names)
            ssdl, ctrl = clip_points(words, vad[clip_id].astype(np.float32), float(rec["duration"]), low, excluded)
            for p in ssdl + ctrl:
                p.update(point_id=f"{clip_id}:{p['kind']}:{p['word']}", clip_id=clip_id, video_id=m["video_id"],
                         speaker_id=m["speaker_id"], split=split, low_align=low)
            all_ssdl += ssdl
            all_ctrl += ctrl
        pairs, matched = [], {}
        for fam in FAMILIES:
            p, n = match_family(all_ssdl, all_ctrl, fam, args.seed)
            pairs += p
            matched[fam] = {str(k): v for k, v in sorted(n.items())}
        write_csv(args.out / f"points_{split}.csv.gz", all_ssdl + all_ctrl)
        write_csv(args.out / f"matches_{split}.csv.gz", pairs)
        fam_counts = {}
        for fam in FAMILIES:
            s = [x for x in all_ssdl if x[f"elig_{fam}"]]
            c = [x for x in all_ctrl if x[f"elig_{fam}"]]
            cells = Counter((x["video_id"], x["pause_bin"], x["pos_bin"]) for x in c)
            covered = sum(cells[(x["video_id"], x["pause_bin"], x["pos_bin"])] > 0 for x in s)
            fam_counts[fam] = {"ssdl": len(s), "ctrl": len(c), "phrase_ctrl": sum(x["phrase"] for x in c),
                               "ssdl_with_cell_controls": covered,
                               "ssdl_by_stratum": dict(Counter(x["stratum"] for x in s)),
                               "ssdl_by_direction": dict(Counter(f"{x['lang_left']}>{x['lang_right']}" for x in s)),
                               "ssdl_by_pause_bin": dict(Counter(x["pause_bin"] for x in s)),
                               "matched_controls_per_point": matched[fam]}
        summary["splits"][split] = {"clips": len(align[split]), "low_align_clips": n_low,
                                    "ssdl_points": len(all_ssdl), "excluded_switches": dict(excluded),
                                    "families": fam_counts}
        print(split, json.dumps(summary["splits"][split]["families"]["emb1"]), flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
