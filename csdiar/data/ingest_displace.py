#!/usr/bin/env python3
"""SS-DL points, DS-DL/DS-SL turns and controls from DISPLACE 2024 references (PREREGISTRATION_M2B.md §2, §7).

Reads the two segment-level reference tracks of DISPLACE 2024 and derives the conversation analysis units:

- The **speaker reference** is the SD-track RTTM (speaker-labelled segments); the **language reference** is the
  LD-track RTTM (language-labelled segments). Both are coarse, segment-level, not word-aligned (§2).
- An **elementary timeline** is cut at every SD and LD segment edge. An interval is CLEAN when exactly one
  speaker and exactly one language are active; speech with 2+ speakers is OVERLAP; speech with 0 or 2+ languages
  is AMBIGUOUS. These are the regions a point must avoid (§2 overlap handling).
- A speaker **TURN** is a maximal stretch where only one speaker is active, clean intervals of that speaker joined
  across silences shorter than JOIN_GAP_S with no other speaker between them. Inside a turn a **language RUN** is a
  maximal same-language stretch; its length is the summed clean speech time of the run.
- An **SS-DL point** is the junction between two consecutive language runs of one turn (§2: "a boundary between two
  adjacent language runs of one speaker"). PLAN.md §2's exclusion zone applies: a point within 2c of any reference
  speaker change, overlap or ambiguous interval is dropped and counted by reason, so DS errors cannot leak in.
- A **DS-DL / DS-SL turn** is a reference speaker change A->B (B != A) within TURN_GAP_S with no overlap across the
  transition; DS-DL when the language of A's last run differs from B's first run, DS-SL when equal (§2).
- A **control (FAR_CTRL)** is a point inside a clean single-speaker single-language run, at least 2c from the run's
  edges and from any other point or turn: a 0.5 s grid plus the midpoints of VAD non-speech stretches in the run
  (§2). Controls with a pause > 150 ms are the phrase-boundary controls (PREREGISTRATION.md §6).
- **Pause** at any time t is the length of the longest VAD non-speech stretch overlapping [t - TOL_S, t + TOL_S]
  (0 if none); pause_bin and stratum are exactly `csdiar.data.boundaries.pause_bin` / `.stratum` (D1 house style,
  decision 10). With no VAD, pause is "none" everywhere and this is flagged.

The core of this module is stdlib-only (DISPLACE references are plain-text RTTMs); numpy is imported lazily inside
main() only, to read the optional per-split .npz VAD files. These are definitions and counts, not outcomes, so
they may be computed on every split.
Usage (dev first, never eval before the freeze):
  python3 -m csdiar.data.ingest_displace --root ~/data/displace --split dev \
      --vad-dir ~/data/displace/vad --out ~/data/displace/units/dev
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

# --- frozen parameters (PREREGISTRATION_M2B.md §2, §7; D1 house style in boundaries.py) -------------------
C_S = 0.5            # primary collar c (PLAN.md §2 / M2B §7)
EXCL_S = 2 * C_S     # 2c exclusion / spacing zone
TURN_GAP_S = 2.0     # DS speaker change within 2 s (PLAN.md §2)
JOIN_GAP_S = 2.0     # clean intervals of one speaker join across silences shorter than this
GRID_S = C_S         # 0.5 s control grid
TOL_S = 0.1          # pause-window widening, matches boundaries.TOL_S
EPS = 1e-9
FRAME_S = 0.032      # Silero frame k covers [0.032k, 0.032(k+1)) s (csdiar.data.vad)
VAD_THRESHOLD = 0.5  # non-speech = p < 0.5

# Archive layout as read from both zips' central directories (2026-10-19; names only, nothing extracted).
# Track 2's top folder is lower-case "track2_..." while Track 1's is "Track1_...": keep the case as is.
# Both archives hold the same 35 dev and 32 eval recording IDs, and their WAVs are byte-identical
# (same length and CRC-32 for all 67), so Track 1's audio serves both references.
LAYOUT = {
    "dev": {
        "wav": "Track1_SD_supervised/Dev/Track1_SD",
        "sd": "Track1_SD_supervised/Dev/Track1_SD_labels",
        "ld": "track2_LD_supervised/Dev/Track2_LD_labels",
    },
    "eval": {
        "wav": "Track1_SD_supervised/Eval/Track1_SD",
        "sd": "Track1_SD_supervised/Eval/SD_eval_rttms",
        "ld": "track2_LD_supervised/Eval/LD_eval_rttms",
    },
}
SD_SUFFIX = "_SPEAKER.rttm"
LD_SUFFIX = "_LANGUAGE.rttm"


# --- house-style helpers, copied verbatim from csdiar.data.boundaries (kept in sync by test) -------------
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


# --- RTTM parsing ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Segment:
    onset: float
    dur: float
    label: str

    @property
    def end(self) -> float:
        return self.onset + self.dur


def check_label_variants(rec_id: str, track: str, labels: set[str]) -> None:
    """Refuse a track whose labels differ only in case (e.g. 'ne' and 'NE'): compared as opaque strings they would
    split one run in two and fabricate a junction, and merging them silently could join distinct labels. The
    summary lists every raw label, so a variant is fixed by an explicit, recorded mapping, never by guessing."""
    groups: dict[str, set[str]] = {}
    for lab in labels:
        groups.setdefault(lab.casefold(), set()).add(lab)
    clash = sorted(sorted(v) for v in groups.values() if len(v) > 1)
    if clash:
        raise ValueError(f"{rec_id} {track}: labels differing only in case {clash}; add an explicit label map")


def parse_rttm(text: str) -> list[Segment]:
    """Parse RTTM lines "<TYPE> <file> <chan> <onset> <dur> <NA> <NA> <label> <NA> <NA>".

    Any TYPE token is accepted; blank and comment (';'/'#') lines are skipped. The label (field 8, 1-based) is
    the speaker (SD) or the language (LD), an opaque string. Segments with non-positive duration are dropped.
    Output is sorted by onset then end for a deterministic timeline.
    """
    segs: list[Segment] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] in ";#":
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            onset = float(parts[3])
            dur = float(parts[4])
        except ValueError:
            continue
        label = parts[7]
        if dur <= 0 or label in ("", "<NA>"):
            continue
        segs.append(Segment(onset, dur, label))
    segs.sort(key=lambda s: (s.onset, s.end))
    return segs


# --- elementary timeline ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    speakers: tuple[str, ...]   # distinct speaker labels active over the whole interval
    languages: tuple[str, ...]  # distinct language labels active over the whole interval

    @property
    def dur(self) -> float:
        return self.end - self.start

    @property
    def is_speech(self) -> bool:
        return bool(self.speakers)

    @property
    def is_clean(self) -> bool:
        return len(self.speakers) == 1 and len(self.languages) == 1

    @property
    def is_overlap(self) -> bool:
        return len(self.speakers) >= 2

    @property
    def is_ambiguous(self) -> bool:
        # speech with 0 or 2+ languages (but not an overlap, which is classified first)
        return self.is_speech and not self.is_overlap and len(self.languages) != 1


def _active(segs: Sequence[Segment], start: float, end: float) -> tuple[str, ...]:
    mid = (start + end) / 2
    labels = {s.label for s in segs if s.onset <= mid < s.end}
    return tuple(sorted(labels))


def elementary_timeline(sd: Sequence[Segment], ld: Sequence[Segment]) -> list[Interval]:
    """Cut a single timeline at every SD and LD edge; label each atom by its active speakers and languages.

    Only intervals with at least one active speaker (speech) are returned; pure silence carries no speaker so it
    is not a timeline atom, but the gaps it leaves are exactly the silences that turn/run joining steps over.
    """
    edges = {0.0}
    for s in list(sd) + list(ld):
        edges.add(s.onset)
        edges.add(s.end)
    cuts = sorted(edges)
    out: list[Interval] = []
    for a, b in zip(cuts, cuts[1:]):
        if b - a <= EPS:
            continue
        spk = _active(sd, a, b)
        if not spk:
            continue  # silence (no active speaker): not a timeline atom
        lang = _active(ld, a, b)
        out.append(Interval(a, b, spk, lang))
    return out


# --- turns and language runs -----------------------------------------------------------------------------
@dataclass
class LangRun:
    lang: str
    start: float          # onset of the run's first clean segment
    end: float            # end of the run's last clean segment
    speech_s: float       # summed clean speech time of the run
    first_start: float    # start of the run's first clean interval
    last_end: float       # end of the run's last clean interval


@dataclass
class Turn:
    speaker: str
    start: float
    end: float
    runs: list[LangRun] = field(default_factory=list)


def _speaker_change_times(sd: Sequence[Segment]) -> list[float]:
    """Times of reference speaker changes, for the 2c exclusion zone (PLAN.md §2).

    A change is any transition between two consecutive SPEECH regions whose active speaker set differs —
    including across an intervening silence (A ends, B begins later: still a speaker change), and into or out
    of overlap. A silence boundary alone is not a change. The recorded time is the boundary: the end of the
    earlier region when a silence separates them, else the shared edge.
    """
    edges = sorted({s.onset for s in sd} | {s.end for s in sd})
    times: list[float] = []
    prev: tuple[str, ...] | None = None
    prev_end: float | None = None
    for a, b in zip(edges, edges[1:]):
        spk = _active(sd, a, b)
        if not spk:
            continue  # silence: carry prev across it
        if prev is not None and spk != prev:
            # change between two speech regions; mark both the previous region's end and this onset so a point
            # near either side of the (possibly silent) transition is excluded.
            times.append(prev_end if prev_end is not None else a)
            if a != prev_end:
                times.append(a)
        prev = spk
        prev_end = b
    return sorted(set(times))


def build_turns(timeline: Sequence[Interval]) -> list[Turn]:
    """Maximal single-speaker stretches, joining clean intervals of one speaker across short silences.

    A turn grows while the active speaker set stays exactly {A}. Non-clean single-speaker-A atoms (A with 0 or 2+
    languages) still belong to A's turn, but only CLEAN atoms contribute to language runs. The turn breaks when a
    different or additional speaker becomes active, or when a silence of at least JOIN_GAP_S separates two of A's
    clean atoms (the join-across-silences rule).
    """
    turns: list[Turn] = []
    cur: Turn | None = None
    last_end: float | None = None        # end of the current speaker's previous atom (clean or not)
    for iv in timeline:
        if len(iv.speakers) != 1:
            cur = None
            last_end = None
            continue
        spk = iv.speakers[0]
        # Silence = no speech at all, so the gap is measured between consecutive atoms of the speaker. The split is
        # decided before the turn is extended, so the earlier turn never absorbs the later atom's end.
        long_gap = (cur is not None and cur.speaker == spk and last_end is not None
                    and iv.start - last_end >= JOIN_GAP_S - EPS)
        if cur is None or cur.speaker != spk or long_gap:
            cur = Turn(spk, iv.start, iv.end)
            turns.append(cur)
        else:
            cur.end = iv.end
        last_end = iv.end
        if iv.is_clean:
            _extend_run(cur, iv)
    return turns


def _extend_run(turn: Turn, iv: Interval) -> None:
    lang = iv.languages[0]
    if turn.runs and turn.runs[-1].lang == lang:
        r = turn.runs[-1]
        r.end = iv.end
        r.last_end = iv.end
        r.speech_s += iv.dur
    else:
        turn.runs.append(LangRun(lang, iv.start, iv.end, iv.dur, iv.start, iv.end))


# --- VAD ------------------------------------------------------------------------------------------------
def nonspeech_stretches(vad: Sequence[float] | None) -> list[tuple[float, float]]:
    """Maximal [start, end) non-speech stretches (p < threshold) from a frame probability sequence.

    Frame k covers [FRAME_S k, FRAME_S (k+1)) s (csdiar.data.vad). With no VAD, returns []."""
    if not vad:
        return []
    out: list[tuple[float, float]] = []
    k = 0
    n = len(vad)
    while k < n:
        if vad[k] < VAD_THRESHOLD:
            j = k
            while j + 1 < n and vad[j + 1] < VAD_THRESHOLD:
                j += 1
            out.append((k * FRAME_S, (j + 1) * FRAME_S))
            k = j + 1
        else:
            k += 1
    return out


def pause_ms_at(t: float, stretches: Sequence[tuple[float, float]]) -> float:
    """Length (ms) of the longest non-speech stretch overlapping [t - TOL_S, t + TOL_S], 0 if none."""
    lo, hi = t - TOL_S, t + TOL_S
    best = 0.0
    for s0, s1 in stretches:
        if s1 > lo and s0 < hi:
            best = max(best, s1 - s0)
    return best * 1000.0


def longest_nonspeech_in(stretches: Sequence[tuple[float, float]], lo: float, hi: float) -> tuple[float, float] | None:
    """Longest non-speech stretch clipped to [lo, hi], or None. Used to place an SS-DL point inside a gap."""
    best: tuple[float, float] | None = None
    for s0, s1 in stretches:
        a, b = max(s0, lo), min(s1, hi)
        if b - a > EPS and (best is None or (b - a) > (best[1] - best[0]) + EPS):
            best = (a, b)
    return best


# --- point / turn / control derivation -------------------------------------------------------------------
def _covered_bad(intervals: Sequence[Interval], t0: float, t1: float,
                 own_gap: tuple[float, float, str] | None = None) -> bool:
    """True if [t0, t1] overlaps any overlap or ambiguous interval.

    `own_gap` = (gap0, gap1, speaker) marks an SS-DL point's own junction gap: the no-language stretch between
    the point's two language runs of one speaker is where the point lives, so a single-speaker-A no-language
    atom lying inside that gap is the expected inter-run silence, not a disqualifying ambiguity. A genuine
    ambiguity (2+ languages) or any overlap still disqualifies even inside the gap.
    """
    for iv in intervals:
        if iv.end <= t0 or iv.start >= t1:
            continue
        if iv.is_overlap:
            return True
        if iv.is_ambiguous:
            if (own_gap is not None and iv.start >= own_gap[0] - EPS and iv.end <= own_gap[1] + EPS
                    and len(iv.speakers) == 1 and iv.speakers[0] == own_gap[2] and len(iv.languages) == 0):
                continue  # the point's own inter-run gap (single speaker, no language): allowed
            return True
    return False


def _near_change(times: Sequence[float], t0: float, t1: float) -> bool:
    return any(t0 <= ch <= t1 for ch in times)


def ssdl_points(turns: Sequence[Turn], timeline: Sequence[Interval], spk_changes: Sequence[float],
                stretches: Sequence[tuple[float, float]], excluded: Counter) -> list[dict]:
    """One row per junction of two consecutive language runs inside a turn; excluded points are counted."""
    rows: list[dict] = []
    for turn in turns:
        for a, b in zip(turn.runs, turn.runs[1:]):
            gap0, gap1 = a.last_end, b.first_start       # end of A's last segment -> start of B's first
            mid_a, mid_b = (a.start + a.end) / 2, (b.start + b.end) / 2
            # Point time t: midpoint of the longest VAD non-speech stretch inside the gap widened by TOL_S on
            # each side (never past the runs' midpoints); else the gap's middle.
            lo = max(min(gap0, gap1) - TOL_S, mid_a)
            hi = min(max(gap0, gap1) + TOL_S, mid_b)
            stretch = longest_nonspeech_in(stretches, lo, hi) if hi > lo else None
            t = (stretch[0] + stretch[1]) / 2 if stretch else (gap0 + gap1) / 2
            excl = t - EXCL_S, t + EXCL_S
            gap_lo, gap_hi = min(gap0, gap1), max(gap0, gap1)
            if _covered_bad(timeline, *excl, own_gap=(gap_lo, gap_hi, turn.speaker)):
                excluded["overlap_or_ambiguous"] += 1
                continue
            if _near_change(spk_changes, *excl):
                excluded["near_speaker_change"] += 1
                continue
            shorter = min(a.speech_s, b.speech_s)
            # elig_p0L1: both runs >= 1 s and [t-1, t], [t, t+1] each lie inside one clean run.
            elig = (a.speech_s >= 1.0 - EPS and b.speech_s >= 1.0 - EPS
                    and _window_in_clean_run(timeline, t - 1.0, t, turn.speaker, a.lang)
                    and _window_in_clean_run(timeline, t, t + 1.0, turn.speaker, b.lang))
            pms = pause_ms_at(t, stretches)
            rows.append({
                "kind": "ssdl", "speaker": turn.speaker, "speaker2": "",
                "lang_left": a.lang, "lang_right": b.lang,
                "t": round(t, 3), "left_s": round(a.speech_s, 3), "right_s": round(b.speech_s, 3),
                "stratum": stratum(shorter), "elig_p0L1": elig,
                "pause_ms": round(pms, 1), "pause_bin": pause_bin(pms), "phrase": False,
            })
    return rows


def _window_in_clean_run(timeline: Sequence[Interval], lo: float, hi: float, speaker: str, lang: str) -> bool:
    """True if [lo, hi] is fully covered by clean intervals of one (speaker, lang) with no gap."""
    covering = sorted((iv for iv in timeline
                       if iv.is_clean and iv.speakers[0] == speaker and iv.languages[0] == lang
                       and iv.end > lo and iv.start < hi),
                      key=lambda iv: iv.start)
    if not covering:
        return False
    cursor = lo
    for iv in covering:
        if iv.start > cursor + EPS:
            return False
        cursor = max(cursor, iv.end)
        if cursor >= hi - EPS:
            return True
    return cursor >= hi - EPS


def ds_turns(turns: Sequence[Turn], timeline: Sequence[Interval], stretches: Sequence[tuple[float, float]],
             excluded: Counter) -> list[dict]:
    """DS-DL / DS-SL transitions: consecutive turns of different speakers within TURN_GAP_S, no overlap across."""
    rows: list[dict] = []
    for a, b in zip(turns, turns[1:]):
        if a.speaker == b.speaker or not a.runs or not b.runs:
            continue
        gap = b.start - a.end
        if gap < -EPS or gap > TURN_GAP_S + EPS:
            continue
        if _covered_bad(timeline, a.end - EPS, b.start + EPS):
            excluded["overlap_or_ambiguous"] += 1
            continue
        la, lb = a.runs[-1].lang, b.runs[0].lang
        kind = "dsdl" if la != lb else "dssl"
        t = (a.end + b.start) / 2
        pms = pause_ms_at(t, stretches)
        shorter = min(a.runs[-1].speech_s, b.runs[0].speech_s)
        rows.append({
            "kind": kind, "speaker": a.speaker, "speaker2": b.speaker,
            "lang_left": la, "lang_right": lb,
            "t": round(t, 3), "left_s": round(a.runs[-1].speech_s, 3), "right_s": round(b.runs[0].speech_s, 3),
            "stratum": stratum(shorter), "elig_p0L1": False,
            "pause_ms": round(pms, 1), "pause_bin": pause_bin(pms), "phrase": False,
        })
    return rows


def control_points(turns: Sequence[Turn], timeline: Sequence[Interval],
                   stretches: Sequence[tuple[float, float]], reserved: Sequence[float]) -> list[dict]:
    """Candidate controls inside clean single-speaker single-language runs (§2 FAR_CTRL).

    Candidates: the midpoint of every non-speech stretch inside the run, placed first so each pause yields a
    control, then a GRID_S grid. Each must be at least EXCL_S from the run's edges and from every already-placed
    point/turn/control time (`reserved`), so the ±c firing regions of two controls never overlap.
    """
    rows: list[dict] = []
    placed = list(reserved)
    for turn in turns:
        for run in turn.runs:
            lo, hi = run.first_start + EXCL_S, run.last_end - EXCL_S
            if hi <= lo:
                continue
            cands: list[tuple[int, float]] = []
            for s0, s1 in stretches:
                mid = (s0 + s1) / 2
                if lo - EPS <= mid <= hi + EPS:
                    cands.append((0, round(mid, 6)))
            g = _ceil_to_grid(lo)
            while g <= hi + EPS:
                cands.append((1, round(g, 6)))
                g += GRID_S
            cands.sort()
            for _prio, t in cands:
                if not _window_in_clean_run(timeline, t - EPS, t + EPS, turn.speaker, run.lang):
                    continue
                if any(abs(t - p) < EXCL_S - EPS for p in placed):
                    continue
                pms = pause_ms_at(t, stretches)
                placed.append(t)
                rows.append({
                    "kind": "ctrl", "speaker": turn.speaker, "speaker2": "",
                    "lang_left": run.lang, "lang_right": run.lang,
                    "t": round(t, 3), "left_s": round(t - run.first_start, 3),
                    "right_s": round(run.last_end - t, 3),
                    "stratum": stratum(min(t - run.first_start, run.last_end - t)),
                    "elig_p0L1": False,
                    "pause_ms": round(pms, 1), "pause_bin": pause_bin(pms), "phrase": pms > 150.0,
                })
    return rows


def _ceil_to_grid(x: float) -> float:
    import math
    return math.ceil(x / GRID_S - EPS) * GRID_S


# --- per-recording driver --------------------------------------------------------------------------------
def process_recording(rec_id: str, split: str, sd_text: str, ld_text: str,
                      vad: Sequence[float] | None) -> tuple[list[dict], dict]:
    """All rows (points, turns, controls) for one recording, plus its summary of counts and exclusions."""
    sd = parse_rttm(sd_text)
    ld = parse_rttm(ld_text)
    for track, segs in (("SD", sd), ("LD", ld)):
        check_label_variants(rec_id, track, {s.label for s in segs})
    timeline = elementary_timeline(sd, ld)
    turns = build_turns(timeline)
    spk_changes = _speaker_change_times(sd)
    stretches = nonspeech_stretches(vad)
    excluded: Counter = Counter()

    ssdl = ssdl_points(turns, timeline, spk_changes, stretches, excluded)
    ds = ds_turns(turns, timeline, stretches, excluded)
    reserved = [r["t"] for r in ssdl] + [r["t"] for r in ds]
    ctrl = control_points(turns, timeline, stretches, reserved)

    rows = ssdl + ds + ctrl
    for r in rows:
        r["recording"] = rec_id
        r["split"] = split
    rows = order_rows(rows)

    summary = {
        "recording": rec_id, "split": split,
        "speakers": sorted({s.label for s in sd}), "languages": sorted({s.label for s in ld}),
        "has_vad": bool(stretches) or (vad is not None and len(vad) > 0),
        "n_turns": len(turns),
        "n_ssdl": len(ssdl),
        "n_dsdl": sum(1 for r in ds if r["kind"] == "dsdl"),
        "n_dssl": sum(1 for r in ds if r["kind"] == "dssl"),
        "n_ctrl": len(ctrl),
        "n_phrase_ctrl": sum(1 for r in ctrl if r["phrase"]),
        "ssdl_by_stratum": dict(Counter(r["stratum"] for r in ssdl)),
        "ssdl_elig_p0L1": sum(1 for r in ssdl if r["elig_p0L1"]),
        "excluded": dict(excluded),
    }
    return rows, summary


FIELDS = ["recording", "split", "kind", "t", "speaker", "speaker2", "lang_left", "lang_right",
          "left_s", "right_s", "stratum", "elig_p0L1", "pause_ms", "pause_bin", "phrase"]

_KIND_ORDER = {"ssdl": 0, "dsdl": 1, "dssl": 1, "ctrl": 2}


def order_rows(rows: list[dict]) -> list[dict]:
    """Deterministic output order: by t, then kind (points, turns, controls), then the rest of the key."""
    return sorted(rows, key=lambda r: (r["t"], _KIND_ORDER.get(r["kind"], 9), r["kind"],
                                       r["speaker"], r["speaker2"], r["lang_left"], r["lang_right"]))


# --- IO -------------------------------------------------------------------------------------------------
def write_csv_gz(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def discover_ids(sd_dir: Path) -> list[str]:
    ids = sorted(p.name[: -len(SD_SUFFIX)] for p in sd_dir.glob(f"*{SD_SUFFIX}"))
    return ids


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--split", choices=["dev", "eval"], required=True)
    ap.add_argument("--vad-dir", type=Path, default=None,
                    help="optional directory with <split>.npz, one float array per recording ID (numpy, lazy)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ld-suffix", default=LD_SUFFIX, help="LD RTTM filename suffix (Track 2 path parameter)")
    args = ap.parse_args(argv)

    lay = LAYOUT[args.split]
    sd_dir = args.root / lay["sd"]
    ld_dir = args.root / lay["ld"]
    # Fail loudly: a wrong path must not turn into "no language labels" and silently empty counts.
    for kind, d in (("speaker", sd_dir), ("language", ld_dir)):
        if not d.is_dir():
            raise SystemExit(f"ingest_displace: {kind} label folder not found: {d}")
    ids = discover_ids(sd_dir)
    if not ids:
        raise SystemExit(f"ingest_displace: no *{SD_SUFFIX} files in {sd_dir}")
    ld_ids = sorted(p.name[: -len(args.ld_suffix)] for p in ld_dir.glob(f"*{args.ld_suffix}"))
    if ld_ids != ids:
        only_sd = sorted(set(ids) - set(ld_ids))
        only_ld = sorted(set(ld_ids) - set(ids))
        raise SystemExit(f"ingest_displace: speaker and language references cover different recordings "
                         f"in {args.split}: speaker only {only_sd}, language only {only_ld}")

    vad_arrays: dict[str, Sequence[float]] = {}
    vad_flagged = False
    if args.vad_dir is not None:
        npz_path = args.vad_dir / f"{args.split}.npz"
        if npz_path.exists():
            import numpy as np  # lazy: only needed to read the optional .npz VAD
            with np.load(npz_path) as z:
                vad_arrays = {k: z[k].astype(float).tolist() for k in z.files}
        else:
            vad_flagged = True
    else:
        vad_flagged = True

    args.out.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    summaries: list[dict] = []
    for rec_id in ids:
        sd_text = (sd_dir / f"{rec_id}{SD_SUFFIX}").read_text(encoding="utf-8")
        ld_text = (ld_dir / f"{rec_id}{args.ld_suffix}").read_text(encoding="utf-8")
        vad = vad_arrays.get(rec_id)
        rows, summary = process_recording(rec_id, args.split, sd_text, ld_text, vad)
        all_rows.extend(rows)
        summaries.append(summary)

    all_rows = order_rows(all_rows)
    for kind in ("ssdl", "turns", "ctrl"):
        subset = [r for r in all_rows if (r["kind"] in ("dsdl", "dssl") if kind == "turns" else r["kind"] == kind)]
        write_csv_gz(args.out / f"{kind}_{args.split}.csv.gz", subset)

    totals: Counter = Counter()
    excl_totals: Counter = Counter()
    for s in summaries:
        for k in ("n_turns", "n_ssdl", "n_dsdl", "n_dssl", "n_ctrl", "n_phrase_ctrl", "ssdl_elig_p0L1"):
            totals[k] += s[k]
        for k, v in s["excluded"].items():
            excl_totals[k] += v
    summary_doc = {
        "split": args.split,
        "params": {"c_s": C_S, "excl_s": EXCL_S, "turn_gap_s": TURN_GAP_S, "join_gap_s": JOIN_GAP_S,
                   "grid_s": GRID_S, "tol_s": TOL_S, "frame_s": FRAME_S, "vad_threshold": VAD_THRESHOLD},
        "ld_suffix": args.ld_suffix,
        "n_recordings": len(ids),
        "vad_absent": vad_flagged,
        "totals": dict(totals),
        "excluded_totals": dict(excl_totals),
        "recordings": summaries,
    }
    (args.out / "summary.json").write_text(json.dumps(summary_doc, indent=2))
    print(args.split, json.dumps({k: totals[k] for k in sorted(totals)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
