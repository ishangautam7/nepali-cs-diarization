#!/usr/bin/env python3
"""M2b post-clustering hooks for P1-AHC (PREREGISTRATION_M2B.md §3, §4).

These are the two hooks `p1_ahc.py` says "act after/inside clustering and are added with the M2b supplement":
B2 as a language-aware AHC affinity offset (§4) and M2 as the P1-AHC local relabel post-processor (§4,
"a local relabel of the shorter segment only, never a global cluster merge"). A pairwise affinity offset cannot
be expressed inside pyannote's centroid-linkage AHC, so both act on P1-AHC's OUTPUT: the speakers and their
segments produced by `p1_ahc.diarize`.

Everything here is stdlib-only (plain lists for vectors, the `math` module): the box's analysis environment has
Python 3.12 without numpy, and these hooks must import and test without it. The numerics mirror
`csdiar/analysis/confirm_h4.py` (`jsd`, base-2) and `csdiar/analysis/m2_lambda.py` (the 0.1 s LID grid, the
k-10/k window pair) so λ here has the same form and scale in bits as on D1.

The scale conventions follow the supplement:
- centroids are unit vectors; cosine distance is ``1 - cos`` of two unit vectors (§4: AHC works on 1 - cos, and
  δ is a cosine difference, so lowering a distance by δ equals raising an affinity by δ);
- λ(t) is the base-2 JSD of the renormalised left/right LID windows, at most 1 bit (§2), and "λ is high" means
  λ >= 0.5 bit (§3, §4);
- B2 merges only cross-language pairs (same-language pairs were already compared by AHC at the threshold, §4);
- M2 decides every adjacent pair on the ORIGINAL labels and relabels the shorter segment only (§4).

Public functions: two_language_jsd, lambda_at, majority_language, b2_merge, m2_relabel.

Integration contract (the adapter is part of the M2b confirmatory code, not this file):
- b2_merge needs, per output speaker of `p1_ahc.diarize`, the unit duration-weighted centroid of the system's own
  embeddings of that speaker's output segments, and `lang_dur` = seconds of that speaker's speech per
  LID-predicted language (argmax of the VoxLingua107 posterior renormalised over the recording's reference
  languages, §2);
- m2_relabel needs the exclusive diarization's segments in time order, each with the unit embedding of that
  segment, and `lam` = lambda_at bound to the recording's LID grid with `pair` = the indices of the recording's
  two reference languages in the posterior (never (ne, en) on DISPLACE; departure 3);
- the same embedding model as the system under test supplies both.
"""
from __future__ import annotations

import math
from typing import Optional

HOP = 0.1          # LID grid hop, seconds (m2_lambda.HOP)
LAG = 10           # windows compared are k-LAG and k (1.0 s apart on the 0.1 s grid)
EPS = 1e-12        # log-floor, matching confirm_h4.jsd's np.maximum(., 1e-12)
UNIT_TOL = 1e-6    # tolerance for the unit-norm check on centroids/embeddings


# --------------------------------------------------------------------------- #
# vector helpers (stdlib-only; plain lists)
# --------------------------------------------------------------------------- #
def _as_floats(v, name: str) -> list:
    try:
        out = [float(x) for x in v]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: expected a sequence of numbers") from exc
    if len(out) == 0:
        raise ValueError(f"{name}: empty vector")
    return out


def _norm(v: list) -> float:
    return math.sqrt(sum(x * x for x in v))


def _check_unit(v, name: str) -> list:
    out = _as_floats(v, name)
    n = _norm(out)
    if abs(n - 1.0) > UNIT_TOL:
        raise ValueError(f"{name}: expected a unit vector, got norm {n!r}")
    return out


def _cos(a: list, b: list) -> float:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def cosine_distance(a, b) -> float:
    """1 - cos of two unit vectors (§4). Validates both are unit and same-dimension."""
    ua = _check_unit(a, "a")
    ub = _check_unit(b, "b")
    return 1.0 - _cos(ua, ub)


# --------------------------------------------------------------------------- #
# 1. two_language_jsd
# --------------------------------------------------------------------------- #
def _kl2(p: list, q: list) -> float:
    # base-2 KL, with the same 1e-12 floor and "0 * log 0 = 0" rule as confirm_h4.jsd
    s = 0.0
    for a, b in zip(p, q):
        if a > 0.0:
            s += a * math.log2(max(a, EPS) / max(b, EPS))
    return s


def _jsd2(p: list, q: list) -> float:
    m = [0.5 * (a + b) for a, b in zip(p, q)]
    return 0.5 * _kl2(p, m) + 0.5 * _kl2(q, m)


def two_language_jsd(p_left, p_right, pair) -> Optional[float]:
    """Base-2 Jensen-Shannon divergence between two posterior vectors renormalised over the two label indices
    in ``pair`` (PREREGISTRATION_M2B.md §2, §3).

    Each side is renormalised to sum 1 over exactly the two entries ``pair = (i, j)``; a side with zero mass over
    the pair returns None. On 2-entry inputs renormalised over (0, 1) this equals ``confirm_h4.jsd``; bounded by
    1 bit.
    """
    i, j = pair
    i = int(i)
    j = int(j)
    if i == j:
        raise ValueError(f"pair must name two distinct indices, got {pair!r}")
    pl = _as_floats(p_left, "p_left")
    pr = _as_floats(p_right, "p_right")
    if len(pl) != len(pr):
        raise ValueError(f"posteriors differ in length ({len(pl)} vs {len(pr)}): misaligned LID windows")
    for idx in (i, j):
        if idx < 0 or idx >= len(pl) or idx >= len(pr):
            raise ValueError(f"pair index {idx} out of range for posteriors of length "
                             f"{len(pl)}/{len(pr)}")
    left = [pl[i], pl[j]]
    right = [pr[i], pr[j]]
    sl, sr = left[0] + left[1], right[0] + right[1]
    if sl <= 0.0 or sr <= 0.0:
        return None
    left = [left[0] / sl, left[1] / sl]
    right = [right[0] / sr, right[1] / sr]
    return _jsd2(left, right)


# --------------------------------------------------------------------------- #
# 2. lambda_at
# --------------------------------------------------------------------------- #
def lambda_at(t, post, pair, hop: float = HOP, lag: int = LAG) -> Optional[float]:
    """λ at time t: the two_language_jsd of LID windows k-lag and k, with k = round(t / hop)
    (m2_lambda.lam, generalised to an explicit label pair per PREREGISTRATION_M2B.md §2).

    ``post`` is a per-window list of posterior vectors on the 0.1 s LID grid; window k covers
    [0.1k, 0.1k + 1.0] s. Returns None when k-lag < 0 or k >= len(post) (out of range), or when
    ``two_language_jsd`` returns None for a window with no mass over ``pair``.
    """
    if hop <= 0.0:
        raise ValueError(f"hop must be positive, got {hop!r}")
    if lag <= 0:
        raise ValueError(f"lag must be positive, got {lag!r}")
    if len(post) == 0:
        raise ValueError("post: empty window list")
    k = int(round(float(t) / hop))
    kl = k - lag
    if kl < 0 or k >= len(post):
        return None
    return two_language_jsd(post[kl], post[k], pair)


# --------------------------------------------------------------------------- #
# 3. majority_language
# --------------------------------------------------------------------------- #
def majority_language(spans):
    """The language with the most total duration over a list of (duration, lang) pairs.

    Ties are broken by the lexicographically smallest language label. Raises ValueError on an empty list or a
    negative duration.
    """
    if len(spans) == 0:
        raise ValueError("spans: empty")
    totals: dict = {}
    for item in spans:
        dur, lang = item
        dur = float(dur)
        if dur < 0.0:
            raise ValueError(f"negative duration {dur!r} for language {lang!r}")
        totals[lang] = totals.get(lang, 0.0) + dur
    # most duration, then lexicographically smallest label
    best_lang = None
    best_dur = None
    for lang in sorted(totals):
        d = totals[lang]
        if best_dur is None or d > best_dur:
            best_dur = d
            best_lang = lang
    return best_lang


def _total_duration(lang_dur: dict) -> float:
    return sum(float(v) for v in lang_dur.values())


# --------------------------------------------------------------------------- #
# 4. b2_merge
# --------------------------------------------------------------------------- #
def _validate_speakers(speakers: dict) -> dict:
    if not isinstance(speakers, dict) or len(speakers) == 0:
        raise ValueError("speakers: expected a non-empty mapping")
    clusters = {}
    dim = None
    for label, rec in speakers.items():
        if not isinstance(label, str):
            raise ValueError(f"speaker labels must be strings (P1-AHC's are), got {label!r}")
        if "centroid" not in rec or "lang_dur" not in rec:
            raise ValueError(f"speaker {label!r}: needs 'centroid' and 'lang_dur'")
        cen = _check_unit(rec["centroid"], f"speaker {label!r} centroid")
        if dim is None:
            dim = len(cen)
        elif len(cen) != dim:
            raise ValueError(f"speaker {label!r}: centroid dimension {len(cen)} != {dim}")
        ld = rec["lang_dur"]
        if not isinstance(ld, dict) or len(ld) == 0:
            raise ValueError(f"speaker {label!r}: lang_dur must be a non-empty mapping")
        clean_ld = {}
        for lang, sec in ld.items():
            sec = float(sec)
            if sec < 0.0:
                raise ValueError(f"speaker {label!r}: negative duration for {lang!r}")
            clean_ld[lang] = sec
        clusters[label] = {"centroid": cen, "lang_dur": clean_ld}
    return clusters


def _weighted_unit_mean(cen_a: list, w_a: float, cen_b: list, w_b: float) -> list:
    mean = [w_a * x + w_b * y for x, y in zip(cen_a, cen_b)]
    n = _norm(mean)
    if n <= 0.0:
        raise ValueError("duration-weighted centroid mean is a zero vector; cannot renormalise")
    return [x / n for x in mean]


def b2_merge(speakers: dict, threshold: float, delta: float) -> dict:
    """B2 as a language-aware AHC affinity offset applied to P1-AHC's output clusters
    (PREREGISTRATION_M2B.md §4).

    ``speakers`` maps a label to ``{"centroid": unit vector, "lang_dur": {lang: seconds}}``. Repeatedly: among
    pairs of CURRENT clusters whose majority languages DIFFER, take the pair with the smallest cosine distance
    (1 - cos of unit centroids; ties broken by the sorted label pair). If that distance < threshold + delta,
    merge the pair and continue; otherwise stop. The merged cluster keeps the label of the member with more total
    duration (ties -> the smaller label), its centroid is the duration-weighted mean of the two unit centroids
    re-normalised, and its lang_dur is the summed per-language durations. Same-language pairs are never merged
    here (AHC already compared them at the threshold). ``delta = 0`` leaves everything unchanged.

    Returns ``{old_label: final_label}`` over the original labels.
    """
    threshold = float(threshold)
    delta = float(delta)
    clusters = _validate_speakers(speakers)
    # resolver maps every ORIGINAL label to the current cluster label it belongs to
    resolve = {lbl: lbl for lbl in clusters}

    while True:
        labels = sorted(clusters)
        majs = {lbl: majority_language([(sec, lang) for lang, sec in clusters[lbl]["lang_dur"].items()])
                for lbl in labels}
        best = None  # (dist, (la, lb))
        for ia in range(len(labels)):
            for ib in range(ia + 1, len(labels)):
                la, lb = labels[ia], labels[ib]
                if majs[la] == majs[lb]:
                    continue  # same-language pairs never merged here
                dist = 1.0 - _cos(clusters[la]["centroid"], clusters[lb]["centroid"])
                key = (dist, (la, lb))
                if best is None or key < best:
                    best = key
        if best is None:
            break
        dist, (la, lb) = best
        if not (dist < threshold + delta):
            break
        # choose the surviving label: more total duration, ties -> smaller label
        da = _total_duration(clusters[la]["lang_dur"])
        db = _total_duration(clusters[lb]["lang_dur"])
        if da > db or (da == db and la <= lb):
            keep, drop = la, lb
        else:
            keep, drop = lb, la
        wk = _total_duration(clusters[keep]["lang_dur"])
        wd = _total_duration(clusters[drop]["lang_dur"])
        tot = wk + wd
        if tot <= 0.0:
            # both clusters have zero duration: fall back to an equal-weight mean
            new_cen = _weighted_unit_mean(clusters[keep]["centroid"], 0.5,
                                          clusters[drop]["centroid"], 0.5)
        else:
            new_cen = _weighted_unit_mean(clusters[keep]["centroid"], wk / tot,
                                          clusters[drop]["centroid"], wd / tot)
        new_ld = dict(clusters[keep]["lang_dur"])
        for lang, sec in clusters[drop]["lang_dur"].items():
            new_ld[lang] = new_ld.get(lang, 0.0) + sec
        clusters[keep] = {"centroid": new_cen, "lang_dur": new_ld}
        del clusters[drop]
        # every original label that pointed at drop now points at keep
        for orig, cur in resolve.items():
            if cur == drop:
                resolve[orig] = keep
    return resolve


# --------------------------------------------------------------------------- #
# 5. m2_relabel
# --------------------------------------------------------------------------- #
def _validate_segments(segments) -> list:
    segs = []
    prev_end = None
    dim = None
    for idx, s in enumerate(segments):
        for key in ("start", "end", "label", "emb"):
            if key not in s:
                raise ValueError(f"segment {idx}: missing {key!r}")
        start = float(s["start"])
        end = float(s["end"])
        if end < start:
            raise ValueError(f"segment {idx}: end {end} < start {start}")
        if prev_end is not None and start < prev_end - 1e-12:
            raise ValueError(f"segment {idx}: starts at {start} before previous end {prev_end} "
                             "(unsorted or overlapping)")
        emb = _check_unit(s["emb"], f"segment {idx} emb")
        if dim is None:
            dim = len(emb)
        elif len(emb) != dim:
            raise ValueError(f"segment {idx}: emb dimension {len(emb)} != {dim}")
        segs.append({"start": start, "end": end, "label": s["label"], "emb": emb})
        prev_end = end
    if len(segs) == 0:
        raise ValueError("segments: empty")
    return segs


def m2_relabel(segments, lam, tau0, alpha, lam_min: float = 0.5, max_gap: float = 2.0):
    """M2 as the P1-AHC local relabel post-processor (PREREGISTRATION_M2B.md §4).

    ``segments`` is a time-ordered, non-overlapping list of dicts ``{"start","end","label","emb"}`` (the
    exclusive diarization). For every pair of CONSECUTIVE segments with DIFFERENT labels and a gap
    ``next.start - prev.end <= max_gap``, the boundary time ``t`` is the middle of the gap. Every pair is decided
    on the ORIGINAL labels: the SHORTER segment (ties -> the later one) is relabelled to the other's label when
    ``lam(t)`` is not None, ``lam(t) >= lam_min`` and ``cosine_distance(emb_prev, emb_next) < tau0 + alpha *
    lam(t)``. A segment two decisions would relabel takes the label of the neighbour at the SMALLER distance
    (ties -> the earlier neighbour). Nothing else is relabelled (no global merge). ``alpha = 0`` with ``tau0``
    below every distance changes nothing.

    Returns ``(new_segments, records)``: a list of COPIES with updated labels, and a list of relabel records
    ``{"index","old","new","t","lam","dist"}`` (one per segment actually relabelled).
    """
    tau0 = float(tau0)
    alpha = float(alpha)
    lam_min = float(lam_min)
    max_gap = float(max_gap)
    segs = _validate_segments(segments)
    n = len(segs)

    def duration(i):
        return segs[i]["end"] - segs[i]["start"]

    # Decide every eligible consecutive pair on the ORIGINAL labels. A decision names the segment to relabel
    # (the shorter one, ties -> later), the source label, and the pair's distance / t / lam.
    # claims[target_index] collects (dist, neighbour_index, new_label, t, lam).
    claims: dict = {}
    for i in range(n - 1):
        j = i + 1
        if segs[i]["label"] == segs[j]["label"]:
            continue
        gap = segs[j]["start"] - segs[i]["end"]
        if gap > max_gap + 1e-12:
            continue
        t = segs[i]["end"] + gap / 2.0
        lv = lam(t)
        if lv is None or lv < lam_min:
            continue
        dist = 1.0 - _cos(segs[i]["emb"], segs[j]["emb"])
        if not (dist < tau0 + alpha * lv):
            continue
        # relabel the shorter segment; ties -> the later one (j)
        if duration(i) < duration(j):
            target, source, neighbour = i, segs[j]["label"], j
        else:
            target, source, neighbour = j, segs[i]["label"], i
        claims.setdefault(target, []).append((dist, neighbour, source, t, lv))

    records = []
    new_segs = [dict(s) for s in segs]
    # Deliberately no cascade: every target takes a neighbour's ORIGINAL label, even when that neighbour is itself
    # relabelled in the same pass (A|B|C|A can give C -> B while B -> A). Cascading would make the result depend
    # on the processing order, which §4 rules out.
    for target in sorted(claims):
        # a double claim is resolved by the smaller distance; ties -> the earlier neighbour (smaller index)
        best = min(claims[target], key=lambda c: (c[0], c[1]))
        dist, _neigh, source, t, lv = best
        old = segs[target]["label"]
        new_segs[target]["label"] = source
        records.append({"index": target, "old": old, "new": source,
                        "t": t, "lam": lv, "dist": dist})
    return new_segs, records
