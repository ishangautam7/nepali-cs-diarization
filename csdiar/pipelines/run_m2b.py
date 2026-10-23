#!/usr/bin/env python3
"""Cached DISPLACE-2024 system runner for the M2b confirmatory study (PREREGISTRATION_M2B.md §3, §4).

Tuning M2b on DISPLACE dev re-clusters every system (B0, B1, B2, M1, M2, M3 on P1-AHC+E3) at 15 clustering
thresholds (0.30-1.00 step 0.05) and M2 over 11 alpha values, on ~20 h of 26-60 min recordings. Re-running
Community-1 segmentation and the (chunk, local-speaker) embeddings for every threshold would be wasteful, so
this module splits `p1_ahc.diarize` into:

  * **Stage 1 (expensive, cached once per recording and embedding model):** everything Community-1 computes up to
    the clustering INPUT -- segmentation, powerset-binarized segmentation, the frame-level speaker count, and the
    (num_chunks, local_num_speakers, dim) embeddings -- saved to an .npz keyed by (model, recording id).
  * **Stage 2 (cheap, per threshold):** an optional embedding transform (the B1/M1 hook; identity for B0/M2/M3),
    pyannote's own AgglomerativeClustering (centroid linkage, cosine, min_cluster_size 12) at the threshold, and
    the pipeline's own `reconstruct` / `to_annotation` / speaker renaming, reproducing EXACTLY what
    `p1_ahc.diarize` returns at that threshold (overlapping + exclusive annotations).

How Stage 2 stays identical to `diarize()` -- `apply()` (pyannote.audio 4.0.7
SpeakerDiarization) does, after `get_embeddings`:
    hard_clusters, _, centroids = self.clustering(embeddings, binarized_segmentations, num_clusters=None, ...)
    inactive = binarized_segmentations.data.sum(1) == 0 ; hard_clusters[inactive] = -2
    count.data = min(count.data, max_speakers) ; discrete = reconstruct(segmentations, hard_clusters, count)
    diar = to_annotation(discrete, 0.0, min_duration_off) ; (overlapping)
    count.data = min(count.data, 1) ; discrete_excl = reconstruct(...) ; excl = to_annotation(...)
    rename labels to SPEAKER_00.. in label order (no reference annotation on our runs).
Stage 2 rebuilds `segmentations`, `binarized_segmentations` and `count` as SlidingWindowFeature from the cache and
calls the SAME pipeline methods, so the only moving part is the clustering threshold and the optional transform.
`load_stage2_pipeline` loads Community-1 WITHOUT the embedding model (segmentation is never re-run in stage 2),
and clustering is instantiated exactly as `p1_ahc.load_pipeline` does.

LID grid (`lid` subcommand) and the M2b adapter follow `csdiar/models/lid.py`, `csdiar/analysis/m2_lambda.py` and
the "Integration contract" in `csdiar/pipelines/m2b_hooks.py`: window k covers [0.1k, 0.1k + 1.0] s, lambda is
the base-2 JSD of windows k-10 and k renormalised over the recording's two reference-language indices, and the
language pair, delta, alpha and tau0 are INPUTS -- no language is hardcoded.

TWO LID-window conventions read the same grid (review m3; do NOT "fix" one to match the other):
  * CENTRED (k = round(t/hop) - round(win/(2*hop)) = g - 5): used by `lang_dur_from_segments` and
    `slot_languages` for B1/B2's majority-language assignment -- the window centred on the step/frame, so a
    segment's own time is credited to the language spoken at that time (brief change 1);
  * D1 convention (k = round(t/hop), the window STARTING near t): used by `make_lambda_closure` -> `m2b_hooks
    .lambda_at`, matching `m2_lambda.lam`, for M2's gate lambda. Changing either to match the other would break
    reproduction of `m2_lambda`.

CLI (run under `.venv-p1` on the box, CPU-only with `CUDA_VISIBLE_DEVICES=''`):
  python -m csdiar.pipelines.run_m2b cache   --model e3 --wav REC.wav --cache-dir DIR [--rid ID]
  python -m csdiar.pipelines.run_m2b lid      --wav REC.wav --cache-dir DIR [--rid ID] [--win 1.0 --hop 0.1]
  python -m csdiar.pipelines.run_m2b cluster  --model MODEL --rid ID --cache-dir DIR \
        --thresholds 0.70 0.90 --out OUTDIR --split dev --system SYS [system args]
    SYS and its required args (each refuses a missing required arg AND an arg irrelevant to it):
      B0 (--model e1|e2|e3): no extra args.
      B1 (--model e1|e2|e3): --pair a,b  --means MEANS.npz   (one (dim,) array per VoxLingua107 code; both
                             pair languages required; LID cache needed).
      M1 (--model e1|e2|e3): --v-hat V.npy                   (dim checked, normalised on load, zero/non-finite
                             refused).
      B2 (--model e1|e2|e3): --pair a,b  --delta D  --wav REC.wav   (LID cache + audio: B2 embeds output
                             segments with the system's own model).
      M2 (--model e1|e2|e3): --pair a,b  --alpha A [--tau0 T]  --wav REC.wav   (deployed tau0=0; alpha grid).
      M3 (--model m3:<ckpt>): no extra args; clusters the m3 cache. --model m3:<ckpt> is REQUIRED for M3 and
                              REFUSED for every other system; e1/e2/e3 is required for B0/B1/B2/M1/M2.
Stage 2 loads NO model for B0/B1/M1/M3 (they only re-cluster the cache), but B2 and M2 DO embed their output
segments with the system's own embedding model (SegmentEmbedder), so stage 2 is not always model-free (review
m5 docstring fix). Statistics -- B1 means, M1's v -- come only from the files passed on the command line, never
from the recording being scored.
M2's deployed form is tau0 = 0 (PREREGISTRATION_M2B.md §3): the relabel gate is distance < tau0 + alpha*lambda,
so --tau0 defaults to 0.0 and alpha = 0 is an exact no-op. Both stay explicit for the alpha grid.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# The pure helpers (centroids, lang_dur, RTTM writing, cache keys) below import nothing heavy, so
# tests/test_run_m2b.py imports this module and exercises them without loading any model. torch / pyannote /
# soundfile are imported lazily inside the functions that actually need a model or audio.

SR = 16000
MIN_CLUSTER_SIZE = 12          # pyannote AHC default; identical to p1_ahc.MIN_CLUSTER_SIZE
MIN_SAMPLES = 4000             # 0.25 s; segments shorter than this get no M2b adapter embedding (p1_ahc.MIN_SAMPLES)
CACHE_VERSION = 2              # bump when the stage-1 .npz layout changes
HOP = 0.1                      # LID grid hop (m2_lambda.HOP / lid.py default)
LAG = 10                       # windows compared are k-LAG and k (m2b_hooks.LAG)
LID_WIN = 1.0                  # LID analysis window (lid.py default)


# --------------------------------------------------------------------------- #
# cache keys  (pure; tested without a model)
# --------------------------------------------------------------------------- #
def recording_id(wav_path) -> str:
    """Recording id = the wav file stem (matches `p1_ahc.main`, which uses `p.stem` as the uri)."""
    return Path(wav_path).stem


def model_tag(model: str) -> str:
    """Filesystem-safe tag for an embedding model id. `m3:<ckpt>` -> `m3-<ckpt name>`; e1/e2/e3 unchanged.

    The `m3:` tag uses the SAME per-seed name `system_tag("M3")` uses (`m3_ckpt_name`): the run-folder name when
    the checkpoint file is the shared `best.pt`, else the file stem. m3_train / run_m3_queue write every
    checkpoint to `<run>/best.pt`, so the file stem `best` is identical across the five all-data seeds; keying
    the stage-1 cache on it would collapse all five into one `m3-best` folder and make them overwrite each
    other. Deriving the tag from `m3_ckpt_name` keeps each seed's stage-1 cache (and thus its hypotheses and
    model tag) distinct, consistent with the output-directory tag."""
    if model.startswith("m3:"):
        return "m3-" + m3_ckpt_name(model)
    if model in ("e1", "e2", "e3"):
        return model
    raise ValueError(f"unknown model id {model!r} (expected e1 | e2 | e3 | m3:<checkpoint>)")


def stage1_cache_path(cache_dir, model: str, rid: str) -> Path:
    """Stage-1 .npz path, keyed by (model, recording id): <cache_dir>/stage1/<model tag>/<rid>.npz."""
    return Path(cache_dir) / "stage1" / model_tag(model) / f"{rid}.npz"


def lid_cache_path(cache_dir, rid: str) -> Path:
    """Per-recording LID grid cache (model-independent): <cache_dir>/lid/<rid>.npz."""
    return Path(cache_dir) / "lid" / f"{rid}.npz"


def threshold_tag(threshold: float) -> str:
    """Directory tag for a clustering threshold, e.g. 0.7 -> 'th0.70', 0.3 -> 'th0.30' (2 dp, grid step 0.05)."""
    return f"th{float(threshold):.2f}"


def rttm_paths(out_dir, split: str, system: str, threshold: float, rid: str) -> tuple[Path, Path]:
    """(overlapping, exclusive) RTTM paths: <out>/<split>/<system>/<th tag>/<rid>.rttm and .exclusive.rttm."""
    base = Path(out_dir) / split / system / threshold_tag(threshold)
    return base / f"{rid}.rttm", base / f"{rid}.exclusive.rttm"


# --------------------------------------------------------------------------- #
# RTTM writing  (pure; tested without a model)
# --------------------------------------------------------------------------- #
def segments_from_annotation(annotation) -> list[tuple[float, float, str]]:
    """Flatten a pyannote Annotation to a list of (start, end, label) in track order."""
    out = []
    for segment, _, label in annotation.itertracks(yield_label=True):
        out.append((float(segment.start), float(segment.end), str(label)))
    return out


def write_rttm(path, rid: str, segments) -> None:
    """Write SPEAKER lines in NIST RTTM format for `segments` = iterable of (start, end, label).

    Columns: SPEAKER <rid> 1 <onset> <dur> <NA> <NA> <label> <NA> <NA>. Onset and duration at 3 dp, as
    pyannote's `Annotation.write_rttm`. Durations are clamped at 0 (a zero-length segment is dropped)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for start, end, label in segments:
        start = float(start)
        dur = float(end) - start
        if dur <= 0.0:
            continue
        lines.append(
            f"SPEAKER {rid} 1 {start:.3f} {dur:.3f} <NA> <NA> {label} <NA> <NA>"
        )
    path.write_text("".join(f"{ln}\n" for ln in lines))


# --------------------------------------------------------------------------- #
# M2b adapter pure helpers  (tested without a model)
# --------------------------------------------------------------------------- #
def unit(vec) -> np.ndarray:
    """Return `vec` scaled to unit L2 norm; raises on a zero (or non-finite) vector."""
    v = np.asarray(vec, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 0.0:
        raise ValueError("cannot unit-normalise a zero or non-finite vector")
    return v / n


def duration_weighted_centroid(embs, durations) -> np.ndarray:
    """Unit duration-weighted mean of per-segment embeddings (b2_merge's cluster centroid).

    `embs` is (n, dim); `durations` is (n,). Each embedding is used AS GIVEN (the system's own embedding of the
    segment); the weighted sum is renormalised to a unit vector. Segments with a non-finite embedding or a
    non-positive duration are dropped first. Raises if nothing usable remains or the weighted sum is zero."""
    embs = np.asarray(embs, dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)
    if embs.ndim != 2:
        raise ValueError(f"embs must be 2-D (n, dim), got shape {embs.shape}")
    if durations.shape != (embs.shape[0],):
        raise ValueError(f"durations shape {durations.shape} != ({embs.shape[0]},)")
    keep = np.isfinite(embs).all(axis=1) & np.isfinite(durations) & (durations > 0.0)
    if not keep.any():
        raise ValueError("no finite embedding with positive duration for the centroid")
    w = durations[keep]
    mean = (w[:, None] * embs[keep]).sum(axis=0)
    return unit(mean)


def _centre_offset(win: float, hop: float) -> int:
    """k - g for the LID window CENTRED on a grid step's left edge t = g*hop.

    Window k covers [hop*k, hop*k + win], so its centre is hop*k + win/2; setting that equal to the step edge
    g*hop gives k = g - win/(2*hop). The brief fixes win/(2*hop) = 5 for win 1.0, hop 0.1 and asks for integer
    arithmetic with no float rounding of an exact x.5 (which round-half-to-even would send the wrong way). We
    therefore round win/(2*hop) on scaled integers: round(win*1e6) and round(hop*1e6) are exact for the 0.1 s
    grid, and (a + b//2)//b is integer round-half-UP of a/b with b = 2*hop_i."""
    win_i = int(round(win * 1_000_000))
    hop_i = int(round(hop * 1_000_000))
    if hop_i <= 0:
        raise ValueError(f"hop must be positive, got {hop!r}")
    b = 2 * hop_i
    return (win_i + b // 2) // b          # round-half-up of win/(2*hop), exact integer arithmetic


def lang_dur_from_segments(segments, post, pair, hop: float = HOP, win: float = LID_WIN) -> dict:
    """Seconds of a speaker's speech per LID-predicted language (b2_merge / integration contract §4).

    The recording's 0.1 s grid is walked STEP BY STEP, not segment-midpoint by segment-midpoint. For each global
    grid step g -- the interval [g*hop, (g+1)*hop) -- its overlap with each (start, end) output segment is
    credited to the language that is the argmax, over the two reference-language indices `pair = (i, j)`, of the
    LID window CENTRED on the step's left edge t = g*hop, i.e. window k = g - round(win/(2*hop)) (= g - 5 for
    win 1.0, hop 0.1), clamped to [0, n-1]. The language label is the pair index (an int), so no language name is
    hardcoded. A step whose centred window has no mass over `pair` credits its overlap to `skipped` SECONDS
    instead (reported to the caller). Ties over the pair go to the SMALLER pair index (review m4), matching
    `majority_language`'s smaller-label rule. This fixes the two bugs of
    the midpoint rule: whole-segment crediting to one window, and k = round(mid/hop) (the window STARTING at the
    midpoint, off by half a window)."""
    i, j = int(pair[0]), int(pair[1])
    if i == j:
        raise ValueError(f"pair must be two distinct indices, got {pair!r}")
    post = np.asarray(post, dtype=np.float64)
    if post.ndim != 2:
        raise ValueError(f"post must be 2-D (n_windows, n_langs), got shape {post.shape}")
    n = post.shape[0]
    off = _centre_offset(win, hop)
    totals: dict = {}
    skipped = 0.0
    for start, end in segments:
        s, e = float(start), float(end)
        if e <= s:
            continue
        g0 = int(np.floor(s / hop + 1e-9))        # first grid step the segment touches
        g1 = int(np.floor((e - 1e-9) / hop))      # last grid step the segment touches
        for g in range(g0, g1 + 1):
            lo = max(s, g * hop)
            hi = min(e, (g + 1) * hop)
            overlap = hi - lo
            if overlap <= 0.0:
                continue
            if n == 0:
                skipped += overlap
                continue
            k = g - off
            if k < 0:
                k = 0
            elif k >= n:
                k = n - 1
            pi, pj = post[k, i], post[k, j]
            if pi + pj <= 0.0:
                skipped += overlap
                continue
            # argmax over the pair; a tie credits the SMALLER LID index (review m4), matching
            # majority_language's smaller-label rule, so a per-step tie and the majority agree.
            # NB: this is the smaller 107-list INDEX, which (because majority_language sees the same integer
            # keys) agrees with it, but is NOT necessarily the alphabetically smaller language code.
            if pi > pj:
                lang = i
            elif pj > pi:
                lang = j
            else:
                lang = min(i, j)
            totals[lang] = totals.get(lang, 0.0) + overlap
    return {"lang_dur": totals, "skipped": skipped}


def make_lambda_closure(post, pair, hop: float = HOP, lag: int = LAG):
    """Return `lam(t)` = m2b_hooks.lambda_at bound to this recording's LID grid and reference-language pair.

    `post` is a (n_windows, n_langs) array of per-window posteriors; it is handed to lambda_at as a list of rows
    so the stdlib-only hook needs no numpy. The returned closure has exactly the signature m2_relabel expects."""
    from csdiar.pipelines.m2b_hooks import lambda_at

    rows = [list(map(float, r)) for r in np.asarray(post, dtype=np.float64)]
    p = (int(pair[0]), int(pair[1]))

    def lam(t):
        return lambda_at(t, rows, p, hop=hop, lag=lag)

    return lam


# --------------------------------------------------------------------------- #
# stage 1: expensive, cached
# --------------------------------------------------------------------------- #
def _read_wav_mono(wav_path) -> np.ndarray:
    import soundfile as sf

    wav, sr = sf.read(str(wav_path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != SR:
        raise ValueError(f"{wav_path}: expected {SR} Hz, got {sr}")
    return np.ascontiguousarray(wav, dtype=np.float32)


def run_stage1(model: str, wav_path, cache_dir, rid: Optional[str] = None,
               device: str = "cpu", overwrite: bool = False) -> dict:
    """Run Community-1 up to the clustering input for one recording / embedding model and cache it.

    Loads `p1_ahc.load_pipeline` at a throwaway threshold (clustering is NOT used here), computes segmentation,
    the powerset-binarized segmentation (identical object for a powerset model, as in `apply`), the frame-level
    speaker count, and the overlap-aware (num_chunks, local_num_speakers, dim) embeddings, then saves everything
    `reconstruct` and clustering need to the stage-1 .npz. Returns a small metadata dict (shapes, timings)."""
    import torch
    from pyannote.audio import Audio

    from csdiar.pipelines.p1_ahc import load_pipeline

    rid = rid or recording_id(wav_path)
    out_path = stage1_cache_path(cache_dir, model, rid)
    if out_path.exists() and not overwrite:
        return {"rid": rid, "model": model, "cache": str(out_path), "cached": True}

    wav = _read_wav_mono(wav_path)
    duration_s = len(wav) / SR

    t0 = time.time()
    pipe = load_pipeline(model, threshold=0.7, device=device)
    load_s = time.time() - t0

    file = {"waveform": torch.from_numpy(wav)[None], "sample_rate": SR, "uri": rid}

    t1 = time.time()
    segmentations = pipe.get_segmentations(file)
    if pipe._segmentation.model.specifications.powerset:
        binarized = segmentations
    else:                                      # Community-1 is powerset; kept for completeness
        from pyannote.audio.utils.signal import binarize
        binarized = binarize(segmentations, onset=pipe.segmentation.threshold, initial_state=False)
    rf = pipe._segmentation.model.receptive_field
    count = pipe.speaker_count(binarized, rf, warm_up=(0.0, 0.0))
    seg_s = time.time() - t1

    t2 = time.time()
    embeddings = pipe.get_embeddings(file, binarized, exclude_overlap=pipe.embedding_exclude_overlap)
    emb_s = time.time() - t2

    sw = segmentations.sliding_window         # chunk-level sliding window
    csw = count.sliding_window                # frame-level sliding window of the count
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        version=np.int64(CACHE_VERSION),
        rid=rid,
        model=model,
        model_tag=model_tag(model),
        dimension=np.int64(pipe._embedding.dimension),
        duration_s=np.float64(duration_s),
        segmentations=segmentations.data.astype(np.float32),
        binarized=binarized.data.astype(np.float32),
        count=count.data.astype(np.int64),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        # chunk sliding window (segmentation / binarized)
        seg_start=np.float64(sw.start), seg_duration=np.float64(sw.duration), seg_step=np.float64(sw.step),
        # frame sliding window (count)
        cnt_start=np.float64(csw.start), cnt_duration=np.float64(csw.duration), cnt_step=np.float64(csw.step),
        # receptive field (frame resolution) for reconstruct's to_diarization
        rf_start=np.float64(rf.start), rf_duration=np.float64(rf.duration), rf_step=np.float64(rf.step),
        min_duration_off=np.float64(pipe.segmentation.min_duration_off),
    )
    return {"rid": rid, "model": model, "cache": str(out_path), "cached": False,
            "duration_s": duration_s, "num_chunks": int(segmentations.data.shape[0]),
            "dim": int(pipe._embedding.dimension),
            "load_s": load_s, "seg_count_s": seg_s, "embed_s": emb_s,
            "stage1_s_per_min": (load_s + seg_s + emb_s) / (duration_s / 60.0)}


# --------------------------------------------------------------------------- #
# stage 2: cheap, from the cache
# --------------------------------------------------------------------------- #
def _sliding(start, duration, step):
    from pyannote.core import SlidingWindow

    return SlidingWindow(start=float(start), duration=float(duration), step=float(step))


def load_stage1(cache_dir, model: str, rid: str) -> dict:
    """Load a stage-1 .npz into a plain dict of numpy arrays / scalars. Validates the cache version."""
    path = stage1_cache_path(cache_dir, model, rid)
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    version = int(data.get("version", -1))
    if version != CACHE_VERSION:
        raise ValueError(f"{path}: cache version {version} != expected {CACHE_VERSION}; re-run the cache step")
    return data


def load_stage2_pipeline(model: str, threshold: float, device: str = "cpu"):
    """Community-1 instantiated for stage-2 clustering + reconstruction ONLY (no embedding model loaded).

    Clustering is set up byte-for-byte as `p1_ahc.load_pipeline`: AgglomerativeClustering(metric='cosine'),
    method 'centroid', the given threshold, min_cluster_size 12. The embedding model is deliberately NOT swapped
    in -- stage 2 never runs it -- which also keeps stage 2 loadable in an env without s3prl."""
    import torch
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.clustering import AgglomerativeClustering

    from csdiar.pipelines.p1_ahc import COMMUNITY1

    pipe = Pipeline.from_pretrained(str(COMMUNITY1))
    pipe.to(torch.device(device))
    clustering = AgglomerativeClustering(metric="cosine")
    clustering.instantiate({"method": "centroid", "threshold": float(threshold),
                            "min_cluster_size": MIN_CLUSTER_SIZE})
    pipe.clustering = clustering
    pipe._expects_num_speakers = False
    return pipe


# --------------------------------------------------------------------------- #
# slot-aware language assignment for B1  (pure; tested without a model)
# --------------------------------------------------------------------------- #
# A "slot" is a (chunk, local speaker) pair -- one embedding row. B1 centres each embedding by the mean of its
# window's LANGUAGE (e' = unit(e - mu[lang])), but the (chunk, speaker) embedding has no row-wise LID label: the
# language of a slot is decided by its ACTIVE FRAMES in the cached binarized segmentation. These helpers compute
# that language per slot from the cache, so B1 can be a slot-aware transform of the (num_chunks, num_speakers,
# dim) array rather than a row-wise map (which a plain transform(flat_rows) cannot be, since it loses the slot's
# chunk/time position). The row-wise path (_apply_transform) is kept for M1, whose projector is position-free.


def slot_frame_times(cache: dict) -> np.ndarray:
    """(num_chunks, num_speakers, num_frames) float32: the absolute time of each active frame of every slot,
    NaN for inactive frames.

    A frame is active for slot (chunk c, speaker p) when the cached binarized segmentation is > 0 there. Frame f
    of chunk c is the segmentation model's receptive-field window f shifted to the chunk's start: its absolute
    MIDDLE time is chunk_start(c) + rf.start + f*rf.step + rf.duration/2, with chunk_start(c) = seg.start +
    c*seg.step (pyannote's SlidingWindow convention, the same mapping `reconstruct` uses). The binarized data is
    (num_chunks, num_frames, num_speakers)."""
    binarized = np.asarray(cache["binarized"])
    num_chunks, num_frames, num_speakers = binarized.shape
    seg_start = float(cache["seg_start"]); seg_step = float(cache["seg_step"])
    rf_start = float(cache["rf_start"]); rf_step = float(cache["rf_step"]); rf_dur = float(cache["rf_duration"])

    chunk_start = seg_start + seg_step * np.arange(num_chunks, dtype=np.float64)        # (num_chunks,)
    frame_mid = rf_start + rf_dur / 2.0 + rf_step * np.arange(num_frames, dtype=np.float64)   # (num_frames,)
    abs_mid = chunk_start[:, None] + frame_mid[None, :]                                  # (num_chunks, num_frames)

    active = binarized > 0.0                                                             # (C, F, S)
    # broadcast the per-(chunk,frame) time onto (chunk, speaker, frame), NaN where the slot is inactive there
    times = np.where(np.transpose(active, (0, 2, 1)),
                     abs_mid[:, None, :], np.float32("nan")).astype(np.float32)
    return times


def slot_languages(cache: dict, post, pair, hop: float = HOP, win: float = LID_WIN) -> np.ndarray:
    """(num_chunks, num_speakers) int array: the LID-predicted language (a `pair` index) of every slot. Two
    distinct "no language" sentinels, because B1 must treat them differently:
      * -1 = INACTIVE slot: no active frame (binarized segmentation 0 everywhere for this (chunk, speaker)).
             Clustering drops such a slot (pyannote's `inactive = sum == 0`), so B1 leaves it untouched.
      * -2 = ACTIVE but UNLABELLED: the slot has active frames, yet their centred LID windows carry no mass over
             `pair` anywhere. B1 cannot centre it, so this must fail loudly rather than be silently skipped.

    For each active frame of a slot, the LID posterior is read from the window CENTRED on that frame -- window
    k = round(frame_time / hop) - round(win/(2*hop)), clamped to [0, n-1], the same centred-window rule as
    `lang_dur_from_segments` (brief change 1). The posterior over `pair = (i, j)` is SUMMED over the slot's active
    frames and the argmax of that sum is the slot's language; ties go to the SMALLER pair index (review m4).
    Robust: an empty
    LID grid leaves every active slot -2."""
    i, j = int(pair[0]), int(pair[1])
    if i == j:
        raise ValueError(f"pair must be two distinct indices, got {pair!r}")
    post = np.asarray(post, dtype=np.float64)
    if post.ndim != 2:
        raise ValueError(f"post must be 2-D (n_windows, n_langs), got shape {post.shape}")
    n = post.shape[0]
    off = _centre_offset(win, hop)

    times = slot_frame_times(cache)                      # (C, S, F) abs frame times, NaN inactive
    num_chunks, num_speakers, _ = times.shape
    langs = np.full((num_chunks, num_speakers), -1, dtype=np.int64)

    for c in range(num_chunks):
        for p in range(num_speakers):
            ft = times[c, p]
            ft = ft[np.isfinite(ft)]
            if ft.size == 0:
                continue                                 # INACTIVE slot -> -1
            if n == 0:
                langs[c, p] = -2                         # active but no LID grid to label it
                continue
            # centred window index per active frame; round-half-up on scaled ints to avoid x.5 bias
            k = np.floor(ft / hop + 0.5).astype(np.int64) - off
            np.clip(k, 0, n - 1, out=k)
            si = float(post[k, i].sum())
            sj = float(post[k, j].sum())
            if si + sj <= 0.0:
                langs[c, p] = -2                         # ACTIVE but no pair mass anywhere -> fail loudly in B1
                continue
            # argmax of the summed pair mass; a tie credits the SMALLER pair index (review m4)
            # (the smaller 107-list INDEX, matching majority_language's integer keys, not the alphabetically
            # smaller language code)
            langs[c, p] = i if si > sj else (j if sj > si else min(i, j))
    return langs


def b1_transform(means: dict, slot_langs: np.ndarray) -> Callable:
    """Return a SLOT-AWARE transform for `cluster_from_cache`: e' = unit(e - mu[lang]) per slot.

    `means` maps a `pair` index (the language label `slot_languages` produces) to its mean embedding vector
    (B1's language mean, `confirm_h4.b1_means` style). `slot_langs` is the (num_chunks, num_speakers) array from
    `slot_languages`, whose two negative sentinels drive the handling:
      * lang >= 0  -> centre the slot: e' = unit(e - mu[lang]);
      * lang == -1 -> INACTIVE slot (no active frame); clustering drops it, so leave the row untouched. This is
                      keyed on slot ACTIVITY, not on embedding finiteness, because a native-embedder slot (e2) can
                      be finite yet inactive -- centring it is pointless and reading its LID label is undefined;
      * lang == -2 -> ACTIVE but UNLABELLED (no LID mass over the pair): FAIL LOUDLY, because silently leaving an
                      active, clustered slot in the raw space would mix a raw and a centred space in one AHC run.
    An active slot (lang >= 0) with a non-finite embedding is also a loud failure (it should have been dropped)."""
    mu = {int(l): np.asarray(v, dtype=np.float64) for l, v in means.items()}

    def transform(embeddings3d: np.ndarray) -> np.ndarray:
        emb = np.asarray(embeddings3d, dtype=np.float64)
        if emb.ndim != 3:
            raise ValueError(f"b1_transform expects (chunks, speakers, dim), got {emb.shape}")
        if slot_langs.shape != emb.shape[:2]:
            raise ValueError(f"slot_langs shape {slot_langs.shape} != embeddings slots {emb.shape[:2]}")
        out = emb.copy()
        C, S, _ = emb.shape
        for c in range(C):
            for p in range(S):
                lang = int(slot_langs[c, p])
                if lang == -1:
                    continue                             # inactive slot: clustering drops it, leave untouched
                if lang == -2:
                    raise ValueError(f"active slot (chunk {c}, speaker {p}) has no LID mass over the pair "
                                     "(language -2); cannot centre it for B1")
                if lang < 0 or lang not in mu:
                    raise ValueError(f"slot (chunk {c}, speaker {p}) has no language mean for label {lang}")
                row = emb[c, p]
                if not np.isfinite(row).all():
                    raise ValueError(f"active slot (chunk {c}, speaker {p}) has a non-finite embedding")
                out[c, p] = unit(row - mu[lang])
        return out

    return transform


def _apply_transform(embeddings: np.ndarray, transform: Optional[Callable]) -> np.ndarray:
    """Apply the ROW-WISE B1/M1 hook to finite embedding rows only, leaving NaN (chunk, speaker) slots untouched.

    `embeddings` is (num_chunks, num_speakers, dim). This mirrors `p1_ahc`'s E2 transform wrapper, which maps
    only the finite rows and leaves pyannote's `filter_embeddings` to drop the NaN ones. Used for M1 (a
    position-free projector); B1 uses the slot-aware path (`slot_transform` / `b1_transform`), not this."""
    if transform is None:
        return embeddings
    emb = embeddings.astype(np.float64, copy=True)
    flat = emb.reshape(-1, emb.shape[-1])
    fin = np.isfinite(flat).all(axis=1)
    if fin.any():
        flat[fin] = np.asarray(transform(flat[fin]), dtype=np.float64)
    return emb.reshape(embeddings.shape)


def cluster_from_cache(pipe, cache: dict, threshold: float, transform: Optional[Callable] = None,
                       slot_transform: Optional[Callable] = None):
    """Reproduce `p1_ahc.diarize` at `threshold` from a stage-1 cache, with an optional embedding transform.

    Returns (overlapping_annotation, exclusive_annotation, info) where info carries the final speaker labels and
    their row order in `centroids`. The clustering threshold on `pipe` must already equal `threshold`
    (load_stage2_pipeline does this); it is re-instantiated here too so a single pipe can serve many thresholds.

    Two transform paths, mutually exclusive:
      * `transform`     -- ROW-WISE, finite (n, dim) rows -> (n, dim). Position-free; used by M1.
      * `slot_transform` -- SLOT-AWARE, the whole (num_chunks, num_speakers, dim) array -> same shape, so a
                            transform (B1) can depend on each slot's chunk/time position. `b1_transform` builds
                            one. It owns the finite/NaN handling itself (B1 leaves NaN slots untouched).
    """
    from pyannote.audio.pipelines.clustering import AgglomerativeClustering
    from pyannote.core import SlidingWindowFeature

    if transform is not None and slot_transform is not None:
        raise ValueError("pass at most one of transform (row-wise) and slot_transform (slot-aware)")

    # keep the pipe's threshold in sync (lets one pipe serve a list of thresholds)
    clustering = AgglomerativeClustering(metric="cosine")
    clustering.instantiate({"method": "centroid", "threshold": float(threshold),
                            "min_cluster_size": MIN_CLUSTER_SIZE})
    pipe.clustering = clustering

    seg_sw = _sliding(cache["seg_start"], cache["seg_duration"], cache["seg_step"])
    rf = _sliding(cache["rf_start"], cache["rf_duration"], cache["rf_step"])
    cnt_sw = _sliding(cache["cnt_start"], cache["cnt_duration"], cache["cnt_step"])

    segmentations = SlidingWindowFeature(cache["segmentations"].astype(np.float32), seg_sw)
    binarized = SlidingWindowFeature(cache["binarized"].astype(np.float32), seg_sw)
    count = SlidingWindowFeature(cache["count"].astype(np.int64).copy(), cnt_sw)

    if slot_transform is not None:
        embeddings = np.asarray(slot_transform(cache["embeddings"]), dtype=np.float64)
        if embeddings.shape != cache["embeddings"].shape:
            raise ValueError("slot_transform must preserve the (chunks, speakers, dim) shape")
    else:
        embeddings = _apply_transform(cache["embeddings"], transform)

    hard_clusters, _, centroids = pipe.clustering(
        embeddings=embeddings,
        segmentations=binarized,
        num_clusters=None, min_clusters=None, max_clusters=None,
    )
    # inactive (chunk, speaker) slots -> throw-away cluster -2  (apply() step)
    inactive = np.sum(binarized.data, axis=1) == 0
    hard_clusters[inactive] = -2

    min_off = float(cache["min_duration_off"])

    # overlapping diarization: count capped at max_speakers (unbounded here, so == count itself), cast to int8
    # as apply() does at line 676 (np.minimum(count.data, max_speakers).astype(np.int8)); review m2.
    count_ov = SlidingWindowFeature(count.data.astype(np.int8), cnt_sw)
    discrete = pipe.reconstruct(segmentations, hard_clusters, count_ov)
    diar = pipe.to_annotation(discrete, min_duration_on=0.0, min_duration_off=min_off)
    diar.uri = str(cache["rid"])

    # exclusive diarization: count capped at 1, cast to int8 as apply() line 702 does; review m2
    count_excl = SlidingWindowFeature(np.minimum(count.data, 1).astype(np.int8), cnt_sw)
    discrete_excl = pipe.reconstruct(segmentations, hard_clusters, count_excl)
    excl = pipe.to_annotation(discrete_excl, min_duration_on=0.0, min_duration_off=min_off)
    excl.uri = str(cache["rid"])

    # no reference annotation on our runs -> rename integer labels to SPEAKER_00.. in label order.
    # review m1: pyannote 4.0.7 classes() (lines 295-299) is an ENDLESS SPEAKER_NN generator (checked on the
    # box, review header), so zip() is bounded by diar.labels() and the rename cannot truncate -- no change.
    mapping = {label: expected for label, expected in zip(diar.labels(), pipe.classes())}
    diar = diar.rename_labels(mapping=mapping)
    excl = excl.rename_labels(mapping={k: v for k, v in mapping.items() if k in excl.labels()})

    info = {"labels": list(diar.labels()), "num_speakers": len(diar.labels())}
    return diar, excl, info


# --------------------------------------------------------------------------- #
# M2b adapter: per-output-segment embeddings (cached) + the two hooks
# --------------------------------------------------------------------------- #
class SegmentEmbedder:
    """The system's own embedding model run on each exclusive-output segment's audio, cached by
    (model, recording, start, end) so a segment repeated across thresholds is embedded once.

    A segment with fewer than MIN_SAMPLES samples gets NO embedding (returns None); such a segment is never
    relabelled by M2 and never contributes to a B2 centroid. Embeddings are returned UNIT-normalised (both hooks
    work on unit vectors). Loads the embedding model lazily on first use."""

    def __init__(self, model: str, rid: str, wav: np.ndarray, device: str = "cpu"):
        self.model = model
        self.rid = rid
        self.wav = np.asarray(wav, dtype=np.float32)
        self.device = device
        self._emb = None
        self._cache: dict = {}
        self.min_samples = MIN_SAMPLES

    def _ensure(self):
        if self._emb is not None:
            return
        if self.model == "e2":
            # e2 has no standalone MaskedEmbedding (its embedding is Community-1's own WeSpeaker masked pooling,
            # built into the pipeline). Borrow that native embedding object; it has the same __call__(waveforms,
            # masks) -> (B, dim) numpy signature, NaN below MIN_SAMPLES, as MaskedEmbedding.
            import torch
            from pyannote.audio import Pipeline
            from csdiar.pipelines.p1_ahc import COMMUNITY1
            pipe = Pipeline.from_pretrained(str(COMMUNITY1))
            pipe.to(torch.device(self.device))
            self._emb = pipe._embedding
        else:
            from csdiar.pipelines.p1_ahc import MaskedEmbedding
            self._emb = MaskedEmbedding(self.model, self.device)

    def embed(self, start: float, end: float):
        key = (self.model, self.rid, round(float(start), 6), round(float(end), 6))
        if key in self._cache:
            return self._cache[key]
        s = max(0, int(round(float(start) * SR)))
        e = min(len(self.wav), int(round(float(end) * SR)))
        if e - s < self.min_samples:
            self._cache[key] = None
            return None
        self._ensure()
        import torch
        seg = torch.from_numpy(self.wav[s:e])[None, None]      # (1, 1, T); full mask (exclusive segment)
        vec = self._emb(seg, masks=None)[0]
        if not np.isfinite(vec).all():
            self._cache[key] = None
            return None
        out = unit(vec)
        self._cache[key] = out
        return out


def run_m2_relabel(exclusive_segments, embedder: SegmentEmbedder, lam, alpha: float, tau0: float = 0.0):
    """Prepare the M2 post-processor inputs from an exclusive annotation and call `m2b_hooks.m2_relabel`.

    `tau0` defaults to 0.0 -- M2's DEPLOYED form (PREREGISTRATION_M2B.md §3, draft v4 fix): the gate is
    `distance < tau0 + alpha*lambda(t)` with tau0 = 0, so alpha = 0 never relabels and is an exact no-op. It
    stays an explicit parameter for the alpha/tau0 grid sweep, but 0.0 is the value the study runs at.

    `exclusive_segments` is a list of (start, end, label) in time order. Each segment is embedded by `embedder`;
    a segment with no embedding (shorter than MIN_SAMPLES) is left in place and cannot be relabelled -- it is
    passed through `m2_relabel` by splitting the run at it. Returns (new_segments, records, skipped) where
    `new_segments` is the full list in order with updated labels and `skipped` counts embeddings not computed."""
    from csdiar.pipelines.m2b_hooks import m2_relabel

    enriched = []
    skipped = 0
    for start, end, label in exclusive_segments:
        emb = embedder.embed(start, end)
        if emb is None:
            skipped += 1
        enriched.append({"start": float(start), "end": float(end), "label": str(label),
                         "emb": None if emb is None else [float(x) for x in emb]})

    # m2_relabel decides consecutive pairs; a segment with no embedding cannot participate, so we run
    # m2_relabel on maximal runs of consecutive embedded segments and stitch the no-embedding ones back in place.
    new_segments = [dict(s) for s in enriched]
    records = []
    run: list = []
    run_idx: list = []

    def flush():
        if len(run) >= 2:
            relabelled, recs = m2_relabel(run, lam, tau0, alpha)
            for local, gidx in enumerate(run_idx):
                new_segments[gidx]["label"] = relabelled[local]["label"]
            for r in recs:
                g = run_idx[r["index"]]
                records.append({**r, "index": g,
                                "start": enriched[g]["start"], "end": enriched[g]["end"]})

    for gidx, s in enumerate(enriched):
        if s["emb"] is None:
            flush()
            run, run_idx = [], []
        else:
            run.append({"start": s["start"], "end": s["end"], "label": s["label"], "emb": s["emb"]})
            run_idx.append(gidx)
    flush()

    out = [(s["start"], s["end"], s["label"]) for s in new_segments]
    return out, records, skipped


def run_b2_merge(overlapping_annotation, exclusive_annotation, embedder: SegmentEmbedder, post, pair,
                 threshold: float, delta: float):
    """Prepare the B2 affinity-offset inputs from the EXCLUSIVE annotation and call `m2b_hooks.b2_merge`.

    B2 merges whole output SPEAKERS, so its per-speaker centroid and language durations must describe each
    speaker's actual, non-overlapping speech -- the EXCLUSIVE diarization -- not the overlapping annotation
    (where the same instant is credited to several speakers and double-counts a region's embedding and language
    mass). The exclusive segments also share `embedder`'s cache with M2, so each distinct segment is embedded by
    the system's own model exactly once across both hooks.

    Per output speaker, from its exclusive segments: the duration-weighted unit centroid of the system's
    embeddings (segments with no embedding are dropped from the centroid), and `lang_dur` = seconds per
    LID-predicted language over `post` and the reference-language `pair` (brief change 1's per-step crediting).
    b2_merge's resolve map is then applied to BOTH the overlapping and the exclusive annotation, so the two stay
    label-consistent after a merge.

    Returns (resolve, skipped) where `resolve` is b2_merge's {old_label: final_label} map over every label that
    appears in EITHER annotation. A speaker with no exclusive segment, no usable embedding, or no language mass
    keeps its own label and is reported in `skipped`."""
    from csdiar.pipelines.m2b_hooks import b2_merge

    excl_by_speaker: dict = {}
    for start, end, label in segments_from_annotation(exclusive_annotation):
        excl_by_speaker.setdefault(label, []).append((float(start), float(end)))

    all_labels = set(overlapping_annotation.labels()) | set(exclusive_annotation.labels())

    speakers: dict = {}
    skipped = {"no_exclusive": [], "no_embedding": [], "no_lang_mass": []}
    for label in sorted(all_labels):
        segs = excl_by_speaker.get(label, [])
        if not segs:                                   # speaker present only in the overlapping annotation
            skipped["no_exclusive"].append(label)
            continue
        embs, durs = [], []
        for start, end in segs:
            vec = embedder.embed(start, end)
            if vec is not None:
                embs.append(vec)
                durs.append(end - start)
        ld = lang_dur_from_segments(segs, post, pair)["lang_dur"]
        if not embs:
            skipped["no_embedding"].append(label)
            continue
        if not ld:
            skipped["no_lang_mass"].append(label)
            continue
        centroid = duration_weighted_centroid(np.asarray(embs), np.asarray(durs))
        speakers[label] = {"centroid": [float(x) for x in centroid], "lang_dur": ld}

    if len(speakers) < 2:
        resolve = {label: label for label in all_labels}
        return resolve, skipped
    resolve = b2_merge(speakers, threshold, delta)
    for label in all_labels:                           # speakers dropped above keep their own label
        resolve.setdefault(label, label)
    return resolve, skipped


def relabel_annotation(annotation, resolve: dict):
    """Return a copy of `annotation` with labels mapped through `resolve` (b2_merge's old->final map)."""
    return annotation.rename_labels(mapping={k: resolve.get(k, k) for k in annotation.labels()})


def apply_m2_overlapping(overlapping_annotation, m2_records):
    """Carry M2's exclusive-segment relabels into the OVERLAPPING annotation (PREREGISTRATION_M2B.md §4).

    `m2_records` is the record list from `run_m2_relabel` -- one dict per relabelled EXCLUSIVE segment with
    `start`, `end`, `old` (source speaker) and `new` (target label). Prereg §4: "M2 relabels exclusive segments,
    and the overlapping annotation takes the same relabel over the same spans: inside a relabelled span, the
    source speaker's overlapping track gets the target label." So for each record, over its [start, end] span,
    the parts of the OLD speaker's overlapping tracks that fall inside the span are reassigned to NEW; the OLD
    speaker's speech OUTSIDE the span, and every OTHER speaker's tracks (the overlap partners), are left
    untouched -- M2 is a local relabel of one speaker's own segment, not a merge of the two speakers. After
    reassigning, same-label pieces are merged (`support`, gap 0) so a track that becomes contiguous with an
    existing NEW track under the same label is not left split.

    Works on the flat (start, end, label) view and rebuilds the annotation, so a track key collision cannot
    arise. Returns a new Annotation; the input is not mutated."""
    from pyannote.core import Annotation, Segment

    segs = segments_from_annotation(overlapping_annotation)      # (start, end, label), overlap-aware
    spans_by_new: dict = {}
    for rec in m2_records:
        if rec["old"] == rec["new"]:
            continue
        spans_by_new.setdefault((rec["old"], rec["new"]), []).append(
            (float(rec["start"]), float(rec["end"])))

    out_segs = []
    for s, e, label in segs:
        pieces = [(s, e, label)]
        # for every (old->new) relabel whose OLD matches this track's label, split out the in-span portion
        for (old, new), spans in spans_by_new.items():
            if old != label:
                continue
            for a, b in spans:
                nxt = []
                for ps, pe, pl in pieces:
                    if pl != label:                     # already reassigned piece -> leave it
                        nxt.append((ps, pe, pl))
                        continue
                    lo, hi = max(ps, a), min(pe, b)
                    if hi <= lo:                        # no overlap with this span
                        nxt.append((ps, pe, pl))
                        continue
                    if ps < lo:
                        nxt.append((ps, lo, label))     # before the span: keep OLD
                    nxt.append((lo, hi, new))           # inside the span: becomes NEW
                    if hi < pe:
                        nxt.append((hi, pe, label))     # after the span: keep OLD
                pieces = nxt
        out_segs.extend(pieces)

    ann = Annotation(uri=getattr(overlapping_annotation, "uri", None))
    for i, (s, e, label) in enumerate(out_segs):
        if e <= s:
            continue
        ann[Segment(float(s), float(e)), i] = str(label)
    return ann.support(collar=0.0)


# --------------------------------------------------------------------------- #
# system tags and threshold validation  (pure; tested without a model)
# --------------------------------------------------------------------------- #
VALID_SYSTEMS = ("B0", "B1", "B2", "M1", "M2", "M3")
GRID_LO, GRID_HI = 0.0, 2.0            # cosine-distance thresholds live in (0, 2]


def m3_ckpt_name(m3_ckpt: str) -> str:
    """The stable, per-seed name of an M3 checkpoint, from its path (optionally `m3:`-prefixed).

    m3_train / run_m3_queue name the RUN FOLDER after the model (`all_s<seed>_w0.3_none`) and write every
    checkpoint to `<run>/best.pt`, so the file stem `best` is shared by all five seeds. Three cases:
      * canonical `<run>/best.pt` -> the RUN-FOLDER name, so each seed stays distinct;
      * a checkpoint FILE that already carries its own name (an ad-hoc `seed0_all.pt`) -> its file stem;
      * the run FOLDER itself, with no file part (`.../all_s0_w0.3_none` or a trailing slash) -> that folder's
        name VERBATIM. `Path.stem` must NOT be used here: a dotted run-folder name like `all_s0_w0.3_none` has
        `.stem == "all_s0_w0"` (everything after the last dot is read as a suffix), so two w-values would
        collide on `all_s0_w0`. `p.name` keeps the component whole.
    A path with no recognised checkpoint-file suffix (`.pt` / `.pth` / `.ckpt`) is treated as a run folder and
    its last component is taken verbatim; a recognised checkpoint file uses the run folder when it is `best.pt`,
    else its own stem."""
    p = Path(m3_ckpt[3:] if m3_ckpt.startswith("m3:") else m3_ckpt).expanduser()
    if p.suffix.lower() not in (".pt", ".pth", ".ckpt"):
        # no file part -> the path IS the run folder; take its last component verbatim (dots survive)
        return p.name
    if p.stem == "best":
        parent = p.parent.name
        if parent:
            return parent
    return p.stem


def system_tag(system: str, *, delta: float = 0.0, alpha: float = 0.0, tau0: float = 0.0,
               m3_ckpt: Optional[str] = None) -> str:
    """Output-directory tag that keeps every system AND every parameter value apart (brief change 2).

    B0/B1/M1 take no operating-point parameter, so their tag is just the system name. B2 carries its merge
    offset delta; M2 carries alpha and tau0 (both needed because the alpha grid sweeps alpha at tau0=0 but the
    P0-gate form and sensitivity runs vary tau0 too); M3 carries the checkpoint name (its run-folder name when
    the file is the shared `best.pt`, so two seeds never collide). These are the leaf directory under
    <out>/<split>/, so distinct operating points never overwrite each other:
      B0, B1, M1, B2_d{delta:.3f}, M2_a{alpha:.2f}_t{tau0:.2f}, M3_<ckpt name>."""
    if system not in VALID_SYSTEMS:
        raise ValueError(f"unknown system {system!r} (expected one of {VALID_SYSTEMS})")
    if system in ("B0", "B1", "M1"):
        return system
    if system == "B2":
        return f"B2_d{float(delta):.3f}"
    if system == "M2":
        return f"M2_a{float(alpha):.2f}_t{float(tau0):.2f}"
    # M3
    if not m3_ckpt:
        raise ValueError("system_tag(M3) needs m3_ckpt")
    return f"M3_{m3_ckpt_name(m3_ckpt)}"


def validate_thresholds(thresholds) -> list:
    """Validate a clustering-threshold sweep and return it unchanged (brief change / review m5).

    Refuses a threshold that is non-finite, outside (0, 2], not equal to its own 2-dp rounding within 1e-9, or
    that duplicates another after rounding -- so a fat-fingered `--thresholds 0.700 0.7000001` (which would share
    the `th0.70` directory and silently overwrite one RTTM with another's) or a value > 1.00 (silent `th1.05`)
    is caught before anything is written. The 2-dp rule pins each threshold onto the frozen 0.05 grid's
    resolution so `threshold_tag` (2 dp) is injective over the accepted set."""
    out = []
    seen = {}
    for th in thresholds:
        th = float(th)
        if not np.isfinite(th):
            raise ValueError(f"threshold {th!r} is not finite")
        if not (GRID_LO < th <= GRID_HI):
            raise ValueError(f"threshold {th!r} outside the cosine-distance range ({GRID_LO}, {GRID_HI}]")
        rounded = round(th, 2)
        if abs(th - rounded) > 1e-9:
            raise ValueError(f"threshold {th!r} is not on the 2-dp grid (nearest {rounded:.2f}); "
                             "the output directory tag is 2 dp, so a finer value would collide silently")
        tag = f"{rounded:.2f}"
        if tag in seen:
            raise ValueError(f"thresholds {seen[tag]!r} and {th!r} both round to {tag} -- duplicate sweep point")
        seen[tag] = th
        out.append(th)
    return out


# --------------------------------------------------------------------------- #
# the system library function: cache -> (overlapping, exclusive) per threshold
# --------------------------------------------------------------------------- #
def run_system_from_cache(pipe, cache: dict, threshold: float, system: str, *,
                          post=None, pair=None, means: Optional[dict] = None,
                          v_hat=None, delta: float = 0.0, alpha: float = 0.0, tau0: float = 0.0,
                          embedder: Optional["SegmentEmbedder"] = None):
    """Produce the (overlapping, exclusive) annotations for ONE system on P1-AHC at ONE threshold, from the
    stage-1 cache (PREREGISTRATION_M2B.md §4, "What each system writes").

    This is the single place that turns a system name + operating point into the two annotations the study
    ships, so the CLI, the acceptance script and the tests all go through it rather than re-deriving the per-
    system write rules. Every OFF setting reproduces B0 exactly in BOTH annotations:
      * B0: re-cluster, no transform.
      * B1: slot-aware e' = unit(e - mu[lang]) through `slot_languages` + `b1_transform`; `means` maps each
            pair language index to its mean embedding. Zero means -> unit(e), which cosine-AHC leaves == B0.
      * M1: row-wise e' = unit(e - (e.v)v) via `m1_subspace.project_out` (the position-free transform path);
            inactive/NaN slots pass through unchanged. v_hat == 0 would be refused on load, so there is no
            "off" M1 beyond "not selected".
      * M3: B0 clustering on the m3:<ckpt> model's OWN cache (the caller loads that cache); no transform here.
      * B2: re-cluster as B0, then `run_b2_merge` and apply its resolve map to BOTH annotations. delta=0 -> B0.
      * M2: re-cluster as B0, then `run_m2_relabel` on the EXCLUSIVE segments; the overlapping annotation takes
            the same relabel over the same spans via `apply_m2_overlapping`. alpha=0 (tau0=0) -> B0.

    B2/M2 need `embedder` (the system's own SegmentEmbedder over the recording wav) and `post`/`pair` (the LID
    grid and reference-language pair); B1 needs `means` + `post`/`pair`; M1 needs `v_hat`. Returns
    (overlapping, exclusive, info) with info carrying per-system diagnostics (skips, relabel count)."""
    if system not in VALID_SYSTEMS:
        raise ValueError(f"unknown system {system!r} (expected one of {VALID_SYSTEMS})")

    info: dict = {"system": system, "threshold": float(threshold)}

    # ----- systems that change the CLUSTERING INPUT: B0, B1, M1, M3 -----
    if system in ("B0", "M3"):
        diar, excl, cinfo = cluster_from_cache(pipe, cache, threshold)
        info.update(cinfo)
        return diar, excl, info

    if system == "B1":
        if post is None or pair is None or means is None:
            raise ValueError("B1 needs post, pair and means")
        slot_langs = slot_languages(cache, post, pair)
        mu = {int(pair[0]): np.asarray(means[int(pair[0])], dtype=np.float64),
              int(pair[1]): np.asarray(means[int(pair[1])], dtype=np.float64)}
        tf = b1_transform(mu, slot_langs)
        diar, excl, cinfo = cluster_from_cache(pipe, cache, threshold, slot_transform=tf)
        info.update(cinfo)
        info["n_active_slots"] = int((slot_langs >= 0).sum())
        info["n_active_unlabelled"] = int((slot_langs == -2).sum())
        return diar, excl, info

    if system == "M1":
        if v_hat is None:
            raise ValueError("M1 needs v_hat")
        from csdiar.mitigations.m1_subspace import project_out
        vh = np.asarray(v_hat, dtype=np.float32)

        def m1_tf(rows):                       # row-wise, position-free: (n, dim) -> (n, dim)
            return project_out(np.asarray(rows, dtype=np.float32), vh)

        diar, excl, cinfo = cluster_from_cache(pipe, cache, threshold, transform=m1_tf)
        info.update(cinfo)
        return diar, excl, info

    # ----- systems that act on the CLUSTERING OUTPUT: B2, M2 -----
    # both start from the B0 clustering at this threshold
    diar, excl, cinfo = cluster_from_cache(pipe, cache, threshold)
    info.update(cinfo)

    if system == "B2":
        if embedder is None or post is None or pair is None:
            raise ValueError("B2 needs embedder, post and pair")
        resolve, skipped = run_b2_merge(diar, excl, embedder, post, pair, threshold, delta)
        info["b2_resolve"] = resolve
        info["b2_skipped"] = skipped
        if all(resolve.get(k, k) == k for k in resolve):
            return diar, excl, info                    # no merge (delta=0) -> leave B0 annotations untouched
        diar = relabel_annotation(diar, resolve)       # B2 applies its merge map to BOTH annotations
        excl = relabel_annotation(excl, resolve)
        return diar, excl, info

    # system == "M2"
    if embedder is None or post is None or pair is None:
        raise ValueError("M2 needs embedder, post and pair")
    lam = make_lambda_closure(post, pair)
    excl_segs = segments_from_annotation(excl)
    new_excl, records, m2_skipped = run_m2_relabel(excl_segs, embedder, lam, alpha=alpha, tau0=tau0)
    info["m2_relabels"] = len(records)
    info["m2_skipped"] = m2_skipped
    if not records:
        # no relabel -> leave BOTH B0 annotations exactly as clustered, so alpha=0 (the deployed no-op) writes
        # RTTMs byte-identical to B0 and never passes through support()'s track re-ordering.
        return diar, excl, info
    excl = annotation_from_segments(excl, new_excl)    # the relabelled exclusive annotation
    diar = apply_m2_overlapping(diar, records)         # overlapping takes the same relabel over the same spans
    return diar, excl, info


def annotation_from_segments(template_annotation, segments):
    """Rebuild an Annotation with the same uri as `template_annotation` from (start, end, label) tuples.

    Used to turn `run_m2_relabel`'s updated exclusive-segment list back into a pyannote Annotation. The exclusive
    diarization is non-overlapping, so each segment is its own track; `support()` then merges adjacent
    same-label pieces a relabel made contiguous."""
    from pyannote.core import Annotation, Segment

    ann = Annotation(uri=getattr(template_annotation, "uri", None))
    for i, (start, end, label) in enumerate(segments):
        s, e = float(start), float(end)
        if e <= s:
            continue
        ann[Segment(s, e), i] = str(label)
    return ann.support(collar=0.0)


# --------------------------------------------------------------------------- #
# LID grid  (expensive; cached per recording)
# --------------------------------------------------------------------------- #
def run_lid(wav_path, cache_dir, rid: Optional[str] = None, win: float = LID_WIN, hop: float = HOP,
            batch: int = 256, device: str = "cpu", overwrite: bool = False) -> dict:
    """VoxLingua107 posteriors on `win`-second windows every `hop` seconds for a whole recording.

    Follows `csdiar/models/lid.py` exactly: window k covers [hop*k, hop*k + win]; the FULL 107-language posterior
    is stored (float16) so the reference-language pair can be chosen per recording at stage 2 without re-running
    the model. Cached at `lid_cache_path`; the VoxLingua107 label list is stored alongside so pair indices can be
    resolved. Returns metadata (shapes, timing)."""
    import torch
    from speechbrain.inference.classifiers import EncoderClassifier

    from csdiar.models.lid import windows

    rid = rid or recording_id(wav_path)
    out_path = lid_cache_path(cache_dir, rid)
    if out_path.exists() and not overwrite:
        return {"rid": rid, "lid_cache": str(out_path), "cached": True}

    wav = _read_wav_mono(wav_path)
    duration_s = len(wav) / SR
    clf = EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-voxlingua107-ecapa",
        savedir=str(Path.home() / "data/models/lang-id-voxlingua107-ecapa"),
        run_opts={"device": device})
    labels = clf.hparams.label_encoder.ind2lab
    label_list = [labels[i].split(":")[0].strip() for i in range(len(labels))]

    frames, centres = windows(wav, win, hop)
    t0 = time.time()
    probs = []
    with torch.inference_mode():
        for b in range(0, len(frames), batch):
            x = torch.from_numpy(frames[b:b + batch]).to(device)
            logp = clf.classify_batch(x)[0]
            probs.append(logp.exp().float().cpu().numpy())
    post = np.concatenate(probs).astype(np.float16)     # (n_windows, 107) full posterior
    lid_s = time.time() - t0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, version=np.int64(CACHE_VERSION), rid=rid,
                        post=post, centres=centres.astype(np.float32),
                        labels=np.asarray(label_list), hop=np.float64(hop), win=np.float64(win))
    return {"rid": rid, "lid_cache": str(out_path), "cached": False, "n_windows": int(post.shape[0]),
            "n_langs": int(post.shape[1]), "duration_s": duration_s, "lid_s": lid_s}


def load_lid(cache_dir, rid: str) -> dict:
    """Load a LID cache: {'post': (n,107) float64, 'centres', 'labels': list[str], 'hop', 'win'}."""
    path = lid_cache_path(cache_dir, rid)
    with np.load(path, allow_pickle=False) as z:
        return {"post": z["post"].astype(np.float64), "centres": z["centres"],
                "labels": [str(x) for x in z["labels"]], "hop": float(z["hop"]), "win": float(z["win"])}


def pair_indices(labels: list[str], lang_a: str, lang_b: str) -> tuple[int, int]:
    """Resolve two language codes to their indices in a VoxLingua107 label list (for the recording's pair)."""
    def find(code):
        for i, lab in enumerate(labels):
            if lab == code:
                return i
        raise ValueError(f"language {code!r} not in the VoxLingua107 label list")
    return find(lang_a), find(lang_b)


def load_means_npz(path, lang_a: str, lang_b: str, dim: int) -> dict:
    """Load B1 language means from an .npz with one (dim,) array per VoxLingua107 code; return {pair_idx: mean}.

    The study's B1 means are per-language mean embeddings (`confirm_h4.b1_means` style). The file is keyed by
    VoxLingua107 code (e.g. 'en', 'hi'); BOTH pair languages must be present, and each array's dimension must
    match the cache embedding dim. Statistics come only from this file (fit on dev), never from the recording
    being scored. Returns {i: mean_a, j: mean_b} keyed by the pair INDICES the slot helpers produce -- the
    caller passes `pair = (i, j)` and `lang_a`/`lang_b` are the codes for i and j respectively."""
    with np.load(Path(path).expanduser(), allow_pickle=False) as z:
        present = set(z.files)
        out = {}
        for code in (lang_a, lang_b):
            if code not in present:
                raise SystemExit(f"--means {path}: missing a mean for language {code!r} "
                                 f"(has {sorted(present)}); both pair languages are required for B1")
            arr = np.asarray(z[code], dtype=np.float64).reshape(-1)
            if arr.shape[0] != dim:
                raise SystemExit(f"--means {path}: language {code!r} mean has dim {arr.shape[0]} != "
                                 f"cache embedding dim {dim}")
            out[code] = arr
    return out


def load_v_hat_npy(path, dim: int) -> np.ndarray:
    """Load M1's subspace direction from a .npy (dim,), check the dim, refuse zero/non-finite, and unit-normalise.

    v is fit on dev (`m1_subspace.estimate_v_lang`) and passed in -- never estimated from the recording scored."""
    v = np.asarray(np.load(Path(path).expanduser(), allow_pickle=False), dtype=np.float64).reshape(-1)
    if v.shape[0] != dim:
        raise SystemExit(f"--v-hat {path}: dim {v.shape[0]} != cache embedding dim {dim}")
    if not np.isfinite(v).all():
        raise SystemExit(f"--v-hat {path}: contains non-finite values")
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        raise SystemExit(f"--v-hat {path}: zero vector; cannot define a subspace direction")
    return (v / n).astype(np.float32)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cmd_cache(args) -> int:
    meta = run_stage1(args.model, args.wav, args.cache_dir, rid=args.rid,
                      device=args.device, overwrite=args.overwrite)
    print(meta, flush=True)
    return 0


def _cmd_lid(args) -> int:
    meta = run_lid(args.wav, args.cache_dir, rid=args.rid, win=args.win, hop=args.hop,
                   device=args.device, overwrite=args.overwrite)
    print(meta, flush=True)
    return 0


def _parse_pair_arg(pair_arg):
    """`--pair a,b` -> (code_a, code_b); refuse anything but exactly two distinct codes."""
    if pair_arg is None:
        return None
    parts = [p.strip() for p in str(pair_arg).split(",")]
    if len(parts) != 2 or not all(parts):
        raise SystemExit(f"--pair must be two VoxLingua107 codes 'a,b', got {pair_arg!r}")
    if parts[0] == parts[1]:
        raise SystemExit(f"--pair must name two DISTINCT languages, got {pair_arg!r}")
    return parts[0], parts[1]


def _validate_cluster_args(args) -> None:
    """Refuse a missing REQUIRED argument AND an argument IRRELEVANT to the chosen system (brief change 3).

    Model rule: --model must be `m3:<ckpt>` exactly when --system is M3, and e1|e2|e3 otherwise.
    Per-system required / forbidden args:
      B0: none required; --pair/--means/--v-hat/--delta/--wav(for embedding)/--alpha(non-default)/--tau0 forbidden
      B1: --pair, --means required; --v-hat/--delta/--alpha/--tau0 forbidden
      M1: --v-hat required; --pair/--means/--delta/--alpha/--tau0/--wav forbidden
      B2: --pair, --delta, --wav required; --means/--v-hat/--alpha/--tau0 forbidden
      M2: --pair, --wav required (--alpha drives it, --tau0 optional); --means/--v-hat/--delta forbidden
      M3: none of the operating-point args; --pair/--means/--v-hat/--delta/--alpha/--tau0/--wav(embedding) forbidden
    --wav is always allowed as the rid source; it is only "irrelevant" as an EMBEDDING input, which only B2/M2
    use, so we never forbid --wav itself -- we forbid the embedding-needing systems from running WITHOUT it."""
    sys_ = args.system
    if sys_ not in VALID_SYSTEMS:
        raise SystemExit(f"--system must be one of {VALID_SYSTEMS}, got {sys_!r}")

    is_m3_model = str(args.model).startswith("m3:")
    if sys_ == "M3":
        if not is_m3_model:
            raise SystemExit("M3 requires --model m3:<ckpt>")
    else:
        if is_m3_model:
            raise SystemExit(f"{sys_} requires --model e1|e2|e3, not an m3:<ckpt> model")
        if args.model not in ("e1", "e2", "e3"):
            raise SystemExit(f"{sys_} requires --model e1|e2|e3, got {args.model!r}")

    given = {
        "--pair": args.pair is not None,
        "--means": args.means is not None,
        "--v-hat": args.v_hat is not None,
        "--delta": args.delta is not None,
        "--alpha": args.alpha is not None,
        "--tau0": args.tau0 is not None,
    }
    required = {
        "B0": set(),
        "B1": {"--pair", "--means"},
        "M1": {"--v-hat"},
        "B2": {"--pair", "--delta"},
        "M2": {"--pair", "--alpha"},
        "M3": set(),
    }[sys_]
    allowed = {
        "B0": set(),
        "B1": {"--pair", "--means"},
        "M1": {"--v-hat"},
        "B2": {"--pair", "--delta"},
        "M2": {"--pair", "--alpha", "--tau0"},
        "M3": set(),
    }[sys_]

    for arg in required:
        if not given[arg]:
            raise SystemExit(f"system {sys_} requires {arg}")
    for arg, was_given in given.items():
        if was_given and arg not in allowed:
            raise SystemExit(f"{arg} is not relevant to system {sys_}; remove it")

    # B2 and M2 embed output segments, so they need the audio
    if sys_ in ("B2", "M2") and args.wav is None:
        raise SystemExit(f"system {sys_} embeds its output segments and needs --wav REC.wav")


def _cmd_cluster(args) -> int:
    _validate_cluster_args(args)
    thresholds = validate_thresholds(args.thresholds)

    rid = (args.rid or recording_id(args.wav)) if args.wav else args.rid
    if rid is None:
        raise SystemExit("cluster: --rid (or --wav to derive it) is required")

    cache = load_stage1(args.cache_dir, args.model, rid)
    dim = int(cache["embeddings"].shape[-1])

    # resolve per-system inputs once
    post = pair = means = v_hat = embedder = None
    alpha = float(args.alpha) if args.alpha is not None else 0.0
    tau0 = float(args.tau0) if args.tau0 is not None else 0.0
    delta = float(args.delta) if args.delta is not None else 0.0

    if args.system in ("B1", "B2", "M2"):
        code_a, code_b = _parse_pair_arg(args.pair)
        lid = load_lid(args.cache_dir, rid)                  # stage-2 LID input (full 107-wide posterior)
        post = lid["post"]
        i, j = pair_indices(lid["labels"], code_a, code_b)
        pair = (i, j)
        if args.system == "B1":
            mu_by_code = load_means_npz(args.means, code_a, code_b, dim)
            means = {i: mu_by_code[code_a], j: mu_by_code[code_b]}
    if args.system == "M1":
        v_hat = load_v_hat_npy(args.v_hat, dim)
    if args.system in ("B2", "M2"):
        wav = _read_wav_mono(args.wav)
        embedder = SegmentEmbedder(args.model, rid, wav, device=args.device)

    tag = system_tag(args.system, delta=delta, alpha=alpha, tau0=tau0, m3_ckpt=args.model)
    pipe = load_stage2_pipeline(args.model, thresholds[0], device=args.device)

    for th in thresholds:
        t0 = time.time()
        diar, excl, info = run_system_from_cache(
            pipe, cache, th, args.system,
            post=post, pair=pair, means=means, v_hat=v_hat,
            delta=delta, alpha=alpha, tau0=tau0, embedder=embedder)
        dt = time.time() - t0
        ov_path, ex_path = rttm_paths(args.out, args.split, tag, th, rid)
        write_rttm(ov_path, rid, segments_from_annotation(diar))
        write_rttm(ex_path, rid, segments_from_annotation(excl))
        rec = {"rid": rid, "system": args.system, "system_tag": tag, "threshold": th,
               "num_speakers": info.get("num_speakers"), "stage2_s": round(dt, 3),
               "rttm": str(ov_path), "exclusive_rttm": str(ex_path)}
        for k in ("m2_relabels", "b2_skipped", "m2_skipped", "n_active_slots", "n_active_unlabelled"):
            if k in info:
                rec[k] = info[k]
        print(rec, flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", default="cpu", help="torch device (default cpu; acceptance runs CPU-only)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cache", help="stage 1: segmentation + embeddings -> .npz")
    c.add_argument("--model", required=True, help="e1 | e2 | e3 | m3:<checkpoint>")
    c.add_argument("--wav", required=True, type=Path)
    c.add_argument("--cache-dir", required=True, type=Path)
    c.add_argument("--rid", default=None, help="recording id (default: wav stem)")
    c.add_argument("--overwrite", action="store_true")
    c.set_defaults(func=_cmd_cache)

    l = sub.add_parser("lid", help="VoxLingua107 posterior grid -> .npz")
    l.add_argument("--wav", required=True, type=Path)
    l.add_argument("--cache-dir", required=True, type=Path)
    l.add_argument("--rid", default=None)
    l.add_argument("--win", type=float, default=LID_WIN)
    l.add_argument("--hop", type=float, default=HOP)
    l.add_argument("--overwrite", action="store_true")
    l.set_defaults(func=_cmd_lid)

    cl = sub.add_parser("cluster", help="stage 2: re-cluster from cache at thresholds -> RTTM")
    cl.add_argument("--model", required=True, help="e1|e2|e3 (B0/B1/B2/M1/M2) or m3:<ckpt> (M3)")
    cl.add_argument("--rid", default=None, help="recording id (default: derive from --wav)")
    cl.add_argument("--wav", default=None, type=Path,
                    help="recording audio: derives --rid when omitted, and is the embedding input B2/M2 need")
    cl.add_argument("--cache-dir", required=True, type=Path)
    cl.add_argument("--thresholds", required=True, type=float, nargs="+")
    cl.add_argument("--out", required=True, type=Path)
    cl.add_argument("--split", default="dev")
    cl.add_argument("--system", default="B0", choices=VALID_SYSTEMS)
    cl.add_argument("--pair", default=None, help="two VoxLingua107 codes 'a,b' (B1, B2, M2)")
    cl.add_argument("--means", default=None, type=Path,
                    help="B1 language means .npz, one (dim,) array per VoxLingua107 code (both pair langs)")
    cl.add_argument("--v-hat", dest="v_hat", default=None, type=Path,
                    help="M1 subspace direction .npy (dim,); normalised on load, zero/non-finite refused")
    cl.add_argument("--delta", type=float, default=None, help="B2 merge offset on the cosine-distance scale")
    cl.add_argument("--alpha", type=float, default=None,
                    help="M2 relabel gate coefficient (grid 0..1.00); 0 is a no-op")
    cl.add_argument("--tau0", type=float, default=None,
                    help="M2 relabel gate offset; deployed form tau0=0 (PREREGISTRATION_M2B.md §3), default 0.0")
    cl.set_defaults(func=_cmd_cluster)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
