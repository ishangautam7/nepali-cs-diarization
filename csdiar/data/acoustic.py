#!/usr/bin/env python3
"""Acoustic-change covariates at every boundary point (PLAN.md §2 "Acoustic covariates"; decision 18).

For each point of a boundary version, two window configurations, each with a left and a right window:
- "emb": W_left = [p0 - 1, p0], W_right = [p1, p1 + 1] (H1's pause-excluded windows at L = 1.0 s);
- "p0":  [t - 1, t] and [t, t + 1] (the windows P0 compares at the point).
A window must lie inside the clip (10 ms tolerance); otherwise every measure of that side is missing.

Measures per window. Speech frames are Silero VAD frames (32 ms) with p >= 0.5 (csdiar.data.vad).
- f0: median F0 of the voiced 10 ms frames, in semitones re 100 Hz. Praat autocorrelation pitch (parselmouth)
  with a per-channel range from a first pass: floor = 0.75 Q1 and ceiling = 1.5 Q3 of the voiced F0 of up to
  200 clips of the channel, analysed at 60-700 Hz (De Looze & Hirst 2008). Missing with < 10 voiced frames.
- energy: 10 log10 of the mean power of the window's speech samples (dB re full scale). Missing with < 0.2 s
  of speech.
- rate: articulation rate, syllable nuclei per second of speech. Nuclei after de Jong & Wempe (2009): local
  maxima of the clip's Praat intensity contour (minimum pitch 50 Hz) above the clip's 0.99 quantile - 25 dB,
  kept when the intensity dips by more than 2 dB before the next peak, and voiced. Missing with < 0.3 s of
  speech. (Not from the transcript: orthographic syllable or letter counts differ systematically between
  romanised Devanagari and English spelling, so they would confound the covariate with the language change.)
- tilt: least-squares slope, in dB per octave, of the long-term average spectrum of the window's speech frames
  over 17 one-third-octave bands centred 125 Hz-5 kHz (Hann, 512-sample frames zero-padded to 2048).
  Missing with < 6 speech frames.
- snr: local SNR proxy, the window's energy minus that side's noise floor, which is the 10th percentile of
  10 ms frame levels over the 2 s on that side of the window's inner edge (clipped to the clip). Missing when
  energy is missing or the noise region is shorter than 0.5 s.
C50 is not estimated: Brouhaha (the plan's source for SNR and C50) is gated and was not accessible at setup.

Output: <out>/acoustic_<split>.csv.gz (point_id, then <cfg>_<measure>_<side> for cfg in emb/p0, side l/r;
empty = missing) and meta.json (parameters, versions, per-channel pitch ranges).
Usage (on the box, after `source ops/env.sh`):
  .venv-pyannote4/bin/python -m csdiar.data.acoustic --root ~/data/d1 --vad ~/data/d1/vad \
    --boundaries ~/data/d1/boundaries/lex_v1b --out ~/data/d1/acoustic/lex_v1b
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
from collections import defaultdict
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path

import numpy as np

SR = 16000
VAD_FRAME = 512
VAD_THRESHOLD = 0.5
LEVEL_FRAME = 160  # 10 ms
WIN_TOL = 0.01
F0_PASS1 = (60.0, 700.0)
F0_CLIPS_PER_CHANNEL = 200
MIN_VOICED = 10
MIN_SPEECH_ENERGY_S = 0.2
MIN_SPEECH_RATE_S = 0.3
MIN_TILT_FRAMES = 6
NOISE_SIDE_S = 2.0
MIN_NOISE_S = 0.5
SILENCE_DB = -25.0
MIN_DIP_DB = 2.0
NFFT = 2048
BAND_CENTRES = 1000.0 * 2.0 ** (np.arange(-9, 8) / 3.0)  # 125 Hz ... 5.04 kHz
MEASURES = ("f0", "energy", "rate", "tilt", "snr")
CONFIGS = ("emb", "p0")
SEED = 20261018


def semitones(f_hz: np.ndarray) -> np.ndarray:
    return 12.0 * np.log2(np.asarray(f_hz, dtype=np.float64) / 100.0)


def band_slope(freqs: np.ndarray, psd: np.ndarray) -> float:
    """dB-per-octave slope of band-averaged power over the one-third-octave bands in BAND_CENTRES."""
    db, x = [], []
    for fc in BAND_CENTRES:
        sel = (freqs >= fc * 2 ** (-1 / 6)) & (freqs < fc * 2 ** (1 / 6))
        if sel.any():
            db.append(10 * np.log10(psd[sel].mean() + 1e-20))
            x.append(np.log2(fc))
    if len(x) < 3:
        return float("nan")
    return float(np.polyfit(np.array(x), np.array(db), 1)[0])


def syllable_nuclei(values: np.ndarray, times: np.ndarray, voiced_times: np.ndarray, voiced: np.ndarray,
                    silence_db: float = SILENCE_DB, min_dip: float = MIN_DIP_DB) -> np.ndarray:
    """Nucleus times after de Jong & Wempe (2009). values/times: intensity contour; voiced flags on voiced_times."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 3:
        return np.empty(0)
    thr = max(np.quantile(v, 0.99) + silence_db, v.min())
    peaks = np.where((v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:]))[0] + 1
    peaks = peaks[v[peaks] > thr]
    keep = [a for a, b in zip(peaks[:-1], peaks[1:]) if v[a] - v[a:b + 1].min() > min_dip]
    if not keep:
        return np.empty(0)
    t = times[np.array(keep)]
    j = np.clip(np.searchsorted(voiced_times, t), 1, len(voiced_times) - 1)
    j = np.where(np.abs(voiced_times[j - 1] - t) <= np.abs(voiced_times[j] - t), j - 1, j)
    return t[voiced[j]]


def frame_levels(x: np.ndarray) -> np.ndarray:
    n = len(x) // LEVEL_FRAME
    fr = x[: n * LEVEL_FRAME].reshape(n, LEVEL_FRAME)
    return 10 * np.log10((fr ** 2).mean(1) + 1e-12)


def pitch_track(x: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    import parselmouth
    p = parselmouth.Sound(x, sampling_frequency=SR).to_pitch_ac(time_step=0.01, pitch_floor=lo, pitch_ceiling=hi)
    return np.asarray(p.xs()), np.asarray(p.selected_array["frequency"])


def window_measures(a: float, b: float, inner: float, side: str, ctx: dict) -> dict:
    nan = float("nan")
    out = dict.fromkeys(MEASURES, nan)
    if a < -WIN_TOL or b > ctx["dur"] + WIN_TOL:
        return out
    a, b = max(a, 0.0), min(b, ctx["dur"])
    f0t, f0 = ctx["f0_t"], ctx["f0"]
    sel = (f0t >= a) & (f0t < b) & (f0 > 0)
    if sel.sum() >= MIN_VOICED:
        out["f0"] = float(np.median(semitones(f0[sel])))
    x, speech_s = ctx["x"], ctx["speech_samples"]
    i0, i1 = int(round(a * SR)), int(round(b * SR))
    seg = x[i0:i1][speech_s[i0:i1]]
    if len(seg) >= MIN_SPEECH_ENERGY_S * SR:
        out["energy"] = float(10 * np.log10((seg ** 2).mean() + 1e-12))
    vad = ctx["speech_frames"]
    centres = (np.arange(len(vad)) + 0.5) * VAD_FRAME / SR
    speech_dur = float(((centres >= a) & (centres < b) & vad).sum()) * VAD_FRAME / SR
    if speech_dur >= MIN_SPEECH_RATE_S:
        nt = ctx["nuclei"]
        out["rate"] = float(((nt >= a) & (nt < b)).sum() / speech_dur)
    k = np.arange(len(vad))
    full = (k * VAD_FRAME >= i0) & ((k + 1) * VAD_FRAME <= min(i1, len(x))) & vad
    if full.sum() >= MIN_TILT_FRAMES:
        frames = np.stack([x[j * VAD_FRAME:(j + 1) * VAD_FRAME] for j in np.where(full)[0]]) * np.hanning(VAD_FRAME)
        psd = (np.abs(np.fft.rfft(frames, NFFT, axis=1)) ** 2).mean(0)
        out["tilt"] = band_slope(np.fft.rfftfreq(NFFT, 1 / SR), psd)
    lo, hi = (max(inner - NOISE_SIDE_S, 0.0), inner) if side == "l" else (inner, min(inner + NOISE_SIDE_S, ctx["dur"]))
    lev = ctx["levels"]
    starts = np.arange(len(lev)) * LEVEL_FRAME / SR
    region = lev[(starts >= lo) & (starts + LEVEL_FRAME / SR <= hi)]
    if out["energy"] == out["energy"] and hi - lo >= MIN_NOISE_S and len(region):
        out["snr"] = out["energy"] - float(np.percentile(region, 10))
    return out


def clip_job(job: tuple) -> list[list]:
    import parselmouth
    import soundfile as sf
    wav, vad_p, lo, hi, pts = job
    x, sr = sf.read(wav, dtype="float64")
    assert sr == SR, wav
    speech_frames = vad_p >= VAD_THRESHOLD
    speech_samples = np.repeat(speech_frames, VAD_FRAME)[: len(x)]
    if len(speech_samples) < len(x):
        speech_samples = np.pad(speech_samples, (0, len(x) - len(speech_samples)))
    f0_t, f0 = pitch_track(x, lo, hi)
    inten = parselmouth.Sound(x, sampling_frequency=SR).to_intensity(minimum_pitch=50.0, subtract_mean=True)
    nuclei = syllable_nuclei(inten.values[0], np.asarray(inten.xs()), f0_t, f0 > 0) if len(f0_t) > 1 else np.empty(0)
    ctx = {"x": x, "dur": len(x) / SR, "f0_t": f0_t, "f0": f0, "speech_frames": speech_frames,
           "speech_samples": speech_samples, "nuclei": nuclei, "levels": frame_levels(x)}
    rows = []
    for pid, t, p0, p1 in pts:
        row = [pid]
        for cfg in CONFIGS:
            (la, lb, li), (ra, rb, ri) = (((p0 - 1, p0, p0), (p1, p1 + 1, p1)) if cfg == "emb"
                                          else ((t - 1, t, t), (t, t + 1, t)))
            left, right = window_measures(la, lb, li, "l", ctx), window_measures(ra, rb, ri, "r", ctx)
            for m in MEASURES:
                row += [left[m], right[m]]
        rows.append(row)
    return rows


def channel_job(job: tuple) -> tuple[str, list[float]]:
    import soundfile as sf
    chan, wavs = job
    voiced = []
    for w in wavs:
        x, _ = sf.read(w, dtype="float64")
        _, f0 = pitch_track(x, *F0_PASS1)
        voiced.append(f0[f0 > 0])
    v = np.concatenate(voiced) if voiced else np.empty(0)
    if len(v) < 1000:
        return chan, [75.0, 500.0]
    q1, q3 = np.percentile(v, [25, 75])
    return chan, [float(max(50.0, round(0.75 * q1))), float(min(800.0, round(1.5 * q3)))]


def fmt(v) -> str:
    return "" if isinstance(v, float) and v != v else (f"{v:.4f}" if isinstance(v, float) else str(v))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--vad", type=Path, required=True)
    ap.add_argument("--boundaries", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["validation", "train", "test"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    rows = list(csv.DictReader(open(args.root / "manifest.csv", encoding="utf-8")))
    by_clip = {r["clip_id"]: r for r in rows}
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    by_chan = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["clip_id"]):
        by_chan[r["speaker_id"]].append(str(args.root / r["wav"]))
    jobs = [(c, sorted(rng.choice(w, min(F0_CLIPS_PER_CHANNEL, len(w)), replace=False).tolist()))
            for c, w in sorted(by_chan.items())]
    t0 = time.time()
    with get_context("fork").Pool(args.workers) as pool:
        ranges = dict(pool.map(channel_job, jobs))
    print(f"pitch ranges for {len(ranges)} channels in {time.time() - t0:.0f} s", flush=True)
    import parselmouth
    meta = {"version": 1, "parselmouth": version("praat-parselmouth"), "praat": parselmouth.PRAAT_VERSION,
            "params": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in globals().items()
                       if k.isupper() and isinstance(v, (int, float, str, tuple, np.ndarray))},
            "pitch_range_hz": ranges, "splits": {}}
    header = ["point_id"] + [f"{c}_{m}_{s}" for c in CONFIGS for m in MEASURES for s in ("l", "r")]
    for split in args.splits:
        pts = defaultdict(list)
        with gzip.open(args.boundaries / f"points_{split}.csv.gz", "rt", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                pts[r["clip_id"]].append((r["point_id"], float(r["t"]), float(r["p0"]), float(r["p1"])))
        vad = np.load(args.vad / f"{split}.npz")
        todo = [(str(args.root / by_clip[c]["wav"]), vad[c].astype(np.float32),
                 *ranges[by_clip[c]["speaker_id"]], p) for c, p in sorted(pts.items())]
        t0, n = time.time(), 0
        out_path = args.out / f"acoustic_{split}.csv.gz"
        with gzip.open(out_path, "wt", encoding="utf-8", newline="") as fh, get_context("fork").Pool(args.workers) as pool:
            w = csv.writer(fh)
            w.writerow(header)
            for i, res in enumerate(pool.imap(clip_job, todo, chunksize=8)):
                for row in res:
                    w.writerow([fmt(v) for v in row])
                    n += 1
                if (i + 1) % 2000 == 0:
                    print(f"{split}: {i + 1}/{len(todo)} clips, {time.time() - t0:.0f} s", flush=True)
        meta["splits"][split] = {"clips": len(todo), "points": n}
        print(f"{split}: done {len(todo)} clips, {n} points in {time.time() - t0:.0f} s", flush=True)
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
