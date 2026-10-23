#!/usr/bin/env python3
"""Screen candidate D2-P podcasts: fetch the newest episode of each show from its RSS feed and summarise it.

For each show: download the newest audio enclosure, convert it to 16 kHz mono WAV (FFmpeg), then report:
- speakers and speaking-time shares from pyannote Community-1. This is screening only; the study's
  reference for any selected episode is human annotation, never this output.
- English share: fraction of 3-second windows where VoxLingua107 prefers English over Nepali, and the share
  where other languages dominate (music, ads or a third language).
Nothing here is a study result; it only decides which shows are worth annotating.

Usage (on the box, after `source ops/env.sh`):
  .venv-pyannote4/bin/python scripts/screen_podcasts.py --candidates ~/data/podcasts/candidates.csv \
      --shows "The Doers Nepal -Podcast" "Millennials Of Nepal" --out ~/data/podcasts/screen
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

FFMPEG = str(Path.home() / "opt/ffmpeg/bin/ffmpeg")
UA = {"User-Agent": "csdiar-research/0.1 (podcast screening)"}


def newest_enclosure(feed_url: str) -> tuple[str, str]:
    req = urllib.request.Request(feed_url, headers=UA)
    root = ET.fromstring(urllib.request.urlopen(req, timeout=30).read(20_000_000))
    best = None
    for it in root.find("channel").findall("item"):
        enc = it.find("enclosure")
        if enc is None or not enc.get("url"):
            continue
        try:
            when = parsedate_to_datetime(it.findtext("pubDate"))
        except Exception:
            continue
        if best is None or when > best[0]:
            best = (when, enc.get("url"), (it.findtext("title") or "").strip())
    if best is None:
        raise ValueError("no dated audio enclosure")
    return best[1], best[2]


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:40] or "show"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--shows", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    feeds = {}
    for r in csv.DictReader(open(args.candidates, encoding="utf-8")):
        feeds.setdefault(r["show"], r["feed"])

    from pyannote.audio import Pipeline
    from speechbrain.inference.classifiers import EncoderClassifier
    pipe = Pipeline.from_pretrained(str(Path.home() / "data/models/pyannote-community-1")).to(torch.device("cuda"))
    lid = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa",
                                         savedir=str(Path.home() / "data/models/lang-id-voxlingua107-ecapa"),
                                         run_opts={"device": "cuda:0"})
    labels = lid.hparams.label_encoder.ind2lab
    i_ne = next(i for i, lab in labels.items() if lab.split(":")[0].strip() == "ne")
    i_en = next(i for i, lab in labels.items() if lab.split(":")[0].strip() == "en")

    args.out.mkdir(parents=True, exist_ok=True)
    out_csv = args.out / "screening.csv"
    fields = ["show", "episode", "minutes", "speakers", "top_shares", "english_share", "other_share", "url", "error"]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for show in args.shows:
            row = {"show": show}
            try:
                url, title = newest_enclosure(feeds[show])
                row.update(episode=title[:80], url=url)
                folder = args.out / slug(show)
                folder.mkdir(exist_ok=True)
                raw = folder / "episode.audio"
                req = urllib.request.Request(url, headers=UA)
                raw.write_bytes(urllib.request.urlopen(req, timeout=120).read())
                wav_path = folder / "episode_16k.wav"
                subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(raw), "-ac", "1", "-ar", "16000",
                                str(wav_path)], check=True)
                wav, sr = sf.read(wav_path, dtype="float32")
                row["minutes"] = round(len(wav) / sr / 60, 1)
                diar = pipe({"waveform": torch.from_numpy(wav)[None], "sample_rate": sr})
                ann = getattr(diar, "speaker_diarization", diar)
                totals = {}
                for seg, _, spk in ann.itertracks(yield_label=True):
                    totals[spk] = totals.get(spk, 0.0) + seg.duration
                talk = sum(totals.values()) or 1.0
                shares = sorted((v / talk for v in totals.values()), reverse=True)
                row.update(speakers=len(totals), top_shares=" ".join(f"{s:.2f}" for s in shares[:4]))
                win = 3 * sr
                frames = np.stack([wav[s : s + win] for s in range(0, len(wav) - win, win)])
                probs = []
                with torch.inference_mode():
                    for b in range(0, len(frames), 256):
                        probs.append(lid.classify_batch(torch.from_numpy(frames[b : b + 256]).cuda())[0].exp().cpu().numpy())
                p = np.concatenate(probs)
                pair = p[:, i_ne] + p[:, i_en]
                row["english_share"] = round(float(np.mean(p[:, i_en] > p[:, i_ne])), 3)
                row["other_share"] = round(float(np.mean(pair < 0.5)), 3)
            except Exception as exc:
                row["error"] = str(exc)[:200]
            w.writerow(row)
            fh.flush()
            print(row, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
