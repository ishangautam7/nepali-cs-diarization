#!/usr/bin/env python3
"""Word-level forced alignment of D1 transcripts on the GPU (PLAN.md §4, step 2).

Each transcript is split into words with csdiar.data.tags.words (the same split the tagger uses), every word
is romanised with uroman (language "nep"; it restores Devanagari inherent vowels, which the aligner
package's unidecode helper drops), and the clip is
aligned with torchaudio's MMS_FA bundle: the MMS-300m forced-alignment model with a <star> token at both
edges to absorb untranscribed speech. Words with no alignable letters (digits, symbols) get no timing and
are marked aligned=false; they are `neutral` anyway.

Output: one JSON line per clip in <out>/<split>.jsonl, with per-word start/end (s), mean token
probability and script tag. Re-running skips clips already written.

Usage (on the box, after `source ops/env.sh`):
  .venv-pyannote4/bin/python -m csdiar.data.align --root ~/data/d1 --out ~/data/d1/align --splits validation
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
import uroman

from csdiar.data.tags import tag_word, words as split_words

LANG = "nep"
_UROMAN = uroman.Uroman()


def romanise(ws: list[str]) -> list[str]:
    """One lowercase romanised string per word (uroman, language 'nep'), as MMS_FA expects.

    uroman restores Devanagari inherent vowels (नमस्ते -> namaste); Latin-script words pass through.
    """
    return [_UROMAN.romanize_string(w, lcode=LANG).lower().replace("\u2019", "'") for w in ws]


def align_clip(wav_path: Path, transcript: str, model, dictionary: dict, aligner, device) -> dict:
    ws = split_words(transcript)
    if not ws:
        return {"status": "empty", "words": []}
    # Keep letters only: "-" is the CTC blank (id 0) and "*" is the star; neither may appear inside a word.
    ids = [[dictionary[ch] for ch in tok if ch in dictionary and ch not in "-*"] for tok in romanise(ws)]
    alignable = [i for i, t in enumerate(ids) if t]
    if not alignable:
        return {"status": "no_alignable_words", "words": [{"w": w, "tag": tag_word(w), "aligned": False} for w in ws]}
    star = dictionary["*"]
    wav, sr = sf.read(wav_path, dtype="float32")
    if sr != 16000 or wav.ndim != 1:
        raise ValueError(f"{wav_path}: expected 16 kHz mono")
    x = torch.from_numpy(wav).unsqueeze(0).to(device)
    with torch.inference_mode():
        emission, _ = model(x)
        spans = aligner(emission[0], [[star]] + [ids[i] for i in alignable] + [[star]])
    sec_per_frame = x.size(1) / emission.size(1) / sr
    timed = {}
    for i, word_spans in zip(alignable, spans[1:-1]):
        timed[i] = (word_spans[0].start * sec_per_frame, word_spans[-1].end * sec_per_frame,
                    sum(s.score for s in word_spans) / len(word_spans))
    out = []
    for i, w in enumerate(ws):
        rec = {"w": w, "tag": tag_word(w), "aligned": i in timed}
        if i in timed:
            rec["start"], rec["end"], rec["score"] = (round(timed[i][0], 3), round(timed[i][1], 3),
                                                      round(float(timed[i][2]), 4))
        out.append(rec)
    return {"status": "ok", "duration": round(len(wav) / sr, 3), "words": out}


def main() -> int:
    global LANG
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True, help="D1 ingest folder with manifest.csv and wav/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["validation", "train", "test"])
    ap.add_argument("--limit", type=int, default=0, help="align at most this many clips per split (testing)")
    ap.add_argument("--lang", default=LANG, help="uroman language code (nep for D1; hin / ben for D4)")
    args = ap.parse_args()
    LANG = args.lang

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=True).to(device).eval()
    dictionary = bundle.get_dict(star="*")
    aligner = bundle.get_aligner()
    rows = list(csv.DictReader(open(args.root / "manifest.csv", encoding="utf-8")))
    args.out.mkdir(parents=True, exist_ok=True)

    for split in [s for s in args.splits if any(r["split"] == s for r in rows)]:
        out_path = args.out / f"{split}.jsonl"
        done = set()
        if out_path.exists():
            done = {json.loads(line)["clip_id"] for line in open(out_path, encoding="utf-8") if line.strip()}
        todo = [r for r in rows if r["split"] == split and r["clip_id"] not in done]
        if args.limit:
            todo = todo[: args.limit]
        t0, n_fail = time.time(), 0
        with open(out_path, "a", encoding="utf-8") as fh:
            for k, r in enumerate(todo, 1):
                try:
                    rec = align_clip(args.root / r["wav"], r["transcription"] or "", model, dictionary, aligner, device)
                except Exception as exc:  # recorded, never silently dropped
                    rec, n_fail = {"status": "error", "error": str(exc)[:300], "words": []}, n_fail + 1
                fh.write(json.dumps({"clip_id": r["clip_id"], **rec}, ensure_ascii=False) + "\n")
                if k % 500 == 0:
                    fh.flush()
                    print(f"{split}: {k}/{len(todo)} clips, {time.time() - t0:.0f} s, {n_fail} errors", flush=True)
        print(f"{split}: done {len(todo)} clips in {time.time() - t0:.0f} s, {n_fail} errors", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
