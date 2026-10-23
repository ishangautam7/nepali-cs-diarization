#!/usr/bin/env python3
"""Draw the D4 residual-contamination sample of PREREGISTRATION_H7.md §3 (descriptive; after the freeze).

English words that the lexicon missed stay tagged native and blur switch points and controls. To estimate how
often, the population is exactly the words the frozen H7 boundary run tagged: the D4 word alignments of both splits
(<d4-root>/<pair>/align/{train,test}.jsonl, written by csdiar.data.align with the frozen command), read the way
csdiar.data.boundaries reads them, keeping only clips whose alignment status is "ok" (the boundary run skips every
other clip), and tagged by csdiar.data.boundaries.load_words with the D4 tagger for the pair's script swapped in,
as csdiar.data.boundaries_d4 does, and with the pair's loan and name lists read by csdiar.data.boundaries.read_list.
From the native-tagged tokens, 200 are drawn uniformly at random with seed 20261018 and written with their clip
context, for an LLM-based labeller to label LOAN (English written in the native script), NATIVE, NAME or UNSURE,
the labels of §3. 50 of the 200, drawn with the same generator, are marked for the analyst's check. A token is one
occurrence, so frequent words are drawn in proportion to their frequency, which is what a share of tokens needs.
Nothing here reads audio, a boundary, or an H7 outcome.

Outputs in --out:
  <pair>_items.csv   item, split, clip_id, word_index, word, context (the word is marked «» in it), check (1 = in
                     the analyst's 50)
  summary.json       per pair and split: clips by alignment status, words in the "ok" clips, native-tagged tokens;
                     SHA-256 of the alignment files and the lexicon lists; seed and sample sizes
  instructions.md    the labelling instructions given to the labeller

Usage (on the box, after the frozen D4 alignment command has written ~/data/d4/<pair>/align):
  .venv-pyannote4/bin/python scripts/d4_contamination_sample.py --d4-root ~/data/d4 \
      --lexicon planning/lexicon/d4 --out ~/data/d4/results/contamination
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csdiar.data.boundaries as boundaries  # noqa: E402
from csdiar.data.tags import NE  # noqa: E402
from csdiar.data.tags_d4 import PAIR_SCRIPT, make_tagger  # noqa: E402

SEED = 20261018
N_ITEMS = 200
N_CHECK = 50
CONTEXT = 8                  # words shown either side of the token
PAIRS = ("hi-en", "bn-en")
SPLITS = ("train", "test")   # boundaries_d4 ... --splits train test
INSTRUCTIONS = """\
# Labelling instructions: D4 residual-contamination audit (PREREGISTRATION_H7.md §3)

Each item is one word from a MUCS 2021 spoken-tutorial transcript, shown inside its sentence and marked «like this».
The tagger counted it as a native-language word (Hindi in hi-en, Bengali in bn-en) because it is written in the
native script. Label that one occurrence, in its context, with exactly one of:

- LOAN: an English word written in the native script (e.g. क्लिक "click", ফাংশন "function"), whether or not it is
  common in everyday speech.
- NATIVE: a word of the native language, including long-established words of Sanskrit, Persian or Arabic origin.
- NAME: a proper name (a person, place, product, software or brand), in any language.
- UNSURE: only when the context does not settle it; add a one-line note.

Judge the word itself, not the sentence's overall language. A word near the start or end of its clip has less
context on that side. Give a short gloss for each item.
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_align(path: Path) -> dict[str, dict]:
    """One record per clip, read as csdiar.data.boundaries reads it (a clip written twice keeps its last record)."""
    with open(path, encoding="utf-8") as fh:
        return {json.loads(line)["clip_id"]: json.loads(line) for line in fh}


def native_tokens(align_dir: Path, pair: str, loans: frozenset[str], names: frozenset[str]) -> tuple[list[dict], dict]:
    """Every native-tagged token of the "ok" clips of both splits, in a fixed order (split, clip_id, position)."""
    saved = boundaries.tag_word
    boundaries.tag_word = make_tagger(PAIR_SCRIPT[pair])   # load_words looks the tagger up in its module, as in D4
    try:
        out, counts = [], {}
        for split in SPLITS:
            recs = read_align(align_dir / f"{split}.jsonl")
            n_words = 0
            for clip_id in sorted(recs):
                rec = recs[clip_id]
                if rec["status"] != "ok":
                    continue
                if any(w.get("aligned") and not ("start" in w and "end" in w) for w in rec["words"]):
                    raise SystemExit(f"refusing: {split}.jsonl clip {clip_id} has an aligned word without timings "
                                     "(truncated or malformed alignment file?)")
                ws = boundaries.load_words(rec, loans, names)
                n_words += len(ws)
                texts = [w.text for w in ws]
                for w in ws:
                    if w.tag == NE:
                        lo, hi = max(0, w.idx - CONTEXT), min(len(ws), w.idx + CONTEXT + 1)
                        out.append({"split": split, "clip_id": clip_id, "word_index": w.idx, "word": w.text,
                                    "context": " ".join(texts[lo:w.idx] + [f"«{w.text}»"] + texts[w.idx + 1:hi])})
            counts[split] = {"clips": len(recs), "status": dict(sorted(Counter(r["status"] for r in recs.values()).items())),
                             "words_in_ok_clips": n_words, "align_sha256": sha256(align_dir / f"{split}.jsonl")}
        counts["native_tokens"] = len(out)
    finally:
        boundaries.tag_word = saved
    return out, counts


def draw(tokens: list[dict], n_items: int, n_check: int, seed: int) -> list[dict]:
    """n_items tokens drawn uniformly without replacement, n_check of them marked for the analyst."""
    if len(tokens) < n_items:
        raise SystemExit(f"refusing: only {len(tokens)} native-tagged tokens, fewer than the {n_items} to draw")
    rng = random.Random(seed)
    items = rng.sample(tokens, n_items)
    check = set(rng.sample(range(n_items), n_check))
    return [{"item": k + 1, **t, "check": int(k in check)} for k, t in enumerate(items)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--d4-root", type=Path, required=True)
    ap.add_argument("--lexicon", type=Path, required=True, help="folder with <pair>_loans.txt and <pair>_names.txt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs", nargs="+", default=list(PAIRS), choices=PAIRS)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"seed": SEED, "n_items": N_ITEMS, "n_check": N_CHECK, "context_words": CONTEXT, "pairs": {}}
    for pair in args.pairs:
        loans, loans_sha = boundaries.read_list(args.lexicon / f"{pair}_loans.txt")
        names, names_sha = boundaries.read_list(args.lexicon / f"{pair}_names.txt")
        tokens, counts = native_tokens(args.d4_root / pair / "align", pair, loans, names)
        items = draw(tokens, N_ITEMS, N_CHECK, SEED)
        with open(args.out / f"{pair}_items.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(items[0]))
            w.writeheader()
            w.writerows(items)
        summary["pairs"][pair] = {**counts, "loans_sha256": loans_sha, "names_sha256": names_sha}
        status = "; ".join(f"{s}: {counts[s]['status']}" for s in SPLITS)
        print(f"{pair}: {counts['native_tokens']} native-tagged tokens in the ok clips ({status}); "
              f"drew {N_ITEMS} ({N_CHECK} for the check)")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    (args.out / "instructions.md").write_text(INSTRUCTIONS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
