#!/usr/bin/env python3
"""Candidate lists for the frozen loan lexicon and name list (PLAN.md §2, "Language tags").

Text only; no audio and no outcome. Three lists, each for human curation:
1. deva_loans.tsv: Devanagari word types whose romanisation (uroman, 'nep') has the same consonant
   skeleton as an English word (corpus Latin vocabulary first, then the wordfreq English list).
2. latin_nonenglish.tsv: Latin word types that are rare or absent in English (wordfreq Zipf < 3.0):
   romanised Nepali, abbreviations, typos and names.
3. latin_names.tsv: Latin word types written capitalised in >= 80% of their occurrences, excluding
   single letters (variables) and very common English words.
Each row carries up to three example contexts so it can be judged quickly.

Usage: python scripts/lexicon_candidates.py --clips data/d1_meta/clips.csv --out data/lexicon
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import uroman
import wordfreq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csdiar.data.tags import EN, NE, tag_word, words  # noqa: E402

EN_RULES = [("tion", "sn"), ("sion", "sn"), ("ph", "f"), ("ck", "k"), ("qu", "kb"), ("q", "k"), ("x", "ks"),
            ("ch", "C"), ("sh", "s"), ("th", "t"), ("wh", "b"), ("gh", "g"), ("dg", "j"), ("w", "b"), ("v", "b"),
            ("z", "j")]
NE_RULES = [("chh", "C"), ("ch", "C"), ("c", "C"), ("kh", "k"), ("gh", "g"), ("jh", "j"), ("tth", "t"), ("th", "t"),
            ("ddh", "d"), ("dh", "d"), ("ph", "f"), ("bh", "b"), ("sh", "s"), ("w", "b"), ("v", "b"), ("z", "j"),
            ("ng", "n")]


def skeleton(s: str, rules) -> str:
    s = s.lower()
    for a, b in rules:
        s = s.replace(a, b)
    if rules is EN_RULES:
        s = re.sub(r"c(?=[eiy])", "s", s).replace("c", "k")
    s = re.sub(r"[aeiouy]", "", s)
    s = re.sub(r"(?<=[^aeiou])h", "", s)
    s = re.sub(r"(.)\1+", r"\1", s)
    return re.sub(r"[^a-zC]", "", s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.clips, encoding="utf-8")))
    dev, lat, lat_cap, contexts = Counter(), Counter(), Counter(), defaultdict(list)
    for r in rows:
        ws = words(r["transcription"] or "")
        for i, w in enumerate(ws):
            t = tag_word(w)
            key = w if t == NE else w.lower()
            if t == NE:
                dev[w] += 1
            elif t == EN:
                lat[key] += 1
                lat_cap[key] += w[:1].isupper()
            else:
                continue
            if len(contexts[key]) < 3:
                contexts[key].append(" ".join(ws[max(i - 3, 0): i + 4]))
    rom = uroman.Uroman()
    en_by_key = defaultdict(list)
    for w in set(lat) | set(wordfreq.top_n_list("en", 50000)):
        if w.isalpha() and len(w) > 2:
            en_by_key[skeleton(w, EN_RULES)].append(w)
    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.out / "deva_loans.tsv", "w", encoding="utf-8") as fh:
        fh.write("deva\tcount\turoman\tskeleton\tenglish\ten_corpus\ten_zipf\tcontexts\n")
        for w, n in dev.most_common():
            r = rom.romanize_string(w, lcode="nep").lower()
            k = skeleton(r, NE_RULES)
            if len(k) < 3 or k not in en_by_key:
                continue
            best = max(en_by_key[k], key=lambda e: (lat.get(e, 0), wordfreq.zipf_frequency(e, "en")))
            fh.write(f"{w}\t{n}\t{r}\t{k}\t{best}\t{lat.get(best, 0)}\t{wordfreq.zipf_frequency(best, 'en'):.2f}\t"
                     f"{' | '.join(contexts[w])}\n")
    with open(args.out / "latin_nonenglish.tsv", "w", encoding="utf-8") as fh:
        fh.write("latin\tcount\ten_zipf\tcontexts\n")
        for w, n in lat.most_common():
            z = wordfreq.zipf_frequency(w, "en")
            if z < 3.0 and len(w) > 1:
                fh.write(f"{w}\t{n}\t{z:.2f}\t{' | '.join(contexts[w])}\n")
    with open(args.out / "latin_names.tsv", "w", encoding="utf-8") as fh:
        fh.write("latin\tcount\tcap_share\ten_zipf\tcontexts\n")
        for w, n in lat.most_common():
            share = lat_cap[w] / n
            z = wordfreq.zipf_frequency(w, "en")
            if len(w) > 1 and share >= 0.8 and z < 5.0:
                fh.write(f"{w}\t{n}\t{share:.2f}\t{z:.2f}\t{' | '.join(contexts[w])}\n")
    print({"devanagari_types": len(dev), "latin_types": len(lat)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
