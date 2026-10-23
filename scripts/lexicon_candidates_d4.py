#!/usr/bin/env python3
"""Candidates for the D4 loan lexicons (H7 supplement; PLAN.md §2 "Language tags"). Text only; no audio, no outcome.

MUCS transcripts often write English words in the native script (e.g. Hindi "ऑफिस" for "office"), which script
tags would call native. Following the D1 method (scripts/lexicon_candidates.py), each native-script word type is
romanised with uroman (hin / ben) and its consonant skeleton is matched against the skeletons of English words:
the corpus's own Latin-script vocabulary first, then the wordfreq English top 50,000. Each candidate carries its
token count, the matched English word, the English word's corpus count and Zipf frequency, the native word's
wordfreq Zipf frequency in the native language, and up to three contexts, for curation.
Output: <out>/<pair>_native_loans.tsv and <out>/<pair>_stats.json
Usage: python scripts/lexicon_candidates_d4.py --manifest data/d4_meta/hi-en_manifest.csv --pair hi-en --out data/lexicon_d4
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import uroman
import wordfreq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csdiar.data.tags import EN, NE, words  # noqa: E402
from csdiar.data.tags_d4 import PAIR_SCRIPT, PAIR_UROMAN, make_tagger  # noqa: E402
from scripts.lexicon_candidates import EN_RULES, NE_RULES, skeleton  # noqa: E402

WF_LANG = {"hi-en": "hi", "bn-en": "bn"}
NATIVE_SIMPLE = [("aa", "a"), ("ii", "i"), ("uu", "u"), ("ee", "e"), ("oo", "o"), ("tt", "t"), ("dd", "d"), ("nn", "n"),
                 ("ss", "s"), ("kh", "k"), ("gh", "g"), ("chh", "ch"), ("jh", "j"), ("th", "t"), ("dh", "d"), ("ph", "f"),
                 ("bh", "b"), ("sh", "s"), ("v", "b"), ("w", "b"), ("y", "i")]
ENGLISH_SIMPLE = [("ph", "f"), ("ck", "k"), ("qu", "kb"), ("x", "ks"), ("th", "t"), ("sh", "s"), ("w", "b"), ("v", "b"),
                  ("y", "i"), ("ee", "i"), ("oo", "u")]


def simple(s: str, rules) -> str:
    s = s.lower()
    for a, b in rules:
        s = s.replace(a, b)
    if rules is ENGLISH_SIMPLE:
        s = re.sub(r"c(?=[eiy])", "s", s).replace("c", "k")
    s = re.sub(r"(.)\1+", r"\1", s)
    return re.sub(r"a$", "", re.sub(r"[^a-z]", "", s))


def similarity(native_roman: str, english: str) -> float:
    """Character similarity of simplified romanised native form and English spelling (difflib ratio)."""
    return SequenceMatcher(None, simple(native_roman, NATIVE_SIMPLE), simple(english, ENGLISH_SIMPLE)).ratio()
LIKELY_SIM = 0.75  # simplified romanised native form vs English spelling (difflib ratio)
LIKELY_SIM_SHORT = 0.85  # the same for two-consonant skeletons (short words), which collide more often
LIKELY_CORPUS = 5  # the English word occurs at least this often in the corpus's own Latin-script text
REVIEW_TOP = 1500  # most frequent native-script types always reviewed
REVIEW_MIN = 5  # candidates seen at least this often are reviewed too


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--pair", choices=sorted(PAIR_SCRIPT), required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tag = make_tagger(PAIR_SCRIPT[args.pair])
    nat, lat, contexts = Counter(), Counter(), defaultdict(list)
    n_tok = Counter()
    for r in csv.DictReader(open(args.manifest, encoding="utf-8")):
        ws = words(r["transcription"] or "")
        for i, w in enumerate(ws):
            t = tag(w)
            n_tok[t] += 1
            if t == NE:
                key = w
                nat[w] += 1
            elif t == EN:
                key = w.lower()
                lat[key] += 1
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
    n_cand = n_cand_tok = n_likely = n_likely_tok = 0
    with open(args.out / f"{args.pair}_native_loans.tsv", "w", encoding="utf-8") as fh:
        fh.write("native\tcount\turoman\tskeleton\tenglish\ten_corpus\ten_zipf\tnative_zipf\tsim\tlikely\tcontexts\n")
        for w, n in nat.most_common():
            r = rom.romanize_string(w, lcode=PAIR_UROMAN[args.pair]).lower()
            k = skeleton(r, NE_RULES)
            if len(k) < 2 or k not in en_by_key:
                continue
            bar = LIKELY_SIM if len(k) >= 3 else LIKELY_SIM_SHORT
            best = max(en_by_key[k], key=lambda e: (similarity(r, e) >= bar, lat.get(e, 0),
                                                    wordfreq.zipf_frequency(e, "en")))
            sim = similarity(r, best)
            likely = sim >= bar and lat.get(best, 0) >= LIKELY_CORPUS
            n_cand += 1
            n_cand_tok += n
            n_likely += likely
            n_likely_tok += n * likely
            fh.write(f"{w}\t{n}\t{r}\t{k}\t{best}\t{lat.get(best, 0)}\t{wordfreq.zipf_frequency(best, 'en'):.2f}\t"
                     f"{wordfreq.zipf_frequency(w, WF_LANG[args.pair]):.2f}\t{sim:.2f}\t{int(likely)}\t"
                     f"{' | '.join(contexts[w])}\n")
    stats = {"pair": args.pair, "tokens_by_tag": dict(n_tok), "native_types": len(nat), "latin_types": len(lat),
             "candidate_types": n_cand, "candidate_tokens": n_cand_tok, "likely_types": n_likely,
             "likely_tokens": n_likely_tok}
    # Review list for labelling: the most frequent native types plus every candidate seen at least REVIEW_MIN times.
    cand = {}
    with open(args.out / f"{args.pair}_native_loans.tsv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            cand[r["native"]] = r
    review = [w for w, _ in nat.most_common(REVIEW_TOP)]
    review += [w for w, n in nat.most_common() if w in cand and n >= REVIEW_MIN and w not in set(review)]
    with open(args.out / f"{args.pair}_review.tsv", "w", encoding="utf-8") as fh:
        fh.write("native\tcount\turoman\tenglish_guess\tsim\tcontexts\n")
        for w in review:
            c = cand.get(w, {})
            r = c.get("uroman") or rom.romanize_string(w, lcode=PAIR_UROMAN[args.pair]).lower()
            fh.write(f"{w}\t{nat[w]}\t{r}\t{c.get('english', '')}\t{c.get('sim', '')}\t{' | '.join(contexts[w])}\n")
    stats.update(review_types=len(review), review_tokens=sum(nat[w] for w in review))
    with open(args.out / f"{args.pair}_latin_review.tsv", "w", encoding="utf-8") as fh:
        fh.write("latin\tcount\ten_zipf\tcontexts\n")
        n_lat = 0
        for w, n in lat.most_common():
            z = wordfreq.zipf_frequency(w, "en")
            if z < 3.0 and len(w) > 1 and n >= REVIEW_MIN:
                fh.write(f"{w}\t{n}\t{z:.2f}\t{' | '.join(contexts[w])}\n")
                n_lat += 1
    stats.update(latin_review_types=n_lat)
    (args.out / f"{args.pair}_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
