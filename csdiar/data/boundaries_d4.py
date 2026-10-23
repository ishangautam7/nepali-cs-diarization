#!/usr/bin/env python3
"""Boundary points, controls and matches for D4 (MUCS 2021; the H7 supplement).

This runs csdiar.data.boundaries (frozen with M2a) unchanged, except that the word tagger is the D4 one for the
pair (csdiar.data.tags_d4: native script -> NE, Latin -> EN, loan lexicon -> LOAN). Every other rule is the
D1 rule: pause finding, window containment, low-alignment exclusion (5th percentile of the pair's train clips),
families, 1:3 matching and the seed.
Usage (on the box, after `source ops/env.sh`):
  .venv-pyannote4/bin/python -m csdiar.data.boundaries_d4 --pair hi-en --root ~/data/d4/hi-en \
    --align ~/data/d4/hi-en/align --vad ~/data/d4/hi-en/vad --out ~/data/d4/hi-en/boundaries/<tag> \
    --loans planning/lexicon/d4/hi-en_loans.txt --names planning/lexicon/d4/hi-en_names.txt --splits train test
"""
from __future__ import annotations

import sys

import csdiar.data.boundaries as boundaries
from csdiar.data.tags_d4 import PAIR_SCRIPT, make_tagger


def main() -> int:
    argv = sys.argv[1:]
    if "--pair" not in argv or argv.index("--pair") + 1 >= len(argv):
        raise SystemExit(f"--pair is required ({' or '.join(sorted(PAIR_SCRIPT))})")
    i = argv.index("--pair")
    pair = argv[i + 1]
    if pair not in PAIR_SCRIPT:
        raise SystemExit(f"unknown pair {pair}")
    boundaries.tag_word = make_tagger(PAIR_SCRIPT[pair])  # load_words looks the tagger up in the module namespace
    sys.argv = [sys.argv[0]] + argv[:i] + argv[i + 2:]
    return boundaries.main()


if __name__ == "__main__":
    sys.exit(main())
