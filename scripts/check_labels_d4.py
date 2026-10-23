#!/usr/bin/env python3
"""Check D4 lexicon label files against their review chunks (text only).

For every chunk in <dir>/chunks with a label file in <dir>/labels: the rows must match one to one, in order (same
word string), and every label must be valid. Prints label counts by type and by token, and exits non-zero on
any mismatch.
Usage: python scripts/check_labels_d4.py data/lexicon_d4
"""
import csv
import sys
from collections import Counter
from pathlib import Path

VALID = {"review": {"LOAN", "NATIVE", "NAME", "UNSURE"}, "latin_review": {"ENGLISH", "NATIVE", "NAME", "UNSURE"}}


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))


def main() -> int:
    d = Path(sys.argv[1])
    bad = 0
    for chunk in sorted((d / "chunks").glob("*.tsv")):
        lab = d / "labels" / chunk.name
        if not lab.exists():
            print(f"{chunk.name}: no label file yet")
            continue
        kind = "latin_review" if "_latin_review_" in chunk.name else "review"
        key = "latin" if kind == "latin_review" else "native"
        src, out = read(chunk), read(lab)
        problems = []
        if len(src) != len(out):
            problems.append(f"{len(src)} input rows vs {len(out)} labelled")
        for i, (a, b) in enumerate(zip(src, out)):
            if a[key] != b.get(key):
                problems.append(f"row {i + 2}: {a[key]!r} vs {b.get(key)!r}")
                break
        invalid = Counter(b.get("label") for b in out if b.get("label") not in VALID[kind])
        if invalid:
            problems.append(f"invalid labels {dict(invalid)}")
        by_type = Counter(b.get("label") for b in out)
        by_tok = Counter()
        for a, b in zip(src, out):
            by_tok[b.get("label")] += int(a["count"])
        status = "OK" if not problems else "PROBLEM: " + "; ".join(problems)
        print(f"{chunk.name}: {status} | types {dict(by_type)} | tokens {dict(by_tok)}")
        bad += bool(problems)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
