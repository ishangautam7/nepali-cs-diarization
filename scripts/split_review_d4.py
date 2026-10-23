#!/usr/bin/env python3
"""Split the D4 lexicon review lists into chunks for labelling (text only).

Native-script review lists go into 3 parts per pair; the Latin-script review list stays whole.
Usage: python scripts/split_review_d4.py data/lexicon_d4
"""
import math
import sys
from pathlib import Path


def main() -> int:
    d = Path(sys.argv[1])
    (d / "chunks").mkdir(exist_ok=True)
    (d / "labels").mkdir(exist_ok=True)
    for pair in ("hi-en", "bn-en"):
        for kind, parts in (("review", 3), ("latin_review", 1)):
            lines = (d / f"{pair}_{kind}.tsv").read_text(encoding="utf-8").splitlines()
            head, body = lines[0], lines[1:]
            size = math.ceil(len(body) / parts)
            for i in range(parts):
                chunk = body[i * size:(i + 1) * size]
                out = d / "chunks" / f"{pair}_{kind}_part{i + 1}.tsv"
                out.write_text("\n".join([head] + chunk) + "\n", encoding="utf-8")
                print(out.name, len(chunk))
    return 0


if __name__ == "__main__":
    sys.exit(main())
