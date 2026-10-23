#!/usr/bin/env python3
"""Build the D4 loan and name lists for the H7 supplement from the validated label files (text only).

loans: native-script words labelled LOAN (English written in the native script), plus Latin-script words labelled
NATIVE (the native language written in Latin letters), as in D1's loan lexicon (decision 12).
names: words labelled NAME in either script. Names are neutral, as in D1.
UNSURE words are left to the script rule and listed in the header with their count.
The label files must pass scripts/check_labels_d4.py first.
Usage: python scripts/build_lexicon_d4.py data/lexicon_d4 planning/lexicon/d4
"""
import csv
import sys
from collections import Counter
from pathlib import Path


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))


def main() -> int:
    src_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    for pair in ("hi-en", "bn-en"):
        loans, names, unsure, tok = [], [], [], Counter()
        for chunk in sorted((src_dir / "chunks").glob(f"{pair}_*review_part*.tsv")):
            latin = "_latin_review_" in chunk.name
            key = "latin" if latin else "native"
            rows, labels = read(chunk), read(src_dir / "labels" / chunk.name)
            assert len(rows) == len(labels) and all(a[key] == b[key] for a, b in zip(rows, labels)), chunk.name
            for a, b in zip(rows, labels):
                lab, w = b["label"], a[key]
                tok[(latin, lab)] += int(a["count"])
                if (not latin and lab == "LOAN") or (latin and lab == "NATIVE"):
                    loans.append((w, int(a["count"]), b.get("gloss") or ""))
                elif lab == "NAME":
                    names.append((w, int(a["count"]), b.get("gloss") or ""))
                elif lab == "UNSURE":
                    unsure.append(w)
        head = (f"# D4 {pair} {{kind}} for the H7 supplement (built by scripts/build_lexicon_d4.py from the labelled "
                f"review lists; text only).\n# One word per line, followed by '  #' with its corpus count and gloss (the "
                f"reader ignores everything after '#').\n# Labelled tokens by script/label: "
                f"{', '.join(f'{'latin' if k[0] else 'native'}/{k[1]}={v}' for k, v in sorted(tok.items()))}.\n"
                f"# UNSURE (left to the script rule): {len(unsure)}.\n")
        for kind, items in (("loans", loans), ("names", names)):
            items.sort(key=lambda x: (-x[1], x[0]))
            lines = [head.replace("{kind}", kind)] + [f"{w}  # {n} {g}\n" for w, n, g in items]
            (out_dir / f"{pair}_{kind}.txt").write_text("".join(lines), encoding="utf-8")
            print(pair, kind, len(items), "types,", sum(n for _, n, _ in items), "tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
