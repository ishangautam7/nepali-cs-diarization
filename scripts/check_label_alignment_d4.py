#!/usr/bin/env python3
"""Detect shifted labels in D4 label files (text only).

For each LOAN row, the English gloss should resemble the native word's romanisation. This checks whether it
resembles the gloss of its own row better than those of the neighbouring rows. It reports, per file, the share of
LOAN rows whose own-row similarity is below 0.5, and any window of 5 consecutive LOAN rows where the gloss of the
next row (or the previous row) fits better than the row's own gloss.
Usage: python scripts/check_label_alignment_d4.py data/lexicon_d4
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.lexicon_candidates_d4 import similarity  # noqa: E402


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))


def main() -> int:
    d = Path(sys.argv[1])
    for chunk in sorted((d / "chunks").glob("*_review_part*.tsv")):
        if "_latin_" in chunk.name or not (d / "labels" / chunk.name).exists():
            continue
        src, lab = read(chunk), read(d / "labels" / chunk.name)
        rows = [(s["uroman"], l["label"], (l.get("gloss") or "").split("(")[0].strip().lower()) for s, l in zip(src, lab)]
        own, shift_next, shift_prev, low = [], 0, 0, 0
        loan_idx = [i for i, r in enumerate(rows) if r[1] == "LOAN" and r[2]]
        for i in loan_idx:
            s0 = similarity(rows[i][0], rows[i][2])
            own.append(s0)
            low += s0 < 0.5
            # does this row's gloss fit the previous / next row's word better?
            if i + 1 < len(rows) and similarity(rows[i + 1][0], rows[i][2]) > s0 + 0.2:
                shift_prev += 1
            if i > 0 and similarity(rows[i - 1][0], rows[i][2]) > s0 + 0.2:
                shift_next += 1
        n = len(loan_idx)
        print(f"{chunk.name}: LOAN rows {n}, own-gloss similarity < 0.5: {low} ({low / max(n, 1):.0%}); "
              f"gloss fits the next word better: {shift_prev}; fits the previous word better: {shift_next}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
