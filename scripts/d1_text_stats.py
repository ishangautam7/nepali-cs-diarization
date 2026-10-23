#!/usr/bin/env python3
"""Text-level statistics of D1 switch points, before any forced alignment.

Checks the tagger against the dataset card (token shares, adjacent-pair switch rate, share of clips
with both languages) and gives rough counts of switch points per split. Run lengths in seconds are
estimated from each clip's mean word duration, so they are approximate until alignment.

These are descriptive data counts, not study outcomes, so they are safe before the freeze.

Usage:
  python scripts/d1_text_stats.py data/d1_meta/clips.csv data/d1_meta/text_stats.json
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csdiar.data.tags import EN, NE, NEUTRAL, adjacent_pairs, switches, tag_words, words  # noqa: E402


def main() -> int:
    clips_csv, out_json = Path(sys.argv[1]), Path(sys.argv[2])
    rows = list(csv.DictReader(open(clips_csv, encoding="utf-8")))

    tag_counts = Counter()
    changed = pairs = 0
    both_langs = 0
    per100 = []
    per_split = defaultdict(lambda: Counter())
    shorter_words = defaultdict(Counter)
    en_insert_words = Counter()
    per_channel = defaultdict(lambda: [0, 0.0])  # switches, minutes

    for r in rows:
        ws = words(r["transcription"] or "")
        tags = tag_words(ws)
        tag_counts.update(tags)
        c, p = adjacent_pairs(tags)
        changed, pairs = changed + c, pairs + p
        langs = {t for t in tags if t in (NE, EN)}
        both_langs += len(langs) == 2
        n_lang = sum(t in (NE, EN) for t in tags)
        sw = switches(tags)
        if n_lang >= 10:
            per100.append(100 * len(sw) / n_lang)

        dur = float(r["duration_s"] or 0)
        sec_per_word = dur / len(ws) if ws else 0.0
        split = r["split"]
        per_split[split]["clips"] += 1
        per_split[split]["switches"] += len(sw)
        per_channel[r["speaker_id"]][0] += len(sw)
        per_channel[r["speaker_id"]][1] += dur / 60
        for s in sw:
            short = min(s.left.n_words, s.right.n_words)
            shorter_words[split]["1" if short == 1 else "2" if short == 2 else "3-4" if short <= 4 else "5+"] += 1
            if s.neutral_between:
                per_split[split]["with_neutral_between"] += 1
            # Rough duration of each run: its word span times the clip's mean word duration.
            left_s = (s.left.last - s.left.first + 1) * sec_per_word
            right_s = (s.right.last - s.right.first + 1) * sec_per_word
            if min(left_s, right_s) >= 1.0:
                per_split[split]["both_runs_ge_1s_est"] += 1
            if min(left_s, right_s) >= 0.5:
                per_split[split]["both_runs_ge_0.5s_est"] += 1
        for run_lang, run_len in ((s.right.lang, s.right.n_words) for s in sw):
            if run_lang == EN:
                en_insert_words["1" if run_len == 1 else "2" if run_len == 2 else "3-4" if run_len <= 4 else "5+"] += 1

    lang_total = tag_counts[NE] + tag_counts[EN]
    rates = sorted(n / m for n, m in per_channel.values() if m > 0)
    out = {
        "clips": len(rows),
        "words": sum(tag_counts.values()),
        "tag_counts": dict(tag_counts),
        "english_share_of_language_words": round(tag_counts[EN] / lang_total, 4),
        "adjacent_pair_switch_rate": round(changed / pairs, 4),
        "clips_with_both_languages": round(both_langs / len(rows), 4),
        "switches_per_100_words_median_clips_ge_10": round(st.median(per100), 1),
        "per_split": {k: dict(v) for k, v in per_split.items()},
        "shorter_run_words_per_split": {k: dict(v) for k, v in shorter_words.items()},
        "english_insertion_words": dict(en_insert_words),
        "switches_per_minute_by_channel": {"min": round(rates[0], 1), "median": round(st.median(rates), 1),
                                           "max": round(rates[-1], 1)},
        "card_reference": {"english_share": 0.332, "adjacent_pair_switch_rate": 0.366,
                           "clips_with_both_languages": 0.942, "switches_per_100_tokens_median": 35.7},
    }
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
