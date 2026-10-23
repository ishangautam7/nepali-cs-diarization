#!/usr/bin/env python3
"""Find candidate Nepali–English conversation podcasts that publish audio through public RSS feeds (D2-P).

1. Query the public Apple Podcasts directory (iTunes Search API, no key) with Nepali-related terms.
2. Fetch each show's RSS feed and summarise it: episode count, median length, newest episode, feed language,
   and a few recent titles.
3. Keep shows with at least 10 audio episodes of 20 minutes or more (median), and write a ranked CSV.
Nothing is downloaded except the RSS text itself; episodes are chosen by a person before any download.

Usage:
  python scripts/find_podcasts.py --out data/podcasts/candidates.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path

TERMS = ["nepali podcast", "nepal podcast", "nepali", "nepal", "kathmandu", "नेपाली", "नेपाली पोडकास्ट",
         "nepali interview", "nepali talk show", "nepali conversation"]
COUNTRIES = ["np", "us", "in", "gb", "au"]
ITUNES = "{http://www.itunes.com/dtds/podcast-1.0.dtd}"
UA = {"User-Agent": "csdiar-research/0.1 (podcast discovery; contact via repository)"}


def get(url: str, limit: int = 8_000_000, timeout: int = 20) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(limit)


def search() -> dict[int, dict]:
    shows: dict[int, dict] = {}
    for term in TERMS:
        for cc in COUNTRIES:
            q = urllib.parse.urlencode({"term": term, "media": "podcast", "limit": 200, "country": cc})
            try:
                data = json.loads(get(f"https://itunes.apple.com/search?{q}"))
            except Exception as exc:
                print(f"search failed: {term!r} {cc}: {exc}", file=sys.stderr)
                continue
            for r in data.get("results", []):
                if r.get("feedUrl") and r.get("collectionId") not in shows:
                    shows[r["collectionId"]] = {"show": r.get("collectionName", ""), "author": r.get("artistName", ""),
                                                "feed": r["feedUrl"], "genre": r.get("primaryGenreName", ""),
                                                "apple_count": r.get("trackCount", 0)}
            time.sleep(3)  # the directory API allows about 20 calls a minute
    return shows


def seconds(text: str | None) -> float | None:
    if not text:
        return None
    text = text.strip()
    if re.fullmatch(r"\d+", text):
        return float(text)
    parts = text.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    total = 0.0
    for n in nums:
        total = total * 60 + n
    return total


def summarise(feed_url: str) -> dict:
    root = ET.fromstring(get(feed_url))
    ch = root.find("channel")
    if ch is None:
        raise ValueError("no channel element")
    items = ch.findall("item")
    audio = [it for it in items if (enc := it.find("enclosure")) is not None
             and (enc.get("type", "").startswith("audio") or enc.get("url", "").split("?")[0].endswith((".mp3", ".m4a")))]
    durs = [d for it in audio if (d := seconds(it.findtext(f"{ITUNES}duration"))) is not None]
    dates = []
    for it in audio:
        try:
            dates.append(parsedate_to_datetime(it.findtext("pubDate")))
        except Exception:
            pass
    return {"feed_language": (ch.findtext("language") or "").strip(), "episodes": len(audio),
            "median_min": round(st.median(durs) / 60, 1) if durs else None,
            "newest": max(dates).date().isoformat() if dates else "",
            "titles": " | ".join((it.findtext("title") or "").strip()[:70] for it in audio[:4])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    shows = search()
    print(f"{len(shows)} shows with RSS feeds found", file=sys.stderr)
    rows = []
    for sid, s in shows.items():
        try:
            s.update(summarise(s["feed"]))
        except Exception as exc:
            s.update({"feed_language": "", "episodes": 0, "median_min": None, "newest": "", "titles": f"ERROR {exc}"[:120]})
        s["apple_id"] = sid
        rows.append(s)
        time.sleep(1)
    keep = [r for r in rows if r["episodes"] >= 10 and (r["median_min"] or 0) >= 20]
    keep.sort(key=lambda r: (r["newest"], r["episodes"]), reverse=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["show", "author", "genre", "feed_language", "episodes", "median_min", "newest", "titles", "feed", "apple_id"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(keep)
    print(f"{len(keep)} shows kept (>= 10 audio episodes, median >= 20 min) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
