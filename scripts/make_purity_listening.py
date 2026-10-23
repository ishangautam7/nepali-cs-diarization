#!/usr/bin/env python3
"""Listening page for the speaker-purity audit's flagged clips (PLAN.md §4 step 6d; planning/DEVIATIONS.md entry 8).

csdiar.analysis.purity_audit flags clips whose voice does not match the rest of their video, and lists a stratified
sample of 200 of them (`listening_sample` in purity_audit.json). Listening to them is descriptive. Each sampled
flagged clip is paired with one unflagged clip of the same video, of the same speech type (en / ne / cs) when the
video has one, and the two are played as A and B in random order under one question: same speaker or not? 30 control
pairs of two unflagged clips from the same videos, built the same way (the second clip of the first one's speech type
when there is one), are mixed in, so the listener's own rate of hearing a different speaker can be read beside the
flagged pairs' rate. Nothing on the page says which clip is flagged or which pairs are controls. A clip is played
whole up to 8 s (its middle 8 s if longer), as 40 kbps MP3, like the D1 listening page.

Outputs in --out: purity_listening_v1.html (self-contained; answers stay in the browser and are exported as CSV) and
key.csv (item -> kind, clips, which side is flagged), which must NOT be given to the listener.

Usage (on the box):
  .venv-pyannote4/bin/python scripts/make_purity_listening.py --root ~/data/d1 \
      --audit ~/data/d1/results/purity/purity_audit.json --out ~/data/d1/listening/purity_v1
"""
from __future__ import annotations

import argparse
import base64
import csv
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

SEED = 20261018
N_CONTROLS = 30
MAX_S = 8.0


def make_pairs(manifest: dict[str, dict], flagged: set[str], sample: list[str], n_controls: int = N_CONTROLS,
               seed: int = SEED) -> tuple[list[dict], list[str]]:
    """The shuffled item list (flagged pairs and controls), and the sampled clips that had no unflagged partner."""
    rng = random.Random(seed)
    unflagged: dict[str, list[dict]] = defaultdict(list)
    for cid in sorted(manifest):
        if cid not in flagged:
            unflagged[manifest[cid]["video_id"]].append(manifest[cid])
    pairs, skipped = [], []
    for cid in sample:
        r = manifest[cid]
        pool = unflagged[r["video_id"]]
        same_type = [u for u in pool if u["speech_type"] == r["speech_type"]]
        if not pool:
            skipped.append(cid)
            continue
        ref = rng.choice(same_type or pool)["clip_id"]
        first = rng.random() < 0.5
        pairs.append({"kind": "flagged", "a": cid if first else ref, "b": ref if first else cid,
                      "flagged_side": "A" if first else "B", "video_id": r["video_id"]})
    videos = sorted({manifest[c]["video_id"] for c in sample if len(unflagged[manifest[c]["video_id"]]) >= 2})
    if videos and n_controls:
        chosen = rng.sample(videos, n_controls) if len(videos) >= n_controls else [rng.choice(videos)
                                                                                   for _ in range(n_controls)]
        for v in chosen:
            x = rng.choice(unflagged[v])
            rest = [u for u in unflagged[v] if u["clip_id"] != x["clip_id"]]
            y = rng.choice([u for u in rest if u["speech_type"] == x["speech_type"]] or rest)
            pairs.append({"kind": "control", "a": x["clip_id"], "b": y["clip_id"], "flagged_side": "-", "video_id": v})
    rng.shuffle(pairs)
    for k, p in enumerate(pairs, 1):
        p["item"] = f"P{k:03d}"
    return pairs, skipped


def window(duration: float) -> tuple[float, float]:
    """The whole clip up to MAX_S, else its middle MAX_S."""
    if duration <= MAX_S:
        return 0.0, duration
    start = (duration - MAX_S) / 2
    return start, start + MAX_S


def page(items: list[dict]) -> str:
    """The HTML page; each item carries only its id, two audio strings and two durations."""
    return TEMPLATE.replace("__N__", str(len(items))).replace("__ITEMS__", json.dumps(items))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True, help="D1 ingest folder with manifest.csv and wav/")
    ap.add_argument("--audit", type=Path, required=True, help="purity_audit.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    spec = importlib.util.spec_from_file_location("make_listening_check", Path(__file__).with_name("make_listening_check.py"))
    lc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lc)                     # the D1 page's MP3 encoder (soundfile + ffmpeg, on the box)
    with open(args.root / "manifest.csv", encoding="utf-8", newline="") as fh:
        manifest = {r["clip_id"]: r for r in csv.DictReader(fh)}
    audit = json.loads(args.audit.read_text())
    pairs, skipped = make_pairs(manifest, {f["clip_id"] for f in audit["flagged"]}, audit["listening_sample"])
    items, key = [], []
    for p in pairs:
        item = {"id": p["item"]}
        for side in ("a", "b"):
            m = manifest[p[side]]
            start, stop = window(float(m["duration_s"]))
            item[side] = base64.b64encode(lc.mp3(args.root / m["wav"], start, stop)).decode()
            item[f"d{side}"] = round(stop - start, 3)
        items.append(item)
        key.append({"item": p["item"], "kind": p["kind"], "clip_a": p["a"], "clip_b": p["b"],
                    "flagged_side": p["flagged_side"], "video_id": p["video_id"],
                    "speech_type_a": manifest[p["a"]]["speech_type"], "speech_type_b": manifest[p["b"]]["speech_type"]})
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "key.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)
    html = args.out / "purity_listening_v1.html"
    html.write_text(page(items), encoding="utf-8")
    n_flag = sum(p["kind"] == "flagged" for p in pairs)
    print(f"{len(items)} items ({n_flag} flagged pairs, {len(items) - n_flag} controls; {len(skipped)} sampled clips "
          f"without an unflagged partner: {skipped}) -> {html} ({html.stat().st_size / 1e6:.1f} MB); key.csv kept separately")
    return 0


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Listening check 2: same speaker?</title>
<style>
:root { color-scheme: light dark; --bg: #fafafa; --fg: #1d1d1f; --muted: #5f6368; --card: #ffffff; --line: #d0d4da;
        --accent: #1a5fb4; }
@media (prefers-color-scheme: dark) { :root { --bg: #16181d; --fg: #e8eaed; --muted: #a0a6ad; --card: #1f232a;
        --line: #3a3f47; --accent: #78aeed; } }
body { margin: 0; font: 16px/1.5 system-ui, sans-serif; background: var(--bg); color: var(--fg); }
main { max-width: 860px; margin: 0 auto; padding: 16px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px; margin: 12px 0; }
h1 { font-size: 1.3rem; margin: 0 0 4px; } .muted { color: var(--muted); }
button { font: inherit; padding: 6px 12px; border-radius: 6px; border: 1px solid var(--line); background: var(--card);
         color: var(--fg); cursor: pointer; } button:focus-visible, input:focus-visible { outline: 3px solid var(--accent); }
button.primary { background: var(--accent); color: var(--bg); border-color: var(--accent); }
fieldset { border: 1px solid var(--line); border-radius: 8px; margin: 10px 0; } legend { font-weight: 600; padding: 0 4px; }
label { margin-right: 14px; white-space: nowrap; } .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
progress { width: 100%; height: 10px; } textarea { width: 100%; font: inherit; background: var(--card); color: var(--fg);
         border: 1px solid var(--line); border-radius: 6px; } kbd { border: 1px solid var(--line); border-radius: 4px; padding: 0 4px; }
</style></head><body><main>
<h1>Listening check 2: same speaker?</h1>
<p class="muted">__N__ pairs of short clips (A and B, up to 8 seconds each) from the same Nepali–English lecture video.
For each pair, say whether the main voice in A and the main voice in B belong to the same person. It takes about an hour.
Your answers are saved in this browser as you go, so you can stop and come back. When you finish, press
<b>Export answers</b> and send me the CSV file.</p>
<details class="card"><summary>How to judge (read once)</summary>
<ul><li><b>Same speaker:</b> judge the voice, not the language, topic or recording quality. The two clips can be in
different languages and still be the same person.</li>
<li><b>Unusable:</b> mark it if either clip has no speech, only music, or two or more people talking so that there is no
single main voice.</li>
<li>Use "Unsure" whenever you cannot tell. Keys: <kbd>A</kbd> play A, <kbd>D</kbd> play B, <kbd>Space</kbd> play A then B,
<kbd>1</kbd> same, <kbd>2</kbd> different, <kbd>3</kbd> unsure, <kbd>4</kbd> both usable, <kbd>5</kbd> unusable,
<kbd>Enter</kbd> or <kbd>→</kbd> next, <kbd>←</kbd> previous.</li></ul></details>
<div class="card">
  <div class="row"><b id="pos"></b><span class="muted" id="done"></span></div>
  <progress id="prog" max="__N__" value="0"></progress>
  <div class="row" style="margin-top:8px">
    <button class="primary" id="playA">▶ Play A</button><button class="primary" id="playB">▶ Play B</button>
    <button id="playAB">Play A then B</button><button id="stop">Stop</button>
  </div>
  <fieldset><legend>1. Is the main voice in A and in B the same person?</legend>
    <label><input type="radio" name="q1" value="same"> Same speaker</label>
    <label><input type="radio" name="q1" value="different"> Different speaker</label>
    <label><input type="radio" name="q1" value="unsure"> Unsure</label></fieldset>
  <fieldset><legend>2. Are both clips usable?</legend>
    <label><input type="radio" name="q2" value="usable"> Both usable</label>
    <label><input type="radio" name="q2" value="unusable"> A or B unusable (no speech, music, several people)</label></fieldset>
  <label for="note" class="muted">Note (optional)</label><textarea id="note" rows="2"></textarea>
  <div class="row" style="margin-top:8px"><button id="prev">← Previous</button><button class="primary" id="next">Next →</button>
    <span style="flex:1"></span><button id="export">Export answers (CSV)</button></div>
</div></main>
<script>
const ITEMS = __ITEMS__;
const KEY = "purity_listening_v1_answers";
const answers = JSON.parse(localStorage.getItem(KEY) || "{}");
let idx = Math.max(0, Math.min(ITEMS.findIndex(it => !complete(answers[it.id])), ITEMS.length - 1));
let ctx = null, buf = {a: null, b: null}, source = null, timer = null;
function complete(a) { return a && a.q1 && a.q2; }
function save() { localStorage.setItem(KEY, JSON.stringify(answers)); }
function b64ToBuf(b64) { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u.buffer; }
function status() {
  const n = ITEMS.filter(i => complete(answers[i.id])).length;
  document.getElementById("pos").textContent = `Pair ${idx + 1} of ${ITEMS.length} (${ITEMS[idx].id})`;
  document.getElementById("done").textContent = ` — ${n} answered`; document.getElementById("prog").value = n;
}
async function load() {
  const it = ITEMS[idx], a = answers[it.id] || {};
  stop(); status();
  for (const q of ["q1", "q2"]) document.querySelectorAll(`input[name=${q}]`).forEach(r => r.checked = r.value === a[q]);
  document.getElementById("note").value = a.note || "";
  ctx = ctx || new (window.AudioContext || window.webkitAudioContext)();
  buf = {a: await ctx.decodeAudioData(b64ToBuf(it.a)), b: await ctx.decodeAudioData(b64ToBuf(it.b))};
}
function stop() { if (timer) { clearTimeout(timer); timer = null; } if (source) try { source.stop(); } catch (e) {} }
function play(side, then) { if (!buf[side]) return; stop(); if (ctx.state === "suspended") ctx.resume();
  source = ctx.createBufferSource(); source.buffer = buf[side]; source.connect(ctx.destination); source.start();
  if (then) timer = setTimeout(() => play(then), buf[side].duration * 1000 + 400); }
function record(q, v) { const id = ITEMS[idx].id; answers[id] = answers[id] || {}; answers[id][q] = v;
  answers[id].at = new Date().toISOString(); save();
  document.querySelectorAll(`input[name=${q}]`).forEach(r => r.checked = r.value === v); status(); }
document.querySelectorAll("input[type=radio]").forEach(r => r.addEventListener("change", e => record(e.target.name, e.target.value)));
document.getElementById("note").addEventListener("input", e => { const id = ITEMS[idx].id; answers[id] = answers[id] || {};
  answers[id].note = e.target.value; save(); });
document.getElementById("playA").onclick = () => play("a");
document.getElementById("playB").onclick = () => play("b");
document.getElementById("playAB").onclick = () => play("a", "b");
document.getElementById("stop").onclick = stop;
document.getElementById("prev").onclick = () => { if (idx > 0) { idx--; load(); } };
document.getElementById("next").onclick = () => { if (idx < ITEMS.length - 1) { idx++; load().then(() => play("a", "b")); } };
document.getElementById("export").onclick = () => {
  const rows = [["item", "q1_same_speaker", "q2_usable", "note", "answered_at"]];
  for (const i of ITEMS) { const a = answers[i.id] || {}; rows.push([i.id, a.q1 || "", a.q2 || "", (a.note || "").replace(/[\r\n]+/g, " "), a.at || ""]); }
  const csv = rows.map(r => r.map(v => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\n");
  const link = document.createElement("a"); link.href = URL.createObjectURL(new Blob([csv], {type: "text/csv"}));
  link.download = "purity_listening_v1_answers.csv"; link.click(); };
const keys = {"1": ["q1", "same"], "2": ["q1", "different"], "3": ["q1", "unsure"], "4": ["q2", "usable"], "5": ["q2", "unusable"]};
document.addEventListener("keydown", e => { if (e.target.tagName === "TEXTAREA") return;
  if (keys[e.key]) record(...keys[e.key]);
  else if (e.key === " ") { e.preventDefault(); play("a", "b"); }
  else if (e.key === "a" || e.key === "A") play("a");
  else if (e.key === "d" || e.key === "D") play("b");
  else if (e.key === "Enter" || e.key === "ArrowRight") document.getElementById("next").click();
  else if (e.key === "ArrowLeft") document.getElementById("prev").click(); });
load();
</script></body></html>
"""

if __name__ == "__main__":
    sys.exit(main())
