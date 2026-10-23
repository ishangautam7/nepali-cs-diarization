#!/usr/bin/env python3
"""Blind listening check for D1 switch points and controls (PLAN.md §4 step 5).

Draws 300 switch points (150 from the embedding family at L = 1.0 s, 150 from the detection family only)
and 150 controls (75 + 75 likewise) from the frozen boundary files, mixes them in random order and writes
one self-contained HTML page. Each item has a 3 s snippet with a marker at the point and a spectrogram
drawn in the browser. Three questions per item:
- a change of language at the marker?
- the same single speaker on both sides?
- the marker within about 0.1 s of a word boundary?
The page shows no transcript and no condition. Answers stay in the browser and are exported as CSV. The key
(item -> point) is written next to the page and must NOT be given to the listener.

Usage (on the box): python scripts/make_listening_check.py --points ~/data/d1/boundaries/lex_v1 \
    --root ~/data/d1 --out ~/data/d1/listening/v1
"""
from __future__ import annotations

import argparse
import base64
import csv
import gzip
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import soundfile as sf

FFMPEG = str(Path.home() / "opt/ffmpeg/bin/ffmpeg")
SNIP_S = 3.0
PLAN = (("ssdl", "emb1", 150), ("ssdl", "det_only", 150), ("ctrl", "emb1", 75), ("ctrl", "det_only", 75))


def load_points(d: Path) -> list[dict]:
    rows = []
    for p in sorted(d.glob("points_*.csv.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            rows += list(csv.DictReader(fh))
    return rows


def pool(rows: list[dict], kind: str, fam: str) -> list[dict]:
    yes = lambda r, f: r[f"elig_{f}"] == "True"
    if fam == "emb1":
        return [r for r in rows if r["kind"] == kind and yes(r, "emb1")]
    return [r for r in rows if r["kind"] == kind and yes(r, "det") and not yes(r, "emb1")]


def mp3(path: Path, start: float, stop: float, sr: int = 16000) -> bytes:
    pcm, file_sr = sf.read(path, start=int(start * sr), stop=int(stop * sr), dtype="int16")
    if file_sr != sr:
        raise ValueError(f"{path}: {file_sr} Hz")
    out = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", str(sr), "-ac", "1",
                          "-i", "pipe:0", "-c:a", "libmp3lame", "-b:a", "40k", "-f", "mp3", "pipe:1"],
                         input=pcm.tobytes(), capture_output=True, check=True,
                         env=dict(os.environ, LD_LIBRARY_PATH=str(Path.home() / "opt/ffmpeg/lib")))
    return out.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--points", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261018)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = load_points(args.points)
    manifest = {r["clip_id"]: r for r in csv.DictReader(open(args.root / "manifest.csv", encoding="utf-8"))}
    chosen = []
    for kind, fam, n in PLAN:
        p = pool(rows, kind, fam)
        chosen += [dict(r, stratum_check=f"{kind}:{fam}") for r in rng.sample(p, min(n, len(p)))]
    rng.shuffle(chosen)
    args.out.mkdir(parents=True, exist_ok=True)
    items, key = [], []
    for i, r in enumerate(chosen, 1):
        m = manifest[r["clip_id"]]
        dur, t = float(m["duration_s"]), float(r["t"])
        start = min(max(t - SNIP_S / 2, 0.0), max(dur - SNIP_S, 0.0))
        stop = min(start + SNIP_S, dur)
        item_id = f"L{i:03d}"
        items.append({"id": item_id, "marker": round(t - start, 3), "dur": round(stop - start, 3),
                      "audio": base64.b64encode(mp3(args.root / m["wav"], start, stop)).decode()})
        key.append({"item": item_id, "point_id": r["point_id"], "condition": r["stratum_check"], "split": r["split"],
                    "clip_id": r["clip_id"], "t": r["t"], "snippet_start": round(start, 3),
                    "lang_left": r["lang_left"], "lang_right": r["lang_right"], "pause_bin": r["pause_bin"]})
    with open(args.out / "key.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)
    page = TEMPLATE.replace("__ITEMS__", json.dumps(items)).replace("__N__", str(len(items)))
    (args.out / "listening_check_v1.html").write_text(page, encoding="utf-8")
    print(f"{len(items)} items -> {args.out / 'listening_check_v1.html'} "
          f"({(args.out / 'listening_check_v1.html').stat().st_size / 1e6:.1f} MB); key.csv kept separately")
    return 0


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Listening check v1</title>
<style>
:root { color-scheme: light dark; --bg: #fafafa; --fg: #1d1d1f; --muted: #5f6368; --card: #ffffff; --line: #d0d4da;
        --accent: #1a5fb4; --mark: #e01b24; }
@media (prefers-color-scheme: dark) { :root { --bg: #16181d; --fg: #e8eaed; --muted: #a0a6ad; --card: #1f232a;
        --line: #3a3f47; --accent: #78aeed; --mark: #ff7b72; } }
body { margin: 0; font: 16px/1.5 system-ui, sans-serif; background: var(--bg); color: var(--fg); }
main { max-width: 860px; margin: 0 auto; padding: 16px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px; margin: 12px 0; }
h1 { font-size: 1.3rem; margin: 0 0 4px; } .muted { color: var(--muted); }
canvas { width: 100%; height: 220px; border-radius: 6px; background: #000; display: block; }
button { font: inherit; padding: 6px 12px; border-radius: 6px; border: 1px solid var(--line); background: var(--card);
         color: var(--fg); cursor: pointer; } button:focus-visible, input:focus-visible { outline: 3px solid var(--accent); }
button.primary { background: var(--accent); color: var(--bg); border-color: var(--accent); }
fieldset { border: 1px solid var(--line); border-radius: 8px; margin: 10px 0; } legend { font-weight: 600; padding: 0 4px; }
label { margin-right: 14px; white-space: nowrap; } .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
progress { width: 100%; height: 10px; } textarea { width: 100%; font: inherit; background: var(--card); color: var(--fg);
         border: 1px solid var(--line); border-radius: 6px; } kbd { border: 1px solid var(--line); border-radius: 4px; padding: 0 4px; }
</style></head><body><main>
<h1>Listening check v1</h1>
<p class="muted">__N__ short clips from Nepali–English lectures, each 3 seconds long. The red line marks a point in time.
For each clip, answer three questions by ear. There is no transcript on purpose. It takes about an hour. Your answers are
saved in this browser as you go, so you can stop and come back. When you finish, press <b>Export answers</b> and send me the CSV file.</p>
<details class="card"><summary>How to judge (read once)</summary>
<ul><li><b>Language change:</b> does the speech switch between Nepali and English exactly at the red line (within a word or so)?
English technical words count as English.</li>
<li><b>Same speaker:</b> is it one and the same person talking on both sides of the line, with nobody else speaking?</li>
<li><b>Boundary:</b> is the red line within about a tenth of a second of the gap between two words (not in the middle of a word)?</li>
<li>Use "Unsure" whenever you cannot tell. Keys: <kbd>Space</kbd> play all, <kbd>A</kbd> before the line, <kbd>D</kbd> after it,
<kbd>1</kbd>–<kbd>3</kbd> question 1, <kbd>4</kbd>–<kbd>7</kbd> question 2, <kbd>8</kbd>–<kbd>0</kbd> question 3,
<kbd>Enter</kbd> or <kbd>→</kbd> next, <kbd>←</kbd> previous.</li></ul></details>
<div class="card">
  <div class="row"><b id="pos"></b><span class="muted" id="done"></span></div>
  <progress id="prog" max="__N__" value="0"></progress>
  <canvas id="spec" width="1200" height="300" role="img" aria-label="Spectrogram of the clip with a red line at the marked point"></canvas>
  <div class="row" style="margin-top:8px">
    <button class="primary" id="playAll">▶ Play all</button><button id="playL">Before the line</button>
    <button id="playR">After the line</button>
  </div>
  <fieldset><legend>1. Change of language at the line?</legend>
    <label><input type="radio" name="q1" value="yes"> Yes</label><label><input type="radio" name="q1" value="no"> No</label>
    <label><input type="radio" name="q1" value="unsure"> Unsure</label></fieldset>
  <fieldset><legend>2. Same single speaker on both sides?</legend>
    <label><input type="radio" name="q2" value="same"> Same speaker</label>
    <label><input type="radio" name="q2" value="different"> Different speaker</label>
    <label><input type="radio" name="q2" value="overlap_noise"> Overlap or noise</label>
    <label><input type="radio" name="q2" value="unsure"> Unsure</label></fieldset>
  <fieldset><legend>3. Line within about 0.1 s of a word boundary?</legend>
    <label><input type="radio" name="q3" value="yes"> Yes</label><label><input type="radio" name="q3" value="no"> No</label>
    <label><input type="radio" name="q3" value="unsure"> Unsure</label></fieldset>
  <label for="note" class="muted">Note (optional)</label><textarea id="note" rows="2"></textarea>
  <div class="row" style="margin-top:8px"><button id="prev">← Previous</button><button class="primary" id="next">Next →</button>
    <span style="flex:1"></span><button id="export">Export answers (CSV)</button></div>
</div></main>
<script>
const ITEMS = __ITEMS__;
const KEY = "listening_check_v1_answers";
const answers = JSON.parse(localStorage.getItem(KEY) || "{}");
let idx = Math.min(ITEMS.findIndex(it => !complete(answers[it.id])), ITEMS.length - 1);
if (idx < 0) idx = 0;
let ctx = null, buffer = null, source = null;
function complete(a) { return a && a.q1 && a.q2 && a.q3; }
function save() { localStorage.setItem(KEY, JSON.stringify(answers)); }
function b64ToBuf(b64) { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u.buffer; }
function fft(re, im) {  // in-place radix-2
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) { let b = n >> 1; for (; j & b; b >>= 1) j ^= b; j ^= b;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
  for (let len = 2; len <= n; len <<= 1) { const a = -2 * Math.PI / len;
    for (let i = 0; i < n; i += len) for (let k = 0; k < len / 2; k++) {
      const c = Math.cos(a * k), s = Math.sin(a * k), xr = re[i + k + len / 2], xi = im[i + k + len / 2];
      const tr = xr * c - xi * s, ti = xr * s + xi * c;
      re[i + k + len / 2] = re[i + k] - tr; im[i + k + len / 2] = im[i + k] - ti; re[i + k] += tr; im[i + k] += ti; } } }
function drawSpec(it) {
  const cv = document.getElementById("spec"), g = cv.getContext("2d"), x = buffer.getChannelData(0), sr = buffer.sampleRate;
  const N = 512, hop = Math.max(64, Math.floor(x.length / cv.width)), cols = Math.floor((x.length - N) / hop);
  const bins = Math.floor(N / 2 * Math.min(1, 8000 / (sr / 2))), img = g.createImageData(cols, bins), win = [];
  for (let i = 0; i < N; i++) win.push(0.5 - 0.5 * Math.cos(2 * Math.PI * i / (N - 1)));
  const mags = [], re = new Float32Array(N), im = new Float32Array(N); let mx = -1e9;
  for (let c = 0; c < cols; c++) { for (let i = 0; i < N; i++) { re[i] = x[c * hop + i] * win[i]; im[i] = 0; }
    fft(re, im); const col = new Float32Array(bins);
    for (let b = 0; b < bins; b++) { col[b] = 10 * Math.log10(re[b] * re[b] + im[b] * im[b] + 1e-10); if (col[b] > mx) mx = col[b]; }
    mags.push(col); }
  for (let c = 0; c < cols; c++) for (let b = 0; b < bins; b++) {
    const v = Math.max(0, Math.min(1, (mags[c][b] - mx + 70) / 70)), p = ((bins - 1 - b) * cols + c) * 4;
    img.data[p] = 255 * Math.pow(v, 0.8); img.data[p + 1] = 255 * Math.pow(v, 1.6); img.data[p + 2] = 255 * Math.pow(v, 3); img.data[p + 3] = 255; }
  const tmp = document.createElement("canvas"); tmp.width = cols; tmp.height = bins; tmp.getContext("2d").putImageData(img, 0, 0);
  g.imageSmoothingEnabled = true; g.drawImage(tmp, 0, 0, cv.width, cv.height);
  const xm = it.marker / it.dur * cv.width; g.strokeStyle = "#ff3b30"; g.lineWidth = 4;
  g.beginPath(); g.moveTo(xm, 0); g.lineTo(xm, cv.height); g.stroke();
}
function status() {
  const n = ITEMS.filter(i => complete(answers[i.id])).length;
  document.getElementById("pos").textContent = `Clip ${idx + 1} of ${ITEMS.length} (${ITEMS[idx].id})`;
  document.getElementById("done").textContent = ` — ${n} answered`; document.getElementById("prog").value = n;
}
async function load() {
  const it = ITEMS[idx], a = answers[it.id] || {};
  status();
  for (const q of ["q1", "q2", "q3"]) document.querySelectorAll(`input[name=${q}]`).forEach(r => r.checked = r.value === a[q]);
  document.getElementById("note").value = a.note || "";
  ctx = ctx || new (window.AudioContext || window.webkitAudioContext)();
  buffer = await ctx.decodeAudioData(b64ToBuf(it.audio)); drawSpec(it);
}
function play(from, to) { if (!buffer) return; if (source) try { source.stop(); } catch (e) {}
  if (ctx.state === "suspended") ctx.resume(); source = ctx.createBufferSource(); source.buffer = buffer;
  source.connect(ctx.destination); source.start(0, from, Math.max(0.05, to - from)); }
function record(q, v) { const id = ITEMS[idx].id; answers[id] = answers[id] || {}; answers[id][q] = v;
  answers[id].at = new Date().toISOString(); save();
  document.querySelectorAll(`input[name=${q}]`).forEach(r => r.checked = r.value === v); status(); }
document.querySelectorAll("input[type=radio]").forEach(r => r.addEventListener("change", e => record(e.target.name, e.target.value)));
document.getElementById("note").addEventListener("input", e => { const id = ITEMS[idx].id; answers[id] = answers[id] || {};
  answers[id].note = e.target.value; save(); });
const it = () => ITEMS[idx];
document.getElementById("playAll").onclick = () => play(0, it().dur);
document.getElementById("playL").onclick = () => play(Math.max(0, it().marker - 1.5), it().marker);
document.getElementById("playR").onclick = () => play(it().marker, Math.min(it().dur, it().marker + 1.5));
document.getElementById("prev").onclick = () => { if (idx > 0) { idx--; load(); } };
document.getElementById("next").onclick = () => { if (idx < ITEMS.length - 1) { idx++; load().then(() => play(0, it().dur)); } };
document.getElementById("export").onclick = () => {
  const rows = [["item", "q1_language_change", "q2_same_speaker", "q3_boundary", "note", "answered_at"]];
  for (const i of ITEMS) { const a = answers[i.id] || {}; rows.push([i.id, a.q1 || "", a.q2 || "", a.q3 || "", (a.note || "").replace(/[\r\n]+/g, " "), a.at || ""]); }
  const csv = rows.map(r => r.map(v => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\n");
  const link = document.createElement("a"); link.href = URL.createObjectURL(new Blob([csv], {type: "text/csv"}));
  link.download = "listening_check_v1_answers.csv"; link.click(); };
const keys = {"1": ["q1", "yes"], "2": ["q1", "no"], "3": ["q1", "unsure"], "4": ["q2", "same"], "5": ["q2", "different"],
              "6": ["q2", "overlap_noise"], "7": ["q2", "unsure"], "8": ["q3", "yes"], "9": ["q3", "no"], "0": ["q3", "unsure"]};
document.addEventListener("keydown", e => { if (e.target.tagName === "TEXTAREA") return;
  if (keys[e.key]) record(...keys[e.key]);
  else if (e.key === " ") { e.preventDefault(); play(0, it().dur); }
  else if (e.key === "a" || e.key === "A") document.getElementById("playL").click();
  else if (e.key === "d" || e.key === "D") document.getElementById("playR").click();
  else if (e.key === "Enter" || e.key === "ArrowRight") document.getElementById("next").click();
  else if (e.key === "ArrowLeft") document.getElementById("prev").click(); });
load();
</script></body></html>
"""

if __name__ == "__main__":
    sys.exit(main())
