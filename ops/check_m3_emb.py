"""Read-only check of M3 embedding folders before a secondary_h4 run (run on the box, e.g.
`ssh <box> '~/nepali-cs-diarization/.venv-e3/bin/python - variants' < ops/check_m3_emb.py`).

Modes (first argument):
  all       the five all-data models, splits validation/train/test (before `--scope test`);
  variants  the seed-0 fold variants (distillation weights and ablations), splits validation/train
            (before `--scope cv --parts m3`); folders not written yet are listed as pending.
Per folder: whether meta.json names the SHA-256 of the run's best.pt, the run settings it records, and per
split whether the clips and window counts match E3's own split with no non-finite values. Computes no outcome
and writes nothing.
"""
import datetime
import hashlib
import json
import os
import sys

import numpy as np

BASE = os.path.expanduser("~/data/m3/emb")
REF = os.path.expanduser("~/data/d1/emb/e3_L1")
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
if MODE == "all":
    SPLITS = ("validation", "train", "test")
    EXPECTED = [f"m3_all_s{s}_w0.3_none_L1" for s in range(5)]
elif MODE == "variants":
    SPLITS = ("validation", "train")
    CONFIGS = ("w0_none", "w0.1_none", "w1_none", "w0.3_nogrl", "w0.3_nocon", "w0.3_noboth", "w0.3_nod5")
    EXPECTED = [f"m3_f{f}_s0_{c}_L1" for f in range(5) for c in CONFIGS]
else:
    sys.exit(f"unknown mode {MODE!r}: use 'all' or 'variants'")


def load_counts(path):
    """Per-clip window counts, the embedding widths, and how many clips hold non-finite values."""
    counts, dims, nonfinite = {}, set(), 0
    with np.load(path, allow_pickle=False) as z:
        for key in z.files:
            arr = z[key]
            counts[key] = arr.shape[0]
            dims.add(arr.shape[1:] if arr.ndim > 1 else ())
            if arr.dtype.kind == "f" and not np.isfinite(arr).all():
                nonfinite += 1
    return counts, dims, nonfinite


def stamp(path):
    t = datetime.datetime.fromtimestamp(os.path.getmtime(path), datetime.timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%MZ")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


ref = {s: load_counts(os.path.join(REF, s + ".npz"))[0] for s in SPLITS}
print("ref e3_L1:", {s: (len(c), sum(c.values())) for s, c in ref.items()})
ok_n, pending, problems = 0, [], []
for name in EXPECTED:
    d = os.path.join(BASE, name)
    meta_p = os.path.join(d, "meta.json")
    if not os.path.exists(meta_p) or not all(os.path.exists(os.path.join(d, s + ".npz")) for s in SPLITS):
        pending.append(name)
        continue
    with open(meta_p) as fh:
        meta = json.load(fh)
    inner = meta.get("m3_meta", {})
    ckpt = meta.get("m3_checkpoint", "")
    hash_ok = bool(ckpt) and os.path.exists(ckpt) and sha256(ckpt) == meta.get("ckpt_sha256")
    notes = []
    for s in SPLITS:
        try:
            counts, dims, bad = load_counts(os.path.join(d, s + ".npz"))
        except Exception as exc:  # report, never raise
            notes.append(f"{s} LOAD FAILED {type(exc).__name__}")
            continue
        same = set(counts) == set(ref[s]) and all(counts[k] == ref[s][k] for k in counts)
        if not same or bad or dims != {(256,)}:
            notes.append(f"{s}: clips {len(counts)} windows {sum(counts.values())} dims {dims} nonfinite {bad}")
    settings = {k: inner.get(k) for k in ("fold", "seed", "w_dist", "ablation") if k in inner}
    line = f"{name}: hash_ok {hash_ok} {settings} written {stamp(meta_p)}"
    if hash_ok and not notes:
        ok_n += 1
        print("  ok  ", line)
    else:
        problems.append(name)
        print("  BAD ", line, "|", "; ".join(notes))
print(f"{MODE}: {ok_n} ok, {len(problems)} with problems {problems}, {len(pending)} pending")
