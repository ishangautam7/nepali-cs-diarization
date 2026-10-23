#!/usr/bin/env bash
# Acceptance for csdiar.pipelines.run_m2b (PREREGISTRATION_M2B.md §3, §4). CPU-ONLY.
#
# Re-runs the brief's acceptance (a)-(d) on pyannote.audio's bundled 2-speaker sample.wav:
#   (a) stage 1 + stage 2 at thresholds 0.7 and 0.9 reproduce p1_ahc.diarize() EXACTLY (DER 0.0, overlap
#       scored, no collar; equal segment counts), for e2 and e3;
#   (b) B2 with delta=0 (inputs from the exclusive annotation, resolve applied to both annotations) and M2 in
#       its DEPLOYED form (alpha=0, tau0=0) reproduce B0 exactly;
#   (c) stage-1 wall time per minute of audio for e2 and e3 on CPU, and stage-2 time per threshold;
#   (d) B1 with ALL language means zero reproduces B0 exactly: the slot-aware transform e'=unit(e-mu) with mu=0
#       is unit(e), and pyannote's cosine-AHC is norm-invariant, so this must not change the clustering; if it
#       does not reproduce B0, that non-invariance is REPORTED rather than forced.
#
# The GPU is full and shared: this NEVER touches it (CUDA_VISIBLE_DEVICES='' below). Models load only in
# .venv-p1 (ops/env_p1.sh), which has pyannote + s3prl for E3. Scratch outputs go under ~/data/m2b_scratch.
#
# Usage on the box, from the repo root:
#   bash ops/accept_run_m2b.sh
# Options (env):  CACHE_DIR (default ~/data/m2b_scratch/accept_cache), SAMPLE (default: bundled sample.wav),
#                 MODELS (default "e2 e3"), THRESHOLDS (default "0.7 0.9").
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=''          # CPU only; the GPU is full and shared
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
source ops/env.sh

PY=.venv-p1/bin/python
[ -x "$PY" ] || { echo "ERROR: $PY not found; build it with ops/env_p1.sh on the box" >&2; exit 2; }

CACHE_DIR="${CACHE_DIR:-$HOME/data/m2b_scratch/accept_cache}"
SAMPLE="${SAMPLE:-}"
MODELS="${MODELS:-e2 e3}"
THRESHOLDS="${THRESHOLDS:-0.7 0.9}"
mkdir -p "$CACHE_DIR"

echo "== run_m2b acceptance (CPU only) =="
echo "cache_dir=$CACHE_DIR models='$MODELS' thresholds='$THRESHOLDS'"

# nice -n 10: the box runs long GPU training + a video service; keep this CPU job low priority.
set +e
nice -n 10 "$PY" - "$CACHE_DIR" "$SAMPLE" "$MODELS" "$THRESHOLDS" <<'PY'
import json, sys, time
from pathlib import Path
import numpy as np
from csdiar.pipelines import p1_ahc, run_m2b

cache_dir = Path(sys.argv[1])
sample_arg = sys.argv[2] or None
models = sys.argv[3].split()
thresholds = [float(x) for x in sys.argv[4].split()]

def sample_path(explicit):
    if explicit:
        return Path(explicit)
    import pyannote.audio
    p = Path(pyannote.audio.__file__).parent / "sample" / "sample.wav"
    if not p.exists():
        raise SystemExit(f"sample.wav not found at {p}; set SAMPLE")
    return p

def der(ref, hyp):
    from pyannote.metrics.diarization import DiarizationErrorRate
    return float(DiarizationErrorRate(collar=0.0, skip_overlap=False)(ref, hyp))

def canon(segs):
    return sorted((round(float(s), 4), round(float(e), 4), str(l)) for s, e, l in segs)

sample = sample_path(sample_arg)
rid = run_m2b.recording_id(sample)
wav = run_m2b._read_wav_mono(sample)
dur_s = len(wav) / run_m2b.SR
print(f"# sample={sample} rid={rid} duration={dur_s:.1f}s")

report = {"a_reproduction": {}, "b_identity": {}, "c_timings": {}, "d_b1_zero_means": {}}
ok = True

for model in models:
    meta = run_m2b.run_stage1(model, sample, cache_dir, rid=rid, device="cpu", overwrite=True)
    s1pm = meta.get("stage1_s_per_min")
    report["c_timings"].setdefault(model, {})["stage1_s_per_min"] = s1pm
    print(f"[c] {model}: stage1 {s1pm:.2f} s/min  (load {meta['load_s']:.1f}s "
          f"seg+count {meta['seg_count_s']:.1f}s embed {meta['embed_s']:.1f}s)")

    cache = run_m2b.load_stage1(cache_dir, model, rid)
    pipe2 = run_m2b.load_stage2_pipeline(model, thresholds[0], device="cpu")

    for th in thresholds:
        t0 = time.time()
        diar2, excl2, _ = run_m2b.cluster_from_cache(pipe2, cache, th)
        s2 = time.time() - t0
        report["c_timings"][model].setdefault("stage2_s", {})[f"th{th:.2f}"] = s2

        ref_pipe = p1_ahc.load_pipeline(model, threshold=th, device="cpu")
        ref_ann, ref_excl = p1_ahc.diarize(ref_pipe, wav, rid)
        der_ov, der_ex = der(ref_ann, diar2), der(ref_excl, excl2)
        same = (len(list(diar2.itertracks())) == len(list(ref_ann.itertracks()))
                and len(list(excl2.itertracks())) == len(list(ref_excl.itertracks())))
        passed = der_ov < 1e-9 and der_ex < 1e-9 and same
        ok = ok and passed
        report["a_reproduction"][f"{model}@{th:.2f}"] = {
            "der_overlapping": der_ov, "der_exclusive": der_ex,
            "segments_hyp_ov": len(list(diar2.itertracks())),
            "segments_ref_ov": len(list(ref_ann.itertracks())), "stage2_s": s2, "pass": passed}
        print(f"[a] {model}@{th:.2f}: DER_ov={der_ov:.4f} DER_excl={der_ex:.4f} "
              f"stage2={s2:.2f}s -> {'PASS' if passed else 'FAIL'}")

    th = thresholds[0]
    diar0, excl0, _ = run_m2b.cluster_from_cache(pipe2, cache, th)
    base_ov = canon(run_m2b.segments_from_annotation(diar0))
    base_excl = canon(run_m2b.segments_from_annotation(excl0))

    embedder = run_m2b.SegmentEmbedder(model, rid, wav, device="cpu")
    n_win = int(dur_s / run_m2b.HOP) + 1
    post = np.tile([0.6, 0.4], (n_win, 1)).astype(np.float64)   # delta=0 / alpha=0 make the pair irrelevant

    # B2 builds its inputs from the EXCLUSIVE annotation (change 2) and applies resolve to BOTH annotations.
    resolve, _ = run_m2b.run_b2_merge(diar0, excl0, embedder, post, (0, 1), threshold=th, delta=0.0)
    b2_ov = canon(run_m2b.segments_from_annotation(run_m2b.relabel_annotation(diar0, resolve)))
    b2_ex = canon(run_m2b.segments_from_annotation(run_m2b.relabel_annotation(excl0, resolve)))
    b2_identity = all(resolve[k] == k for k in resolve) and b2_ov == base_ov and b2_ex == base_excl

    # M2's DEPLOYED form is tau0 = 0 (change 4): check alpha = 0 is an exact no-op with tau0 defaulted to 0.
    lam = run_m2b.make_lambda_closure(post, (0, 1))
    excl_segs = run_m2b.segments_from_annotation(excl0)
    m2_out, _, _ = run_m2b.run_m2_relabel(excl_segs, embedder, lam, alpha=0.0)   # tau0 defaults to 0.0
    m2_identity = canon(m2_out) == canon(excl_segs)

    passed_b = b2_identity and m2_identity
    ok = ok and passed_b
    report["b_identity"][model] = {"b2_delta0_identity": b2_identity,
                                   "m2_alpha0_tau0_0_noop": m2_identity, "pass": passed_b}
    print(f"[b] {model}: B2(delta=0) identity={b2_identity} | M2(alpha=0,tau0=0 deployed) no-op={m2_identity} "
          f"-> {'PASS' if passed_b else 'FAIL'}")

    # (d) B1 with ALL language means zero must reproduce B0 exactly (change 3). B1's slot transform is
    # e' = unit(e - mu[lang]); with mu = 0 this is unit(e). pyannote's AHC-cosine operates on cosine distance,
    # which is invariant to each embedding row's norm, so unit() must not change the clustering. We assign every
    # finite slot a language with slot_languages on a constant posterior (so no slot is -1), zero the means, and
    # cluster through the slot-aware path; the result must equal B0 at the same threshold.
    n_lid = n_win
    lid_post = np.tile([0.6, 0.4], (n_lid, 1)).astype(np.float64)     # constant, two-language, pair (0,1)
    slot_langs = run_m2b.slot_languages(cache, lid_post, (0, 1))
    dim = int(cache["embeddings"].shape[-1])
    zero_means = {0: np.zeros(dim), 1: np.zeros(dim)}
    try:
        b1_tf = run_m2b.b1_transform(zero_means, slot_langs)
        diar_b1, excl_b1, _ = run_m2b.cluster_from_cache(pipe2, cache, th, slot_transform=b1_tf)
        b1_ov = canon(run_m2b.segments_from_annotation(diar_b1))
        b1_ex = canon(run_m2b.segments_from_annotation(excl_b1))
        b1_reproduces_b0 = (b1_ov == base_ov and b1_ex == base_excl)
        der_b1_ov, der_b1_ex = der(diar0, diar_b1), der(excl0, excl_b1)
        note = None
    except Exception as exc:                 # noqa: BLE001 - report a finite-slot/label failure, don't crash
        b1_reproduces_b0 = False
        der_b1_ov = der_b1_ex = None
        note = f"{type(exc).__name__}: {exc}"
    report["d_b1_zero_means"][model] = {"reproduces_b0": b1_reproduces_b0,
                                        "der_overlapping": der_b1_ov, "der_exclusive": der_b1_ex,
                                        "n_active_slots": int((slot_langs >= 0).sum()),
                                        "n_active_unlabelled": int((slot_langs == -2).sum()),
                                        "n_inactive_but_finite": int(((slot_langs == -1) &
                                            np.isfinite(cache["embeddings"]).all(-1)).sum()),
                                        "note": note, "pass": b1_reproduces_b0}
    ok = ok and b1_reproduces_b0
    if note:
        print(f"[d] {model}: B1(zero means) -> ERROR {note} -> FAIL")
    elif b1_reproduces_b0:
        print(f"[d] {model}: B1(zero means) reproduces B0 (DER_ov={der_b1_ov:.4f} DER_excl={der_b1_ex:.4f}) "
              f"-> PASS")
    else:
        print(f"[d] {model}: B1(zero means) does NOT reproduce B0 "
              f"(DER_ov={der_b1_ov} DER_excl={der_b1_ex}); pyannote cosine-AHC is not norm-invariant -> FAIL")

print("\n# SUMMARY")
print(json.dumps(report, indent=2, default=float))
print(f"\nACCEPTANCE {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
AD_STATUS=$?
set -e

# --------------------------------------------------------------------------- #
# (e) CLI-level identity: through the `cluster` CLI on the bundled sample, B1 (zero means), B2 (delta 0) and
# M2 (alpha 0) must write RTTMs byte-identical to B0's -- BOTH the overlapping and the exclusive RTTM -- for
# e2 and e3 at 0.7 and 0.9. This is the artefact the study ships (the review's MAJOR M1: the CLI must honour
# --system), which the (a)-(d) Python checks bypass by calling the hooks directly. CPU-only, niced, scratch in
# a NEW subfolder; deletes nothing.
# --------------------------------------------------------------------------- #
E_DIR="${E_DIR:-$HOME/data/m2b_scratch/accept_cli_$(date +%Y%m%d_%H%M%S)}"
E_CACHE="$E_DIR/cache"
E_OUT="$E_DIR/out"
E_PAIR="${E_PAIR:-en,hi}"
mkdir -p "$E_CACHE" "$E_OUT"
echo
echo "== (e) CLI-level B1/B2/M2 == B0 identity =="
echo "e_dir=$E_DIR pair=$E_PAIR"

nice -n 10 env CUDA_VISIBLE_DEVICES='' "$PY" - \
    "$E_CACHE" "$E_OUT" "$SAMPLE" "$MODELS" "$THRESHOLDS" "$E_PAIR" <<'PY'
import subprocess, sys, json
from pathlib import Path
import numpy as np
from csdiar.pipelines import run_m2b

e_cache, e_out = Path(sys.argv[1]), Path(sys.argv[2])
sample_arg = sys.argv[3] or None
models = sys.argv[4].split()
thresholds = [f"{float(x):.2f}" for x in sys.argv[5].split()]
code_a, code_b = [c.strip() for c in sys.argv[6].split(",")]

def sample_path(explicit):
    if explicit:
        return Path(explicit)
    import pyannote.audio
    return Path(pyannote.audio.__file__).parent / "sample" / "sample.wav"

sample = sample_path(sample_arg)
rid = run_m2b.recording_id(sample)
CLI = [sys.executable, "-m", "csdiar.pipelines.run_m2b"]

def run(args):
    print("  $", " ".join(str(a) for a in args[3:]), flush=True)
    subprocess.run(args, check=True, env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""})

# LID grid once on the sample (the `lid` subcommand, per the brief).
run(CLI + ["lid", "--wav", str(sample), "--cache-dir", str(e_cache), "--rid", rid])
lid = run_m2b.load_lid(e_cache, rid)
i, j = run_m2b.pair_indices(lid["labels"], code_a, code_b)

ok = True
report = {}
for model in models:
    # stage-1 cache in this run's own subfolder
    run(CLI + ["cache", "--model", model, "--wav", str(sample), "--cache-dir", str(e_cache),
               "--rid", rid, "--overwrite"])
    cache = run_m2b.load_stage1(e_cache, model, rid)
    dim = int(cache["embeddings"].shape[-1])
    # zero-means .npz for the pair (one (dim,) array per VoxLingua107 code), written into this run's scratch
    means_path = e_cache / f"zero_means_{model}.npz"
    np.savez(means_path, **{code_a: np.zeros(dim, np.float32), code_b: np.zeros(dim, np.float32)})

    specs = {
        "B0": ["--system", "B0"],
        "B1": ["--system", "B1", "--pair", f"{code_a},{code_b}", "--means", str(means_path)],
        "B2": ["--system", "B2", "--pair", f"{code_a},{code_b}", "--delta", "0.0", "--wav", str(sample)],
        "M2": ["--system", "M2", "--pair", f"{code_a},{code_b}", "--alpha", "0.0", "--wav", str(sample)],
    }
    tags = {
        "B0": run_m2b.system_tag("B0"),
        "B1": run_m2b.system_tag("B1"),
        "B2": run_m2b.system_tag("B2", delta=0.0),
        "M2": run_m2b.system_tag("M2", alpha=0.0, tau0=0.0),
    }
    for name, extra in specs.items():
        run(CLI + ["cluster", "--model", model, "--rid", rid, "--cache-dir", str(e_cache),
                   "--thresholds", *thresholds, "--out", str(e_out), "--split", "dev"] + extra)

    for th in thresholds:
        b0_ov, b0_ex = run_m2b.rttm_paths(e_out, "dev", tags["B0"], float(th), rid)
        base_ov, base_ex = b0_ov.read_text(), b0_ex.read_text()
        for name in ("B1", "B2", "M2"):
            ov, ex = run_m2b.rttm_paths(e_out, "dev", tags[name], float(th), rid)
            same = (ov.read_text() == base_ov) and (ex.read_text() == base_ex)
            ok = ok and same
            report[f"{model}@{th} {name}"] = same
            print(f"[e] {model}@{th}: {name} == B0 (both RTTMs) -> {'PASS' if same else 'FAIL'}", flush=True)

print("\n# (e) SUMMARY")
print(json.dumps(report, indent=2))
print(f"\n(e) {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
E_STATUS=$?

echo
if [ "$AD_STATUS" -eq 0 ] && [ "$E_STATUS" -eq 0 ]; then
    echo "ACCEPTANCE PASS (a-d: pass, e: pass)"
    exit 0
else
    echo "ACCEPTANCE FAIL (a-d status=$AD_STATUS, e status=$E_STATUS)"
    exit 1
fi
