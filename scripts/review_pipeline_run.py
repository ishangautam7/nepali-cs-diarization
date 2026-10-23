#!/usr/bin/env python3
"""EXPLORATORY, post hoc (external review, issue 2): run a full diarization pipeline over the evaluation set.

Systems, both as released, with their own default hyperparameters (no tuning, no oracle number of speakers):
  p1  pyannote Community-1 (pyannote.audio 4.0.7) at the pinned revision 3533c8cf8e369892e6b79ff1bf80f7b0286a54ee,
      loaded from a local snapshot (--model-dir); gated on Hugging Face, so fetching it needs a login.
      This is the stock pipeline (VBx clustering), not the paper's P1-AHC variant, whose AHC threshold was to be
      tuned on DISPLACE dev data that never became available (DEVIATIONS.md entry 7).
  p2  DiariZen (BUT-FIT/diarizen-wavlm-large-s80-md at 6285693ddd5b38e8229acb93f864f3d04a82bee1, embedding
      pyannote/wespeaker-voxceleb-resnet34-LM at 837717ddb9ff5507820346191109dc79c958d614), code at DiariZen
      commit 844f5555b0a98acd0931511fc641a8c5b8ba92c7; both model repos are public.
Writes <out>/hyp_<system>.rttm (appended file by file, so a stopped run resumes where it left off) and
<out>/timing_<system>.jsonl (seconds per file, for the real-time factor). Runs on CPU unless --device says otherwise.

Usage (repo root; p1 in an env with pyannote.audio 4.0.7, p2 in DiariZen's env):
  python -m scripts.review_pipeline_run --set results/review_followups/pipeline/set.json --d1 ~/data/d1 \\
      --system p1 --model-dir ~/data/models/pyannote-community-1 --out results/review_followups/pipeline
  python -m scripts.review_pipeline_run --set ... --d1 ~/data/d1 --system p2 \\
      --model-dir ~/data/models/diarizen-wavlm-large-s80-md --embedding ~/data/models/wespeaker-resnet34-LM/pytorch_model.bin --out ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PINS = {"p1": {"repo": "pyannote/speaker-diarization-community-1", "revision": "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"},
        "p2": {"repo": "BUT-FIT/diarizen-wavlm-large-s80-md", "revision": "6285693ddd5b38e8229acb93f864f3d04a82bee1",
               "embedding_repo": "pyannote/wespeaker-voxceleb-resnet34-LM",
               "embedding_revision": "837717ddb9ff5507820346191109dc79c958d614",
               "code_commit": "844f5555b0a98acd0931511fc641a8c5b8ba92c7"}}


def audio_path(f: dict, set_dir: Path, d1: Path) -> Path:
    return set_dir / f["audio"] if f["audio"] else d1 / f["parts"][0]["wav"]


def load(system: str, model_dir: Path, embedding: Path | None, device: str):
    import torch
    if system == "p1":
        from pyannote.audio import Pipeline
        pipe = Pipeline.from_pretrained(str(model_dir))
        pipe.to(torch.device(device))
        import soundfile as sf

        def run(path: Path, uri: str):
            wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
            out = pipe({"waveform": torch.from_numpy(wav)[None], "sample_rate": sr, "uri": uri})
            return out.speaker_diarization
        return run
    from diarizen.pipelines.inference import DiariZenPipeline
    pipe = DiariZenPipeline(diarizen_hub=model_dir.expanduser().absolute(), embedding_model=str(embedding))
    if device != "cpu":
        pipe.to(torch.device(device))
    return lambda path, uri: pipe(str(path), sess_name=uri)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", type=Path, required=True)
    ap.add_argument("--d1", type=Path, required=True)
    ap.add_argument("--system", choices=("p1", "p2"), required=True)
    ap.add_argument("--model-dir", type=Path, required=True, help="local snapshot of the pinned model repo")
    ap.add_argument("--embedding", type=Path, default=None, help="p2: pinned WeSpeaker ResNet34-LM pytorch_model.bin")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=3, help="torch CPU threads (leave a core for the host)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N files not yet done (timing trial)")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    files = json.loads(args.set.read_text())["files"]
    hyp_dir, timing = args.out / f"hyp_{args.system}", args.out / f"timing_{args.system}.jsonl"
    hyp_dir.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["file_id"] for line in open(timing)} if timing.exists() else set()
    # a fixed pseudo-random order (by hash), so that a partial run already covers every boundary type
    todo = sorted((fid for fid in files if fid not in done), key=lambda k: hashlib.sha1(k.encode()).hexdigest())
    if args.limit:
        todo = todo[: args.limit]
    run = load(args.system, args.model_dir, args.embedding, args.device)
    print(f"{args.system}: {len(done)} done, {len(todo)} to run", flush=True)
    for i, fid in enumerate(todo, 1):
        f = files[fid]
        t0 = time.perf_counter()
        ann = run(audio_path(f, args.set.parent, args.d1), fid)
        secs = time.perf_counter() - t0
        ann.uri = fid
        tmp = hyp_dir / f"{fid}.rttm.tmp"
        tmp.write_text(ann.to_rttm())  # may be empty: no speech found
        tmp.replace(hyp_dir / f"{fid}.rttm")  # then the timing line marks the file done
        with open(timing, "a") as fh:
            fh.write(json.dumps({"file_id": fid, "seconds": round(secs, 3), "audio_s": round(f["duration"], 3),
                                 "n_speakers": len(ann.labels())}) + "\n")
        if i % 25 == 0 or i == len(todo):
            print(f"{i}/{len(todo)} files", flush=True)
    done = {json.loads(line)["file_id"] for line in open(timing)}
    if done >= set(files):  # complete: one RTTM for the record
        (args.out / f"hyp_{args.system}.rttm").write_text(
            "".join((hyp_dir / f"{fid}.rttm").read_text() for fid in sorted(files)))
    meta = args.out / f"run_{args.system}.json"
    meta.write_text(json.dumps({"label": "EXPLORATORY (post hoc, external review issue 2)", "system": args.system,
                                "pins": PINS[args.system], "device": args.device, "threads": args.threads,
                                "torch": torch.__version__}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
