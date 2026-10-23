#!/usr/bin/env python3
"""P1-AHC: pyannote Community-1 with swappable embeddings and AHC-cosine clustering (PLAN.md §5).

Everything except the embedding and the clustering is Community-1 itself (pyannote.audio 4.0.7, local copy in
~/data/models/pyannote-community-1): powerset segmentation, speaker counting, overlap-aware embedding masks
(embedding_exclude_overlap), reconstruction and the annotation. Swapped in:
- the embedding of each (chunk, local speaker): E2 is Community-1's own WeSpeaker ResNet34 with its masked
  pooling; E1 (SpeechBrain ECAPA), E3 (WavLM-Large + ECAPA) and M3 checkpoints embed the masked samples of the
  chunk, as pyannote's SpeechBrain wrapper does, with segments under MIN_SAMPLES left as NaN (pyannote then
  ignores them);
- an optional transform of those embeddings (the hook B1 and M1 use);
- pyannote's AgglomerativeClustering with metric "cosine" (centroid linkage on unit vectors), the threshold given
  per run (PLAN.md §7: grid 0.30-1.00, tuned on dev) and min_cluster_size 12 (pyannote's AHC default).
B2's affinity offset and M2's local relabelling act after or inside clustering and are added with the M2b
supplement. Real speech only.
Usage (on the box, after `source ops/env.sh`; .venv-e3 for E3/M3, .venv-pyannote4 for E1/E2):
  python -m csdiar.pipelines.p1_ahc --model e2 --threshold 0.7 --wav a.wav b.wav --out rttm/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

SR = 16000
MIN_SAMPLES = 4000                       # 0.25 s of masked speech, below which an embedding is NaN
MIN_CLUSTER_SIZE = 12
MODELS = Path.home() / "data/models"
COMMUNITY1 = MODELS / "pyannote-community-1"


class MaskedEmbedding:
    """pyannote-compatible embedding: __call__(waveforms (B, 1, T), masks (B, F)) -> (B, D) numpy, NaN when the
    masked speech is shorter than MIN_SAMPLES. `transform` maps finite (n, D) embeddings to (n, D)."""

    metric = "cosine"
    sample_rate = SR
    min_num_samples = MIN_SAMPLES

    def __init__(self, model: str, device: str, transform=None):
        self.device = torch.device(device)
        self.transform = transform
        if model == "e1":
            from speechbrain.inference.speaker import EncoderClassifier
            enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                 savedir=str(MODELS / "spkrec-ecapa-voxceleb"),
                                                 run_opts={"device": str(self.device)})
            self.dimension = 192
            self._batched = lambda x, lens: enc.encode_batch(x, wav_lens=lens)[:, 0].float().cpu().numpy()
            self._single = None
        else:
            if model == "e3":
                from csdiar.models.e3 import load_e3
                net, _ = load_e3(str(self.device))
            elif model.startswith("m3:"):
                from csdiar.mitigations.m3_train import load_m3
                net, _ = load_m3(Path(model[3:]).expanduser(), str(self.device))
            else:
                raise ValueError(model)
            self.dimension = 256
            self._batched = None
            self._single = lambda x: net(x).float().cpu().numpy()

    def to(self, device):                # pyannote moves its embedding with the pipeline
        return self

    @torch.inference_mode()
    def __call__(self, waveforms: torch.Tensor, masks: torch.Tensor | None = None) -> np.ndarray:
        wav = waveforms[:, 0].float()
        n, t = wav.shape
        if masks is None:
            keep = torch.ones_like(wav, dtype=torch.bool)
        else:
            keep = F.interpolate(masks[:, None].float(), size=t, mode="nearest")[:, 0] > 0.5
        segs = [wav[i][keep[i]] for i in range(n)]
        out = np.full((n, self.dimension), np.nan, np.float32)
        ok = [i for i in range(n) if len(segs[i]) >= MIN_SAMPLES]
        if not ok:
            return out
        if self._batched is not None:
            lens = torch.tensor([len(segs[i]) for i in ok], dtype=torch.float32)
            x = torch.zeros(len(ok), int(lens.max()))
            for j, i in enumerate(ok):
                x[j, : len(segs[i])] = segs[i].cpu()
            emb = self._batched(x.to(self.device), (lens / lens.max()).to(self.device))
        else:                            # E3 / M3 pool over the whole input: one segment per forward, no padding
            emb = np.concatenate([self._single(segs[i][None].to(self.device)) for i in ok])
        if self.transform is not None:
            emb = self.transform(emb)
        out[ok] = emb
        return out


def load_pipeline(model: str, threshold: float, device: str = "cuda:0", transform=None):
    """Community-1 with the embedding of `model` (e1, e2, e3, m3:<checkpoint>) and AHC-cosine at `threshold`."""
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.clustering import AgglomerativeClustering
    pipe = Pipeline.from_pretrained(str(COMMUNITY1))
    pipe.to(torch.device(device))
    if model != "e2":
        pipe._embedding = MaskedEmbedding(model, device, transform)
    elif transform is not None:
        native = pipe._embedding
        class _T:                        # E2 keeps pyannote's masked pooling; only the output is transformed
            metric, sample_rate, dimension = native.metric, native.sample_rate, native.dimension
            min_num_samples = native.min_num_samples
            def __call__(self, w, masks=None):
                e = native(w, masks=masks)
                fin = np.isfinite(e).all(1)
                e[fin] = transform(e[fin])
                return e
            def to(self, device):
                return self
        pipe._embedding = _T()
    clustering = AgglomerativeClustering(metric="cosine")
    clustering.instantiate({"method": "centroid", "threshold": float(threshold),
                            "min_cluster_size": MIN_CLUSTER_SIZE})
    pipe.clustering = clustering
    pipe._expects_num_speakers = False
    return pipe


def diarize(pipe, wav: np.ndarray, uri: str):
    """Exclusive and overlapping diarization of a mono 16 kHz waveform."""
    out = pipe({"waveform": torch.from_numpy(wav.astype(np.float32))[None], "sample_rate": SR, "uri": uri})
    return out.speaker_diarization, out.exclusive_speaker_diarization


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="e1 | e2 | e3 | m3:<checkpoint>")
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--wav", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    pipe = load_pipeline(args.model, args.threshold)
    args.out.mkdir(parents=True, exist_ok=True)
    for p in args.wav:
        wav, sr = sf.read(p, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        if sr != SR:
            raise ValueError(f"{p}: expected 16 kHz")
        ann, _ = diarize(pipe, wav, p.stem)
        with open(args.out / f"{p.stem}.rttm", "w") as fh:
            ann.write_rttm(fh)
        print(f"{p.stem}: {len(ann.labels())} speakers, {len(list(ann.itertracks()))} segments", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
