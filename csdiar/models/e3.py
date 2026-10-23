"""E3: WavLM-Large + ECAPA-TDNN speaker embedding (UniSpeech SV checkpoint; PLAN.md §5).

The upstream script loads the checkpoint with strict=False, which would silently leave the fine-tuned
WavLM at its pre-trained values if key names drifted between s3prl versions. Here every model parameter
must be covered by the checkpoint, and every checkpoint key must be used, or loading fails.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

CKPT = Path.home() / "data/models/unispeech_sv/wavlm_large_finetune.pth"
CKPT_SHA256 = "51f07e3b94d9e0262a6a675ef5a087be3dd09e8c62e9d886827f44f82fe7f94b"


def load_e3(device: str, ckpt: Path = CKPT, check_sha: bool = True):
    """Return (model in eval mode on device, provenance dict). Input: float32 (B, T) at 16 kHz -> (B, 256)."""
    from third_party.unispeech.ecapa_tdnn import ECAPA_TDNN_SMALL

    if check_sha:
        h = hashlib.sha256()
        with open(ckpt, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 24), b""):
                h.update(block)
        if h.hexdigest() != CKPT_SHA256:
            raise ValueError(f"{ckpt}: unexpected sha256 {h.hexdigest()}")
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    # The AAM-softmax training head is not part of the embedding model; nothing else may be left unused.
    state = {k: v for k, v in state.items() if not k.startswith("loss_calculator.")}
    own = model.state_dict()
    missing = sorted(k for k in own if k not in state)
    unexpected = sorted(k for k in state if k not in own)
    mismatched = sorted(k for k in own if k in state and tuple(state[k].shape) != tuple(own[k].shape))
    if missing or unexpected or mismatched:
        raise RuntimeError(f"E3 checkpoint does not match the model: missing {missing[:5]} ({len(missing)}), "
                           f"unexpected {unexpected[:5]} ({len(unexpected)}), shape {mismatched[:5]}")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, {"source": "UniSpeech wavlm_large_finetune.pth", "sha256": CKPT_SHA256,
                   "code": "third_party/unispeech (UniSpeech 6112826, CC BY-SA 3.0)"}
