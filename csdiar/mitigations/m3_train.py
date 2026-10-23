#!/usr/bin/env python3
"""M3 LanguageAdversarialFineTuner: training one model (PREREGISTRATION_H4.md §7).

Starts from E3 (WavLM-Large + ECAPA-TDNN, UniSpeech SV checkpoint). WavLM stays frozen; its layer weights, the
ECAPA-TDNN and two new heads are trained:
  L = L_AAM + L_SupCon + L_GRL + w_dist * L_dist
- L_AAM: additive angular margin softmax (m 0.2, s 30) over the fold's D1 speakers and the D5 speakers, all items;
- L_SupCon: supervised contrastive loss (temperature 0.1) over the D1 items, positives = same speaker;
- L_GRL: language head (ne / en / mixed) behind a gradient-reversal layer, D1 items only; the reversal
  coefficient ramps linearly from 0 to 1 over the first 30% of steps;
- L_dist: squared distance between unit embeddings and those of the frozen E3 head on the same WavLM features.
Batch: 12 D1 speakers x 4 windows (2 ne + 2 en, mixed substituted) + 48 D5 speakers x 1 crop, one window length
per batch from {1.0, 1.25, 1.5, 1.75, 2.0} s. AdamW (2e-5 layer weights + ECAPA, 1e-3 heads, weight decay 0.01;
the first 500 steps train only the new heads while batch-norm statistics adapt), 500 warm-up steps, cosine decay
to 0 at 6,000 steps, fp32 (E3's numerics), clipping at 5. Every 500 steps: dev EER on the
calibration-channel trials (m3_data dev); patience 3; the best checkpoint among steps >= 500 is kept.
The H4 statistic is never computed here.

Output: <out>/<tag>/best.pt (trainable state), log.jsonl, meta.json.
Usage (on the box, .venv-e3):
  .venv-e3/bin/python -m csdiar.mitigations.m3_train --fold 0 --seed 0 --w-dist 0.3 --out ~/data/m3/runs
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from csdiar.mitigations.folds import fitting_channels, h4_folds, train_channels

SR = 16000
HOP = 0.1
STEPS = 6000
WARMUP = 500
EVAL_EVERY = 500
PATIENCE = 3
GRL_RAMP = 0.3
D1_SPEAKERS, D1_PER_SPEAKER, D5_ITEMS = 12, 4, 48
LR_BODY, LR_HEADS, WD, CLIP = 2e-5, 1e-3, 0.01, 5.0     # body lr and warm-up chosen on dev (decision 28)
HEAD_WARMUP = 500
AAM_M, AAM_S, SUPCON_T = 0.2, 30.0, 0.1
ABLATIONS = ("none", "nogrl", "nocon", "noboth", "nod5")


# ------------------------------------------------------------------ model ------------------------------------
def wavlm_hidden(base, wav: torch.Tensor, bf16: bool = False) -> torch.Tensor:
    """Stacked WavLM hidden states (layers, B, frames, 1024) for E3's layer weighting; WavLM is frozen.
    fp32 by default, the numerics of E3's frozen dense features (csdiar/models/embed.py)."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        hs = base.feature_extract([w for w in wav])[base.feature_selection]
    return torch.stack([h.float() for h in hs], 0)


def head(m, hs: torch.Tensor) -> torch.Tensor:
    """ECAPA_TDNN.forward after the feature extractor (third_party/unispeech/ecapa_tdnn.py), on given hidden states."""
    w = F.softmax(m.feature_weight, dim=-1)[:, None, None, None]
    x = torch.transpose((w * hs).sum(0), 1, 2) + 1e-6
    x = m.instance_norm(x)
    out1 = m.layer1(x)
    out2 = m.layer2(out1)
    out3 = m.layer3(out2)
    out4 = m.layer4(out3)
    out = F.relu(m.conv(torch.cat([out2, out3, out4], dim=1)))
    return m.linear(m.bn(m.pooling(out)))


def trainable_state(base) -> dict:
    return {k: v.detach().cpu() for k, v in base.state_dict().items() if not k.startswith("feature_extract.")}


def load_m3(ckpt: Path, device: str):
    """E3 with an M3 checkpoint's trained state; returns (model in eval mode, provenance)."""
    from csdiar.models.e3 import load_e3
    base, prov = load_e3(device)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = base.load_state_dict(state["model"], strict=False)
    bad = [k for k in missing if not k.startswith("feature_extract.")]
    if bad or unexpected:
        raise RuntimeError(f"M3 checkpoint mismatch: missing {bad[:5]}, unexpected {list(unexpected)[:5]}")
    return base.eval(), {**prov, "m3_checkpoint": str(ckpt), "m3_meta": state.get("meta", {})}


class AAM(nn.Module):
    def __init__(self, dim: int, n: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n, dim) * 0.01)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cos = F.linear(F.normalize(x), F.normalize(self.w)).clamp(-1 + 1e-7, 1 - 1e-7)
        target = torch.cos(torch.acos(cos) + AAM_M)
        logits = AAM_S * torch.where(F.one_hot(y, cos.shape[1]).bool(), target, cos)
        return F.cross_entropy(logits, y)


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.coeff * g, None


def supcon(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al. 2020) on unit vectors z with labels y."""
    z = F.normalize(z)
    sim = z @ z.T / SUPCON_T
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (y[:, None] == y[None, :]) & ~eye
    n_pos = pos.sum(1)
    ok = n_pos > 0
    return -((logp * pos).sum(1)[ok] / n_pos[ok]).mean()


def eer(scores_t: np.ndarray, scores_n: np.ndarray) -> float:
    s = np.concatenate([scores_t, scores_n])
    lab = np.concatenate([np.ones(len(scores_t)), np.zeros(len(scores_n))])
    order = np.argsort(-s)
    lab = lab[order]
    tpr = np.cumsum(lab) / lab.sum()
    fpr = np.cumsum(1 - lab) / (1 - lab).sum()
    i = int(np.argmin(np.abs((1 - tpr) - fpr)))
    return float(((1 - tpr[i]) + fpr[i]) / 2)


# ------------------------------------------------------------------ data -------------------------------------
class Data:
    def __init__(self, m3: Path, manifest: Path, fold: int | None, use_d5: bool):
        z = np.load(m3 / "d1_windows.npz")
        self.d1_audio = np.memmap(m3 / "d1_audio.int16", dtype=np.int16, mode="r")
        self.d1_off = np.load(m3 / "d1_offsets.npy")
        n_samples = {r["clip_id"]: int(r["n_samples"]) for r in csv.DictReader(open(manifest, encoding="utf-8"))}
        if len(self.d1_off) != len(z["clip_id"]) + 1 or \
                not np.array_equal(np.diff(self.d1_off), [n_samples[c] for c in z["clip_id"]]):
            raise RuntimeError("d1_audio.int16 does not match d1_windows.npz (rebuild the memmap)")
        self.lengths = [float(x) for x in z["lengths"]]
        fit = set(fitting_channels(train_channels(manifest), fold))
        clip_ok = (z["clip_split"] == "train") & ~z["clip_flagged"] & np.isin(
            np.array([s.split("::")[0] for s in z["clip_speaker"]]), list(fit))
        spk_names = sorted(set(z["clip_speaker"][clip_ok]))
        self.d1_speakers = spk_names
        spk_idx = {s: i for i, s in enumerate(spk_names)}
        wc = z["win_clip"]
        sel = clip_ok[wc]
        self.win_clip, self.win_k = wc[sel], z["win_k"][sel].astype(np.int64)
        self.win_len, self.win_lab = z["win_len"][sel], z["win_label"][sel]
        self.win_spk = np.array([spk_idx[s] for s in z["clip_speaker"][self.win_clip]])
        self.index = {}
        for s in range(len(spk_names)):
            for li in range(len(self.lengths)):
                for lab in range(3):
                    self.index[(s, li, lab)] = np.nonzero((self.win_spk == s) & (self.win_len == li) &
                                                          (self.win_lab == lab))[0]
        self.use_d5 = use_d5
        self.d5_speakers: list[str] = []
        if use_d5:
            rows = list(csv.DictReader(open(Path.home() / "data/d5/manifest.csv", encoding="utf-8")))
            self.d5_audio = np.memmap(m3 / "d5_audio.int16", dtype=np.int16, mode="r")
            self.d5_off = np.load(m3 / "d5_offsets.npy")
            if len(self.d5_off) != len(rows) + 1 or not np.allclose(np.diff(self.d5_off) / SR,
                                                                    [float(r["duration"]) for r in rows], atol=2e-3):
                raise RuntimeError("d5_audio.int16 does not match the D5 manifest (rebuild the memmap)")
            self.d5_speakers = sorted({r["speaker_id"] for r in rows})
            d5i = {s: i for i, s in enumerate(self.d5_speakers)}
            self.d5_spk = np.array([d5i[r["speaker_id"]] for r in rows])
            self.d5_by = [np.nonzero(self.d5_spk == s)[0] for s in range(len(self.d5_speakers))]
        self.n_classes = len(self.d1_speakers) + len(self.d5_speakers)
        self.substituted = {"mixed": 0, "other_language": 0}

    def d1_crop(self, w: int, n: int) -> np.ndarray:
        c = self.win_clip[w]
        a = self.d1_off[c] + int(round(self.win_k[w] * HOP * SR))
        if a + n > self.d1_off[c + 1]:
            raise ValueError(f"window {w} runs past the end of its clip")
        return np.asarray(self.d1_audio[a : a + n], dtype=np.float32) / 32768.0

    def batch(self, rng: np.random.Generator):
        li = int(rng.integers(len(self.lengths)))
        n = int(round(self.lengths[li] * SR))
        wavs, spk, lang = [], [], []
        able = [s for s in range(len(self.d1_speakers)) if any(len(self.index[(s, li, x)]) for x in range(3))]
        if len(able) < D1_SPEAKERS:
            raise RuntimeError(f"only {len(able)} D1 speakers have windows of {self.lengths[li]} s")
        speakers = rng.choice(able, size=D1_SPEAKERS, replace=False)
        for s in speakers:
            for lab in (0, 0, 1, 1):
                pool = self.index[(s, li, lab)]
                if len(pool) == 0:
                    pool = self.index[(s, li, 2)]
                    self.substituted["mixed"] += 1
                if len(pool) == 0:
                    pool = np.concatenate([self.index[(s, li, x)] for x in range(3)])
                    self.substituted["other_language"] += 1
                w = int(pool[rng.integers(len(pool))])
                wavs.append(self.d1_crop(w, n)); spk.append(int(s)); lang.append(int(self.win_lab[w]))
        n_d1 = len(wavs)
        if self.use_d5:
            for s in rng.choice(len(self.d5_speakers), size=D5_ITEMS, replace=False):
                u = int(self.d5_by[s][rng.integers(len(self.d5_by[s]))])
                a0, a1 = int(self.d5_off[u]), int(self.d5_off[u + 1])
                st = a0 + int(rng.integers(0, max(1, a1 - a0 - n + 1)))
                wav = np.asarray(self.d5_audio[st : st + n], dtype=np.float32) / 32768.0
                wavs.append(np.pad(wav, (0, n - len(wav)))); spk.append(len(self.d1_speakers) + int(s)); lang.append(-1)
        return np.stack(wavs), np.array(spk), np.array(lang), n_d1


class Dev:
    """Dev trials (m3_data dev) on the calibration channels: 1 s windows at clean-window positions."""

    def __init__(self, m3: Path, data: Data):
        z = np.load(m3 / "dev_trials.npz")
        w = np.load(m3 / "d1_windows.npz")
        pos = {c: i for i, c in enumerate(w["clip_id"])}
        crops = []
        for c, k in zip(z["clip_id"], z["k"]):
            a = data.d1_off[pos[c]] + int(k) * int(HOP * SR)
            if a + SR > data.d1_off[pos[c] + 1]:
                raise ValueError(f"dev window {c}:{k} runs past the end of its clip")
            crops.append(np.asarray(data.d1_audio[a : a + SR], dtype=np.float32) / 32768.0)
        self.crops = np.stack(crops)
        self.targets, self.nontargets = z["targets"], z["nontargets"]

    @torch.no_grad()
    def score(self, base, device: str) -> float:
        base.eval()
        embs = []
        for b in range(0, len(self.crops), 256):
            x = torch.from_numpy(self.crops[b : b + 256]).to(device)
            embs.append(F.normalize(head(base, wavlm_hidden(base, x))).cpu().numpy())
        e = np.concatenate(embs)
        cos = lambda p: (e[p[:, 0]] * e[p[:, 1]]).sum(1)
        return eer(cos(self.targets), cos(self.nontargets))


# ------------------------------------------------------------------ training ---------------------------------
def lr_factor(step: int, total: int) -> float:
    if step < WARMUP:
        return (step + 1) / WARMUP
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - WARMUP) / max(1, total - WARMUP))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fold", type=int, choices=range(5))
    g.add_argument("--all", action="store_true", help="the all-data model: every train channel")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--w-dist", type=float, default=0.3)
    ap.add_argument("--ablation", choices=ABLATIONS, default="none")
    ap.add_argument("--m3", type=Path, default=Path.home() / "data/m3")
    ap.add_argument("--manifest", type=Path, default=Path.home() / "data/d1/manifest.csv")
    ap.add_argument("--out", type=Path, default=Path.home() / "data/m3/runs")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    ap.add_argument("--smoke", action="store_true", help="short run for speed/memory; never a result")
    ap.add_argument("--bf16-wavlm", action="store_true", help="bf16 WavLM features in training (off = fp32)")
    ap.add_argument("--lr-body", type=float, default=LR_BODY, help="lr of the layer weights and ECAPA-TDNN")
    ap.add_argument("--head-warmup", type=int, default=HEAD_WARMUP, help="steps with the body frozen")
    ap.add_argument("--stop-after", type=int, default=0, help="dev sweep: stop at this step, schedule unchanged")
    args = ap.parse_args()
    if args.stop_after and not args.smoke:
        ap.error("--stop-after is for --smoke dev sweeps only")
    if not args.smoke and (args.lr_body != LR_BODY or args.head_warmup != HEAD_WARMUP or args.steps != STEPS
                           or args.eval_every != EVAL_EVERY or args.bf16_wavlm):
        ap.error("training settings are frozen (PREREGISTRATION_H4.md §7); overrides need --smoke")

    fold = None if args.all else args.fold
    tag = f"{'all' if fold is None else f'f{fold}'}_s{args.seed}_w{args.w_dist:g}_{args.ablation}"
    if args.smoke:
        tag = f"smoke_lr{args.lr_body:g}_hw{args.head_warmup}_" + tag
    out = args.out / tag
    if (out / "meta.json").exists() and not args.smoke:
        print(f"{tag}: finished earlier (meta.json exists), skipping", flush=True)
        return 0
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    from csdiar.models.e3 import load_e3
    base, prov = load_e3(device)
    fe = base.feature_extract
    base.feature_extract = None
    frozen = copy.deepcopy(base).eval()
    base.feature_extract = fe
    for p in frozen.parameters():
        p.requires_grad = False

    data = Data(args.m3, args.manifest, fold, use_d5=args.ablation != "nod5")
    dev = Dev(args.m3, data)
    aam = AAM(256, data.n_classes).to(device)
    lang_head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 3)).to(device)
    body = [p for n, p in base.named_parameters() if not n.startswith("feature_extract.") and p.requires_grad]
    if not body:
        raise RuntimeError("no trainable E3 body parameters (layer weights / ECAPA-TDNN)")
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr_body},
                             {"params": list(aam.parameters()) + list(lang_head.parameters()), "lr": LR_HEADS}],
                            weight_decay=WD)
    use_grl = args.ablation not in ("nogrl", "noboth")
    use_con = args.ablation not in ("nocon", "noboth")

    meta = {"tag": tag, "fold": fold, "seed": args.seed, "w_dist": args.w_dist, "ablation": args.ablation,
            "steps": args.steps, "bf16_wavlm": args.bf16_wavlm, "lr_body": args.lr_body,
            "head_warmup": args.head_warmup, "d1_speakers": data.d1_speakers,
            "n_d5_speakers": len(data.d5_speakers), "e3": prov, "dev_eer": {}}
    meta["dev_eer"]["0"] = dev.score(base, device)   # E3 itself; never selectable
    log = open(out / "log.jsonl", "w", encoding="utf-8")
    best, best_step, bad, t0 = float("inf"), None, 0, time.time()
    for step in range(args.steps):
        # Exactly the procedure evaluated in the dev sweep (decision 28): train mode throughout, so during the
        # classifier warm-up the ECAPA batch-norm statistics adapt to the batches while the body's weights stay
        # fixed (lr 0; Adam moments still accumulate). WavLM stays frozen and in eval mode.
        base.train()
        base.feature_extract.eval()
        for gi, lr in enumerate((args.lr_body, LR_HEADS)):
            opt.param_groups[gi]["lr"] = lr * lr_factor(step, args.steps)
        if step < args.head_warmup:
            opt.param_groups[0]["lr"] = 0.0
        wav, spk, lang, n_d1 = data.batch(rng)
        x = torch.from_numpy(wav).to(device)
        y = torch.from_numpy(spk).to(device)
        hs = wavlm_hidden(base, x, bf16=args.bf16_wavlm)
        e = head(base, hs)
        with torch.no_grad():
            e0 = head(frozen, hs)
        loss_aam = aam(e, y)
        loss_con = supcon(e[:n_d1], y[:n_d1]) if use_con else torch.zeros((), device=device)
        coeff = min(1.0, step / (GRL_RAMP * args.steps))
        lab = torch.from_numpy(lang[:n_d1]).to(device)
        loss_grl = (F.cross_entropy(lang_head(Reverse.apply(e[:n_d1], coeff)), lab)
                    if use_grl else torch.zeros((), device=device))
        loss_dist = ((F.normalize(e) - F.normalize(e0)) ** 2).sum(1).mean()
        loss = loss_aam + loss_con + loss_grl + args.w_dist * loss_dist
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(body + list(aam.parameters()) + list(lang_head.parameters()), CLIP)
        opt.step()
        rec = {"step": step, "loss": float(loss), "aam": float(loss_aam), "con": float(loss_con),
               "grl": float(loss_grl), "dist": float(loss_dist), "len": wav.shape[1] / SR}
        if (step + 1) % args.eval_every == 0:
            d = dev.score(base, device)
            meta["dev_eer"][str(step + 1)] = d
            rec["dev_eer"] = d
            if d < best:
                best, best_step, bad = d, step + 1, 0
                torch.save({"model": trainable_state(base), "meta": {k: meta[k] for k in
                            ("tag", "fold", "seed", "w_dist", "ablation")} | {"step": step + 1, "dev_eer": d}},
                           out / "best.pt")
            else:
                bad += 1
            rec["elapsed_s"] = round(time.time() - t0, 1)
            print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        if bad >= PATIENCE or (args.stop_after and step + 1 >= args.stop_after):
            break
    meta.update({"best_step": best_step, "best_dev_eer": best, "stopped_at": step + 1,
                 "elapsed_s": round(time.time() - t0, 1),
                 "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                 "batch_substitutions": data.substituted,
                 "scored_channels": None if fold is None else h4_folds(train_channels(args.manifest))[fold],
                 "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "reproducibility": "seeded; cuDNN backward is not bitwise deterministic"})
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("tag", "best_step", "best_dev_eer", "stopped_at", "elapsed_s",
                                           "peak_mem_gb")} | {"e3_dev_eer": meta["dev_eer"]["0"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
