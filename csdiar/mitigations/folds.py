"""H4 cross-validation folds (PREREGISTRATION_H4.md §3).

The 15 D1 train channels are scored in 5 folds of 3: `numpy.random.default_rng(20261018).permutation` of the
sorted channel IDs, taken in consecutive triples. The pilot (validation) channels are calibration only and the
test channels are descriptive; neither is in a fold.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

SEED = 20261018
N_FOLDS = 5
PER_FOLD = 3


def train_channels(manifest: Path) -> list[str]:
    with open(manifest, encoding="utf-8") as fh:
        return sorted({r["speaker_id"] for r in csv.DictReader(fh) if r["split"] == "train"})


def h4_folds(channels: list[str]) -> list[list[str]]:
    if len(channels) != N_FOLDS * PER_FOLD:
        raise ValueError(f"expected {N_FOLDS * PER_FOLD} train channels, got {len(channels)}")
    order = [sorted(channels)[i] for i in np.random.default_rng(SEED).permutation(len(channels))]
    return [order[PER_FOLD * f : PER_FOLD * (f + 1)] for f in range(N_FOLDS)]


def fitting_channels(channels: list[str], fold: int | None) -> list[str]:
    """Train channels a fold's systems are fitted on: every train channel outside the fold (all 15 if None)."""
    if fold is None:
        return sorted(channels)
    held = set(h4_folds(channels)[fold])
    return sorted(c for c in channels if c not in held)
