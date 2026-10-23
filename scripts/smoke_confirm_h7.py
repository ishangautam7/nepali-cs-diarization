#!/usr/bin/env python3
"""Smoke test of csdiar.analysis.confirm_h7 on the D1 pilot (validation) channels only.

H7's code must run once, after its freeze, on D4. To exercise every code path beforehand without computing any
D4 outcome, this builds a D4-shaped folder in which both "pairs" point at D1's validation features (as the
"train" and the "test" split), stubs the freeze check, and runs the H7 primary and sensitivity analyses.
Validation-channel outcomes are pilot data (PREREGISTRATION.md §2), so nothing confirmatory is computed.
Usage (on the box): .venv-pyannote4/bin/python scripts/smoke_confirm_h7.py /mnt/nvme/tmp/h7_smoke
"""
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

import csdiar.analysis.confirm_h7 as H  # noqa: E402

D1 = Path.home() / "data" / "d1"


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.symlink(src, dst)


def main() -> int:
    root = Path(sys.argv[1])
    for pair in H.PAIRS:
        base = root / pair
        for s in H.SPLITS:
            for kind in ("points", "matches"):
                link(D1 / "boundaries/lex_v1b" / f"{kind}_validation.csv.gz", base / "boundaries/lex_d4v1" / f"{kind}_{s}.csv.gz")
            for m in ("e1", "e2", "e3"):
                link(D1 / "emb" / f"{m}_L1/validation.npz", base / "emb" / f"{m}_L1" / f"{s}.npz")
                for L in ("0.5", "1", "1.5"):
                    link(D1 / "point_emb/lex_v1b" / m / f"validation_L{L}.npz",
                         base / "point_emb/lex_d4v1" / m / f"{s}_L{L}.npz")
        rows = [r for r in csv.DictReader(open(D1 / "manifest.csv", encoding="utf-8")) if r["split"] == "validation"]
        for r in rows:
            r["split"] = "train"
        with open(base / "manifest.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    taus = H.frozen_taus(D1 / "results/pilot/pilot_v2.json")
    rng = np.random.default_rng(1)
    out = {"mode": "smoke test on D1 validation channels", "primary": {p: H.run_pair(p, root, taus, rng) for p in H.PAIRS}}
    H.grade(out["primary"])
    sens, info = {}, {}
    theta, info["d1_calibration"] = H.d1_voice_threshold(D1)
    for p in H.PAIRS:
        manifest = list(csv.DictReader(open(root / p / "manifest.csv", encoding="utf-8")))
        cluster_of, info[p] = H.voice_clusters(root / p / "emb", manifest, theta)
        sens[p] = H.run_pair(p, root, taus, rng, cluster_of)
    H.grade(sens)
    out["sensitivity_voice_clusters"] = {"clustering": info, **sens}
    (root / "out").mkdir(exist_ok=True)
    H.write(out, root / "out")
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
