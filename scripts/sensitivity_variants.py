#!/usr/bin/env python3
"""Variant boundary runs for the pre-registered D1 sensitivity analyses (PREREGISTRATION.md §5 items 2, 3 and 8).

Each variant changes one rule of the frozen boundary run and nothing else. It swaps one function of the frozen
module csdiar.data.boundaries for the duration of the run, the way csdiar.data.boundaries_d4 swaps in its tagger;
no frozen file is edited, and the swap is undone afterwards.

  retag_ia   item 2: "I" and capital "A", which the single-letter rule of decision 12 tags neutral, are tagged en.
  low_align  item 3: the low-alignment clips are included. clip_score returns +inf for every clip that has an aligned
             word, so the 5th-percentile threshold is +inf and no clip falls below it.
  loan_lang  item 8: loan-lexicon words are re-included with the language they are pronounced in (PLAN.md §2):
             wholly Devanagari entries become en and wholly Latin ones ne; anything else stays loan. The committed
             planning/lexicon/loans.txt is all Devanagari, so on D1 every re-included loan becomes en.
  base       no variant: the reference run on the rebuilt, unvaried boundaries (lex_v1b and its point embeddings).
             The variants run on the inputs rebuilt after the box loss (planning/DEVIATIONS.md entry 6), whose point
             tables differ slightly from the ones M2a's confirmatory run read, so each variant is compared with this
             run, not with the M2a numbers; base against M2a shows what the rebuild alone changes.

Subcommands (on the box, from the repo root, in .venv-pyannote4, on the rebuilt ~/data/d1):
  boundaries --variant V   the frozen boundary command (as in ops/rebuild_features.sh, step d1_boundaries) with V
                           swapped in, writing ~/data/d1/boundaries/lex_v1b_<V>. It refuses an existing output folder.
  analyse --variant V      the frozen M2a analysis (csdiar.analysis.confirm.primary_block: H1, H2-P0 and H3, at the
                           frozen tau of results/pilot/pilot_v2.json, seed 20261018, splits train and test) on the
                           variant's points and their point embeddings, which csdiar.models.point_embed writes
                           beforehand to ~/data/d1/point_emb/lex_v1b_<V> (for base: lex_v1b itself). Output:
                           ~/data/d1/results/sensitivity/<V>/confirm_sens_<V>.{json,md}. There is no freeze check,
                           because the inputs are by design not the frozen ones; the output is labelled a descriptive
                           sensitivity run, and H3, which reads the windows rather than the boundaries, is the same
                           for every variant. The input checks are copied from confirm.main, which is frozen.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import csdiar.data.boundaries as B  # noqa: E402
from csdiar.data.tags import EN, LOAN, NE, NEUTRAL, _char_class, _strip_punct, loan_key  # noqa: E402

VARIANTS = {"retag_ia": 2, "low_align": 3, "loan_lang": 8}
SEED = 20261018
SPLITS = ["train", "test"]
PILOT = REPO / "results/pilot/pilot_v2.json"
FREEZE_M2A = REPO / "planning/FREEZE_M2A.json"


def retag_ia(orig):
    """Tag "I" and capital "A" en where the single-letter rule made them neutral; nothing else changes."""
    def tag_word(word: str, loans: frozenset[str] = frozenset(), names: frozenset[str] = frozenset(),
                 letters_neutral: bool = False) -> str:
        t = orig(word, loans, names, letters_neutral)
        if t == NEUTRAL and letters_neutral and _strip_punct(word) in ("I", "A") and loan_key(word) not in names:
            return EN
        return t
    return tag_word


def loan_lang(orig):
    """Re-include loan-lexicon words: wholly Devanagari -> en, wholly Latin -> ne, anything else stays loan."""
    def tag_word(word: str, loans: frozenset[str] = frozenset(), names: frozenset[str] = frozenset(),
                 letters_neutral: bool = False) -> str:
        t = orig(word, loans, names, letters_neutral)
        if t != LOAN:
            return t
        classes = {_char_class(c) for c in word} - {"mark", "punct"}
        return EN if classes == {"deva"} else NE if classes == {"latin"} else t
    return tag_word


def low_align(orig):
    """Every clip with an aligned word scores +inf, so none is below the low-alignment threshold."""
    def clip_score(rec: dict) -> float | None:
        return None if orig(rec) is None else float("inf")
    return clip_score


@contextlib.contextmanager
def swapped(variant: str):
    """Swap the variant's function into csdiar.data.boundaries, and put the frozen one back afterwards."""
    name, make = {"retag_ia": ("tag_word", retag_ia), "loan_lang": ("tag_word", loan_lang),
                  "low_align": ("clip_score", low_align)}[variant]
    frozen = getattr(B, name)
    setattr(B, name, make(frozen))
    try:
        yield
    finally:
        setattr(B, name, frozen)


def boundary_argv(d1: Path, out: Path) -> list[str]:
    """The frozen d1_boundaries command of ops/rebuild_features.sh, with only --out changed."""
    return ["csdiar.data.boundaries", "--root", str(d1), "--align", str(d1 / "align"), "--vad", str(d1 / "vad"),
            "--out", str(out), "--loans", str(REPO / "planning/lexicon/loans.txt"),
            "--names", str(REPO / "planning/lexicon/names.txt"), "--splits", "validation", "train", "test"]


def run_boundaries(variant: str, d1: Path) -> Path:
    out = d1 / "boundaries" / f"lex_v1b_{variant}"
    if out.exists():
        raise SystemExit(f"refusing: {out} exists (a variant run is written once)")
    saved = sys.argv
    sys.argv = boundary_argv(d1, out)
    try:
        with swapped(variant):
            B.main()
    finally:
        sys.argv = saved
    return out


def check_pilot(path: Path) -> None:
    """The frozen tau must come from the pilot file the M2a freeze record pins."""
    want = json.loads(FREEZE_M2A.read_text())["feature_sha256"]["~/data/d1/results/pilot/pilot_v2.json"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != want:
        raise SystemExit(f"refusing: {path} is not the pilot file pinned in FREEZE_M2A.json")


def analyse(variant: str, d1: Path) -> Path:
    import numpy as np
    from csdiar.analysis import confirm as C
    check_pilot(PILOT)
    name = "lex_v1b" if variant == "base" else f"lex_v1b_{variant}"
    args = argparse.Namespace(boundaries=d1 / "boundaries" / name, point_emb=d1 / "point_emb" / name,
                              emb=d1 / "emb", windows=d1 / "windows" / "lex_v1", align=d1 / "align", pilot=PILOT,
                              out=d1 / "results" / "sensitivity" / variant, splits=list(SPLITS),
                              tag=f"sens_{variant}", seed=SEED)
    rng = np.random.default_rng(args.seed)
    taus = C.frozen_taus(args.pilot)
    pts, matches = C.load_points(args.boundaries, args.splits)
    need = {"split", "score_left", "score_right", "pause_bin", "pos_bin", "speaker_id", "video_id", "clip_id"}
    if pts and not need <= set(next(iter(pts.values()))):            # the same input checks as confirm.main
        raise SystemExit(f"refusing: point tables lack {sorted(need - set(next(iter(pts.values()))))}")
    if any(r["low_align"] and any(v for k, v in r.items() if k.startswith("elig_")) for r in pts.values()):
        raise SystemExit("refusing: a low-alignment point is marked eligible")
    what = ("reference run on the rebuilt, unvaried boundaries (planning/DEVIATIONS.md entry 6)" if variant == "base"
            else f"variant boundary run {variant} (PREREGISTRATION.md §5 item {VARIANTS[variant]})")
    out = {"tag": args.tag, "splits": args.splits, "seed": args.seed,
           "mode": f"sensitivity, {what}; descriptive, no freeze check",
           "variant": variant, "points": len(pts), "tau": taus,
           **C.primary_block(pts, matches, args.splits, args, taus, rng)}
    args.out.mkdir(parents=True, exist_ok=True)
    C.write_outputs(out, args)
    return args.out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["boundaries", "analyse"])
    ap.add_argument("--variant", choices=[*VARIANTS, "base"], required=True)
    ap.add_argument("--d1", type=Path, default=Path.home() / "data/d1")
    args = ap.parse_args(argv)
    if args.command == "boundaries" and args.variant == "base":
        raise SystemExit("refusing: base has no boundary run of its own (it reads lex_v1b)")
    out = (run_boundaries if args.command == "boundaries" else analyse)(args.variant, args.d1)
    print(f"{args.command} {args.variant}: wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
