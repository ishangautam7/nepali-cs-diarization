#!/usr/bin/env python3
"""Score the two listening checks (PLAN.md §4 steps 5 and 6d; planning/DEVIATIONS.md entries 8 and 10). Descriptive.

Written, reviewed and hashed (DEVIATIONS.md entry 10) before any answer was read. Every choice below is fixed here.

d1      The 450-item blind check of scripts/make_listening_check.py. Inputs: the answers CSV the page exports, the
        page's key.csv, the point tables the page was drawn from (lex_v1b) and results/confirm/confirm_m2a.json.
        - Answers allowed: q1 (language change at the line) yes/no/unsure; q2 (one speaker on both sides)
          same/different/overlap_noise/unsure; q3 (line within about 0.1 s of a word boundary) yes/no/unsure.
          A blank or any other value stops the run.
        - Weights: each item counts with the inverse of its stratum's sampling fraction (stratum = the page's four
          pools, ssdl/ctrl x emb1/det_only, rebuilt with make_listening_check.pool on the same tables), so a rate
          estimates a share of points, not of the sample.
        - Domain: the rates below use only items of the confirmatory splits (train, test), the 17 channels whose
          dFAR M2a reports. The page also drew from the pilot split; those items appear only in the per-stratum
          tables, which describe the whole sample.
        - Primary cell (p0L1 switch points against det_w1 controls, as in pilot.h2_cell). Among switch items
          eligible for p0L1: pi = weighted share answered "no language change"; delta = weighted share answered
          "different" or "overlap_noise"; boundary_ok_p0L1 = weighted share answered "within 0.1 s". Among control
          items eligible for det_w1: kappa = weighted share answered "language change". boundary_ok_all_switch is
          the same share over all switch items. An item answered "unsure" on a question is left out of that
          question's rates, and the counts are reported.
        - Correction: dFAR_corr = dFAR / (1 - pi - kappa) for each embedding's primary-cell dFAR
          (confirm_m2a.json h2_p0.primary.<e>.delta_far_std). It rests on two assumptions: (a) a mislabelled switch
          item fires at the control rate and a mislabelled control at the switch rate, so the observed excess is
          (1 - pi - kappa) times the true one; (b) contamination does not depend on channel or cell, so one
          population-level pi and kappa stand in for dFAR, which is a mean over channels of cell-standardised
          differences. Both are approximations, and the corrected value is descriptive.
        - delta and the boundary shares are reported beside the correction, not folded in: a point with a different
          speaker on its two sides is a real speaker change, which would have to be removed rather than reweighted;
          and a marker up to about 0.1 s off still falls inside P0's +-0.5 s collar.
        - Intervals: 10,000 bootstraps of the items within strata (seed 20261018), weights and dFAR held fixed;
          2.5 and 97.5 percentiles with linear interpolation (numpy.quantile). Per-stratum answer shares are
          unweighted, with Wilson 95% intervals.
        - --first N (DEVIATIONS.md entry 11, fixed before any answer was read): score only the first N items in
          page order (L001..), for a listener who stopped early. The page shows the items in a seeded random
          order, so the first N are a simple random sample of the 450. Items after N must be wholly unanswered.
          Inside the first N, q1 and q2 must be answered; a blank q3 is left out of the q3 shares like "unsure"
          and counted. Each stratum's weight uses its count among the scored items. Without --first, everything
          above applies unchanged.
purity  The 230-pair check of scripts/make_purity_listening.py. Answers allowed: q1 same/different/unsure, q2
        usable/unusable. For flagged pairs and for control pairs: the share answered "different" among the usable
        pairs answered same or different (the headline; a pair marked unusable is left out), the same share among
        all pairs answered same or different, the share marked unusable, all with counts and Wilson 95% intervals,
        and the differences flagged minus control.
        - --first N (DEVIATIONS.md entry 12, fixed before any answer was read): score only the first N pairs in page
          order (P001..), for a listener who stopped early. The page shows its pairs in one seeded shuffle of all of
          them, so the first N are a simple random sample of the flagged pairs and of the control pairs. Pairs after
          N must be wholly unanswered, and both questions are required inside N. The d1 path is unchanged.

Usage (on the box, answers in ~/data/d1/listening/answers/):
  .venv-pyannote4/bin/python scripts/score_listening.py d1 --answers <csv> --key ~/data/d1/listening/v1/key.csv \
      --points ~/data/d1/boundaries/lex_v1b --out ~/data/d1/results/listening [--first N]
  .venv-pyannote4/bin/python scripts/score_listening.py purity --answers <csv> \
      --key ~/data/d1/listening/purity_v1/key.csv --out ~/data/d1/results/listening [--first N]
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SEED = 20261018
N_BOOT = 10_000
M2A = REPO / "results/confirm/confirm_m2a.json"
STRATA = ("ssdl:emb1", "ssdl:det_only", "ctrl:emb1", "ctrl:det_only")
CONFIRMATORY = ("train", "test")
ALLOWED_D1 = {"q1": {"yes", "no", "unsure"}, "q2": {"same", "different", "overlap_noise", "unsure"},
              "q3": {"yes", "no", "unsure"}}
D1_COLS = ("q1_language_change", "q2_same_speaker", "q3_boundary")
ALLOWED_PURITY = {"q1_same_speaker": {"same", "different", "unsure"}, "q2_usable": {"usable", "unusable"}}
RATES = ("pi", "delta", "kappa", "boundary_ok_p0L1", "boundary_ok_all_switch")


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float] | None:
    if n == 0:
        return None
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def listening_check_module():
    spec = importlib.util.spec_from_file_location("make_listening_check", REPO / "scripts/make_listening_check.py")
    lc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lc)
    return lc


def pool_sizes(points_dir: Path) -> dict[str, int]:
    """The size of each stratum's pool, computed with the page generator's own load_points() and pool()."""
    lc = listening_check_module()
    rows = lc.load_points(points_dir)
    return {f"{kind}:{fam}": len(lc.pool(rows, kind, fam)) for kind in ("ssdl", "ctrl") for fam in ("emb1", "det_only")}


def weighted_share(items: list[dict], hit, keep) -> dict:
    """Weighted share of items with hit(item) among those with keep(item); counts reported unweighted."""
    kept = [i for i in items if keep(i)]
    w = sum(i["w"] for i in kept)
    return {"share": (sum(i["w"] for i in kept if hit(i)) / w) if w else None, "n": len(kept),
            "n_hit": sum(1 for i in kept if hit(i)), "n_left_out": len(items) - len(kept)}


def rates(items: list[dict]) -> dict:
    dom = [i for i in items if i["split"] in CONFIRMATORY]
    sw = [i for i in dom if i["kind"] == "ssdl" and i["elig_p0L1"]]
    sw_all = [i for i in dom if i["kind"] == "ssdl"]
    ct = [i for i in dom if i["kind"] == "ctrl" and i["elig_det_w1"]]
    yes_no = lambda q: (lambda i: i[q] in ("yes", "no"))
    return {"pi": weighted_share(sw, lambda i: i["q1"] == "no", yes_no("q1")),
            "delta": weighted_share(sw, lambda i: i["q2"] in ("different", "overlap_noise"), lambda i: i["q2"] != "unsure"),
            "kappa": weighted_share(ct, lambda i: i["q1"] == "yes", yes_no("q1")),
            "boundary_ok_p0L1": weighted_share(sw, lambda i: i["q3"] == "yes", yes_no("q3")),
            "boundary_ok_all_switch": weighted_share(sw_all, lambda i: i["q3"] == "yes", yes_no("q3"))}


def corrected(dfar: float, pi: float | None, kappa: float | None) -> float | None:
    if pi is None or kappa is None or 1 - pi - kappa <= 0:
        return None
    return dfar / (1 - pi - kappa)


def percentile_ci(xs: list[float]) -> tuple[float, float] | None:
    if not xs:
        return None
    lo, hi = np.quantile(np.asarray(xs, dtype=float), [0.025, 0.975])
    return float(lo), float(hi)


def check_answers(rows: list[dict], allowed: dict[str, set]) -> None:
    if any(not r[q] for r in rows for q in allowed):
        raise SystemExit("refusing: some items are not fully answered")
    bad = sorted({(q, r[q]) for r in rows for q in allowed if r[q] not in allowed[q]})
    if bad:
        raise SystemExit(f"refusing: unexpected answers {bad}")


def require_columns(rows: list[dict], cols: set[str], what: str) -> None:
    if rows and not cols <= set(rows[0]):
        raise SystemExit(f"refusing: the {what} file lacks columns {sorted(cols - set(rows[0]))}")


def score_d1(answers: list[dict], key: list[dict], points: dict[str, dict], pools: dict[str, int],
             dfar: dict[str, float], n_boot: int = N_BOOT, seed: int = SEED, first: int | None = None) -> dict:
    require_columns(answers, {"item", *D1_COLS}, "answers")
    require_columns(key, {"item", "point_id", "condition", "split"}, "key")
    ans = {a["item"]: a for a in answers}
    if set(ans) != {k["item"] for k in key}:
        raise SystemExit("refusing: the answers and the key do not list the same items")
    if first is not None:                                  # DEVIATIONS.md entry 11
        order = sorted(k["item"] for k in key)             # page order: L001, L002, ... (zero-padded)
        if not 0 < first <= len(order):
            raise SystemExit(f"refusing: --first must be between 1 and {len(order)}")
        later = [i for i in order[first:] if any(ans[i][c].strip() for c in D1_COLS)]
        if later:
            raise SystemExit(f"refusing: {len(later)} items after the first {first} have answers")
        kept = set(order[:first])
        key = [k for k in key if k["item"] in kept]
    n_sample = Counter(k["condition"] for k in key)
    items = []
    for k in key:
        a, p = ans[k["item"]], points[k["point_id"]]
        items.append({"item": k["item"], "stratum": k["condition"], "kind": k["condition"].split(":")[0],
                      "split": k["split"], "w": pools[k["condition"]] / n_sample[k["condition"]],
                      "elig_p0L1": p.get("elig_p0L1") == "True", "elig_det_w1": p.get("elig_det_w1") == "True",
                      "q1": a["q1_language_change"], "q2": a["q2_same_speaker"], "q3": a["q3_boundary"]})
    allowed, n_q3_blank = ALLOWED_D1, 0
    if first is not None:
        for i in items:
            if not i["q3"].strip():
                i["q3"], n_q3_blank = "blank", n_q3_blank + 1
        allowed = {**ALLOWED_D1, "q3": ALLOWED_D1["q3"] | {"blank"}}
    check_answers(items, allowed)
    by_stratum = {}
    for s in STRATA:
        its = [i for i in items if i["stratum"] == s]
        by_stratum[s] = {"n": len(its), "pool": pools[s], "n_confirmatory": sum(i["split"] in CONFIRMATORY for i in its)}
        for q in ("q1", "q2", "q3"):
            c = Counter(i[q] for i in its)
            by_stratum[s][q] = {v: {"n": n, "share": n / len(its), "ci95": wilson(n, len(its))} for v, n in sorted(c.items())}
    point = rates(items)
    in_pool = lambda r: r.get("elig_emb1") == "True" or r.get("elig_det") == "True"   # the page's two pools
    coverage = {}
    for name, kind, fam in (("p0L1_switch_points", "ssdl", "elig_p0L1"), ("det_w1_controls", "ctrl", "elig_det_w1")):
        pop = [r for r in points.values() if r.get("kind") == kind and r.get(fam) == "True"
               and r.get("split") in CONFIRMATORY]
        coverage[name] = {"population": len(pop), "outside_the_sampled_pools": sum(not in_pool(r) for r in pop)}
    strata_items = {s: [i for i in items if i["stratum"] == s] for s in STRATA}
    rng = random.Random(seed)
    boot = defaultdict(list)
    for _ in range(n_boot):
        resample = [rng.choice(its) for s in STRATA for its in [strata_items[s]] for _ in its]
        r = rates(resample)
        for name in RATES:
            if r[name]["share"] is not None:
                boot[name].append(r[name]["share"])
        for e, d in dfar.items():
            c = corrected(d, r["pi"]["share"], r["kappa"]["share"])
            if c is not None:
                boot[f"dfar_corr_{e}"].append(c)
    return {"n_items": len(items), "first": first, "n_q3_blank": n_q3_blank, "strata": by_stratum,
            "coverage": coverage,
            "primary_cell": {name: {**point[name], "ci95_boot": percentile_ci(boot[name])} for name in RATES},
            "dfar": {e: {"observed": d, "corrected": corrected(d, point["pi"]["share"], point["kappa"]["share"]),
                         "corrected_ci95_boot": percentile_ci(boot[f"dfar_corr_{e}"]),
                         "boot_draws_used": len(boot[f"dfar_corr_{e}"])} for e, d in dfar.items()},
            "n_boot": n_boot, "seed": seed}


def different_share(pairs: list[dict]) -> dict:
    judged = [a for a in pairs if a["q1_same_speaker"] in ("same", "different")]
    d = sum(a["q1_same_speaker"] == "different" for a in judged)
    return {"n": len(pairs), "n_judged": len(judged), "n_unsure": len(pairs) - len(judged), "different": d,
            "share_different": d / len(judged) if judged else None, "ci95": wilson(d, len(judged))}


def score_purity(answers: list[dict], key: list[dict], first: int | None = None) -> dict:
    require_columns(answers, {"item", *ALLOWED_PURITY}, "answers")
    require_columns(key, {"item", "kind"}, "key")
    ans = {a["item"]: a for a in answers}
    if set(ans) != {k["item"] for k in key}:
        raise SystemExit("refusing: the answers and the key do not list the same items")
    if first is not None:                                  # DEVIATIONS.md entry 12
        order = sorted(k["item"] for k in key)             # page order: P001, P002, ... (zero-padded)
        if not 0 < first <= len(order):
            raise SystemExit(f"refusing: --first must be between 1 and {len(order)}")
        later = [i for i in order[first:] if any(ans[i][c].strip() for c in ALLOWED_PURITY)]
        if later:
            raise SystemExit(f"refusing: {len(later)} items after the first {first} have answers")
        kept = set(order[:first])
        key = [k for k in key if k["item"] in kept]
    check_answers([ans[k["item"]] for k in key], ALLOWED_PURITY)
    out = {"first": first}
    for kind in ("flagged", "control"):
        its = [ans[k["item"]] for k in key if k["kind"] == kind]
        usable = [a for a in its if a["q2_usable"] == "usable"]
        out[kind] = {"n": len(its), "unusable": len(its) - len(usable),
                     "share_unusable": (len(its) - len(usable)) / len(its) if its else None,
                     "unusable_ci95": wilson(len(its) - len(usable), len(its)),
                     "usable_pairs": different_share(usable), "all_pairs": different_share(its)}
    for scope in ("usable_pairs", "all_pairs"):
        f, c = out["flagged"][scope]["share_different"], out["control"][scope]["share_different"]
        out[f"difference_flagged_minus_control_{scope}"] = (f - c) if f is not None and c is not None else None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("check", choices=["d1", "purity"])
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--points", type=Path, help="d1: the lex_v1b folder the page was drawn from")
    ap.add_argument("--first", type=int,
                    help="score only the first N items in page order (DEVIATIONS.md 11 for d1, 12 for purity)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    answers, key = read_csv(args.answers), read_csv(args.key)
    if args.check == "d1":
        if args.points is None:
            raise SystemExit("refusing: d1 needs --points")
        points = {r["point_id"]: r for r in listening_check_module().load_points(args.points)}
        m2a = json.loads(M2A.read_text())["h2_p0"]["primary"]
        res = score_d1(answers, key, points, pool_sizes(args.points), {e: m2a[e]["delta_far_std"] for e in sorted(m2a)},
                       first=args.first)
    else:
        res = score_purity(answers, key, first=args.first)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"listening_{args.check}.json"
    path.write_text(json.dumps(res, indent=1) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
