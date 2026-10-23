#!/usr/bin/env python3
"""Summary tables of the M2a confirmatory results (reads the frozen outputs; computes nothing new).

PREREGISTRATION.md §7 requires the pooled `det` AUC next to the primary H2-P0 result in every table, which the
frozen confirm.py table does not carry. This script writes results/confirm/summary_m2a.md from
confirm_m2a.json and attribution_m2a.json: the primary outcomes, the pooled `det` and `det_w1` AUCs with their
99% BCa intervals, unadjusted delta-FAR sign-flip p-values, and each attribution grade with its outcome category.
Usage: python scripts/summarize_confirm.py results/confirm
"""
import json
import sys
from pathlib import Path

MODELS = ("e1", "e2", "e3")


def ci(d: dict, scale: float = 1.0) -> str:
    return f"{d['lo'] * scale:.3f}–{d['hi'] * scale:.3f}"


def main() -> int:
    d = Path(sys.argv[1])
    c = json.loads((d / "confirm_m2a.json").read_text())
    a = json.loads((d / "attribution_m2a.json").read_text())
    out = ["# M2a confirmatory results: summary (17 D1 channels; frozen run, tag m2a-freeze)", "",
           "Primary outcomes. Exact sign-flip over channels, Holm-adjusted; BCa intervals at 1 − 0.05/m.", "",
           "| H | Model | Channels | Estimate (CI) | Holm p (one-sided) | Outcome |", "|---|---|---|---|---|---|"]
    for h, key, est, ck, fmt in (("H1", "h1", "dz", "dz_ci", "dz {:.3f}"), ("H2-P0", "h2_p0", "auc", "auc_ci", "AUC {:.3f}"),
                                 ("H3", "h3", "shift_in_target_sd", "ci", "{:+.3f} SD")):
        for m in MODELS:
            r = c[key]["primary"][m]
            out.append(f"| {h} | {m.upper()} | {r['n_channels']} | {fmt.format(r[est])} ({ci(r[ck])}) | "
                       f"{r['p_one_holm']:.1e} | {r['outcome']['category']} |")
    out += ["", "H2-P0 next to the detection cells that PLAN.md §7 first named (secondary; 99% BCa intervals) and the "
            "operating point at τ (cell-standardised ΔFAR, exact one-sided sign-flip, unadjusted).", "",
            "| Model | `p0L1` AUC (primary) | pooled `det` AUC | `det_w1` AUC | ΔFAR at τ (p) |", "|---|---|---|---|---|"]
    for m in MODELS:
        p, s = c["h2_p0"]["primary"][m], c["h2_p0"]["secondary"][m]
        out.append(f"| {m.upper()} | {p['auc']:.3f} ({ci(p['auc_ci'])}) | {s['det']['auc']:.3f} ({ci(s['det']['auc_ci'])}) | "
                   f"{s['det_w1']['auc']:.3f} ({ci(s['det_w1']['auc_ci'])}) | {p['delta_far_std'] * 100:+.2f} pp "
                   f"({p['delta_far_signflip']['p_one']:.3f}) |")
    out += ["", "Pooled `det` AUC by the shorter run's length (mean of channels):", "",
            "| Model | < 0.5 s | 0.5–1.0 s | ≥ 1.0 s |", "|---|---|---|---|"]
    for m in MODELS:
        st = c["h2_p0"]["secondary"][m]["det"]["strata"]
        out.append(f"| {m.upper()} | {st['lt0.5']['auc_mean_of_channels']:.3f} | {st['0.5to1']['auc_mean_of_channels']:.3f} | "
                   f"{st['ge1']['auc_mean_of_channels']:.3f} |")
    out += ["", "Attribution (PREREGISTRATION.md §6): grade (primary category); criteria p-values.", "",
            "| H | Model | Grade | (b) adjusted: channels in predicted direction, p | (c) pause > 150 ms: channels, p |",
            "|---|---|---|---|---|"]
    for m in MODELS:
        for h, name in (("h1", "H1"), ("h2_p0", "H2-P0")):
            r = a["models"][m][h]
            adj, pc = r["adjusted"], r["phrase_contrast"]
            pos = sum(v["beta_cond"] * adj["direction"] > 0 for v in adj["channels"].values())
            out.append(f"| {name} | {m.upper()} | {r['label']} | {pos}/{adj['n_channels']}, {adj['signflip']['p_one']:.4f} | "
                       f"{pc['n_channels']}, {pc['signflip']['p_one']:.4f} |")
    (d / "summary_m2a.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
