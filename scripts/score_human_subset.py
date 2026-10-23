#!/usr/bin/env python3
"""EXPLORATORY, post hoc (external review, issue 4): score two raters' answers on the human-verified D1 subset.

Reads key.csv (written by scripts/review_human_subset.py; which item is a switch point and which a control),
two filled copies of template.csv answered independently and blind to the key, and an OPTIONAL adjudication
sheet resolving the raters' disagreements. The analysis is pre-specified in PROTOCOL.md. Reports:

  agreement   Cohen's kappa and raw agreement per question on the two independent sheets, on the items both
              raters answered; for same_speaker also over y/n only (dropping "unsure"); plus a timing-agreement
              summary from switch_time_s on items both raters marked a switch.
  rates       per kind (switch / control) and per reading (rater 1, rater 2, both agree, either flags), with
              Wilson 95% intervals: different_speaker, no_audible_switch (switch items), audible_switch
              (controls), boundary_off_gt_0.1s, and unusable. The sample is proportional to channel size, so
              unweighted rates estimate point-level rates over the population in subset.json.
  verified    after adjudication (resolved labels), the verified switch items and verified controls, per channel,
              against the pre-specified minimum (>=100 switch, >=100 control, >=10 channels).
  central     when the minimum is met AND --boundaries/--emb are given, two estimators close to the paper's on
              the verified items for E1-E3: an H1-type same-speaker similarity difference from the grid embeddings
              (1 - d(t); not the paper's point-anchored H1) and the cell-standardised dFAR change-score (H2/P0,
              frozen tau from results/confirm/confirm_m2a.json), beside the paper's full-subset dFAR.

It refuses to run on any sheet that carries key columns (kind, point_id, t, clip_id, split, lang_*, ...).

Usage (repo root):
  python -m scripts.score_human_subset --subset results/review_followups/human_subset \\
      --rater1 rater1.csv --rater2 rater2.csv [--adjudication adjudication.csv] \\
      [--boundaries <lex_v1b> --emb <grid emb dir>] --out results/review_followups/human_subset/scores
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

QUESTIONS = {"same_speaker": ("y", "n", "unsure"), "audible_language_switch": ("y", "n", "unsure"),
             "boundary_within_0.1s": ("y", "n", "unsure"), "unusable": ("y", "n")}
# columns that only ever appear in key.csv / subset.json; a rater sheet carrying any of them is refused
KEY_COLUMNS = {"kind", "point_id", "paired_item", "match_level", "speaker_id", "split", "video_id", "clip_id",
               "t", "lang_left", "lang_right", "left_s", "right_s", "pause_bin", "pos_bin", "stratum",
               "snippet_start_s", "pad_left_s", "pad_right_s"}
BOUNDARY_TOL_S = 0.1
CENTRE_S = 2.0
ADJ_FIELDS = ("same_speaker", "audible_language_switch", "switch_time_s", "boundary_within_0.1s", "unusable")


def refuse_key_columns(path: Path, header: list[str]) -> None:
    leaked = sorted(KEY_COLUMNS & {h.strip() for h in header})
    if leaked:
        raise SystemExit(f"refusing: {path} carries key columns {leaked}; a rater sheet must not come from key.csv")


def parse_time(v: str) -> float | None:
    v = (v or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        raise SystemExit(f"switch_time_s = {v!r}; expected a number of seconds or blank")


def read_answers(path: Path, items: set[str]) -> dict[str, dict]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        refuse_key_columns(path, reader.fieldnames or [])
        out = {}
        for row in reader:
            iid = (row.get("item_id") or "").strip()
            if iid not in items:
                raise SystemExit(f"{path}: unknown item {iid!r}")
            ans = {}
            for q, allowed in QUESTIONS.items():
                v = (row.get(q) or "").strip().lower()
                if v and v not in allowed:
                    raise SystemExit(f"{path}: item {iid}, {q} = {v!r}; allowed {allowed} or blank")
                ans[q] = v or None
            ans["switch_time_s"] = parse_time(row.get("switch_time_s", ""))
            out[iid] = ans
    return out


def read_adjudication(path: Path, items: set[str]) -> dict[str, dict]:
    """item_id -> {field: resolved_value}. Only listed (item, field) rows override the agreed answer."""
    resolved: dict[str, dict] = defaultdict(dict)
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        refuse_key_columns(path, reader.fieldnames or [])
        need = {"item_id", "field", "resolved"}
        if not need <= set(reader.fieldnames or []):
            raise SystemExit(f"{path}: adjudication sheet needs columns {sorted(need)}")
        for row in reader:
            iid = (row.get("item_id") or "").strip()
            field = (row.get("field") or "").strip()
            if iid not in items:
                raise SystemExit(f"{path}: unknown item {iid!r}")
            if field not in ADJ_FIELDS:
                raise SystemExit(f"{path}: item {iid}, field {field!r}; allowed {ADJ_FIELDS}")
            val = (row.get("resolved") or "").strip().lower()
            if field == "switch_time_s":
                resolved[iid][field] = parse_time(row.get("resolved", ""))
            else:
                if val and val not in QUESTIONS[field]:
                    raise SystemExit(f"{path}: item {iid}, {field} resolved = {val!r}; allowed {QUESTIONS[field]}")
                resolved[iid][field] = val or None
    return dict(resolved)


def cohen_kappa(a: list[str], b: list[str]) -> dict:
    """Cohen's kappa for two raters' paired nominal labels."""
    n = len(a)
    if n == 0:
        return {"kappa": None, "agreement": None, "n": 0}
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    kappa = (po - pe) / (1 - pe) if pe < 1 else None  # undefined when both raters use one category only
    return {"kappa": kappa, "agreement": po, "n": n, "expected_agreement": pe}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> dict:
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "lo": None, "hi": None}
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"k": k, "n": n, "rate": p, "lo": max(0.0, mid - half), "hi": min(1.0, mid + half)}


def flag_rate(ids: list[str], r1: dict, r2: dict, q: str, flag: str, valid: tuple[str, ...]) -> dict:
    out = {}
    for name, pick in (("rater1", lambda i: [r1[i][q]]), ("rater2", lambda i: [r2[i][q]])):
        vals = [v for i in ids for v in pick(i) if v in valid]
        out[name] = wilson(sum(v == flag for v in vals), len(vals))
    both = [(r1[i][q], r2[i][q]) for i in ids if r1[i][q] in valid and r2[i][q] in valid]
    agree = [x for x, y in both if x == y]
    out["both_agree"] = wilson(sum(v == flag for v in agree), len(agree))
    out["either_flags"] = wilson(sum(flag in (x, y) for x, y in both), len(both))
    return out


def timing_agreement(ids: list[str], r1: dict, r2: dict) -> dict:
    """On items both raters marked a switch and gave a time, |t1 - t2| summary and within-0.1s share."""
    diffs = []
    for i in ids:
        if (r1[i]["audible_language_switch"] == "y" and r2[i]["audible_language_switch"] == "y"
                and r1[i]["switch_time_s"] is not None and r2[i]["switch_time_s"] is not None):
            diffs.append(abs(r1[i]["switch_time_s"] - r2[i]["switch_time_s"]))
    if not diffs:
        return {"n": 0, "mean_abs_diff_s": None, "p95_abs_diff_s": None, "within_0.1s_rate": None}
    diffs.sort()
    p95 = diffs[min(len(diffs) - 1, int(math.ceil(0.95 * len(diffs)) - 1))]
    within = sum(d <= BOUNDARY_TOL_S for d in diffs)
    return {"n": len(diffs), "mean_abs_diff_s": sum(diffs) / len(diffs), "p95_abs_diff_s": p95,
            "within_0.1s_rate": within / len(diffs)}


def resolved_label(iid: str, field: str, r1: dict, r2: dict, adj: dict) -> object:
    """The resolved answer: the adjudicated value if present, else the raters' value when they agree, else None."""
    if iid in adj and field in adj[iid]:
        return adj[iid][field]
    a, b = r1[iid][field], r2[iid][field]
    return a if a == b else None


def verified_subset(key: dict, r1: dict, r2: dict, adj: dict) -> dict:
    """Per the protocol: verified switch = same_speaker y & switch y & boundary y & usable; control = y / n / usable."""
    sw_ids, ct_ids = [], []
    per_channel = defaultdict(lambda: {"switch": 0, "control": 0})
    for iid, k in key.items():
        rl = {f: resolved_label(iid, f, r1, r2, adj) for f in
              ("same_speaker", "audible_language_switch", "boundary_within_0.1s", "unusable")}
        usable = rl["unusable"] == "n"
        if k["kind"] == "switch":
            ok = (rl["same_speaker"] == "y" and rl["audible_language_switch"] == "y"
                  and rl["boundary_within_0.1s"] == "y" and usable)
            if ok:
                sw_ids.append(iid)
                per_channel[k["speaker_id"]]["switch"] += 1
        else:
            ok = (rl["same_speaker"] == "y" and rl["audible_language_switch"] == "n" and usable)
            if ok:
                ct_ids.append(iid)
                per_channel[k["speaker_id"]]["control"] += 1
    channels = sorted({k["speaker_id"] for k in key.values()})
    n_ch = sum(1 for c in channels if per_channel[c]["switch"] or per_channel[c]["control"])
    min_ok = len(sw_ids) >= 100 and len(ct_ids) >= 100 and n_ch >= 10
    return {"switch_item_ids": sw_ids, "control_item_ids": ct_ids,
            "n_verified_switch": len(sw_ids), "n_verified_control": len(ct_ids), "n_channels": n_ch,
            "per_channel": {c: dict(per_channel[c]) for c in channels},
            "minimum": {"switch": 100, "control": 100, "channels": 10, "met": min_ok}}


def central_comparison(key: dict, verified: dict, boundaries: Path, emb: Path) -> dict:
    """The paper's two estimators on the verified items for E1-E3 (DESCRIPTIVE). Frozen code, frozen inputs."""
    import numpy as np  # noqa: local import so the agreement/rates path needs no scientific stack
    from csdiar.analysis.confirm import load_points
    from csdiar.analysis.pilot import p0_curves, score_points, unit

    m2a = json.loads((Path(__file__).resolve().parents[1] / "results/confirm/confirm_m2a.json").read_text())
    taus = m2a["tau"]                                            # frozen tau per family
    full = {e: m2a["h2_p0"]["primary"][e]["delta_far_std"] for e in ("e1", "e2", "e3")} \
        if "h2_p0" in m2a else {}

    pts, _ = load_points(boundaries, ["train", "test"])
    # the verified switch points and their verified matched controls, by key.point_id
    sw_pts = {key[i]["point_id"] for i in verified["switch_item_ids"]}
    ct_pts = {key[i]["point_id"] for i in verified["control_item_ids"]}
    # control d(t)/peak need the frozen cell structure; restrict the points dict to verified ids for both roles
    verified_pts = {p: r for p, r in pts.items() if p in sw_pts or p in ct_pts}

    families = [("e3", "e3_L1"), ("e1", "e1_L1"), ("e2", "e2_L1")]  # e3 first: only e3 is hash-verified
    out = {"note": "DESCRIPTIVE; verified items only; E3 is hash-verified, E1/E2 secondary cross-check.",
           "n_verified_switch": verified["n_verified_switch"], "n_verified_control": verified["n_verified_control"],
           "paper_full_subset_delta_far_std": full, "families": {}}
    for fam, _dir in families:
        emb_name = fam  # p0_curves reads <emb>/<model>_L1/<split>.npz with model=fam
        try:
            curves = p0_curves(emb, emb_name, ("train", "test"))
        except FileNotFoundError as exc:
            out["families"][fam] = {"error": f"grid embeddings not found: {exc}"}
            continue
        vals, _ = score_points(verified_pts, curves)
        tau = taus.get(fam)
        # cell-standardised dFAR on verified items, cell = (video, pause_bin, pos_bin), per channel
        cell = lambda r: (r["video_id"], r["pause_bin"], r["pos_bin"])
        ctrl_hit = defaultdict(lambda: defaultdict(list))
        for p in ct_pts:
            if p in vals:
                r = pts[p]
                ctrl_hit[r["speaker_id"]][cell(r)].append(vals[p]["peak"] > tau)
        per_chan_dfar, sw_cos, ct_cos_by_pt = [], {}, {}
        # same-speaker similarity: left/right cosine per point (H1-style), from the family's grid embeddings is
        # not available per-point here; use the P0 d(t) complement 1 - d as the on-point left/right similarity.
        for p in sw_pts | ct_pts:
            if p in vals:
                # d(t) = 1 - cos(left,right) at the boundary, so cos = 1 - d
                (sw_cos if p in sw_pts else ct_cos_by_pt)[p] = 1.0 - vals[p]["d"]
        # dFAR per channel
        chan_dfar = {}
        for p in sw_pts:
            if p not in vals:
                continue
            r = pts[p]
            c = cell(r)
            if not ctrl_hit[r["speaker_id"]].get(c):
                continue
            chan_dfar.setdefault(r["speaker_id"], {"hits": [], "rates": []})
            chan_dfar[r["speaker_id"]]["hits"].append(float(vals[p]["peak"] > tau))
            chan_dfar[r["speaker_id"]]["rates"].append(float(np.mean(ctrl_hit[r["speaker_id"]][c])))
        dfs = [float(np.mean(v["hits"]) - np.mean(v["rates"])) for v in chan_dfar.values() if v["hits"]]
        # same-speaker similarity difference: per channel mean (control cos) - (switch cos), equal-weight over channels
        chan_cos = defaultdict(lambda: {"sw": [], "ct": []})
        for p in sw_pts:
            if p in sw_cos:
                chan_cos[pts[p]["speaker_id"]]["sw"].append(sw_cos[p])
        for p in ct_pts:
            if p in ct_cos_by_pt:
                chan_cos[pts[p]["speaker_id"]]["ct"].append(ct_cos_by_pt[p])
        diffs = [float(np.mean(v["ct"]) - np.mean(v["sw"])) for v in chan_cos.values() if v["sw"] and v["ct"]]
        ctrl_all = [c for v in chan_cos.values() for c in v["ct"]]
        tsd = float(np.std(ctrl_all, ddof=1)) if len(ctrl_all) > 1 else None
        mean_diff = float(np.mean(diffs)) if diffs else None
        out["families"][fam] = {
            "tau": tau,
            "delta_far_std_mean_of_channels": float(np.mean(dfs)) if dfs else None,
            "n_channels_with_dfar": len(dfs),
            "same_speaker_cos_diff_mean_of_channels": mean_diff,
            "same_speaker_cos_diff_in_control_sd": (mean_diff / tsd) if (mean_diff is not None and tsd) else None,
            "n_channels_with_cos_diff": len(diffs),
            "hash_verified": fam == "e3"}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--subset", type=Path, required=True, help="folder with key.csv")
    ap.add_argument("--rater1", type=Path, required=True)
    ap.add_argument("--rater2", type=Path, required=True)
    ap.add_argument("--adjudication", type=Path, help="optional adjudication sheet (item_id, field, resolved, ...)")
    ap.add_argument("--boundaries", type=Path, help="lex_v1b folder, for the central comparison on verified items")
    ap.add_argument("--emb", type=Path, help="grid embeddings folder (holds e1_L1/e2_L1/e3_L1)")
    ap.add_argument("--out", type=Path, required=True, help="output prefix; writes <out>.json and <out>.md")
    args = ap.parse_args()

    key = {r["item_id"]: r for r in csv.DictReader(open(args.subset / "key.csv", encoding="utf-8"))}
    r1, r2 = read_answers(args.rater1, set(key)), read_answers(args.rater2, set(key))
    for name, r in (("rater1", r1), ("rater2", r2)):
        missing = sorted(set(key) - set(r))
        if missing:
            raise SystemExit(f"{name}: {len(missing)} items have no row, e.g. {missing[:3]}")
    adj = read_adjudication(args.adjudication, set(key)) if args.adjudication else {}

    # per-question agreement on the independent sheets
    agreement = {}
    for q in QUESTIONS:
        pairs = [(r1[i][q], r2[i][q]) for i in key if r1[i][q] and r2[i][q]]
        agreement[q] = cohen_kappa([x for x, _ in pairs], [y for _, y in pairs])
    yn = [(r1[i]["same_speaker"], r2[i]["same_speaker"]) for i in key
          if r1[i]["same_speaker"] in ("y", "n") and r2[i]["same_speaker"] in ("y", "n")]
    agreement["same_speaker_yn_only"] = cohen_kappa([x for x, _ in yn], [y for _, y in yn])
    agreement["switch_timing"] = timing_agreement(list(key), r1, r2)

    # disagreement list (any field differs on the independent sheets)
    disagreements = []
    for i in key:
        diff = {q: [r1[i][q], r2[i][q]] for q in QUESTIONS if r1[i][q] != r2[i][q]}
        if r1[i]["switch_time_s"] != r2[i]["switch_time_s"]:
            diff["switch_time_s"] = [r1[i]["switch_time_s"], r2[i]["switch_time_s"]]
        if diff:
            disagreements.append({"item_id": i, **diff})

    rates = {}
    for kind in ("switch", "control"):
        ids = [i for i, r in key.items() if r["kind"] == kind]
        rates[kind] = {
            "n_items": len(ids),
            "different_speaker": flag_rate(ids, r1, r2, "same_speaker", "n", ("y", "n")),
            "unsure_speaker": flag_rate(ids, r1, r2, "same_speaker", "unsure", ("y", "n", "unsure")),
            "no_audible_switch": flag_rate(ids, r1, r2, "audible_language_switch", "n", ("y", "n")),
            "audible_switch": flag_rate(ids, r1, r2, "audible_language_switch", "y", ("y", "n")),
            "boundary_off_gt_0.1s": flag_rate(ids, r1, r2, "boundary_within_0.1s", "n", ("y", "n")),
            "unusable": flag_rate(ids, r1, r2, "unusable", "y", ("y", "n"))}

    verified = verified_subset(key, r1, r2, adj)
    central = None
    if args.boundaries and args.emb:
        if verified["minimum"]["met"]:
            central = central_comparison(key, verified, args.boundaries, args.emb)
        else:
            central = {"skipped": "verified subset below the pre-specified minimum; descriptive only",
                       "minimum": verified["minimum"]}

    verified_report = {k: v for k, v in verified.items() if k not in ("switch_item_ids", "control_item_ids")}
    out = {"label": "EXPLORATORY (post hoc, external review issue 4)", "raters": [str(args.rater1), str(args.rater2)],
           "adjudication": str(args.adjudication) if args.adjudication else None,
           "agreement": agreement, "disagreements": disagreements, "rates": rates,
           "verified_subset": verified_report, "central_comparison": central}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(out, indent=2))

    fmt = lambda w: "n/a" if w["rate"] is None else f"{100 * w['rate']:.1f}% ({w['k']}/{w['n']}; {100 * w['lo']:.1f}-{100 * w['hi']:.1f})"
    lines = ["# Human-verified subset: agreement and contamination (EXPLORATORY)", "",
             "| Question | Cohen's kappa | Agreement | Items |", "|---|---|---|---|"]
    for q, a in agreement.items():
        if q in ("switch_timing",):
            continue
        k = "n/a" if a["kappa"] is None else f"{a['kappa']:.3f}"
        po = "n/a" if a["agreement"] is None else f"{100 * a['agreement']:.1f}%"
        lines.append(f"| {q} | {k} | {po} | {a['n']} |")
    t = agreement["switch_timing"]
    if t["n"]:
        lines += ["", f"Timing agreement (both marked a switch, n={t['n']}): mean |t1-t2| = {t['mean_abs_diff_s']:.3f} s, "
                      f"95th pct = {t['p95_abs_diff_s']:.3f} s, within 0.1 s = {100 * t['within_0.1s_rate']:.1f}%."]
    lines += ["", "| Kind | Measure | Rater 1 | Rater 2 | Both agree | Either flags |", "|---|---|---|---|---|---|"]
    for kind, block in rates.items():
        for m, v in block.items():
            if m != "n_items":
                lines.append(f"| {kind} | {m} | {fmt(v['rater1'])} | {fmt(v['rater2'])} | {fmt(v['both_agree'])} "
                             f"| {fmt(v['either_flags'])} |")
    vr = verified_report
    lines += ["", "## Verified subset (after adjudication)",
              f"- verified switch items: {vr['n_verified_switch']} (minimum {vr['minimum']['switch']})",
              f"- verified controls: {vr['n_verified_control']} (minimum {vr['minimum']['control']})",
              f"- channels represented: {vr['n_channels']} (minimum {vr['minimum']['channels']})",
              f"- minimum met: {'yes' if vr['minimum']['met'] else 'no — central comparison reported descriptively only'}"]
    if central and "families" in central:
        lines += ["", "## Central comparison on verified items (DESCRIPTIVE)", central["note"],
                  "", "| Family | dFAR (verified) | dFAR (paper full subset) | same-speaker cos diff | hash-verified |",
                  "|---|---|---|---|---|"]
        full = central.get("paper_full_subset_delta_far_std", {})
        for fam, b in central["families"].items():
            if "error" in b:
                lines.append(f"| {fam} | error: {b['error']} | | | |")
                continue
            dfar = "n/a" if b["delta_far_std_mean_of_channels"] is None else f"{100 * b['delta_far_std_mean_of_channels']:+.2f} pp"
            fullv = "n/a" if fam not in full else f"{100 * full[fam]:+.2f} pp"
            cosd = "n/a" if b["same_speaker_cos_diff_mean_of_channels"] is None else f"{b['same_speaker_cos_diff_mean_of_channels']:+.4f}"
            lines.append(f"| {fam} | {dfar} | {fullv} | {cosd} | {'yes' if b['hash_verified'] else 'no'} |")
    elif central and "skipped" in central:
        lines += ["", "## Central comparison", f"Skipped: {central['skipped']}."]
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
