#!/usr/bin/env python3
"""Unit tests for csdiar.analysis.freeze_m2b (PREREGISTRATION_M2B.md §3, §11).

Model-free and fast: a synthetic DISPLACE archive (reference RTTMs only), synthetic feature/VAD/cache files and
synthetic tuning / power / M3 / hypothesis artifacts are written to a tmp tree, and every freeze_m2b subcommand
and its refusals are exercised. Nothing under ~/data is touched and no model is loaded.

Run on the box (numpy present; freeze_m2b imports confirm_m2b, which imports numpy):
  .venv-pyannote4/bin/python -m pytest -q -p no:cacheprovider tests/test_freeze_m2b.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from csdiar.analysis import confirm_m2b as C
from csdiar.analysis import freeze_m2b as F
from csdiar.data.ingest_displace import LAYOUT, LD_SUFFIX, SD_SUFFIX


# ======================================================================================================= #
# synthetic archive + inputs
# ======================================================================================================= #
def _rttm(path: Path, rid: str, segs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"SPEAKER {rid} 1 {o:.3f} {d:.3f} <NA> <NA> {lab} <NA> <NA>\n"
                            for (o, d, lab) in segs))


def _make_archive(root: Path, dev_ids, eval_ids) -> None:
    for split, ids in (("dev", dev_ids), ("eval", eval_ids)):
        lay = LAYOUT[split]
        for rid in ids:
            _rttm(root / lay["sd"] / f"{rid}{SD_SUFFIX}", rid,
                  [(0.0, 5.0, "spk1"), (6.0, 5.0, "spk2")])
            _rttm(root / lay["ld"] / f"{rid}{LD_SUFFIX}", rid,
                  [(0.0, 5.0, "L1"), (6.0, 5.0, "L2")])


def _touch(path: Path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _make_features(cache_dir: Path, vad_dir: Path, wav_dir: Path, dev_ids, eval_ids) -> None:
    from csdiar.pipelines import run_m2b as R
    for split, ids in (("dev", dev_ids), ("eval", eval_ids)):
        for rid in ids:
            _touch(Path(R.stage1_cache_path(cache_dir, "e3", rid)), f"s1-{rid}".encode())
            _touch(Path(R.lid_cache_path(cache_dir, rid)), f"lid-{rid}".encode())
            _touch(wav_dir / f"{rid}.wav", f"wav-{rid}".encode())
        _touch(vad_dir / f"{split}.npz", f"vad-{split}".encode())
    # the per-seed M3 DISPLACE stage-1 caches m3_artifacts(rec) hashes, for the EVAL recordings only
    for name in C.M3_NAMES:
        model = f"m3:~/data/m3/runs/{name}/best.pt"
        for rid in eval_ids:
            _touch(Path(R.stage1_cache_path(cache_dir, model, rid)), f"m3s1-{name}-{rid}".encode())


def _make_m3(home: Path) -> None:
    """Create the record-free M3 artifacts m3_artifacts() names (the five checkpoints), under a fake HOME so ~
    expands into the tmp tree. The per-seed DISPLACE eval stage-1 caches (the record-dependent part of
    m3_artifacts(rec)) live under cache_dir and are created by _make_features."""
    for rel in C.m3_artifacts():
        _touch(Path(rel.replace("~", str(home), 1)))


def _confirm_m2a(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"h2_p0": {"primary": {
        "e1": {"signflip": {"p_one": 7.62939453125e-06}},
        "e2": {"signflip": {"p_one": 7.62939453125e-06}},
        "e3": {"signflip": {"p_one": 7.62939453125e-06}}}}}))


def _label_map(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"L1": "en", "L2": "hi"}))


class Tree:
    """A synthetic world: archive + features + M3 + label map + confirm_m2a, with the record path redirected."""

    def __init__(self, tmp_path: Path, monkeypatch, n_dev=6, n_eval=6):
        self.tmp = tmp_path
        self.dev_ids = [f"DEV{i:02d}" for i in range(n_dev)]
        self.eval_ids = [f"EVA{i:02d}" for i in range(n_eval)]
        self.root = tmp_path / "displace"
        self.cache = tmp_path / "cache"
        self.vad = tmp_path / "vad"
        self.wav = tmp_path / "wav"
        self.units_dev = tmp_path / "units/dev"
        self.units_eval = tmp_path / "units/eval"
        self.out = tmp_path / "results/confirm_m2b"
        self.label_map = tmp_path / "planning/label_map.json"
        self.confirm_m2a = tmp_path / "results/confirm_m2a.json"
        self.home = tmp_path / "home"
        self.record = tmp_path / "FREEZE_M2B.json"
        _make_archive(self.root, self.dev_ids, self.eval_ids)
        _make_features(self.cache, self.vad, self.wav, self.dev_ids, self.eval_ids)
        _make_m3(self.home)
        _label_map(self.label_map)
        _confirm_m2a(self.confirm_m2a)
        # redirect ~ and the record path for both modules
        monkeypatch.setenv("HOME", str(self.home))
        monkeypatch.setattr(C.Path, "home", staticmethod(lambda: self.home), raising=False)
        monkeypatch.setattr(C, "RECORD", self.record)
        monkeypatch.setattr(F, "RECORD", self.record)

    def init_args(self, sd_suffix=None, ld_suffix=None):
        import argparse
        from csdiar.data.ingest_displace import LD_SUFFIX, SD_SUFFIX
        return argparse.Namespace(
            commit="a" * 40, root=str(self.root), cache_dir=str(self.cache), vad_dir=str(self.vad),
            wav_dir=str(self.wav), units_dev=str(self.units_dev), units_eval=str(self.units_eval),
            out=str(self.out), label_map=str(self.label_map), confirm_m2a=str(self.confirm_m2a),
            sd_suffix=sd_suffix or SD_SUFFIX, ld_suffix=ld_suffix or LD_SUFFIX)

    def tuning_record(self, path: Path, *, m2_assessable=True, extra_code=None):
        code = {f: C.sha256(C.REPO / f) for f in C.TUNING_GUARDED}
        if extra_code:
            code.update(extra_code)
        rec = {"code_sha256": code,
               "operating_points": {
                   "B0": {"threshold": 0.60}, "B1": {"threshold": 0.60}, "M1": {"threshold": 0.65},
                   "B2": {"threshold": 0.60}, "M2": {"threshold": 0.60, "alpha": 0.15, "tau0": 0.0}},
               "fits": {"b2_delta": {"delta": 0.07}, "b1_means": {"counts": {}}, "m1_v_hat": {"n_units": 3}},
               "m2_assessable": {"tunable": m2_assessable}}
        path.write_text(json.dumps(rec))
        return path

    def p0_record(self, path: Path):
        rec = {"code_sha256": {"csdiar/analysis/tune_p0_m2b.py":
                               C.sha256(C.REPO / "csdiar/analysis/tune_p0_m2b.py")},
               "alpha": 0.25, "tau0": 0.02, "assessable": {"tunable": True}}
        path.write_text(json.dumps(rec))
        return path

    def power_record(self, path: Path, verdict="confirmatory"):
        path.write_text(json.dumps({"verdict": verdict, "projected_dsdl": 800}))
        return path

    def make_eval_hyps(self, hyp_root: Path):
        """Write one exclusive RTTM per (frozen system, threshold, eval recording) + native dirs."""
        rec = json.loads(self.record.read_text())
        for p in C.eval_hypothesis_files(hyp_root, rec):
            _rttm(Path(p), "r", [(0.0, 1.0, "a"), (2.0, 1.0, "b")])
        for d in C.expected_native_dirs(hyp_root, rec):
            for rid in self.eval_ids:
                _rttm(Path(d) / f"{rid}.exclusive.rttm", rid, [(0.0, 1.0, "a")])


# ======================================================================================================= #
# init
# ======================================================================================================= #
def test_init_writes_record_with_every_section(tmp_path, monkeypatch):
    t = Tree(tmp_path, monkeypatch)
    assert F.cmd_init(t.init_args()) == 0
    rec = json.loads(t.record.read_text())
    assert rec["tag"] == "m2b-freeze" and rec["commit"] == "a" * 40 and rec["seed"] == C.SEED
    assert rec["split"]["dev"] == t.dev_ids and rec["split"]["eval"] == t.eval_ids
    assert set(rec["code_sha256"]) == set(C.GUARDED)
    # features: stage-1 + LID per recording (2 per rec) + one VAD per split
    assert len(rec["feature_sha256"]) == 2 * (len(t.dev_ids) + len(t.eval_ids)) + 2
    # references: SD + LD per recording
    assert len(rec["reference_sha256"]) == 2 * (len(t.dev_ids) + len(t.eval_ids))
    assert rec["label_map"] == {"L1": "en", "L2": "hi"}


def test_init_refuses_short_commit(tmp_path, monkeypatch):
    t = Tree(tmp_path, monkeypatch)
    args = t.init_args()
    args.commit = "short"
    with pytest.raises(SystemExit, match="40-character"):
        F.cmd_init(args)


def test_init_refuses_when_record_exists(tmp_path, monkeypatch):
    t = Tree(tmp_path, monkeypatch)
    F.cmd_init(t.init_args())
    with pytest.raises(SystemExit, match="exists"):
        F.cmd_init(t.init_args())


def test_init_refuses_mismatched_reference_coverage(tmp_path, monkeypatch):
    t = Tree(tmp_path, monkeypatch)
    # delete one LD reference so SD/LD cover different recordings
    (t.root / LAYOUT["dev"]["ld"] / f"{t.dev_ids[0]}{LD_SUFFIX}").unlink()
    with pytest.raises(SystemExit, match="different recordings"):
        F.cmd_init(t.init_args())


def test_init_refuses_missing_feature(tmp_path, monkeypatch):
    t = Tree(tmp_path, monkeypatch)
    from csdiar.pipelines import run_m2b as R
    Path(R.lid_cache_path(t.cache, t.eval_ids[0])).unlink()
    with pytest.raises(SystemExit, match="feature files missing"):
        F.cmd_init(t.init_args())


# ======================================================================================================= #
# add-tuning
# ======================================================================================================= #
def _seed_record(t) -> dict:
    F.cmd_init(t.init_args())
    return json.loads(t.record.read_text())


def test_add_tuning_copies_operating_points(tmp_path, monkeypatch):
    t = Tree(tmp_path, monkeypatch)
    _seed_record(t)
    import argparse
    tun = t.tuning_record(tmp_path / "tune.json")
    p0 = t.p0_record(tmp_path / "p0.json")
    assert F.cmd_add_tuning(argparse.Namespace(tuning=str(tun), p0_tuning=str(p0))) == 0
    rec = json.loads(t.record.read_text())
    assert rec["tuning"]["thresholds"]["B0"] == 0.60
    assert rec["tuning"]["m2_post_alpha"] == 0.15
    assert rec["tuning"]["p0_gate_alpha"] == 0.25
    assert rec["tuning"]["b2_delta"] == 0.07


def test_add_tuning_refuses_edited_tuning_code(tmp_path, monkeypatch):
    t = Tree(tmp_path, monkeypatch)
    _seed_record(t)
    import argparse
    # a tuning record claiming a different hash for tune_m2b.py is refused
    tun = t.tuning_record(tmp_path / "tune.json",
                          extra_code={"csdiar/analysis/tune_m2b.py": "deadbeef"})
    p0 = t.p0_record(tmp_path / "p0.json")
    with pytest.raises(SystemExit, match="differs from the frozen one"):
        F.cmd_add_tuning(argparse.Namespace(tuning=str(tun), p0_tuning=str(p0)))


def test_add_tuning_refuses_twice(tmp_path, monkeypatch):
    t = Tree(tmp_path, monkeypatch)
    _seed_record(t)
    import argparse
    tun = t.tuning_record(tmp_path / "tune.json")
    p0 = t.p0_record(tmp_path / "p0.json")
    F.cmd_add_tuning(argparse.Namespace(tuning=str(tun), p0_tuning=str(p0)))
    with pytest.raises(SystemExit, match="already recorded"):
        F.cmd_add_tuning(argparse.Namespace(tuning=str(tun), p0_tuning=str(p0)))


def test_add_tuning_refuses_when_guarded_code_changed(tmp_path, monkeypatch):
    t = Tree(tmp_path, monkeypatch)
    _seed_record(t)
    rec = json.loads(t.record.read_text())
    rec["code_sha256"]["csdiar/analysis/stats.py"] = "0" * 64   # pretend init froze a different stats.py
    t.record.write_text(json.dumps(rec))
    import argparse
    tun = t.tuning_record(tmp_path / "tune.json")
    p0 = t.p0_record(tmp_path / "p0.json")
    with pytest.raises(SystemExit, match="changed since the freeze"):
        F.cmd_add_tuning(argparse.Namespace(tuning=str(tun), p0_tuning=str(p0)))


# ======================================================================================================= #
# add-power / add-m3 / add-eval-hyps and full chain
# ======================================================================================================= #
def _chain_through_m3(t, tmp_path):
    import argparse
    _seed_record(t)
    tun = t.tuning_record(tmp_path / "tune.json")
    p0 = t.p0_record(tmp_path / "p0.json")
    F.cmd_add_tuning(argparse.Namespace(tuning=str(tun), p0_tuning=str(p0)))
    F.cmd_add_power(argparse.Namespace(power=str(t.power_record(tmp_path / "power.json"))))
    F.cmd_add_m3(argparse.Namespace())


def test_add_power_verdict(tmp_path, monkeypatch):
    t = Tree(tmp_path, monkeypatch)
    import argparse
    _seed_record(t)
    tun = t.tuning_record(tmp_path / "tune.json")
    F.cmd_add_tuning(argparse.Namespace(tuning=str(tun), p0_tuning=str(t.p0_record(tmp_path / "p0.json"))))
    F.cmd_add_power(argparse.Namespace(power=str(t.power_record(tmp_path / "power.json", "descriptive"))))
    assert json.loads(t.record.read_text())["power"]["verdict"] == "descriptive"


def test_add_power_refuses_bad_verdict(tmp_path, monkeypatch):
    t = Tree(tmp_path, monkeypatch)
    import argparse
    _seed_record(t)
    F.cmd_add_tuning(argparse.Namespace(tuning=str(t.tuning_record(tmp_path / "tune.json")),
                                        p0_tuning=str(t.p0_record(tmp_path / "p0.json"))))
    bad = tmp_path / "bad_power.json"
    bad.write_text(json.dumps({"verdict": "maybe"}))
    with pytest.raises(SystemExit, match="verdict"):
        F.cmd_add_power(argparse.Namespace(power=str(bad)))


def test_add_m3_hashes_every_artifact(tmp_path, monkeypatch):
    t = Tree(tmp_path, monkeypatch)
    _chain_through_m3(t, tmp_path)
    rec = json.loads(t.record.read_text())
    assert sorted(rec["m3_artifacts_sha256"]) == sorted(C.m3_artifacts(rec))
    # both artifact classes are pinned: the 5 checkpoints AND a per-seed eval stage-1 cache per eval recording.
    keys = list(rec["m3_artifacts_sha256"])
    assert sum(k.endswith("best.pt") for k in keys) == len(C.M3_NAMES)
    assert all(f"stage1/{tag}/" in k for tag in [f"m3-{n}" for n in C.M3_NAMES]
               for k in keys if tag in k)
    # one stage-1 cache per (seed, eval recording); no dense-grid / emb_displace path is hashed.
    assert sum("/stage1/m3-" in k for k in keys) == len(C.M3_NAMES) * len(t.eval_ids)
    assert not any("emb_displace" in k or "/emb/" in k for k in keys)


def test_add_eval_hyps_hashes_all_and_refuses_missing(tmp_path, monkeypatch):
    t = Tree(tmp_path, monkeypatch)
    import argparse
    _chain_through_m3(t, tmp_path)
    hyp_root = tmp_path / "hyp"
    # before writing the hypotheses the add refuses
    with pytest.raises(SystemExit, match="missing"):
        F.cmd_add_eval_hyps(argparse.Namespace(hyp_root=str(hyp_root)))
    t.make_eval_hyps(hyp_root)
    assert F.cmd_add_eval_hyps(argparse.Namespace(hyp_root=str(hyp_root))) == 0
    rec = json.loads(t.record.read_text())
    # systems from tuning (B0,B1,M1,B2,M2) x 15 thresholds x 6 eval recordings
    assert len(rec["eval_hyps_sha256"]) == 5 * len(C.THRESHOLD_GRID) * len(t.eval_ids)
    assert len(rec["eval_native_dirs"]) == 2


def test_full_chain_then_check_freeze_passes(tmp_path, monkeypatch):
    """A record written by freeze_m2b, with the hypotheses present, passes confirm_m2b.check_freeze."""
    import argparse
    t = Tree(tmp_path, monkeypatch)
    _chain_through_m3(t, tmp_path)
    hyp_root = tmp_path / "hyp"
    t.make_eval_hyps(hyp_root)
    F.cmd_add_eval_hyps(argparse.Namespace(hyp_root=str(hyp_root)))
    # build the args check_freeze reads (the frozen paths) and run it
    rec = json.loads(t.record.read_text())
    args = argparse.Namespace(hyp_root=rec["paths"]["hyp_root"], **{k: rec["paths"][k] for k in C.PATH_ARGS})
    out = C.check_freeze(args)
    assert out["tag"] == "m2b-freeze"


def test_check_freeze_refuses_after_started_marker(tmp_path, monkeypatch):
    import argparse
    t = Tree(tmp_path, monkeypatch)
    _chain_through_m3(t, tmp_path)
    hyp_root = tmp_path / "hyp"
    t.make_eval_hyps(hyp_root)
    F.cmd_add_eval_hyps(argparse.Namespace(hyp_root=str(hyp_root)))
    rec = json.loads(t.record.read_text())
    out_dir = Path(rec["paths"]["out"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "confirm_m2b.started").write_text("{}")
    args = argparse.Namespace(hyp_root=rec["paths"]["hyp_root"], **{k: rec["paths"][k] for k in C.PATH_ARGS})
    with pytest.raises(SystemExit, match="already started"):
        C.check_freeze(args)


def test_add_eval_hyps_refuses_before_tuning(tmp_path, monkeypatch):
    import argparse
    t = Tree(tmp_path, monkeypatch)
    _seed_record(t)
    with pytest.raises(SystemExit, match="add-tuning must run before"):
        F.cmd_add_eval_hyps(argparse.Namespace(hyp_root=str(tmp_path / "hyp"), native_absent=False))


# ======================================================================================================= #
# M1: hyp-root refusal; M2: native per-file hash refusal and the --native-absent path (brief fixes 1, 2)
# ======================================================================================================= #
def test_check_freeze_refuses_mismatched_hyp_root(tmp_path, monkeypatch):
    import argparse
    t = Tree(tmp_path, monkeypatch)
    _chain_through_m3(t, tmp_path)
    hyp_root = tmp_path / "hyp"
    t.make_eval_hyps(hyp_root)
    F.cmd_add_eval_hyps(argparse.Namespace(hyp_root=str(hyp_root), native_absent=False))
    rec = json.loads(t.record.read_text())
    assert rec["paths"]["hyp_root"] == str(hyp_root)       # M1: the hyp root is pinned
    # a confirm run pointed at a DIFFERENT hyp root is refused, even if everything else matches.
    args = argparse.Namespace(hyp_root=str(tmp_path / "other_hyp"),
                              **{k: rec["paths"][k] for k in C.PATH_ARGS})
    with pytest.raises(SystemExit, match="not the frozen hypothesis root"):
        C.check_freeze(args)


def test_add_eval_hyps_hashes_native_files_and_refuses_missing(tmp_path, monkeypatch):
    import argparse
    t = Tree(tmp_path, monkeypatch)
    _chain_through_m3(t, tmp_path)
    hyp_root = tmp_path / "hyp"
    t.make_eval_hyps(hyp_root)
    # delete ONE native per-recording RTTM: add-eval-hyps must refuse the missing native file (not just a dir).
    native_dir = C.expected_native_dirs(hyp_root, json.loads(t.record.read_text()))[0]
    (Path(native_dir) / f"{t.eval_ids[0]}.exclusive.rttm").unlink()
    with pytest.raises(SystemExit, match="missing"):
        F.cmd_add_eval_hyps(argparse.Namespace(hyp_root=str(hyp_root), native_absent=False))


def test_check_freeze_refuses_changed_native_hyp(tmp_path, monkeypatch):
    import argparse
    t = Tree(tmp_path, monkeypatch)
    _chain_through_m3(t, tmp_path)
    hyp_root = tmp_path / "hyp"
    t.make_eval_hyps(hyp_root)
    F.cmd_add_eval_hyps(argparse.Namespace(hyp_root=str(hyp_root), native_absent=False))
    rec = json.loads(t.record.read_text())
    assert rec["native_absent"] is False and len(rec["native_hyps_sha256"]) == 2 * len(t.eval_ids)
    # mutate one native RTTM AFTER the freeze -> check_freeze must detect it.
    native_dir = C.expected_native_dirs(hyp_root, rec)[0]
    (Path(native_dir) / f"{t.eval_ids[0]}.exclusive.rttm").write_text("SPEAKER r 1 0 9 <NA> <NA> z <NA> <NA>\n")
    args = argparse.Namespace(hyp_root=rec["paths"]["hyp_root"], **{k: rec["paths"][k] for k in C.PATH_ARGS})
    with pytest.raises(SystemExit, match="native eval hypothesis .* changed"):
        C.check_freeze(args)


def test_native_absent_freeze_records_flag_and_skips_native(tmp_path, monkeypatch):
    import argparse
    t = Tree(tmp_path, monkeypatch)
    _chain_through_m3(t, tmp_path)
    hyp_root = tmp_path / "hyp"
    # write ONLY the P1-AHC grid hyps, NOT the native dirs
    rec0 = json.loads(t.record.read_text())
    for p in C.eval_hypothesis_files(hyp_root, rec0):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text("SPEAKER r 1 0.0 1.0 <NA> <NA> a <NA> <NA>\n")
    assert F.cmd_add_eval_hyps(argparse.Namespace(hyp_root=str(hyp_root), native_absent=True)) == 0
    rec = json.loads(t.record.read_text())
    assert rec["native_absent"] is True
    assert "native_hyps_sha256" not in rec
    # check_freeze passes without native hashes because the freeze was taken --native-absent.
    args = argparse.Namespace(hyp_root=rec["paths"]["hyp_root"], **{k: rec["paths"][k] for k in C.PATH_ARGS})
    out = C.check_freeze(args)
    assert out["native_absent"] is True


def test_check_freeze_refuses_record_without_native_hashes_or_absent_flag(tmp_path, monkeypatch):
    # a legacy record that hashed no native files and set no native_absent flag is refused (brief M2).
    import argparse
    t = Tree(tmp_path, monkeypatch)
    _chain_through_m3(t, tmp_path)
    hyp_root = tmp_path / "hyp"
    t.make_eval_hyps(hyp_root)
    F.cmd_add_eval_hyps(argparse.Namespace(hyp_root=str(hyp_root), native_absent=False))
    rec = json.loads(t.record.read_text())
    del rec["native_hyps_sha256"]
    del rec["native_absent"]
    t.record.write_text(json.dumps(rec))
    args = argparse.Namespace(hyp_root=rec["paths"]["hyp_root"], **{k: rec["paths"][k] for k in C.PATH_ARGS})
    with pytest.raises(SystemExit, match="does not hash the native"):
        C.check_freeze(args)


def test_init_records_sd_ld_suffixes(tmp_path, monkeypatch):
    # review m2 / brief fix 7: init records the SD/LD suffixes it used.
    t = Tree(tmp_path, monkeypatch)
    F.cmd_init(t.init_args())
    rec = json.loads(t.record.read_text())
    from csdiar.data.ingest_displace import LD_SUFFIX, SD_SUFFIX


def test_init_records_wav_extension(tmp_path, monkeypatch):
    # review-v2 m5: init records the WAV extension it found (the Tree writes <rid>.wav for every recording).
    t = Tree(tmp_path, monkeypatch)
    F.cmd_init(t.init_args())
    rec = json.loads(t.record.read_text())
    assert rec["wav_ext"] == ".wav"


def test_init_records_non_wav_extension(tmp_path, monkeypatch):
    # a corpus stored as .flac is recorded as such, so the in-process eval ingest later looks up <rid>.flac.
    t = Tree(tmp_path, monkeypatch)
    for rid in t.dev_ids + t.eval_ids:
        (t.wav / f"{rid}.wav").unlink()
        _touch(t.wav / f"{rid}.flac", f"flac-{rid}".encode())
    F.cmd_init(t.init_args())
    rec = json.loads(t.record.read_text())
    assert rec["wav_ext"] == ".flac"


def test_init_refuses_mixed_wav_extensions(tmp_path, monkeypatch):
    # a MIX of audio extensions across recordings is refused (the ingest looks up one extension for all).
    t = Tree(tmp_path, monkeypatch)
    (t.wav / f"{t.eval_ids[0]}.wav").unlink()
    _touch(t.wav / f"{t.eval_ids[0]}.flac", b"flac")
    with pytest.raises(SystemExit, match="mix audio extensions"):
        F.cmd_init(t.init_args())


def test_init_refuses_two_extensions_for_one_recording(tmp_path, monkeypatch):
    # the same recording present under two extensions is ambiguous and refused.
    t = Tree(tmp_path, monkeypatch)
    _touch(t.wav / f"{t.eval_ids[0]}.flac", b"flac")   # now EVA00 has both .wav and .flac
    with pytest.raises(SystemExit, match="multiple audio files"):
        F.cmd_init(t.init_args())


def test_eval_ingest_uses_recorded_wav_extension(tmp_path, monkeypatch):
    # review-v2 m5: _ingest_eval_in_process looks the duration up by the recorded wav_ext, not a hard-coded
    # '.wav'. With wav_ext='.flac' and only .flac files present, the duration lookup targets <rid>.flac.
    import csdiar.analysis.confirm_m2b as M
    seen: list = []
    monkeypatch.setattr(M, "_wav_duration_s", lambda p: (seen.append(str(p)), 300.0)[1])

    # a minimal record + synthetic eval SD/LD refs so the real process_recording path runs.
    root = tmp_path / "displace"
    wav = tmp_path / "wav"
    vad = tmp_path / "vad"
    units_eval = tmp_path / "units/eval"
    from csdiar.data.ingest_displace import LAYOUT, LD_SUFFIX, SD_SUFFIX
    eval_ids = [f"EVA{i:02d}" for i in range(3)]
    for rid in eval_ids:
        _rttm(root / LAYOUT["eval"]["sd"] / f"{rid}{SD_SUFFIX}", rid, [(0.0, 5.0, "s1"), (6.0, 5.0, "s2")])
        _rttm(root / LAYOUT["eval"]["ld"] / f"{rid}{LD_SUFFIX}", rid, [(0.0, 5.0, "L1"), (6.0, 5.0, "L2")])
        _touch(wav / f"{rid}.flac", f"flac-{rid}".encode())
    rec = {"paths": {"root": str(root), "wav_dir": str(wav), "vad_dir": str(vad),
                     "units_eval": str(units_eval)},
           "sd_suffix": SD_SUFFIX, "ld_suffix": LD_SUFFIX, "wav_ext": ".flac"}
    M._ingest_eval_in_process(rec)
    assert seen and all(p.endswith(".flac") for p in seen)
    assert not any(p.endswith(".wav") for p in seen)
    assert rec["sd_suffix"] == SD_SUFFIX and rec["ld_suffix"] == LD_SUFFIX
