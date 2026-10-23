"""Tests for scripts/sensitivity_variants.py (stdlib + pytest + numpy via csdiar.data.boundaries; no data)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import csdiar.data.boundaries as B
from csdiar.data.tags import EN, LOAN, NE, NEUTRAL, tag_word

SPEC = importlib.util.spec_from_file_location(
    "sensitivity_variants", Path(__file__).resolve().parents[1] / "scripts/sensitivity_variants.py")
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)


def test_retag_ia_changes_only_I_and_capital_A():
    t = V.retag_ia(tag_word)
    assert t("I", letters_neutral=True) == EN and t("A", letters_neutral=True) == EN
    assert t("i", letters_neutral=True) == NEUTRAL and t("x", letters_neutral=True) == NEUTRAL
    assert t("a", letters_neutral=True) == EN and t("Ohm", letters_neutral=True) == EN
    assert t("नमस्ते", letters_neutral=True) == NE
    assert t("I", names=frozenset({"i"}), letters_neutral=True) == NEUTRAL      # a listed name stays a name
    words = ["I", "A", "i", "x", "a", "two", "म", "R2"]
    assert [t(w, letters_neutral=True) for w in words if w not in ("I", "A")] == \
           [tag_word(w, letters_neutral=True) for w in words if w not in ("I", "A")]


def test_loan_lang_reincludes_loans_with_their_spoken_language():
    loans = frozenset({"कम्प्युटर", "ho", "abcक"})
    t = V.loan_lang(tag_word)
    assert tag_word("कम्प्युटर", loans) == LOAN and t("कम्प्युटर", loans) == EN     # Devanagari loan -> en
    assert t("Ho", loans) == NE                                                    # Latin entry -> ne
    assert t("abcक", loans) == LOAN                                                # mixed script stays loan
    assert t("hello", loans) == EN and t("नमस्ते", loans) == NE
    assert t("राम", loans, frozenset({"राम"})) == NEUTRAL                          # names still come first


def test_low_align_scores_every_aligned_clip_infinite():
    s = V.low_align(B.clip_score)
    ok = {"status": "ok", "words": [{"w": "a", "aligned": True, "score": 0.01}]}
    assert B.clip_score(ok) == pytest.approx(0.01) and s(ok) == float("inf")
    assert s({"status": "ok", "words": [{"w": "a", "aligned": False}]}) is None
    assert s({"status": "error", "words": []}) is None


@pytest.mark.parametrize("variant,name", [("retag_ia", "tag_word"), ("loan_lang", "tag_word"),
                                          ("low_align", "clip_score")])
def test_the_swap_is_undone_even_on_error(variant, name):
    frozen = getattr(B, name)
    with pytest.raises(RuntimeError):
        with V.swapped(variant):
            assert getattr(B, name) is not frozen
            raise RuntimeError("boom")
    assert getattr(B, name) is frozen


def test_boundary_command_is_the_frozen_one_with_a_new_out(tmp_path):
    rebuild = (Path(V.REPO) / "ops/rebuild_features.sh").read_text()
    assert 'step d1_boundaries cpu "$PY4" -m csdiar.data.boundaries --root "$D1" --align "$D1/align" --vad "$D1/vad"' in rebuild
    assert 'LEX=(--loans planning/lexicon/loans.txt --names planning/lexicon/names.txt)' in rebuild
    argv = V.boundary_argv(tmp_path, tmp_path / "boundaries/lex_v1b_retag_ia")
    assert argv[argv.index("--out") + 1].endswith("lex_v1b_retag_ia")
    assert argv[argv.index("--splits") + 1:] == ["validation", "train", "test"]
    assert argv[argv.index("--loans") + 1].endswith("planning/lexicon/loans.txt")


def test_run_boundaries_swaps_in_the_variant_and_refuses_an_existing_output(tmp_path, monkeypatch):
    seen = {}

    def fake_main():
        seen["argv"], seen["tag_word"] = list(sys.argv), B.tag_word
        return 0
    monkeypatch.setattr(B, "main", fake_main)
    frozen = B.tag_word
    out = V.run_boundaries("retag_ia", tmp_path)
    assert out == tmp_path / "boundaries/lex_v1b_retag_ia"
    assert seen["tag_word"] is not frozen and seen["tag_word"]("I", letters_neutral=True) == EN
    assert B.tag_word is frozen and sys.argv[0] != "csdiar.data.boundaries"
    out.mkdir(parents=True)
    with pytest.raises(SystemExit, match="exists"):
        V.run_boundaries("retag_ia", tmp_path)


def test_the_pilot_file_must_be_the_frozen_one(tmp_path):
    V.check_pilot(V.PILOT)                                        # the committed copy matches FREEZE_M2A.json
    bad = tmp_path / "pilot_v2.json"
    bad.write_bytes(V.PILOT.read_bytes() + b" ")
    with pytest.raises(SystemExit, match="not the pilot file"):
        V.check_pilot(bad)


def _fake_confirm(monkeypatch, point: dict) -> dict:
    C = pytest.importorskip("csdiar.analysis.confirm")
    seen: dict = {}

    def load_points(bdir, splits):
        seen["load"] = (bdir, list(splits))
        return {"p1": point}, []

    def primary_block(pts, matches, splits, args, taus, rng):
        seen["args"], seen["pts"] = args, pts
        return {"h1": {}, "h2_p0": {}, "h3": {}}

    monkeypatch.setattr(C, "frozen_taus", lambda path: {"e3": 0.9})
    monkeypatch.setattr(C, "load_points", load_points)
    monkeypatch.setattr(C, "primary_block", primary_block)
    monkeypatch.setattr(C, "write_outputs", lambda out, args: seen.setdefault("out", out))
    return seen


POINT = {"split": "train", "score_left": 0.9, "score_right": 0.8, "pause_bin": 0, "pos_bin": 1, "speaker_id": "s",
         "video_id": "v", "clip_id": "c", "low_align": False, "elig_det": True}


def test_analyse_hands_the_variant_inputs_to_the_frozen_primary_block(tmp_path, monkeypatch):
    seen = _fake_confirm(monkeypatch, POINT)
    V.analyse("loan_lang", tmp_path)
    a = seen["args"]
    assert a.boundaries == tmp_path / "boundaries/lex_v1b_loan_lang" == seen["load"][0]
    assert a.point_emb == tmp_path / "point_emb/lex_v1b_loan_lang"
    assert (a.emb, a.windows, a.align) == (tmp_path / "emb", tmp_path / "windows/lex_v1", tmp_path / "align")
    assert a.splits == seen["load"][1] == ["train", "test"] and a.seed == 20261018 and a.pilot == V.PILOT
    assert "item 8" in seen["out"]["mode"] and seen["out"]["points"] == 1


def test_analyse_base_reads_the_rebuilt_unvaried_boundaries(tmp_path, monkeypatch):
    seen = _fake_confirm(monkeypatch, POINT)
    V.analyse("base", tmp_path)
    assert seen["args"].boundaries == tmp_path / "boundaries/lex_v1b"
    assert seen["args"].point_emb == tmp_path / "point_emb/lex_v1b"
    assert "reference run" in seen["out"]["mode"]
    with pytest.raises(SystemExit, match="no boundary run"):
        V.main(["boundaries", "--variant", "base", "--d1", str(tmp_path)])


def test_analyse_keeps_confirms_input_checks(tmp_path, monkeypatch):
    _fake_confirm(monkeypatch, dict(POINT, low_align=True))
    with pytest.raises(SystemExit, match="low-alignment point is marked eligible"):
        V.analyse("retag_ia", tmp_path)
    _fake_confirm(monkeypatch, {k: v for k, v in POINT.items() if k != "pos_bin"})
    with pytest.raises(SystemExit, match="lack"):
        V.analyse("retag_ia", tmp_path)
