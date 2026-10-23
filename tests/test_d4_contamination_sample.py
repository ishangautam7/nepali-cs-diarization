"""Tests for scripts/d4_contamination_sample.py (stdlib + pytest + numpy via csdiar.data.boundaries; no data)."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

import csdiar.data.boundaries as boundaries
from csdiar.data.tags import words as split_words

SPEC = importlib.util.spec_from_file_location(
    "d4_contamination_sample", Path(__file__).resolve().parents[1] / "scripts/d4_contamination_sample.py")
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)


def ok(clip_id: str, text: str) -> dict:
    """An alignment record as csdiar.data.align writes it for a clip that aligned (every word kept)."""
    ws = [{"w": w, "aligned": True, "start": 0.1 * k, "end": 0.1 * k + 0.08, "score": 0.9}
          for k, w in enumerate(split_words(text))]
    return {"clip_id": clip_id, "status": "ok", "duration": 3.0, "words": ws}


def write_align(folder: Path, split: str, recs: list[dict]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{split}.jsonl", "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_lexicon(folder: Path, pair: str, loans: list[str], names: list[str]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{pair}_loans.txt").write_text("# header\n" + "".join(f"{w}  # 1 gloss\n" for w in loans), "utf-8")
    (folder / f"{pair}_names.txt").write_text("".join(f"{w}\n" for w in names), "utf-8")
    return folder


def test_only_ok_clips_and_native_tags_are_kept(tmp_path):
    align = tmp_path / "hi-en/align"
    write_align(align, "train", [
        ok("c002", "अब हम क्लिक करेंगे Python में, राम x फ़ाइल 42"),
        {"clip_id": "c001", "status": "error", "error": "boom", "words": []},
        {"clip_id": "c003", "status": "no_alignable_words", "words": [{"w": "हम", "aligned": False}]},
    ])
    write_align(align, "test", [{"clip_id": "c000", "status": "empty", "words": []}, ok("c004", "नमस्ते दोस्तों")])
    lex = write_lexicon(tmp_path / "lex", "hi-en", ["क्लिक"], ["राम"])
    loans, _ = boundaries.read_list(lex / "hi-en_loans.txt")
    names, _ = boundaries.read_list(lex / "hi-en_names.txt")
    tokens, counts = S.native_tokens(align, "hi-en", loans, names)
    # the loan (क्लिक), the name (राम), Latin (Python), a single Latin letter (x) and digits (42) are not native;
    # the error and no_alignable_words clips are skipped, as the boundary run skips them
    assert [(t["split"], t["clip_id"], t["word"]) for t in tokens] == [
        ("train", "c002", "अब"), ("train", "c002", "हम"), ("train", "c002", "करेंगे"), ("train", "c002", "में"),
        ("train", "c002", "फ़ाइल"), ("test", "c004", "नमस्ते"), ("test", "c004", "दोस्तों")]
    assert counts["train"]["status"] == {"error": 1, "no_alignable_words": 1, "ok": 1}
    assert counts["test"]["status"] == {"empty": 1, "ok": 1}
    assert counts["train"]["words_in_ok_clips"] == 10 and counts["native_tokens"] == 7
    assert len(counts["train"]["align_sha256"]) == 64
    assert tokens[2]["word_index"] == 3 and tokens[2]["context"].split().count("«करेंगे»") == 1


def test_a_clip_written_twice_keeps_its_last_record(tmp_path):
    align = tmp_path / "hi-en/align"
    write_align(align, "train", [{"clip_id": "c1", "status": "error", "words": []}, ok("c1", "हम")])
    write_align(align, "test", [])
    tokens, counts = S.native_tokens(align, "hi-en", frozenset(), frozenset())
    assert [t["word"] for t in tokens] == ["हम"] and counts["train"]["clips"] == 1


def test_the_tagger_swap_is_undone(tmp_path):
    align = tmp_path / "bn-en/align"
    write_align(align, "train", [ok("c1", "আমরা ফাংশন লিখব আর हम")])
    write_align(align, "test", [])
    before = boundaries.tag_word
    tokens, _ = S.native_tokens(align, "bn-en", frozenset({"ফাংশন"}), frozenset())
    assert boundaries.tag_word is before
    assert [t["word"] for t in tokens] == ["আমরা", "লিখব", "আর"]      # Devanagari is another script for bn-en


def test_a_malformed_ok_record_is_refused(tmp_path):
    align = tmp_path / "hi-en/align"
    bad = ok("c1", "हम दोनों")
    del bad["words"][1]["start"]
    write_align(align, "train", [bad])
    write_align(align, "test", [])
    before = boundaries.tag_word
    with pytest.raises(SystemExit, match="aligned word without timings"):
        S.native_tokens(align, "hi-en", frozenset(), frozenset())
    assert boundaries.tag_word is before                             # restored on the error path too


def test_draw_is_seeded_fixed_size_and_marks_the_check_subset():
    tokens = [{"split": "train", "clip_id": f"c{k}", "word_index": 0, "word": f"w{k}", "context": "x"}
              for k in range(1000)]
    a = S.draw(tokens, 200, 50, S.SEED)
    assert a == S.draw(tokens, 200, 50, S.SEED) and len(a) == 200 and sum(r["check"] for r in a) == 50
    assert len({r["clip_id"] for r in a}) == 200                       # without replacement
    assert [r["item"] for r in a] == list(range(1, 201))
    assert S.draw(tokens, 200, 50, S.SEED + 1) != a


def test_too_small_a_pool_is_refused():
    with pytest.raises(SystemExit, match="fewer than the 200"):
        S.draw([{"word": "w"}] * 10, 200, 50, S.SEED)


def test_main_writes_items_summary_and_instructions(tmp_path):
    text = " ".join(f"शब्द{chr(0x0915 + k % 30)}" for k in range(150))
    write_align(tmp_path / "d4/hi-en/align", "train", [ok("c1", text)])
    write_align(tmp_path / "d4/hi-en/align", "test", [ok("c2", text)])
    lex = write_lexicon(tmp_path / "lex", "hi-en", ["क्लिक"], ["राम"])
    assert S.main(["--d4-root", str(tmp_path / "d4"), "--lexicon", str(lex), "--out", str(tmp_path / "out"),
                   "--pairs", "hi-en"]) == 0
    rows = list(csv.DictReader(open(tmp_path / "out/hi-en_items.csv", encoding="utf-8")))
    assert len(rows) == 200 and sum(int(r["check"]) for r in rows) == 50 and {r["split"] for r in rows} == {"train", "test"}
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["seed"] == 20261018 and summary["pairs"]["hi-en"]["native_tokens"] == 300
    assert len(summary["pairs"]["hi-en"]["loans_sha256"]) == 64
    assert "LOAN" in (tmp_path / "out/instructions.md").read_text()
