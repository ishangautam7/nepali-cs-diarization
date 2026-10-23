"""Word-level language tags for D4 (MUCS 2021 Hindi-English and Bengali-English; the H7 supplement).

The convention of csdiar.data.tags (D1), with the native script as a parameter. A word written wholly in the
pair's native script is tagged NE, the label the boundary code uses for "the native language". A wholly Latin word
is EN. Digits, mixed-script and other-script words are NEUTRAL. Words in the pair's loan lexicon (English words
written in the native script) are LOAN. The single-letter rule of decision 12 applies as in D1.
The frozen D1 tagger (csdiar.data.tags) is not modified; boundaries_d4 swaps this tagger in for D4 runs.
"""
from __future__ import annotations

import unicodedata
from typing import Callable

from csdiar.data.tags import EN, LOAN, NE, NEUTRAL, loan_key

SCRIPTS = {"devanagari": ((0x0900, 0x097F), (0xA8E0, 0xA8FF)), "bengali": ((0x0980, 0x09FF),)}
PAIR_SCRIPT = {"hi-en": "devanagari", "bn-en": "bengali"}
PAIR_UROMAN = {"hi-en": "hin", "bn-en": "ben"}


def char_class(ch: str, script: str) -> str:
    cat = unicodedata.category(ch)
    if cat[0] in "PSZC":
        return "punct"
    if cat[0] == "N":
        return "digit"
    if cat[0] == "M":
        return "mark"
    if cat[0] == "L":
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in SCRIPTS[script]):
            return "native"
        return "latin" if unicodedata.name(ch, "").startswith("LATIN") else "other"
    return "other"


def make_tagger(script: str) -> Callable[..., str]:
    """A tag_word with the signature of csdiar.data.tags.tag_word, for the given native script."""
    if script not in SCRIPTS:
        raise ValueError(script)

    def tag_word(word: str, loans: frozenset[str] = frozenset(), names: frozenset[str] = frozenset(),
                 letters_neutral: bool = False) -> str:
        key = loan_key(word) if (loans or names) else ""
        if names and key in names:
            return NEUTRAL
        if loans and key in loans:
            return LOAN
        classes = {char_class(c, script) for c in word} - {"mark", "punct"}
        if classes == {"native"}:
            return NE
        if classes == {"latin"}:
            letters = [c for c in word if char_class(c, script) == "latin"]
            if letters_neutral and len(letters) == 1 and letters[0] != "a":
                return NEUTRAL
            return EN
        return NEUTRAL

    return tag_word
