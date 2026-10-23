"""Word-level language tags from script, following the D1 transcription convention (PLAN.md §2).

The corpus writes English words in Latin script and Nepali words in Devanagari, so a word's script is
its language tag. Digits, mixed-script words and other scripts are `neutral`. Words found in a loan
lexicon are `loan`.

A *run* is a maximal sequence of same-language words. Neutral words do not break a run; loan words
do, and two runs separated by a loan never form a clean switch.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Iterable, Sequence

NE = "ne"
EN = "en"
NEUTRAL = "neutral"
LOAN = "loan"


def _char_class(ch: str) -> str:
    cat = unicodedata.category(ch)
    if cat[0] in "PSZC":  # punctuation (incl. danda), symbols, separators, format chars such as ZWJ
        return "punct"
    if cat[0] == "N":
        return "digit"
    if cat[0] == "M":  # vowel signs, virama, nukta: they belong to the surrounding letters
        return "mark"
    if cat[0] == "L":
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F or 0xA8E0 <= cp <= 0xA8FF:
            return "deva"
        return "latin" if unicodedata.name(ch, "").startswith("LATIN") else "other"
    return "other"


def _strip_punct(token: str) -> str:
    start, end = 0, len(token)
    while start < end and _char_class(token[start]) == "punct":
        start += 1
    while end > start and _char_class(token[end - 1]) == "punct":
        end -= 1
    return token[start:end]


def words(text: str) -> list[str]:
    """Split a transcript into words: whitespace tokens with edge punctuation removed.

    Tokens without any letter or digit (e.g. a lone danda) are dropped, because nothing is spoken.
    """
    out = []
    for token in unicodedata.normalize("NFC", text).split():
        core = _strip_punct(token)
        if any(_char_class(c) in ("deva", "latin", "other", "digit") for c in core):
            out.append(core)
    return out


def loan_key(word: str) -> str:
    """Normalised form used to look a word up in the loan lexicon."""
    return unicodedata.normalize("NFC", _strip_punct(word)).lower()


def tag_word(word: str, loans: frozenset[str] = frozenset(), names: frozenset[str] = frozenset(),
             letters_neutral: bool = False) -> str:
    """Language tag of one word. Proper names (curated list) are neutral; loan-lexicon words are loan.

    With letters_neutral, a single Latin letter other than the article "a" is neutral: in these STEM
    lectures it is almost always a variable or a symbol (decision 12).
    """
    key = loan_key(word) if (loans or names) else ""
    if names and key in names:
        return NEUTRAL
    if loans and key in loans:
        return LOAN
    classes = {_char_class(c) for c in word} - {"mark", "punct"}
    if classes == {"deva"}:
        return NE
    if classes == {"latin"}:
        letters = [c for c in word if _char_class(c) == "latin"]
        if letters_neutral and len(letters) == 1 and letters[0] != "a":
            return NEUTRAL
        return EN
    return NEUTRAL


def tag_words(ws: Iterable[str], loans: frozenset[str] = frozenset(),
              names: frozenset[str] = frozenset(), letters_neutral: bool = False) -> list[str]:
    return [tag_word(w, loans, names, letters_neutral) for w in ws]


@dataclass(frozen=True)
class Run:
    lang: str  # NE or EN
    first: int  # index of the first word of the run
    last: int  # index of the last word of the run (inclusive)
    n_words: int  # language words in the run; neutral words inside it are not counted


@dataclass(frozen=True)
class Switch:
    left: Run
    right: Run
    neutral_between: int  # neutral words between the two runs (boundary timing is then ambiguous)


def runs(tags: Sequence[str]) -> list[Run]:
    out: list[Run] = []
    lang = first = last = None
    n = 0

    def close() -> None:
        if lang is not None:
            out.append(Run(lang, first, last, n))

    for i, tag in enumerate(tags):
        if tag == NEUTRAL:
            continue
        if tag == LOAN:
            close()
            lang, n = None, 0
            continue
        if tag == lang:
            last, n = i, n + 1
        else:
            close()
            lang, first, last, n = tag, i, i, 1
    close()
    return out


def switches(tags: Sequence[str]) -> list[Switch]:
    rs = runs(tags)
    out = []
    for a, b in zip(rs, rs[1:]):
        between = tags[a.last + 1 : b.first]
        if a.lang != b.lang and LOAN not in between:
            out.append(Switch(a, b, sum(t == NEUTRAL for t in between)))
    return out


def adjacent_pairs(tags: Sequence[str]) -> tuple[int, int]:
    """(language-changing pairs, all pairs) over consecutive ne/en words, ignoring neutral words.

    This is the corpus card's "language-switch rate of adjacent token pairs", so the tagger can be
    checked against the published statistics.
    """
    lang_tags = [t for t in tags if t in (NE, EN)]
    pairs = list(zip(lang_tags, lang_tags[1:]))
    return sum(a != b for a, b in pairs), len(pairs)
