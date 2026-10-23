"""Tests for csdiar.data.tags_d4 (text only)."""
from csdiar.data.tags import EN, LOAN, NE, NEUTRAL
from csdiar.data.tags_d4 import make_tagger


def test_devanagari_pair():
    t = make_tagger("devanagari")
    assert t("सीखेंगे") == NE and t("tutorial") == EN and t("334") == NEUTRAL and t("gnu/linux") == EN
    assert t("ऑफिस", loans=frozenset({"ऑफिस"})) == LOAN
    assert t("x", letters_neutral=True) == NEUTRAL and t("a", letters_neutral=True) == EN
    assert t("আমি") == NEUTRAL  # Bengali script is "other" for the Hindi pair


def test_bengali_pair():
    t = make_tagger("bengali")
    assert t("আমরা") == NE and t("slide") == EN and t("सीखेंगे") == NEUTRAL
    assert t("স্লাইড", loans=frozenset({"স্লাইড"})) == LOAN
