from csdiar.data.tags import EN, LOAN, NE, NEUTRAL, Run, adjacent_pairs, runs, switches, tag_word, words


def test_words_strip_punctuation_and_drop_silent_tokens():
    assert words("यो algorithm हो। , ok?") == ["यो", "algorithm", "हो", "ok"]
    assert words("।  ॥ --") == []


def test_tag_by_script():
    assert tag_word("कम्प्युटर") == NE  # vowel signs and virama do not change the script
    assert tag_word("computer") == EN
    assert tag_word("Don't") == EN
    assert tag_word("२०२४") == NEUTRAL  # Devanagari digits
    assert tag_word("2024") == NEUTRAL
    assert tag_word("COVID19") == NEUTRAL  # letters plus digits
    assert tag_word("algorithmको") == NEUTRAL  # mixed script inside one word
    assert tag_word("क्\u200dष") == NE  # zero-width joiner is ignored


def test_loan_lexicon_takes_precedence():
    loans = frozenset({"सो", "okay"})
    assert tag_word("सो", loans) == LOAN
    assert tag_word("Okay,", loans) == LOAN
    assert tag_word("सो") == NE


def test_neutral_words_do_not_break_runs():
    tags = [NE, NE, EN, NEUTRAL, EN, NE]
    assert runs(tags) == [Run(NE, 0, 1, 2), Run(EN, 2, 4, 2), Run(NE, 5, 5, 1)]
    sw = switches(tags)
    assert [(s.left.lang, s.right.lang, s.neutral_between) for s in sw] == [(NE, EN, 0), (EN, NE, 0)]


def test_loan_breaks_runs_and_blocks_switches():
    assert runs([NE, LOAN, NE]) == [Run(NE, 0, 0, 1), Run(NE, 2, 2, 1)]
    assert switches([NE, LOAN, EN]) == []


def test_neutral_between_runs_is_counted():
    (sw,) = switches([NE, NEUTRAL, NEUTRAL, EN])
    assert sw.neutral_between == 2


def test_adjacent_pairs_ignore_neutral():
    assert adjacent_pairs([NE, NEUTRAL, EN, EN, NE]) == (2, 3)
