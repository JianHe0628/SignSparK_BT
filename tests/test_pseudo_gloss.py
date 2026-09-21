"""Pseudo-gloss generation for datasets without glosses (How2Sign).

Uses a stub tagger so the tests need no spaCy model download.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from make_pseudo_glosses import SELECTED_POS, to_pseudo_gloss  # noqa: E402


class FakeToken:
    def __init__(self, lemma, pos):
        self.lemma_, self.pos_ = lemma, pos


def fake_nlp(tagged):
    """Build an nlp callable returning fixed (lemma, pos) pairs."""
    return lambda sentence: [FakeToken(lemma, pos) for lemma, pos in tagged]


def test_keeps_only_content_parts_of_speech():
    nlp = fake_nlp([("the", "DET"), ("dog", "NOUN"), ("be", "AUX"),
                    ("in", "ADP"), ("park", "NOUN"), (".", "PUNCT")])
    assert to_pseudo_gloss("...", nlp) == "dog park"


def test_pronouns_are_kept():
    """PRON is in the whitelist: 'you' and 'we' survive."""
    nlp = fake_nlp([("you", "PRON"), ("and", "CCONJ"), ("we", "PRON"), ("go", "VERB")])
    assert to_pseudo_gloss("...", nlp) == "you we go"


def test_output_is_lowercase_lemmas():
    nlp = fake_nlp([("Dog", "NOUN"), ("RUN", "VERB")])
    assert to_pseudo_gloss("...", nlp) == "dog run"


def test_exclude_list_drops_extra_lemmas():
    """German drops 'es' and 'sich' on top of the POS filter."""
    nlp = fake_nlp([("es", "PRON"), ("regnen", "VERB"), ("sich", "PRON")])
    assert to_pseudo_gloss("...", nlp, exclude=("es", "sich")) == "regnen"


def test_sentence_with_no_content_words_is_empty():
    nlp = fake_nlp([("the", "DET"), ("of", "ADP"), ("!", "PUNCT")])
    assert to_pseudo_gloss("...", nlp) == ""


def test_whitelist_matches_the_original_notebook():
    assert SELECTED_POS == ["NOUN", "NUM", "ADV", "PRON", "PROPN", "ADJ", "VERB"]


def test_output_is_whitespace_tokenisable():
    """The loader splits glosses on whitespace, so output must survive that."""
    nlp = fake_nlp([("today", "NOUN"), ("show", "VERB"), ("paper", "NOUN")])
    assert to_pseudo_gloss("...", nlp).split() == ["today", "show", "paper"]
