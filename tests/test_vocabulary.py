"""Vocabulary construction, now independent of torchtext."""
from signspark_bt.data.vocabulary import (
    PAD_TOKEN,
    SIL_TOKEN,
    UNK_TOKEN,
    GlossVocabulary,
    TextVocabulary,
    build_vocab,
)


class FakeDataset:
    """Stands in for a dataset, exposing the `.examples` build_vocab reads."""

    class Example:
        def __init__(self, gls, txt):
            self.gls, self.txt = gls, txt

    def __init__(self, rows):
        self.examples = [self.Example(gls, txt) for gls, txt in rows]


def test_gloss_vocabulary_keeps_silence_at_zero():
    """CTC decoding requires the blank symbol to be index 0."""
    vocab = GlossVocabulary(tokens=["HELLO", "WORLD"])
    assert vocab.stoi[SIL_TOKEN] == 0


def test_text_vocabulary_maps_unknown_words_to_unk():
    vocab = TextVocabulary(tokens=["hello"])
    assert vocab.is_unk("absent")
    assert not vocab.is_unk("hello")
    assert vocab.itos[vocab.DEFAULT_UNK_ID()] == UNK_TOKEN


def test_build_vocab_from_dataset_respects_min_freq():
    dataset = FakeDataset([
        (["A"], ["common", "rare"]),
        (["A"], ["common"]),
        (["B"], ["common"]),
    ])
    vocab = build_vocab(field="txt", max_size=100, min_freq=2, dataset=dataset)
    assert "common" in vocab.stoi.keys()
    assert vocab.is_unk("rare")


def test_build_vocab_respects_max_size():
    dataset = FakeDataset([(["A", "B", "C", "D"], ["w"])])
    vocab = build_vocab(field="gls", max_size=2, min_freq=1, dataset=dataset)
    # two tokens plus the specials
    assert len(vocab) == 2 + len(vocab.specials)


def test_round_trip_through_file(tmp_path="/tmp"):
    import os
    vocab = TextVocabulary(tokens=["hello", "world"])
    path = os.path.join(tmp_path, "_vocab_roundtrip.txt")
    try:
        vocab.to_file(path)
        reloaded = TextVocabulary(file=path)
        assert reloaded.itos == vocab.itos
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_arrays_to_sentences_cuts_at_eos():
    vocab = TextVocabulary(tokens=["hello", "world"])
    ids = [vocab.stoi["hello"], vocab.stoi["world"], vocab.stoi["</s>"], vocab.stoi["hello"]]
    assert vocab.arrays_to_sentences([ids])[0] == ["hello", "world"]
