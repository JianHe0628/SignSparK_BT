"""Collation and masking, which replaced torchtext's Field/BucketIterator."""
import torch

from signspark_bt.data.batch import Batch, SignCollator, lengths_to_mask
from signspark_bt.data.lmdb_dataset import Example
from signspark_bt.data.vocabulary import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    GlossVocabulary,
    TextVocabulary,
)

FEATURE_SIZE = 240


def make_example(num_frames, txt, gls):
    return Example(
        sequence=f"clip{num_frames}",
        signer="signer",
        gls=gls,
        txt=txt,
        num_frames=num_frames,
        sgn=torch.ones(num_frames, FEATURE_SIZE),
    )


def make_collator():
    txt_vocab = TextVocabulary(tokens=["hello", "world", "there"])
    gls_vocab = GlossVocabulary(tokens=["HELLO", "WORLD"])
    return SignCollator(txt_vocab=txt_vocab, gls_vocab=gls_vocab), txt_vocab, gls_vocab


def test_lengths_to_mask_marks_padding():
    mask = lengths_to_mask(torch.tensor([3, 1]), 4)
    assert mask.shape == (2, 1, 4)
    assert mask[0, 0].tolist() == [True, True, True, False]
    assert mask[1, 0].tolist() == [True, False, False, False]


def test_collate_pads_to_longest_sequence():
    collator, _, _ = make_collator()
    batch = collator([
        make_example(5, ["hello"], ["HELLO"]),
        make_example(2, ["world", "there"], ["WORLD"]),
    ])
    sgn, lengths = batch.sgn
    assert sgn.shape == (2, 5, FEATURE_SIZE)
    assert lengths.tolist() == [5, 2]
    # padded frames are zero
    assert sgn[1, 2:].abs().sum() == 0


def test_text_is_wrapped_in_bos_and_eos():
    collator, txt_vocab, _ = make_collator()
    batch = collator([make_example(3, ["hello", "world"], ["HELLO"])])
    txt, lengths = batch.txt
    ids = txt[0].tolist()
    assert ids[0] == txt_vocab.stoi[BOS_TOKEN]
    assert ids[len(ids) - 1] == txt_vocab.stoi[EOS_TOKEN]
    assert lengths.tolist() == [4]


def test_unknown_words_map_to_unk():
    collator, txt_vocab, _ = make_collator()
    batch = collator([make_example(3, ["notinvocab"], ["HELLO"])])
    txt, _ = batch.txt
    assert txt[0, 1].item() == txt_vocab.DEFAULT_UNK_ID()


def test_batch_mask_comes_from_lengths_not_zero_features():
    """A genuinely zero frame inside a sequence must stay unmasked.

    The old implementation derived the mask from non-zero features, which is
    why every feature had 1e-8 added to it.
    """
    collator, txt_vocab, _ = make_collator()
    example = make_example(4, ["hello"], ["HELLO"])
    example.sgn[2] = 0.0  # a real, all-zero frame
    raw = collator([example])
    batch = Batch(
        torch_batch=raw,
        txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
        sgn_dim=FEATURE_SIZE,
        use_cuda=False,
    )
    assert batch.sgn_mask[0, 0].tolist() == [True, True, True, True]


def test_batch_shifts_text_for_teacher_forcing():
    collator, txt_vocab, _ = make_collator()
    raw = collator([make_example(3, ["hello", "world"], ["HELLO"])])
    batch = Batch(
        torch_batch=raw,
        txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
        sgn_dim=FEATURE_SIZE,
        use_cuda=False,
    )
    # input drops the last token, target drops BOS
    assert batch.txt_input.shape == batch.txt.shape
    assert batch.txt_input[0, 0].item() == txt_vocab.stoi[BOS_TOKEN]
    assert batch.txt[0, -1].item() == txt_vocab.stoi[EOS_TOKEN]
    assert batch.num_seqs == 1
