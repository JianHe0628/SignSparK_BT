"""CTC decoding, which replaced the TensorFlow beam search decoder."""
import torch

from signspark_bt.model.ctc_decode import (
    ctc_beam_search_decode,
    ctc_decode,
    ctc_greedy_decode,
)


def _one_hot(path, num_classes=4):
    """Build near-deterministic log probabilities following a class path."""
    probs = torch.full((len(path), 1, num_classes), 1e-6)
    for t, c in enumerate(path):
        probs[t, 0, c] = 1.0
    return probs.log_softmax(-1)


def test_greedy_collapses_repeats_and_drops_blanks():
    # 1,1 collapses to one 1; the blank separates it from the next 1.
    log_probs = _one_hot([1, 1, 0, 1, 2, 2, 0])
    assert ctc_greedy_decode(log_probs, torch.tensor([7])) == [[1, 1, 2]]


def test_greedy_respects_sequence_length():
    log_probs = _one_hot([1, 0, 2, 0, 3])
    assert ctc_greedy_decode(log_probs, torch.tensor([3])) == [[1, 2]]


def test_all_blank_decodes_to_empty():
    log_probs = _one_hot([0, 0, 0])
    assert ctc_greedy_decode(log_probs, torch.tensor([3])) == [[]]


def test_beam_of_one_matches_greedy():
    torch.manual_seed(0)
    log_probs = torch.randn(12, 3, 5).log_softmax(-1)
    lengths = torch.tensor([12, 9, 5])
    assert ctc_beam_search_decode(log_probs, lengths, beam_width=1) == ctc_greedy_decode(
        log_probs, lengths
    )


def test_ctc_decode_dispatches_on_beam_size():
    torch.manual_seed(1)
    log_probs = torch.randn(8, 2, 4).log_softmax(-1)
    lengths = torch.tensor([8, 6])
    assert ctc_decode(log_probs, lengths, beam_size=1) == ctc_greedy_decode(log_probs, lengths)
    assert ctc_decode(log_probs, lengths, beam_size=3) == ctc_beam_search_decode(
        log_probs, lengths, 3
    )


def test_decoded_output_never_contains_blank():
    torch.manual_seed(2)
    log_probs = torch.randn(20, 4, 6).log_softmax(-1)
    lengths = torch.tensor([20, 17, 11, 3])
    for decoded in ctc_beam_search_decode(log_probs, lengths, beam_width=4):
        assert 0 not in decoded
