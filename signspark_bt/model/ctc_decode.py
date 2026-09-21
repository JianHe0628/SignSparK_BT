# coding: utf-8
"""CTC decoding for the gloss recognition head.

Replaces tf.nn.ctc_beam_search_decoder, the only reason this codebase needed
TensorFlow. Silence sits at index 0 in the gloss vocabulary, so blank is 0 and
no class rotation is needed.
"""
from __future__ import annotations

import math
from collections import defaultdict
from itertools import groupby
from typing import List

from torch import Tensor

NEG_INF = -float("inf")


def _logaddexp(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def ctc_greedy_decode(
    log_probs: Tensor, lengths: Tensor, blank: int = 0
) -> List[List[int]]:
    """Best path: argmax per frame, collapse repeats, drop blanks.

    `log_probs` is T x N x C; returns one list of class indices per sequence.
    """
    best_path = log_probs.argmax(dim=-1).transpose(0, 1).cpu()  # N x T
    lengths = lengths.cpu()

    decoded = []
    for sequence, length in zip(best_path, lengths):
        frames = sequence[: int(length)].tolist()
        collapsed = [key for key, _ in groupby(frames)]
        decoded.append([c for c in collapsed if c != blank])
    return decoded


def ctc_beam_search_decode(
    log_probs: Tensor, lengths: Tensor, beam_width: int, blank: int = 0
) -> List[List[int]]:
    """Prefix beam search. Only the top `beam_width` classes per frame are
    expanded, which keeps the search tractable without changing the result."""
    log_probs = log_probs.transpose(0, 1).cpu()  # N x T x C
    lengths = lengths.cpu()
    num_classes = log_probs.size(-1)
    top_k = min(beam_width, num_classes)

    decoded = []
    for sequence, length in zip(log_probs, lengths):
        # prefix -> (log prob ending in blank, log prob ending in a real class)
        beams = {(): (0.0, NEG_INF)}

        for t in range(int(length)):
            frame = sequence[t]
            candidate_scores, candidate_classes = frame.topk(top_k)
            next_beams = defaultdict(lambda: (NEG_INF, NEG_INF))

            for prefix, (p_blank, p_non_blank) in beams.items():
                p_total = _logaddexp(p_blank, p_non_blank)
                for score, symbol in zip(candidate_scores.tolist(), candidate_classes.tolist()):
                    if symbol == blank:
                        nb, nnb = next_beams[prefix]
                        next_beams[prefix] = (_logaddexp(nb, p_total + score), nnb)
                        continue

                    last = prefix[-1] if prefix else None
                    if symbol == last:
                        # Repeating a symbol without an intervening blank
                        # extends the current prefix rather than growing it.
                        nb, nnb = next_beams[prefix]
                        next_beams[prefix] = (nb, _logaddexp(nnb, p_non_blank + score))
                        extended = prefix + (symbol,)
                        nb, nnb = next_beams[extended]
                        next_beams[extended] = (nb, _logaddexp(nnb, p_blank + score))
                    else:
                        extended = prefix + (symbol,)
                        nb, nnb = next_beams[extended]
                        next_beams[extended] = (nb, _logaddexp(nnb, p_total + score))

            beams = dict(
                sorted(
                    next_beams.items(),
                    key=lambda item: _logaddexp(*item[1]),
                    reverse=True,
                )[:beam_width]
            )

        best = max(beams.items(), key=lambda item: _logaddexp(*item[1]))[0]
        decoded.append(list(best))
    return decoded


def ctc_decode(
    log_probs: Tensor, lengths: Tensor, beam_size: int = 1, blank: int = 0
) -> List[List[int]]:
    """Decode with beam search, or best-path when `beam_size` is 1."""
    assert beam_size > 0, "recognition beam size must be positive"
    if beam_size == 1:
        return ctc_greedy_decode(log_probs, lengths, blank=blank)
    return ctc_beam_search_decode(log_probs, lengths, beam_size, blank=blank)
