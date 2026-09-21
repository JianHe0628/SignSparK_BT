# coding: utf-8
"""Batching and collation, replacing torchtext's Field/BucketIterator.

SignCollator produces a RawBatch with the same attributes the torchtext batch
had, so Batch and its callers are unchanged apart from how the mask is built.
"""
from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence

import numpy as np
import torch

from .vocabulary import EOS_TOKEN, BOS_TOKEN, PAD_TOKEN, GlossVocabulary, TextVocabulary


class RawBatch:
    """Collated tensors, mirroring the old torchtext batch interface."""

    def __init__(self, sequence, signer, sgn, txt=None, gls=None):
        self.sequence = sequence
        self.signer = signer
        self.sgn = sgn
        if txt is not None:
            self.txt = txt
        if gls is not None:
            self.gls = gls


class SignCollator:
    """Turns a list of Example into a batch."""

    def __init__(
        self,
        txt_vocab: TextVocabulary,
        gls_vocab: Optional[GlossVocabulary] = None,
        sgn_dim: int = 240,
    ):
        self.txt_vocab = txt_vocab
        self.gls_vocab = gls_vocab
        self.sgn_dim = sgn_dim

    def _encode_txt(self, tokens: Sequence[str]) -> List[int]:
        stoi = self.txt_vocab.stoi
        return (
            [stoi[BOS_TOKEN]]
            + [stoi[t] for t in tokens]
            + [stoi[EOS_TOKEN]]
        )

    def __call__(self, examples) -> RawBatch:
        batch_size = len(examples)
        sgn_lengths = torch.tensor([ex.sgn.shape[0] for ex in examples], dtype=torch.long)
        max_frames = int(sgn_lengths.max())

        sgn = torch.zeros(batch_size, max_frames, self.sgn_dim, dtype=torch.float32)
        for i, ex in enumerate(examples):
            sgn[i, : ex.sgn.shape[0]] = ex.sgn

        txt_ids = [self._encode_txt(ex.txt) for ex in examples]
        txt_lengths = torch.tensor([len(t) for t in txt_ids], dtype=torch.long)
        txt_pad = self.txt_vocab.stoi[PAD_TOKEN]
        txt = torch.full((batch_size, int(txt_lengths.max())), txt_pad, dtype=torch.long)
        for i, ids in enumerate(txt_ids):
            txt[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        gls_pair = None
        if self.gls_vocab is not None:
            gls_ids = [[self.gls_vocab.stoi[t] for t in ex.gls] for ex in examples]
            gls_lengths = torch.tensor([max(len(g), 1) for g in gls_ids], dtype=torch.long)
            gls_pad = self.gls_vocab.stoi[PAD_TOKEN]
            gls = torch.full((batch_size, int(gls_lengths.max())), gls_pad, dtype=torch.long)
            for i, ids in enumerate(gls_ids):
                if ids:
                    gls[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            gls_pair = (gls, gls_lengths)

        return RawBatch(
            sequence=[ex.sequence for ex in examples],
            signer=[ex.signer for ex in examples],
            sgn=(sgn, sgn_lengths),
            txt=(txt, txt_lengths),
            gls=gls_pair,
        )


def lengths_to_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    """[B] lengths -> [B, 1, max_length] boolean mask."""
    positions = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    return (positions < lengths.unsqueeze(1)).unsqueeze(1)


class Batch:
    """A batch of sign/gloss/text data with lengths, masks and token counts."""

    def __init__(
        self,
        torch_batch,
        txt_pad_index,
        sgn_dim,
        is_train: bool = False,
        use_cuda: bool = False,
        frame_subsampling_ratio: int = None,
        random_frame_subsampling: bool = None,
        random_frame_masking_ratio: float = None,
    ):
        """The frame subsampling and masking options are training-only."""
        # Sequence Information
        self.sequence = torch_batch.sequence
        self.signer = torch_batch.signer
        # Sign
        self.sgn, self.sgn_lengths = torch_batch.sgn

        # Here be dragons
        if frame_subsampling_ratio:
            tmp_sgn = torch.zeros_like(self.sgn)
            tmp_sgn_lengths = torch.zeros_like(self.sgn_lengths)
            for idx, (features, length) in enumerate(zip(self.sgn, self.sgn_lengths)):
                features = features.clone()
                if random_frame_subsampling and is_train:
                    init_frame = random.randint(0, (frame_subsampling_ratio - 1))
                else:
                    init_frame = math.floor((frame_subsampling_ratio - 1) / 2)

                tmp_data = features[: length.long(), :]
                tmp_data = tmp_data[init_frame::frame_subsampling_ratio]
                tmp_sgn[idx, 0 : tmp_data.shape[0]] = tmp_data
                tmp_sgn_lengths[idx] = tmp_data.shape[0]

            self.sgn = tmp_sgn[:, : tmp_sgn_lengths.max().long(), :]
            self.sgn_lengths = tmp_sgn_lengths

        if random_frame_masking_ratio and is_train:
            tmp_sgn = torch.zeros_like(self.sgn)
            num_mask_frames = (
                (self.sgn_lengths * random_frame_masking_ratio).floor().long()
            )
            for idx, features in enumerate(self.sgn):
                features = features.clone()
                mask_frame_idx = np.random.permutation(
                    int(self.sgn_lengths[idx].long().numpy())
                )[: num_mask_frames[idx]]
                features[mask_frame_idx, :] = 0.0
                tmp_sgn[idx] = features
            self.sgn = tmp_sgn

        self.sgn_dim = sgn_dim
        # Derived from explicit lengths: a padded frame is one past the sequence
        # length, not one that happens to be all zeros.
        self.sgn_mask = lengths_to_mask(self.sgn_lengths, self.sgn.size(1))

        # Text
        self.txt = None
        self.txt_mask = None
        self.txt_input = None
        self.txt_lengths = None

        # Gloss
        self.gls = None
        self.gls_lengths = None

        # Other
        self.num_txt_tokens = None
        self.num_gls_tokens = None
        self.use_cuda = use_cuda
        self.num_seqs = self.sgn.size(0)

        if getattr(torch_batch, "txt", None) is not None:
            txt, txt_lengths = torch_batch.txt
            # txt_input is used for teacher forcing, last one is cut off
            self.txt_input = txt[:, :-1]
            self.txt_lengths = txt_lengths
            # txt is used for loss computation, shifted by one since BOS
            self.txt = txt[:, 1:]
            # we exclude the padded areas from the loss computation
            self.txt_mask = (self.txt_input != txt_pad_index).unsqueeze(1)
            self.num_txt_tokens = (self.txt != txt_pad_index).data.sum().item()

        if getattr(torch_batch, "gls", None) is not None:
            self.gls, self.gls_lengths = torch_batch.gls
            self.num_gls_tokens = self.gls_lengths.sum().detach().clone().numpy()

        if use_cuda:
            self._make_cuda()

    def _make_cuda(self):
        """Move the batch to GPU."""
        self.sgn = self.sgn.cuda()
        self.sgn_mask = self.sgn_mask.cuda()

        if self.txt_input is not None:
            self.txt = self.txt.cuda()
            self.txt_mask = self.txt_mask.cuda()
            self.txt_input = self.txt_input.cuda()

    def sort_by_sgn_lengths(self):
        """Sort by sgn length (descending) and return index to revert sort."""
        _, perm_index = self.sgn_lengths.sort(0, descending=True)
        rev_index = [0] * perm_index.size(0)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
            rev_index[old_pos] = new_pos

        self.sgn = self.sgn[perm_index]
        self.sgn_mask = self.sgn_mask[perm_index]
        self.sgn_lengths = self.sgn_lengths[perm_index]

        self.signer = [self.signer[pi] for pi in perm_index]
        self.sequence = [self.sequence[pi] for pi in perm_index]

        if self.gls is not None:
            self.gls = self.gls[perm_index]
            self.gls_lengths = self.gls_lengths[perm_index]

        if self.txt is not None:
            self.txt = self.txt[perm_index]
            self.txt_mask = self.txt_mask[perm_index]
            self.txt_input = self.txt_input[perm_index]
            self.txt_lengths = self.txt_lengths[perm_index]

        if self.use_cuda:
            self._make_cuda()

        return rev_index
