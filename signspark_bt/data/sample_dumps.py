# coding: utf-8
"""Reading SignSparK sample dumps into back-translation features.

Sampling writes one .npy per stream, each a dict with `gt_poses`, `pred_poses`,
`lengths`, `text` and `video_names`. Poses are (repetition, B, C, T); the hand
file holds 2*B entries, left and right interleaved per clip.

Scoring `gt` gives the upper bound the model can reach on this data; `pred` is
the generated output to compare against it.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

HAND_DIM = 90
BODY_DIM = 60
FEATURE_SIZE = 240

#: Conjugates a SMPL-X left hand into right-hand convention. Vendored from
#: SignSparK's loader so the transform matches bit for bit; applying it twice
#: returns the original, so the same function also undoes it.
LEFT_HAND_FLIP_MASK = np.array(
    [[1.0, -1.0, -1.0], [-1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]], dtype=np.float32
)

NUM_HAND_JOINTS = 15


def rot6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """6D rotation (Zhou et al., Gram-Schmidt) -> (..., 3, 3)."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> 6D by keeping the first two rows."""
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].copy().reshape(*batch_dim, 6)


def flip_left_hand(left_feats: np.ndarray) -> np.ndarray:
    """Flip a T x 90 left-hand rot6D block between SMPL-X and right-hand frames."""
    left = left_feats.astype(np.float32, copy=False)
    frames = left.shape[0]
    pose_6d = left[:, :HAND_DIM].reshape(frames * NUM_HAND_JOINTS, 6)
    mats = rot6d_to_matrix(pose_6d) * LEFT_HAND_FLIP_MASK
    return matrix_to_rot6d(mats).reshape(frames, HAND_DIM)


def _stream(dump: dict, poses: str, repetition: int) -> np.ndarray:
    """Pull one pose stream as B x T x C."""
    if poses not in dump:
        raise KeyError(f"{poses!r} not in dump; available: {sorted(dump)}")
    array = dump[poses]
    if array.ndim == 4:
        array = array[repetition]
    return np.transpose(array, (0, 2, 1))


def _strip_language_tag(text: str) -> str:
    """'<American> hello there' -> 'hello there'."""
    return text.split(">", 1)[1].lstrip() if ">" in text else text


def load_sample_dump(
    body_path: str,
    hand_path: str,
    poses: str = "pred_poses",
    repetition: int = 0,
    unflip_left_hand: bool = True,
) -> Tuple[List[torch.Tensor], List[str], List[str]]:
    """
    Assemble 240-d features from sample dumps.

    :param body_path: .npy holding the body stream (C = 60)
    :param hand_path: .npy holding the hand stream (C = 90, left/right interleaved)
    :param poses: "pred_poses" or "gt_poses"
    :param repetition: which sampling repetition to take
    :param unflip_left_hand: undo SignSparK's left-hand conjugation, which the
        LMDB features the model trained on do not have
    :return: per-clip T x 240 tensors, reference sentences, clip names
    """
    body_dump = np.load(body_path, allow_pickle=True).item()
    hand_dump = np.load(hand_path, allow_pickle=True).item()

    body = _stream(body_dump, poses, repetition)
    hand = _stream(hand_dump, poses, repetition)
    # The hand dump holds 2*B entries: left and right interleaved per clip.
    left, right = hand[::2], hand[1::2]

    lengths = np.asarray(body_dump["lengths"])
    names = np.asarray(body_dump["video_names"])
    texts = np.asarray(body_dump["text"])
    if lengths.ndim == 2:
        lengths = lengths[repetition]
    if names.ndim == 2:
        names = names[repetition]
    if texts.ndim == 2:
        texts = texts[repetition]

    count = min(len(body), len(left), len(right), len(lengths))
    features, references, clip_names = [], [], []
    for i in range(count):
        length = int(lengths[i])
        if length <= 0:
            continue
        b, l, r = body[i][:length], left[i][:length], right[i][:length]
        if not (b.shape[0] == l.shape[0] == r.shape[0]):
            continue
        if unflip_left_hand:
            l = flip_left_hand(l[:, :HAND_DIM])
        stacked = np.concatenate([b[:, :BODY_DIM], l[:, :HAND_DIM], r[:, :HAND_DIM]], axis=-1)
        if stacked.shape[-1] != FEATURE_SIZE:
            raise ValueError(
                f"Assembled {stacked.shape[-1]}-d features, expected {FEATURE_SIZE}. "
                f"Check that body is {BODY_DIM}-d and hand {HAND_DIM}-d."
            )
        features.append(torch.from_numpy(np.ascontiguousarray(stacked)).float())
        references.append(_strip_language_tag(str(texts[i])))
        clip_names.append(str(names[i]))

    return features, references, clip_names
