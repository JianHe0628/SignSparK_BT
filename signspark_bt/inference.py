# coding: utf-8
"""Running a trained back-translation model on pose features."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch

from .data.lmdb_dataset import FEATURE_SIZE
from .data.vocabulary import SIL_TOKEN, build_vocab
from .helpers import load_checkpoint, load_config
from .model.model import SignModel, build_model

logger = logging.getLogger(__name__)

#: Sequences per forward pass during inference.
INFERENCE_BATCH_SIZE = 32


class InferenceBatch:
    """The minimal batch interface SignModel.run_batch needs."""

    def __init__(self, sgn: torch.Tensor, sgn_mask: torch.Tensor, sgn_lengths: torch.Tensor):
        self.sgn = sgn
        self.sgn_mask = sgn_mask
        self.sgn_lengths = sgn_lengths

    def to(self, device: torch.device) -> "InferenceBatch":
        """Move the batch to a device."""
        self.sgn = self.sgn.to(device)
        self.sgn_mask = self.sgn_mask.to(device)
        self.sgn_lengths = self.sgn_lengths.to(device)
        return self


def _find_checkpoint(model_dir: Path) -> Path:
    """best.ckpt, else the highest-numbered checkpoint."""
    best = model_dir / "best.ckpt"
    if best.exists():
        return best

    numbered = []
    for path in model_dir.iterdir():
        if path.suffix == ".ckpt" and path.stem.isdigit():
            numbered.append((int(path.stem), path))
    if not numbered:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")
    return max(numbered)[1]


def _resolve_vocab(model_dir: Path, data_cfg: dict, field: str) -> str:
    """Prefer the vocabulary shipped with the model, else the config's."""
    local = model_dir / f"{field}.vocab"
    if local.exists():
        return str(local)
    configured = data_cfg.get(f"{field}_vocab", None)
    if configured is None:
        raise FileNotFoundError(f"No {field} vocabulary in {model_dir} or in the config")
    return configured


def make_back_translation_model(
    model_dir: Union[str, Path],
    use_cuda: Optional[bool] = None,
) -> SignModel:
    """Load a trained model, in eval mode, on the chosen device.

    `model_dir` holds config.yaml, a checkpoint and the two vocab files -- the
    layout tools/download_models.py produces. Beam settings come from the
    config, so each model carries the values it was tuned with.
    """
    model_dir = Path(model_dir)
    cfg = load_config(model_dir / "config.yaml")
    data_cfg = cfg["data"]

    if use_cuda is None:
        use_cuda = cfg["training"].get("use_cuda", torch.cuda.is_available())
    use_cuda = bool(use_cuda) and torch.cuda.is_available()

    gls_vocab = build_vocab(
        field="gls", max_size=0, min_freq=0,
        dataset=None, vocab_file=_resolve_vocab(model_dir, data_cfg, "gls"),
    )
    txt_vocab = build_vocab(
        field="txt", max_size=0, min_freq=0,
        dataset=None, vocab_file=_resolve_vocab(model_dir, data_cfg, "txt"),
    )

    feature_size = data_cfg["feature_size"]
    if isinstance(feature_size, list):
        feature_size = sum(feature_size)

    model = build_model(
        cfg=cfg["model"],
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        sgn_dim=feature_size,
        do_recognition=cfg["training"].get("recognition_loss_weight", 1.0) > 0.0,
        do_translation=cfg["training"].get("translation_loss_weight", 1.0) > 0.0,
    )

    checkpoint = _find_checkpoint(model_dir)
    logger.info("Loading checkpoint %s", checkpoint)
    model.load_state_dict(load_checkpoint(str(checkpoint), use_cuda=use_cuda)["model_state"])
    model.eval()
    if use_cuda:
        model.cuda()

    testing_cfg = cfg.get("testing", {})
    model.beam_size = testing_cfg.get("eval_translation_beam_size",
                                      cfg["training"].get("eval_translation_beam_size", 1))
    model.beam_alpha = testing_cfg.get("eval_translation_beam_alpha",
                                       cfg["training"].get("eval_translation_beam_alpha", -1))
    model.max_output_len = cfg["training"].get("translation_max_output_length", None)
    return model


def _pad_features(
    features: Sequence[torch.Tensor],
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """Right-pad variable-length sequences into one batch."""
    lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
    max_len = int(lengths.max())
    feature_dim = features[0].shape[-1]

    padded = torch.zeros(len(features), max_len, feature_dim, dtype=torch.float32)
    for i, f in enumerate(features):
        padded[i, : f.shape[0]] = f

    positions = torch.arange(max_len).unsqueeze(0)
    mask = (positions < lengths.unsqueeze(1)).unsqueeze(1)
    return padded, mask, lengths


def back_translate(
    model: SignModel,
    poses: Union[torch.Tensor, Sequence[Optional[torch.Tensor]]],
    batch_size: int = INFERENCE_BATCH_SIZE,
) -> List[str]:
    """Translate pose features into sentences, one per input sequence.

    `poses` is B x T x 240, or a list of T x 240 tensors of differing length;
    SMPL+MANO ordered body (60), left hand (90), right hand (90). Empty and
    None entries come back as "" in place, so output lines up with input.
    """
    sequences = list(poses)

    kept: List[torch.Tensor] = []
    empty_positions: List[int] = []
    for index, sequence in enumerate(sequences):
        if sequence is None or sequence.numel() == 0:
            empty_positions.append(index)
        else:
            kept.append(torch.as_tensor(sequence, dtype=torch.float32))

    if not kept:
        return [""] * len(sequences)

    feature_dim = kept[0].shape[-1]
    if feature_dim != FEATURE_SIZE:
        raise ValueError(
            f"Expected {FEATURE_SIZE}-d SMPL+MANO features, got {feature_dim}. "
            f"Poses should be B x T x {FEATURE_SIZE}."
        )

    assert model.gls_vocab.stoi[SIL_TOKEN] == 0, "CTC blank must be index 0"

    device = next(model.parameters()).device
    was_recognising, model.do_recognition = model.do_recognition, False
    model.eval()

    decoded: List[str] = []
    try:
        with torch.no_grad():
            for start in range(0, len(kept), batch_size):
                chunk = kept[start : start + batch_size]
                sgn, sgn_mask, sgn_lengths = _pad_features(chunk)
                batch = InferenceBatch(sgn, sgn_mask, sgn_lengths).to(device)

                _, txt_predictions, _ = model.run_batch(
                    batch=batch,
                    recognition_beam_size=None,
                    translation_beam_size=model.beam_size,
                    translation_beam_alpha=model.beam_alpha,
                    translation_max_output_length=model.max_output_len,
                )
                for tokens in model.txt_vocab.arrays_to_sentences(arrays=txt_predictions):
                    decoded.append(" ".join(tokens))
    finally:
        model.do_recognition = was_recognising

    # Put the skipped sequences back so the output aligns with the input.
    for index in empty_positions:
        decoded.insert(index, "")
    return decoded
