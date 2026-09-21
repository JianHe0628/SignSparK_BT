# coding: utf-8
"""Building datasets, vocabularies and batch iterators."""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Sampler

from .batch import SignCollator
from .lmdb_dataset import (
    FEATURE_SIZE,
    SignBackTranslationDataset,
    build_tokenizer,
    load_pseudo_glosses,
    resolve_lmdb_paths,
)
from .vocabulary import GlossVocabulary, TextVocabulary, Vocabulary, build_vocab

#: Sort this many batches' worth of examples at a time when bucketing by
#: length, matching torchtext's BucketIterator.
BUCKET_POOL_BATCHES = 100


def _resolve_base_path(data_cfg: dict) -> str:
    """The LMDB root. load_config has already expanded ${VAR} and ~."""
    return data_cfg.get("base_path") or os.path.join(
        os.environ.get("DATA_ROOT", "./data"), "lmdb"
    )


def _pseudo_gloss_path(data_cfg: dict, split: str) -> Optional[str]:
    """Path to the pseudo-gloss file for a split, or None if not configured."""
    directory = data_cfg.get("pseudo_gloss_dir")
    if not directory:
        return None
    return str(Path(directory) / f"{data_cfg['dataset']}_{split}.json")


def _index_cache_dir(data_cfg: dict) -> Optional[str]:
    """Where to cache split indexes, or None if disabled. Indexing is I/O
    bound and slow on network storage, so it is reused across runs."""
    if not data_cfg.get("index_cache", True):
        return None
    configured = data_cfg.get("index_cache_dir")
    if configured:
        return configured
    return str(Path(_resolve_base_path(data_cfg)).parent / ".bt_index_cache")


def _build_split(
    data_cfg: dict, split: str, max_sent_length: Optional[int]
) -> SignBackTranslationDataset:
    """Build the dataset for one split."""
    dataset_names = data_cfg["dataset"]
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    tokenize_txt = build_tokenizer(
        level=data_cfg.get("level", "word"),
        tokenizer=data_cfg.get("tokenizer", "whitespace"),
    )
    return SignBackTranslationDataset(
        lmdb_paths=resolve_lmdb_paths(_resolve_base_path(data_cfg), split, dataset_names),
        tokenize_txt=tokenize_txt,
        txt_lowercase=data_cfg.get("txt_lowercase", True),
        normalize_punctuation=data_cfg.get("normalize_punctuation", True),
        max_sgn_length=data_cfg.get("max_sgn_length", 300),
        max_sent_length=max_sent_length,
        pseudo_gloss=load_pseudo_glosses(_pseudo_gloss_path(data_cfg, split)),
        split=split,
        index_cache_dir=_index_cache_dir(data_cfg),
    )


def _random_subset(
    dataset: SignBackTranslationDataset, size: int
) -> SignBackTranslationDataset:
    """Keep `size` randomly chosen examples, or all of them if size is -1."""
    if size <= -1 or size >= len(dataset):
        return dataset
    dataset.examples = random.sample(dataset.examples, size)
    return dataset


def load_data(
    data_cfg: dict,
) -> Tuple[
    SignBackTranslationDataset,
    SignBackTranslationDataset,
    SignBackTranslationDataset,
    Vocabulary,
    Vocabulary,
]:
    """Load the three splits and build vocabularies from train.

    Training data is filtered to `max_sent_length`; dev and test are left whole.
    """
    feature_size = data_cfg.get("feature_size", FEATURE_SIZE)
    if feature_size != FEATURE_SIZE:
        raise ValueError(
            f"This release only supports {FEATURE_SIZE}-d SMPL+MANO features, "
            f"got feature_size: {feature_size}"
        )

    max_sent_length = data_cfg.get("max_sent_length", None)
    train_data = _build_split(data_cfg, "train", max_sent_length)
    dev_data = _build_split(data_cfg, "dev", None)
    test_data = _build_split(data_cfg, "test", None)

    gls_vocab = build_vocab(
        field="gls",
        min_freq=data_cfg.get("gls_voc_min_freq", 1),
        max_size=data_cfg.get("gls_voc_limit", sys.maxsize),
        dataset=train_data,
        vocab_file=data_cfg.get("gls_vocab", None),
    )
    txt_vocab = build_vocab(
        field="txt",
        min_freq=data_cfg.get("txt_voc_min_freq", 1),
        max_size=data_cfg.get("txt_voc_limit", sys.maxsize),
        dataset=train_data,
        vocab_file=data_cfg.get("txt_vocab", None),
    )

    train_data = _random_subset(train_data, data_cfg.get("random_train_subset", -1))
    dev_data = _random_subset(dev_data, data_cfg.get("random_dev_subset", -1))

    return train_data, dev_data, test_data, gls_vocab, txt_vocab


def token_batch_size(examples) -> int:
    """Batch size in tokens: the largest of the padded sgn/gls/txt counts."""
    count = len(examples)
    if count == 0:
        return 0
    max_sgn = max(ex.num_frames for ex in examples)
    max_gls = max(len(ex.gls) for ex in examples)
    max_txt = max(len(ex.txt) + 2 for ex in examples)
    return max(count * max_sgn, count * max_gls, count * max_txt)


class BucketBatchSampler(Sampler):
    """Batches of dataset indices.

    With `bucket`, sorts by length within pools so batches hold similar-length
    sequences, then shuffles batch order. Otherwise batches in order, which
    keeps evaluation output aligned with the input.
    """

    def __init__(
        self,
        dataset: SignBackTranslationDataset,
        batch_size: int,
        batch_type: str = "sentence",
        bucket: bool = False,
        shuffle: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_type = batch_type
        self.bucket = bucket
        self.shuffle = shuffle

    def _pack(self, indices: List[int]) -> List[List[int]]:
        """Pack indices into batches, by sentence or token count."""
        batches: List[List[int]] = []
        current: List[int] = []
        for index in indices:
            current.append(index)
            if self.batch_type == "token":
                size = token_batch_size([self.dataset.examples[i] for i in current])
                if size >= self.batch_size and len(current) > 1:
                    batches.append(current[:-1])
                    current = [index]
            elif len(current) == self.batch_size:
                batches.append(current)
                current = []
        if current:
            batches.append(current)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        indices = list(range(len(self.dataset)))
        if not self.bucket:
            return iter(self._pack(indices))

        if self.shuffle:
            random.shuffle(indices)

        pool_size = self.batch_size * BUCKET_POOL_BATCHES
        batches: List[List[int]] = []
        for start in range(0, len(indices), pool_size):
            pool = indices[start : start + pool_size]
            pool.sort(key=lambda i: self.dataset.examples[i].num_frames)
            batches.extend(self._pack(pool))

        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        if self.batch_type == "token":
            # Only known once packed, so count a pass.
            return sum(1 for _ in iter(self))
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def make_data_iter(
    dataset: SignBackTranslationDataset,
    batch_size: int,
    txt_vocab: TextVocabulary,
    gls_vocab: Optional[GlossVocabulary] = None,
    batch_type: str = "sentence",
    train: bool = False,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """DataLoader yielding RawBatch. Batches are length-bucketed when training."""
    sampler = BucketBatchSampler(
        dataset,
        batch_size=batch_size,
        batch_type=batch_type,
        bucket=train,
        shuffle=shuffle and train,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=SignCollator(txt_vocab=txt_vocab, gls_vocab=gls_vocab),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
