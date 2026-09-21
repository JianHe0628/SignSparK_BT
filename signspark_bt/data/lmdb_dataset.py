# coding: utf-8
"""Dataset backed by the SignSparK LMDB pose archives.

One LMDB per split. Key b"__meta__" holds a pickled dict of clip ids; every
other key is a clip id whose value is an np.savez blob.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

# The LMDB blobs were written with numpy 2.x.  Under numpy 1.x, np.load raises
# "ModuleNotFoundError: No module named 'numpy._core'" without these aliases.
if int(np.__version__.split(".")[0]) < 2:  # pragma: no cover - environment dependent
    import numpy.core  # noqa: F401

    for _new, _old in (
        ("numpy._core", "numpy.core"),
        ("numpy._core.multiarray", "numpy.core.multiarray"),
        ("numpy._core.numeric", "numpy.core.numeric"),
    ):
        sys.modules.setdefault(_new, sys.modules[_old])

BODY_OFFSET = 11 * 6
HAND_DIM = 90
FEATURE_SIZE = 240

META_KEY = b"__meta__"


@dataclass
class Example:
    """One sample. `sgn` is None in the index and filled in by __getitem__."""

    sequence: str
    signer: str
    gls: List[str]
    txt: List[str]
    num_frames: int
    shard: int = 0
    sgn: Optional[torch.Tensor] = None


def build_tokenizer(level: str = "word", tokenizer: str = "whitespace") -> Callable[[str], List[str]]:
    """Return a text tokenizer. jieba is needed for Chinese."""
    if level == "char":
        return list

    if tokenizer == "jieba":
        try:
            import jieba
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "tokenizer: jieba requires the 'zh' extra -- pip install signspark-bt[zh]"
            ) from exc
        return lambda text: [t for t in jieba.cut(text) if t.strip()]

    if tokenizer != "whitespace":
        raise ValueError(f"Unknown tokenizer: {tokenizer!r}")
    return str.split


def normalize_text(
    text: str,
    tokenize_txt: Callable[[str], List[str]],
    txt_lowercase: bool = True,
    normalize_punctuation: bool = True,
) -> List[str]:
    """Turn a raw translation into the token list the model is trained on.

    `normalize_punctuation` detaches the sentence-final period: in Latin script
    it is glued to the last word, so whitespace tokenization would yield
    "world." as one token. Chinese needs none of this.
    """
    if normalize_punctuation:
        text = text.replace(".", "").replace(",", "") + " ."
    if txt_lowercase:
        text = text.lower()
    return tokenize_txt(text.strip())


def deserialize_npz(blob: bytes) -> Dict[str, np.ndarray]:
    """Decode one LMDB value into all of its arrays."""
    with io.BytesIO(blob) as buffer:
        npz = np.load(buffer, allow_pickle=True)
        return {
            "language": npz["language"][0],
            "translation": npz["translation"][0],
            "gloss": npz["gloss"][0],
            "segment": npz["segment"],
            "left_features": npz["left_features"],
            "right_features": npz["right_features"],
            "body_features": npz["body_features"],
            "face_features": npz["face_features"],
        }


def read_metadata(blob: bytes) -> Tuple[str, str, int]:
    """Read text and frame count only -- ~100x faster than decoding the poses,
    since np.load is lazy and `segment` has one entry per frame."""
    with io.BytesIO(blob) as buffer:
        npz = np.load(buffer, allow_pickle=True)
        return npz["translation"][0], npz["gloss"][0], int(npz["segment"].shape[0])


def read_features(blob: bytes) -> np.ndarray:
    """Decode an LMDB value into the 240-d feature vector (face is unused)."""
    with io.BytesIO(blob) as buffer:
        npz = np.load(buffer, allow_pickle=True)
        body = npz["body_features"][:, BODY_OFFSET:]
        left = npz["left_features"][:, :HAND_DIM]
        right = npz["right_features"][:, :HAND_DIM]
    return np.concatenate([body, left, right], axis=-1)


def assemble_features(record: Dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate body and hand streams into the 240-d vector.

    Hands are sliced to 90: CSL-Daily and PHOENIX store 96, How2Sign 90.
    """
    body = record["body_features"][:, BODY_OFFSET:]
    left = record["left_features"][:, :HAND_DIM]
    right = record["right_features"][:, :HAND_DIM]
    return np.concatenate([body, left, right], axis=-1)


def resolve_lmdb_paths(base_path: str, split: str, dataset_names: Sequence[str]) -> List[str]:
    """Find the LMDBs for ``split`` whose path contains one of ``dataset_names``."""
    split_dir = Path(base_path) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"No LMDB directory at {split_dir}. Set DATA_ROOT, or run "
            f"tools/download_data.py from the SignSparK repo."
        )
    paths = sorted(str(p) for p in split_dir.rglob("*.lmdb"))
    matched = [p for p in paths if any(name in p for name in dataset_names)]
    if not matched:
        raise FileNotFoundError(
            f"No LMDB under {split_dir} matched {list(dataset_names)}. Found: {paths}"
        )
    return matched


def _index_fingerprint(lmdb_paths: Sequence[str], options: Dict) -> str:
    """Identify an index by its inputs, so a stale cache is never reused."""
    parts = []
    for path in lmdb_paths:
        data_file = Path(path) / "data.mdb"
        stat = data_file.stat() if data_file.exists() else None
        parts.append(f"{path}:{stat.st_size if stat else 0}:{stat.st_mtime_ns if stat else 0}")
    parts.append(repr(sorted(options.items())))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _load_index_cache(cache_file: Path) -> Optional[List["Example"]]:
    """Load a cached index, or return None if it is missing or unreadable."""
    if not cache_file.is_file():
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as handle:
            rows = json.load(handle)
    except (json.JSONDecodeError, OSError):
        logger.warning("Ignoring unreadable index cache %s", cache_file)
        return None
    return [
        Example(
            sequence=r["sequence"],
            signer=r["signer"],
            gls=r["gls"],
            txt=r["txt"],
            num_frames=r["num_frames"],
            shard=r["shard"],
        )
        for r in rows
    ]


def _save_index_cache(cache_file: Path, examples: Sequence["Example"]) -> None:
    """Write an index to the cache, ignoring failures (it is only a cache)."""
    rows = [
        {
            "sequence": e.sequence,
            "signer": e.signer,
            "gls": e.gls,
            "txt": e.txt,
            "num_frames": e.num_frames,
            "shard": e.shard,
        }
        for e in examples
    ]
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as handle:
            json.dump(rows, handle)
    except OSError as error:
        logger.warning("Could not write index cache %s: %s", cache_file, error)


class SignBackTranslationDataset(Dataset):
    """One split, from one or more LMDBs.

    Text is tokenized up front so vocabularies and length-bucketing can be
    built; pose features are read on demand to keep memory flat.
    """

    def __init__(
        self,
        lmdb_paths: Sequence[str],
        tokenize_txt: Callable[[str], List[str]],
        txt_lowercase: bool = True,
        normalize_punctuation: bool = True,
        max_sgn_length: int = 300,
        max_sent_length: Optional[int] = None,
        pseudo_gloss: Optional[Dict[str, str]] = None,
        split: str = "",
        index_cache_dir: Optional[str] = None,
    ):
        """`max_sent_length` filters clips (train only); None keeps everything.

        `normalize_punctuation` detaches the sentence-final period. In Latin
        script it is glued to the last word, so whitespace tokenization would
        yield "world." as one token; stripping it and appending " ." makes it
        its own token. Chinese needs none of this: its punctuation is
        full-width and jieba already emits it separately, and CSL BLEU is
        scored per character, where an extra "." would count as a character in
        every sentence.
        """
        self.lmdb_paths = list(lmdb_paths)
        self.max_sgn_length = max_sgn_length
        self._envs: Optional[List[lmdb.Environment]] = None

        cache_file = None
        if index_cache_dir:
            fingerprint = _index_fingerprint(
                self.lmdb_paths,
                {
                    "txt_lowercase": txt_lowercase,
                    "normalize_punctuation": normalize_punctuation,
                    "max_sgn_length": max_sgn_length,
                    "max_sent_length": max_sent_length,
                    "tokenizer": getattr(tokenize_txt, "__name__", repr(tokenize_txt)),
                    "pseudo_gloss": len(pseudo_gloss) if pseudo_gloss else 0,
                },
            )
            cache_file = Path(index_cache_dir) / f"{split or 'index'}-{fingerprint}.json"
            cached = _load_index_cache(cache_file)
            if cached is not None:
                self.examples = cached
                logger.info("%s: %d samples (from index cache)", split or "dataset", len(cached))
                return

        self.examples: List[Example] = []
        skipped = {"missing_text": 0, "empty": 0, "too_long": 0}

        for shard, path in enumerate(self.lmdb_paths):
            with lmdb.open(
                path, readonly=True, lock=False, readahead=False, meminit=False, max_readers=1024
            ) as env:
                with env.begin(write=False) as txn:
                    meta = pickle.loads(txn.get(META_KEY))
                    clip_ids = meta.get("clip_ids", [])
                    self._check_feature_layout(txn, clip_ids, path)

                    progress = tqdm(
                        clip_ids,
                        desc=f"Indexing {Path(path).name}",
                        leave=False,
                        # Progress bars would otherwise flood redirected logs.
                        disable=not sys.stderr.isatty(),
                    )
                    for clip_id in progress:
                        blob = txn.get(clip_id.encode("utf-8"))
                        if blob is None:
                            continue

                        translation, gloss, n_frames = read_metadata(blob)
                        if pseudo_gloss is not None:
                            gloss = pseudo_gloss.get(clip_id, gloss)
                        if not translation or gloss is None:
                            skipped["missing_text"] += 1
                            continue
                        if n_frames == 0:
                            skipped["empty"] += 1
                            continue
                        n_frames = min(n_frames, max_sgn_length)

                        txt_tokens = normalize_text(
                            translation, tokenize_txt, txt_lowercase, normalize_punctuation
                        )
                        gls_tokens = str(gloss).strip().split()

                        if max_sent_length is not None and (
                            n_frames > max_sent_length or len(txt_tokens) > max_sent_length
                        ):
                            skipped["too_long"] += 1
                            continue

                        self.examples.append(
                            Example(
                                sequence=clip_id,
                                signer="signer",  # placeholder, kept for config compatibility
                                gls=gls_tokens,
                                txt=txt_tokens,
                                num_frames=n_frames,
                                shard=shard,
                            )
                        )

        logger.info(
            "%s: %d samples (skipped %d without text, %d empty, %d over length)",
            split or "dataset",
            len(self.examples),
            skipped["missing_text"],
            skipped["empty"],
            skipped["too_long"],
        )

        if cache_file is not None:
            _save_index_cache(cache_file, self.examples)

    @staticmethod
    def _check_feature_layout(txn, clip_ids: Sequence[str], path: str) -> None:
        """Check the first clip's width. Extraction is uniform within an LMDB."""
        if not clip_ids:
            return
        blob = txn.get(clip_ids[0].encode("utf-8"))
        if blob is None:
            return
        width = read_features(blob).shape[-1]
        if width != FEATURE_SIZE:
            raise ValueError(
                f"{Path(path).name} assembles to {width}-d features, expected "
                f"{FEATURE_SIZE}. This release supports SMPL+MANO data only."
            )

    @property
    def envs(self) -> List[lmdb.Environment]:
        """Opened lazily so each DataLoader worker gets its own handle."""
        if self._envs is None:
            self._envs = [
                lmdb.open(
                    path,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                    max_readers=1024,
                    map_size=1 << 40,
                )
                for path in self.lmdb_paths
            ]
        return self._envs

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        example = self.examples[index]
        with self.envs[example.shard].begin(write=False) as txn:
            blob = txn.get(example.sequence.encode("utf-8"))
        features = read_features(blob)[: self.max_sgn_length]
        return Example(
            sequence=example.sequence,
            signer=example.signer,
            gls=example.gls,
            txt=example.txt,
            num_frames=example.num_frames,
            shard=example.shard,
            sgn=torch.from_numpy(np.ascontiguousarray(features)).float(),
        )

    def __getstate__(self):
        # LMDB handles cannot cross a process boundary.
        state = self.__dict__.copy()
        state["_envs"] = None
        return state


def load_pseudo_glosses(path: Optional[str]) -> Optional[Dict[str, str]]:
    """Load a {clip_id: gloss} map from tools/make_pseudo_glosses.py."""
    if not path:
        return None
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"pseudo_gloss_file {path} not found. Generate it with tools/make_pseudo_glosses.py"
        )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
