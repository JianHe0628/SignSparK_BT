#!/usr/bin/env python3
"""Derive pseudo-glosses from translations for datasets shipped without them.

How2Sign stores an empty gloss field, so the CTC head has nothing to train on.
A stand-in gloss sequence is built by tagging the translation with spaCy and
keeping the lemmas of content-bearing parts of speech.

    python tools/make_pseudo_glosses.py --dataset How2Sign

Writes ${DATA_ROOT}/pseudo_glosses/How2Sign_<split>.json, which the loader
picks up via `pseudo_gloss_dir`.

These are a deterministic function of the target sentence: an auxiliary
alignment signal for the encoder, not extra supervision.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signspark_bt.data.lmdb_dataset import META_KEY, read_metadata  # noqa: E402

#: Parts of speech kept in a pseudo-gloss. Everything else (determiners,
#: adpositions, conjunctions, auxiliaries, particles, punctuation) is dropped.
SELECTED_POS = ["NOUN", "NUM", "ADV", "PRON", "PROPN", "ADJ", "VERB"]

#: spaCy model and extra lemma exclusions per language.
LANGUAGES = {
    "How2Sign": ("en_core_web_lg", ()),
    "PHOENIX-2014T": ("de_core_news_lg", ("es", "sich")),
    "CSL-Daily": ("zh_core_web_sm", ()),
}


def load_spacy(model: str):
    """Load a spaCy model, with a pointer to the download command if missing."""
    try:
        import spacy
    except ImportError:
        sys.exit("spaCy is required: pip install signspark-bt[pseudogloss]")
    try:
        return spacy.load(model)
    except OSError:
        sys.exit(f"spaCy model {model} is missing: python -m spacy download {model}")


def to_pseudo_gloss(sentence, nlp, exclude=()):
    """Sentence -> space-joined lowercase lemmas of content words."""
    lemmas = [
        token.lemma_.lower()
        for token in nlp(sentence)
        if token.pos_ in SELECTED_POS and token.lemma_.lower() not in exclude
    ]
    return " ".join(lemmas)


def lmdb_path_for(base_path: Path, dataset: str, split: str) -> Path:
    """Locate the LMDB for one dataset split."""
    matches = [p for p in sorted((base_path / split).rglob("*.lmdb")) if dataset in p.name]
    if not matches:
        raise FileNotFoundError(f"No {dataset} LMDB in {base_path / split}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="How2Sign", choices=sorted(LANGUAGES))
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument(
        "--base-path",
        default=os.path.join(os.getenv("DATA_ROOT", "./data"), "lmdb"),
        help="directory holding train/, dev/ and test/",
    )
    parser.add_argument(
        "--dest", default=os.path.join(os.getenv("DATA_ROOT", "./data"), "pseudo_glosses")
    )
    parser.add_argument("--spacy-model", default=None, help="override the spaCy model")
    args = parser.parse_args()

    import lmdb
    from tqdm import tqdm

    model, exclude = LANGUAGES[args.dataset]
    nlp = load_spacy(args.spacy_model or model)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        path = lmdb_path_for(Path(args.base_path), args.dataset, split)
        glosses = {}
        with lmdb.open(
            str(path), readonly=True, lock=False, readahead=False, meminit=False
        ) as env:
            with env.begin(write=False) as txn:
                clip_ids = pickle.loads(txn.get(META_KEY)).get("clip_ids", [])
                for clip_id in tqdm(clip_ids, desc=split, disable=not sys.stderr.isatty()):
                    blob = txn.get(clip_id.encode("utf-8"))
                    if blob is None:
                        continue
                    translation, _, _ = read_metadata(blob)
                    if translation:
                        glosses[clip_id] = to_pseudo_gloss(translation, nlp, exclude)

        out_file = dest / f"{args.dataset}_{split}.json"
        with open(out_file, "w", encoding="utf-8") as handle:
            json.dump(glosses, handle, ensure_ascii=False)
        print(f"{split}: {len(glosses)} pseudo-glosses -> {out_file}")


if __name__ == "__main__":
    main()
