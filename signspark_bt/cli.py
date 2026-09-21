# coding: utf-8
"""Command line interface for training, evaluating and running the model."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

#: Config values that can be overridden on the command line, as dotted paths.
OVERRIDABLE = [
    ("--name", str, "name of this run"),
    ("--data.dataset", str, "dataset to train on, e.g. PHOENIX-2014T"),
    ("--data.version", str, "dataset version tag used for gloss clean-up"),
    ("--training.learning_rate", float, "learning rate"),
    ("--training.batch_size", int, "batch size"),
    ("--training.model_dir", str, "where to write checkpoints and logs"),
]


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="signspark-bt",
        description="Sign language back-translation: pose features to text.",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train", help="train a model")
    train_parser.add_argument("config_path", type=str, help="path to a YAML config")
    for flag, flag_type, help_text in OVERRIDABLE:
        train_parser.add_argument(flag, type=flag_type, default=None, help=help_text)

    test_parser = subparsers.add_parser("test", help="evaluate a trained model")
    test_parser.add_argument("config_path", type=str, help="path to a YAML config")
    test_parser.add_argument("--ckpt", type=str, default=None, help="checkpoint to load")
    test_parser.add_argument(
        "--output_path", type=str, default=None, help="where to write translations"
    )

    translate_parser = subparsers.add_parser(
        "translate", help="back-translate pose features into text"
    )
    translate_parser.add_argument("model_dir", type=str, help="trained model directory")
    translate_parser.add_argument(
        "features",
        type=str,
        help="a .npy or .pt file holding B x T x 240 SMPL+MANO features",
    )
    translate_parser.add_argument(
        "--output_path", type=str, default=None, help="write one sentence per line here"
    )
    translate_parser.add_argument("--batch_size", type=int, default=32)

    score_parser = subparsers.add_parser(
        "score", help="back-translate poses and score them against references"
    )
    score_parser.add_argument("model_dir", type=str, help="trained model directory")
    score_parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="a .npy or .pt file of generated poses, B x T x 240 (or a list of T x 240)",
    )
    score_parser.add_argument(
        "--references",
        type=str,
        default=None,
        help="text file with one reference sentence per sample, in input order; "
        "required with --features",
    )
    score_parser.add_argument(
        "--body-npy", type=str, default=None, help="SignSparK body sample dump"
    )
    score_parser.add_argument(
        "--hand-npy", type=str, default=None, help="SignSparK hand sample dump"
    )
    score_parser.add_argument(
        "--poses",
        type=str,
        default="both",
        choices=["both", "pred_poses", "gt_poses"],
        help="which streams to score; the default scores both and reports the drop",
    )
    score_parser.add_argument("--repetition", type=int, default=0)
    score_parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "dev", "test"],
        help="score the ground-truth poses of this split instead, which gives "
        "the upper bound the model can reach on real data",
    )
    score_parser.add_argument(
        "--output_path", type=str, default=None, help="also write the hypotheses here"
    )
    score_parser.add_argument("--batch_size", type=int, default=32)

    for subparser in (train_parser, test_parser, translate_parser, score_parser):
        subparser.add_argument(
            "--gpu_id", type=str, default=None, help="GPU to run on, e.g. 0"
        )
    return parser


def collect_overrides(args: argparse.Namespace) -> dict:
    """Gather the dotted config overrides that were actually given."""
    overrides = {}
    for flag, _, _ in OVERRIDABLE:
        key = flag[2:]
        value = getattr(args, key.replace(".", "_"), None) or getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    return overrides


def _load_features(path: str):
    """Load pose features from a .npy or .pt file."""
    import numpy as np
    import torch

    suffix = Path(path).suffix
    if suffix == ".npy":
        return torch.from_numpy(np.load(path)).float()
    if suffix in (".pt", ".pth"):
        return torch.load(path, map_location="cpu")
    raise ValueError(f"Unsupported feature file {path!r}: expected .npy, .pt or .pth")


def _normalize(sentence: str, data_cfg: dict) -> str:
    """Put a reference through the same pipeline the training text went through."""
    from .data.lmdb_dataset import build_tokenizer, normalize_text

    tokenize = build_tokenizer(
        level=data_cfg.get("level", "word"),
        tokenizer=data_cfg.get("tokenizer", "whitespace"),
    )
    return " ".join(
        normalize_text(
            sentence,
            tokenize,
            data_cfg.get("txt_lowercase", True),
            data_cfg.get("normalize_punctuation", True),
        )
    )


def _report(results: dict, count: int, label: str) -> None:
    """Print ground-truth and predicted scores side by side with the drop."""
    columns = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "chrF", "ROUGE"]

    def row(scores):
        return [scores["bleu"][f"bleu{n}"] for n in range(1, 5)] + [
            scores["chrf"],
            scores["rouge"],
        ]

    print(f"\n{label}  ({count} clips)\n")
    print(f"{'':14s}" + "".join(f"{c:>9s}" for c in columns))
    for name, scores in results.items():
        print(f"{name:14s}" + "".join(f"{v:9.2f}" for v in row(scores)))

    if len(results) == 2:
        ceiling, predicted = row(results["ground truth"]), row(results["predicted"])
        drops = [
            (p - c) / c * 100 if c else float("nan") for c, p in zip(ceiling, predicted)
        ]
        print(f"{'drop':14s}" + "".join(f"{d:8.1f}%" for d in drops))


def _score(args) -> None:
    """Back-translate poses and report BLEU, chrF and ROUGE.

    Given a sample dump, scores the ground-truth poses and the predicted poses
    in one pass and reports the drop between them: the ground truth is the
    ceiling this back-translation model can reach on this data, so the drop is
    what the generated poses actually cost.

    Scoring goes through the same helpers as `signspark-bt test`, so the numbers
    are directly comparable.
    """
    from .data.loading import _build_split
    from .helpers import load_config
    from .inference import back_translate, make_back_translation_model
    from .metrics import score_translations

    sources = [bool(args.features), bool(args.split), bool(args.body_npy)]
    if sum(sources) != 1:
        raise SystemExit("Pass exactly one of --features, --split or --body-npy.")
    if args.features and not args.references:
        raise SystemExit("--features needs --references.")
    if args.body_npy and not args.hand_npy:
        raise SystemExit("--body-npy needs --hand-npy.")

    data_cfg = load_config(Path(args.model_dir) / "config.yaml")["data"]
    model = make_back_translation_model(args.model_dir)

    def run(poses, references):
        hypotheses = back_translate(
            model=model, poses=poses, batch_size=args.batch_size
        )
        if len(hypotheses) != len(references):
            raise SystemExit(
                f"{len(hypotheses)} hypotheses but {len(references)} references."
            )
        scores = score_translations(
            references=references,
            hypotheses=hypotheses,
            dataset_version=data_cfg.get("version", ""),
            level=data_cfg.get("level", "word"),
        )
        return scores, hypotheses

    results, hypotheses, count = {}, [], 0

    if args.body_npy:
        from .data.sample_dumps import load_sample_dump

        streams = (
            [("ground truth", "gt_poses"), ("predicted", "pred_poses")]
            if args.poses == "both"
            else [(args.poses.replace("_poses", ""), args.poses)]
        )
        for name, stream in streams:
            poses, raw_refs, _ = load_sample_dump(
                args.body_npy,
                args.hand_npy,
                poses=stream,
                repetition=args.repetition,
            )
            references = [_normalize(r, data_cfg) for r in raw_refs]
            results[name], hypotheses = run(poses, references)
            count = len(poses)
        label = Path(args.body_npy).name

    elif args.split:
        dataset = _build_split(data_cfg, args.split, None)
        poses = [dataset[i].sgn for i in range(len(dataset))]
        references = [" ".join(e.txt) for e in dataset.examples]
        results["ground truth"], hypotheses = run(poses, references)
        count, label = len(poses), f"ground truth, {args.split} split"

    else:
        poses = _load_features(args.features)
        with open(args.references, "r", encoding="utf-8") as handle:
            raw = [line.rstrip("\n") for line in handle]
        references = [_normalize(r, data_cfg) for r in raw]
        results["predicted"], hypotheses = run(poses, references)
        count, label = len(poses), Path(args.features).name

    _report(results, count, label)

    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(hypotheses) + "\n")
        print(f"\nhypotheses -> {args.output_path}")


def main(argv: List[str] = None) -> None:
    """Entry point for the `signspark-bt` command."""
    args = build_parser().parse_args(argv)

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.mode == "train":
        from .training.trainer import train

        train(cfg_file=args.config_path, change_cfg=collect_overrides(args))

    elif args.mode == "test":
        from .training.prediction import test

        test(cfg_file=args.config_path, ckpt=args.ckpt, output_path=args.output_path)

    elif args.mode == "score":
        _score(args)

    elif args.mode == "translate":
        from .inference import back_translate, make_back_translation_model

        model = make_back_translation_model(args.model_dir)
        sentences = back_translate(
            model=model,
            poses=_load_features(args.features),
            batch_size=args.batch_size,
        )
        if args.output_path:
            with open(args.output_path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(sentences) + "\n")
            print(f"Wrote {len(sentences)} translations to {args.output_path}")
        else:
            for sentence in sentences:
                print(sentence)


if __name__ == "__main__":
    main()
