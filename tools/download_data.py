#!/usr/bin/env python3
"""Download the prebuilt SignSparK LMDBs from the HF Hub into ${DATA_ROOT}/lmdb/.

    python tools/download_data.py --datasets PHOENIX-2014T --dest ./data
    export DATA_ROOT=$(pwd)/data

The same script ships with the SignSparK repo; it is duplicated here so this
repo can be set up on its own.
"""

import argparse
import os
from pathlib import Path

DEFAULT_REPO_ID = "LionelLow/SignSparK_data"
DATASETS = ["CSL-Daily", "How2Sign", "PHOENIX-2014T"]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo-id",
        default=os.getenv("SIGNSPARK_DATA_REPO", DEFAULT_REPO_ID),
        help="HuggingFace dataset repo holding the LMDBs",
    )
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument(
        "--dest",
        default=os.getenv("DATA_ROOT", "./data"),
        help="DATA_ROOT; LMDBs land under <dest>/lmdb/<split>/",
    )
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    dest = Path(args.dest) / "lmdb"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"### Downloading {args.datasets} from {args.repo_id} -> {dest}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=[f"*{d}*" for d in args.datasets],
    )
    print(f"### Done -> export DATA_ROOT={Path(args.dest).resolve()}")


if __name__ == "__main__":
    main()
