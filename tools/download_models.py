#!/usr/bin/env python3
"""Download back-translation checkpoints from the HF Hub.

    python tools/download_models.py --datasets PHOENIX-2014T --dest ./checkpoints

Each dataset directory holds config.yaml, best.ckpt and the two vocab files.
"""

import argparse
import os
from pathlib import Path

DEFAULT_REPO_ID = "LionelLow/SignSparK_BT"
DATASETS = ["CSL-Daily", "How2Sign", "PHOENIX-2014T"]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo-id", default=os.getenv("SIGNSPARK_BT_MODEL_REPO", DEFAULT_REPO_ID)
    )
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument(
        "--dest", default=os.getenv("SIGNSPARK_BT_CKPT_DIR", "./checkpoints")
    )
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(dest),
        allow_patterns=[f"{name}/*" for name in args.datasets],
    )
    print(f"### Done -> {dest.resolve()}  (set SIGNSPARK_BT_CKPT_DIR to this)")


if __name__ == "__main__":
    main()
