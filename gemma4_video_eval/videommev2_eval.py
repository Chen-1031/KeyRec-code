"""CLI entry point for the canonical Video-MME-v2 main evaluation."""

from __future__ import annotations

import argparse

from gemma4_video_eval.cli import add_model_arguments
from gemma4_video_eval.storage import configure_storage_env


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser, num_frames=64, max_new_tokens=64)
    parser.add_argument("--videommev2-root", default="data/Video-MME-v2")
    parser.add_argument("--annotation-file", default="Video-MME-v2.tsv")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    configure_storage_env()
    from benchmarks.videommev2 import run_from_args

    run_from_args(parse_args())


if __name__ == "__main__":
    main()
