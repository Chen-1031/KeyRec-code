"""CLI entrypoint for LongVideoBench validation evaluation."""

from __future__ import annotations

import argparse

from gemma4_video_eval.cli import add_model_arguments
from gemma4_video_eval.storage import configure_storage_env

DEFAULT_LONGVIDEOBENCH_ROOT = "data/LongVideoBench"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser, num_frames=64, max_new_tokens=64)
    parser.add_argument("--longvideobench-root", default=DEFAULT_LONGVIDEOBENCH_ROOT)
    parser.add_argument("--annotation-file", default="lvb_val.json")
    parser.add_argument(
        "--duration-groups",
        type=int,
        nargs="+",
        default=None,
        help="Evaluate only samples in the selected LongVideoBench duration groups.",
    )
    parser.add_argument(
        "--sort-by-duration-desc",
        action="store_true",
        help="Run the longest videos first, useful for capacity-limit experiments.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--abort-on-capacity-error",
        action="store_true",
        help="Stop and mark the run when CUDA OOM or context overflow is observed.",
    )
    parser.add_argument(
        "--count-errors-as-incorrect",
        action="store_true",
        help="Keep failed samples in the accuracy denominator.",
    )
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")

    return parser.parse_args()


def main():
    configure_storage_env()
    from benchmarks.longvideobench import run_from_args

    run_from_args(parse_args())


if __name__ == "__main__":
    main()
