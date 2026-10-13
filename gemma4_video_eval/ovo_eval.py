"""CLI entrypoint for OVO-Bench evaluation."""

import argparse
from gemma4_video_eval.cli import add_model_arguments
from gemma4_video_eval.storage import configure_storage_env


def parse_args():
    from benchmarks.ovo import DEFAULT_TASKS

    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser, num_frames=32, max_new_tokens=512)
    parser.add_argument("--ovo-root", default="data/OVO-Bench")
    parser.add_argument("--anno-path", default="data/ovo_bench_new.json")
    parser.add_argument("--video-dir", default="data/src_videos")
    parser.add_argument("--chunked-dir", default="data/chunked_videos")
    parser.add_argument("--mode", choices=["offline"], default="offline")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--save-tag", default="keyrec")
    return parser.parse_args()


def main():
    configure_storage_env()
    from benchmarks.ovo import run_from_args

    run_from_args(parse_args())


if __name__ == "__main__":
    main()
