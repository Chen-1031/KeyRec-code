"""CLI entrypoint for StreamingBench real-subset evaluation."""

import argparse
from gemma4_video_eval.cli import add_model_arguments
from gemma4_video_eval.storage import configure_storage_env


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser, num_frames=32, max_new_tokens=128)
    parser.add_argument("--streamingbench-root", default="data/StreamingBench")
    parser.add_argument("--data-file", default="src/data/questions_real.json")
    parser.add_argument("--task", choices=["real"], default="real")
    parser.add_argument("--context-time", type=float, default=-1.0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    configure_storage_env()
    from benchmarks.streamingbench import run_from_args

    run_from_args(parse_args())


if __name__ == "__main__":
    main()
