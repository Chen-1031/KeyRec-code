"""Shared command-line options for video evaluation."""

import argparse

BACKENDS = ("gemma", "neo_ov", "keyrec", "keyrec_neo")
READ_POLICIES = (
    "llm_router",
    "all_recent",
    "heavy_recent",
    "balanced",
    "heavy_key_event",
    "all_key_event",
)


def add_model_arguments(
    parser: argparse.ArgumentParser,
    *,
    num_frames: int,
    max_new_tokens: int,
    sampling: str = "linspace",
) -> None:
    parser.add_argument("--backend", required=True, choices=BACKENDS)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-frames", type=int, default=num_frames)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument(
        "--sampling",
        choices=["step", "linspace", "uniform", "tail", "recent"],
        default=sampling,
    )
    parser.add_argument("--max-soft-tokens", type=int, default=70)
    parser.add_argument("--max-new-tokens", type=int, default=max_new_tokens)
    parser.add_argument("--add-timestamp-overlay", action="store_true")
    parser.add_argument("--read-policy", choices=READ_POLICIES, default="llm_router")
    parser.add_argument("--router-max-new-tokens", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
