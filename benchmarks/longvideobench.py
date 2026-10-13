"""LongVideoBench validation wrapper for the shared video-model pipeline."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from models import build_video_model

logger = logging.getLogger(__name__)

ANSWER_PREFIXES = (
    "The best answer is",
    "The correct answer is",
    "The answer is",
    "The answer",
    "The best option is",
    "The correct option is",
    "Best answer:",
    "Best option:",
)


def build_prompt(sample: dict) -> str:
    lines = [f"Question: {sample['question']}"]
    lines.extend(
        f"{chr(ord('A') + index)}. {candidate}"
        for index, candidate in enumerate(sample["candidates"])
    )
    lines.append("Answer with the option's letter from the given choices directly.")
    return "\n".join(lines)


def parse_multi_choice_response(
    response: str | None,
    choices: list[str],
    *,
    sample_id: str,
) -> tuple[str, bool]:
    """Match the official parser, with a deterministic random-guess fallback."""
    text = (response or "").strip()
    for prefix in ANSWER_PREFIXES:
        text = text.replace(prefix, "")

    match = re.search(r"[ABCDE]", text)
    if match is not None:
        return match.group(0), False

    # LongVideoBench assigns a random choice when parsing fails. Deriving the
    # choice from the sample id preserves that policy without score drift.
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return choices[int.from_bytes(digest[:8], "big") % len(choices)], True


def _summarize(rows: list[dict]) -> dict:
    total = len(rows)
    correct = sum(bool(row.get("correct")) for row in rows)
    return {
        "total": total,
        "correct": correct,
        "accuracy": 100.0 * correct / total if total else 0.0,
        "accuracy_fraction": correct / total if total else 0.0,
    }


def compute_metrics(
    predictions: list[dict], *, count_errors_as_incorrect: bool = False
) -> dict:
    successful = [row for row in predictions if row.get("response") is not None]
    evaluated = predictions if count_errors_as_incorrect else successful
    by_duration: dict[str, list[dict]] = defaultdict(list)
    by_question_category: dict[str, list[dict]] = defaultdict(list)
    for row in evaluated:
        by_duration[str(row.get("duration_group", "unknown"))].append(row)
        by_question_category[str(row.get("question_category", "unknown"))].append(row)

    metrics = _summarize(evaluated)
    metrics.update(
        {
            "processed": len(predictions),
            "errors": len(predictions) - len(successful),
            "parse_fallbacks": sum(
                bool(row.get("parse_fallback")) for row in successful
            ),
            "by_duration_group": {
                key: _summarize(rows) for key, rows in sorted(by_duration.items())
            },
            "by_question_category": {
                key: _summarize(rows)
                for key, rows in sorted(by_question_category.items())
            },
        }
    )
    return metrics


def load_existing_predictions(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    return {str(row["id"]): row for row in rows if row.get("response") is not None}


def build_model_from_args(args):
    return build_video_model(
        args.backend,
        args.model_path,
        num_frames=args.num_frames,
        sample_fps=args.sample_fps,
        max_soft_tokens=args.max_soft_tokens,
        max_new_tokens=args.max_new_tokens,
    )


def _write_outputs(
    predictions_path: Path,
    metrics_path: Path,
    predictions: list[dict],
    *,
    count_errors_as_incorrect: bool = False,
) -> dict:
    metrics = compute_metrics(
        predictions, count_errors_as_incorrect=count_errors_as_incorrect
    )
    predictions_path.write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def run_from_args(args) -> None:
    root = Path(args.longvideobench_root)
    annotation_path = Path(args.annotation_file)
    if not annotation_path.is_absolute():
        annotation_path = root / annotation_path
    with annotation_path.open(encoding="utf-8") as f:
        samples = json.load(f)

    duration_groups = set(args.duration_groups or [])
    if duration_groups:
        samples = [
            sample
            for sample in samples
            if int(sample.get("duration_group", -1)) in duration_groups
        ]
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError(
            f"Invalid shard {args.shard_index} for num_shards={args.num_shards}"
        )
    if args.num_shards > 1:
        samples = [
            sample
            for sample in samples
            if int.from_bytes(
                hashlib.sha256(str(sample["video_path"]).encode("utf-8")).digest()[:8],
                "big",
            )
            % args.num_shards
            == args.shard_index
        ]
    if args.sort_by_duration_desc:
        samples.sort(
            key=lambda sample: float(sample.get("duration", 0.0)), reverse=True
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.json"
    metrics_path = output_dir / "metrics.json"
    normalized_path = output_dir / "normalized.json"

    existing = load_existing_predictions(predictions_path) if args.resume else {}
    predictions: list[dict] = []
    model = build_model_from_args(args)
    selected = samples[: args.max_samples] if args.max_samples is not None else samples
    run_status = "complete"

    for processed, sample in enumerate(tqdm(selected, total=len(selected)), start=1):
        sample_id = str(sample["id"])
        if sample_id in existing:
            predictions.append(existing[sample_id])
            continue

        video_path = root / "videos" / sample["video_path"]
        prompt = build_prompt(sample)
        error = None
        sample_start = time.perf_counter()
        torch_module = None
        try:
            import torch

            torch_module = torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        try:
            response = model.generate(
                str(video_path),
                prompt,
                num_frames=args.num_frames,
                sample_fps=args.sample_fps,
                sampling=args.sampling,
                max_new_tokens=args.max_new_tokens,
                add_timestamp_overlay=args.add_timestamp_overlay,
                read_policy=args.read_policy,
                router_max_new_tokens=args.router_max_new_tokens,
            )
        except Exception as exc:
            logger.exception("Failed sample %s (%s)", sample_id, video_path)
            response = None
            error = str(exc)
            message = f"{type(exc).__name__}: {exc}".lower()
            if "out of memory" in message and "cuda" in message:
                error_type = "oom"
            elif any(
                marker in message
                for marker in (
                    "maximum context",
                    "max_position_embeddings",
                    "sequence length",
                    "context length",
                    "index out of range in self",
                )
            ):
                error_type = "context_overflow"
            else:
                error_type = "error"
            if error_type == "oom":
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
                gc.collect()
        else:
            error_type = None

        elapsed_sec = time.perf_counter() - sample_start
        peak_gpu_memory_mib = (
            torch_module.cuda.max_memory_allocated() / (1024**2)
            if torch_module is not None and torch_module.cuda.is_available()
            else None
        )
        actual_sampled_frames = getattr(model, "last_frame_count", None)

        router_decision = (
            getattr(model, "last_router_decision", None) if error is None else None
        )
        read_policy = (
            getattr(model, "last_read_policy", None) if error is None else None
        )
        memory_segments = (
            getattr(model, "last_memory_segments", None) if error is None else None
        )
        actual_visual_tokens = (
            sum(int(segment.token_count) for segment in memory_segments)
            if memory_segments
            else getattr(model, "last_visual_token_count", None)
        )

        choices = [chr(ord("A") + index) for index in range(len(sample["candidates"]))]
        prediction, parse_fallback = parse_multi_choice_response(
            response, choices, sample_id=sample_id
        )
        answer = chr(ord("A") + int(sample["correct_choice"]))
        predictions.append(
            {
                "id": sample_id,
                "video_path": str(video_path),
                "duration": sample.get("duration"),
                "duration_group": sample.get("duration_group"),
                "question_category": sample.get("question_category"),
                "topic_category": sample.get("topic_category"),
                "level": sample.get("level"),
                "question": sample["question"],
                "candidates": sample["candidates"],
                "prompt": prompt,
                "router_category": getattr(
                    getattr(router_decision, "state", None), "value", None
                ),
                "router_reason": getattr(router_decision, "reason", None),
                "router_used_default": getattr(router_decision, "used_fallback", None),
                "router_raw_output": getattr(router_decision, "raw_output", None),
                "read_policy": getattr(read_policy, "profile_name", None),
                "read_recent_tokens": sum(
                    segment.token_count
                    for segment in (memory_segments or [])
                    if segment.source == "recent"
                ),
                "read_num_events": sum(
                    segment.source == "event" for segment in (memory_segments or [])
                ),
                "read_tokens_per_event": getattr(read_policy, "tokens_per_event", None),
                "actual_visual_tokens": actual_visual_tokens,
                "actual_sampled_frames": actual_sampled_frames,
                "elapsed_sec": elapsed_sec,
                "peak_gpu_memory_mib": peak_gpu_memory_mib,
                "response": response,
                "error": error,
                "error_type": error_type,
                "prediction": prediction,
                "parse_fallback": parse_fallback,
                "ground_truth": answer,
                "correct": response is not None and prediction == answer,
            }
        )

        if processed % args.save_every == 0:
            _write_outputs(
                predictions_path,
                metrics_path,
                predictions,
                count_errors_as_incorrect=args.count_errors_as_incorrect,
            )

        if args.abort_on_capacity_error and error_type in {"oom", "context_overflow"}:
            run_status = error_type
            logger.error(
                "Stopping LongVideoBench run after capacity error: %s", error_type
            )
            break

    metrics = _write_outputs(
        predictions_path,
        metrics_path,
        predictions,
        count_errors_as_incorrect=args.count_errors_as_incorrect,
    )
    normalized_path.write_text(
        json.dumps(
            {
                "benchmark": "longvideobench_val",
                "dataset": str(annotation_path),
                "sampling": args.sampling,
                "num_frames": args.num_frames,
                "sample_fps": args.sample_fps,
                "duration_groups": sorted(duration_groups),
                "sort_by_duration_desc": args.sort_by_duration_desc,
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "count_errors_as_incorrect": args.count_errors_as_incorrect,
                "status": run_status,
                "subtitle_mode": "none",
                "metrics": metrics,
                "predictions_file": str(predictions_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("LongVideoBench val metrics: %s", metrics)
