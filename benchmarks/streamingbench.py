"""StreamingBench wrapper for the Streaming VU evaluation pipeline."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from tqdm import tqdm

from models import build_video_model

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

Question: {}

Options:
{}
{}
{}
{}"""

PROMPT_TEMPLATE_WITHOUT_OPTIONS = """You are an advanced video question-answering AI assistant. You have been provided with a video and a question related to the video. Your task is to carefully analyze the video and provide the answer to the question.

Question: {}
"""


def timestamp_to_seconds(value: str | int | float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    parts = [float(part) for part in str(value).split(":")]
    total = 0.0
    for part in parts:
        total = total * 60 + part
    return total


def format_options(options: list[str]) -> list[str]:
    if options and options[0].startswith("A."):
        return options
    labels = ["A", "B", "C", "D"]
    return [f"{label}. {option}" for label, option in zip(labels, options)]


def build_prompt(question: dict) -> str:
    query = question["question"]
    if "options" in question:
        options = format_options(question["options"])
        prompt = PROMPT_TEMPLATE.format(query, *options)
        return prompt + "\n\nThe best option is:"
    prompt = PROMPT_TEMPLATE_WITHOUT_OPTIONS.format(query)
    return prompt + "\n\nAnswer:"


def official_first_char(response: str | None) -> str:
    if not response:
        return ""
    stripped = response.strip()
    return stripped[0].upper() if stripped else ""


def normalized_choice(response: str | None) -> str:
    if not response:
        return ""
    match = re.search(r"\b([ABCD])\b", response.strip(), flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def iter_real_questions(data: list[dict]):
    for entry_index, entry in enumerate(data):
        for question_index, question in enumerate(entry.get("questions", [])):
            yield entry_index, question_index, entry, question


def load_existing_predictions(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open() as f:
        rows = json.load(f)
    return {row["id"]: row for row in rows if row.get("response") is not None}


def compute_metrics(predictions: list[dict]) -> dict:
    by_task: dict[str, dict] = {}
    total = 0
    correct = 0
    normalized_correct = 0

    for row in predictions:
        if row.get("response") is None:
            continue
        task = row.get("task_type", "unknown")
        bucket = by_task.setdefault(
            task, {"total": 0, "correct": 0, "normalized_correct": 0}
        )
        total += 1
        bucket["total"] += 1
        if row.get("correct"):
            correct += 1
            bucket["correct"] += 1
        if row.get("normalized_correct"):
            normalized_correct += 1
            bucket["normalized_correct"] += 1

    for bucket in by_task.values():
        denom = bucket["total"] or 1
        bucket["accuracy"] = 100.0 * bucket["correct"] / denom
        bucket["official_first_char_accuracy"] = bucket["accuracy"]
        bucket["normalized_accuracy"] = 100.0 * bucket["normalized_correct"] / denom

    denom = total or 1
    return {
        "total": total,
        "correct": correct,
        "accuracy": 100.0 * correct / denom,
        "official_first_char_accuracy": 100.0 * correct / denom,
        "normalized_correct": normalized_correct,
        "normalized_accuracy": 100.0 * normalized_correct / denom,
        "by_task": by_task,
    }


def build_model_from_args(args):
    return build_video_model(
        args.backend,
        args.model_path,
        num_frames=args.num_frames,
        sample_fps=args.sample_fps,
        max_soft_tokens=args.max_soft_tokens,
        max_new_tokens=args.max_new_tokens,
    )


def run_from_args(args) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.json"
    metrics_path = output_dir / "metrics.json"
    normalized_path = output_dir / "normalized.json"

    data_path = Path(args.data_file)
    if not data_path.is_absolute():
        data_path = Path(args.streamingbench_root) / data_path
    with data_path.open() as f:
        data = json.load(f)

    existing = load_existing_predictions(predictions_path) if args.resume else {}
    predictions: list[dict] = []
    model = build_model_from_args(args)

    total_questions = sum(len(entry.get("questions", [])) for entry in data)
    if args.max_samples is not None:
        total_questions = min(total_questions, args.max_samples)

    processed = 0
    for entry_index, question_index, entry, question in tqdm(
        iter_real_questions(data), total=total_questions
    ):
        if args.max_samples is not None and processed >= args.max_samples:
            break
        sample_id = f"real-{entry_index}-{question_index}"
        if sample_id in existing:
            predictions.append(existing[sample_id])
            processed += 1
            continue

        timestamp = timestamp_to_seconds(question["time_stamp"])
        if args.context_time > 0:
            start_time = max(0.0, timestamp - args.context_time)
        else:
            start_time = 0.0
        end_time = timestamp

        video_path = entry["video_path"]
        prompt = build_prompt(question)
        error = None
        try:
            response = model.generate(
                video_path,
                prompt,
                num_frames=args.num_frames,
                sample_fps=args.sample_fps,
                sampling=args.sampling,
                max_new_tokens=args.max_new_tokens,
                add_timestamp_overlay=args.add_timestamp_overlay,
                video_start_time=start_time,
                video_end_time=end_time,
                read_policy=args.read_policy,
                router_max_new_tokens=args.router_max_new_tokens,
            )
        except Exception as exc:
            logger.exception(
                "Failed sample %s (%s); marking response as None and continuing.",
                sample_id,
                video_path,
            )
            response = None
            error = str(exc)
        official = official_first_char(response)
        normalized = normalized_choice(response)
        answer = str(question["answer"]).strip().upper()
        row = {
            "id": sample_id,
            "entry_index": entry_index,
            "question_index": question_index,
            "video_path": video_path,
            "start_time": start_time,
            "end_time": end_time,
            "time_stamp": question["time_stamp"],
            "task_type": question.get("task_type"),
            "required_ability": question.get("required_ability"),
            "question": question.get("question"),
            "options": question.get("options"),
            "prompt": prompt,
            "response": response,
            "error": error,
            "prediction": official,
            "normalized_prediction": normalized,
            "ground_truth": answer,
            "correct": official == answer,
            "normalized_correct": normalized == answer,
        }
        predictions.append(row)
        processed += 1

        if processed % args.save_every == 0:
            predictions_path.write_text(json.dumps(predictions, indent=2))
            metrics_path.write_text(json.dumps(compute_metrics(predictions), indent=2))

    metrics = compute_metrics(predictions)
    predictions_path.write_text(json.dumps(predictions, indent=2))
    metrics_path.write_text(json.dumps(metrics, indent=2))
    normalized_path.write_text(
        json.dumps(
            {
                "benchmark": "streamingbench_real",
                "dataset": str(data_path),
                "context_time": args.context_time,
                "sampling": args.sampling,
                "num_frames": args.num_frames,
                "sample_fps": args.sample_fps,
                "metrics": metrics,
                "predictions_file": str(predictions_path),
            },
            indent=2,
        )
    )
    logger.info("StreamingBench real metrics: %s", metrics)
