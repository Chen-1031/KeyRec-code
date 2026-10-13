"""Video-MME-v2 main-result runner for the shared video-model interface."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import logging
import random
import re
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from models import build_video_model

logger = logging.getLogger(__name__)
ANSWER_PREFIXES = (
    "Final Answer:",
    "The best answer is",
    "The correct answer is",
    "The answer is",
    "The answer",
    "The best option is",
    "The correct option is",
    "Best answer:",
    "Best option:",
    "Answer:",
    "Option:",
)


def resolve_dataset_root(root: str | Path) -> Path:
    """Resolve an extracted snapshot or Hugging Face cache repository."""
    root = Path(root).expanduser().resolve()
    if (root / "Video-MME-v2.tsv").is_file() and (root / "video").is_dir():
        return root
    snapshots = root / "snapshots"
    candidates = (
        [path.parent for path in snapshots.glob("*/Video-MME-v2.tsv")]
        if snapshots.is_dir()
        else []
    )
    candidates = [path for path in candidates if (path / "video").is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"could not find an extracted Video-MME-v2 snapshot under {root}"
        )
    if len(candidates) != 1:
        raise ValueError(
            "multiple Video-MME-v2 snapshots found; pass the exact immutable "
            "snapshot directory"
        )
    return candidates[0]


def load_samples(root: Path, annotation_file: str) -> list[dict]:
    annotation_path = Path(annotation_file)
    if not annotation_path.is_absolute():
        annotation_path = root / annotation_path
    with annotation_path.open(encoding="utf-8", newline="") as handle:
        samples = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "index",
        "video",
        "video_path",
        "question",
        "options",
        "answer",
        "level",
        "group_type",
        "group_structure",
        "second_head",
        "third_head",
    }
    if not samples:
        raise ValueError("Video-MME-v2 annotation contains no samples")
    missing = required - set(samples[0])
    if missing:
        raise ValueError(
            f"Video-MME-v2 annotation is missing columns: {sorted(missing)}"
        )
    return samples


def build_prompt(sample: dict) -> str:
    """Match the non-thinking, no-subtitle Video-MME-v2 prompt."""
    return "\n".join(
        (
            "These are the frames of a video.",
            f"Question: {sample['question']}",
            sample["options"],
            "Select the best answer to the multiple-choice question based on the "
            "video. Respond with only the letter (A, B, C, D, E, F, G, or H) "
            "of the correct option.",
        )
    )


def parse_multi_choice_response(response: str | None) -> tuple[str, bool]:
    """Apply Video-MME-v2's exact A-H parser without a judge model."""
    text = (response or "").strip()
    for prefix in ANSWER_PREFIXES:
        text = text.replace(prefix, "")
    match = re.search(r"[A-H]", text)
    return (match.group(0), False) if match else ("", True)


def cal_relevance(scores: list[bool | int]) -> tuple[float, float]:
    correct_count = sum(int(score) for score in scores)
    nonlinear = {0: 0.0, 1: 100.0 / 16, 2: 25.0, 3: 56.25, 4: 100.0}
    return nonlinear.get(correct_count, 0.0), correct_count * 25.0


def cal_logic(scores: list[bool | int], group_structure: str) -> float:
    structure = ast.literal_eval(group_structure)
    last_correct_index = -1
    for index, score in enumerate(scores):
        if score:
            last_correct_index = index
        else:
            break
    if structure == [1, 2, 3, 4]:
        score_map = {0: 0.0, 1: 6.25, 2: 25.0, 3: 56.25, 4: 100.0}
    elif structure == [1, [2, 3], 4]:
        score_map = {
            0: 0.0,
            1: 100.0 / 12,
            2: 100.0 * 4 / 12,
            3: 100.0 * 7 / 12,
            4: 100.0,
        }
        if last_correct_index == 0 and scores[2]:
            last_correct_index += 1
    elif structure == [[1, 2], 3, 4]:
        score_map = {0: 0.0, 1: 10.0, 2: 20.0, 3: 50.0, 4: 100.0}
        if last_correct_index == -1 and scores[1]:
            last_correct_index += 1
    else:
        raise ValueError(f"unknown Video-MME-v2 group structure: {structure}")
    return score_map.get(last_correct_index + 1, 0.0)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compute_official_rating(predictions: list[dict]) -> dict:
    """Compute the official group metric over complete four-question videos."""
    by_video: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(predictions, key=lambda item: int(item["index"])):
        by_video[str(row["video"])].append(row)
    final = {
        key: []
        for key in (
            "level_1",
            "level_2",
            "level_3",
            "relevance_score",
            "relevance_linear_score",
            "logic_score",
            "total",
        )
    }
    second_heads: dict[str, list[float]] = defaultdict(list)
    third_heads: dict[str, list[float]] = defaultdict(list)
    complete_groups = 0
    incomplete_groups = 0
    for group in by_video.values():
        if len(group) != 4:
            incomplete_groups += 1
            continue
        complete_groups += 1
        tail = group[-1]
        scores = [bool(item.get("correct")) for item in group]
        if tail["group_type"] == "relevance":
            group_score, linear_score = cal_relevance(scores)
            final["relevance_score"].append(group_score)
            final["relevance_linear_score"].append(linear_score)
        elif tail["group_type"] == "logic":
            group_score = cal_logic(scores, tail["group_structure"])
            final["logic_score"].append(group_score)
        else:
            raise ValueError(f"unknown group type: {tail['group_type']}")
        level = str(tail.get("level", ""))
        if level in {"1", "2", "3"}:
            final[f"level_{level}"].append(group_score)
        final["total"].append(group_score)
        second_heads[str(tail.get("second_head"))].append(group_score)
        third_heads[str(tail.get("third_head"))].append(group_score)
    return {
        "final_rating": {key: _mean(values) for key, values in final.items()},
        "second_head_rating": {
            key: _mean(values) for key, values in sorted(second_heads.items())
        },
        "third_head_rating": {
            key: _mean(values) for key, values in sorted(third_heads.items())
        },
        "complete_groups": complete_groups,
        "incomplete_groups": incomplete_groups,
    }


def _summary(rows: list[dict]) -> dict:
    correct = sum(bool(row.get("correct")) for row in rows)
    return {
        "total": len(rows),
        "correct": correct,
        "accuracy": 100.0 * correct / len(rows) if rows else 0.0,
        "accuracy_fraction": correct / len(rows) if rows else 0.0,
    }


def compute_metrics(predictions: list[dict]) -> dict:
    metrics = _summary(predictions)
    metrics.update(
        {
            "processed": len(predictions),
            "errors": sum(row.get("response") is None for row in predictions),
            "parse_fallbacks": sum(
                bool(row.get("parse_fallback")) for row in predictions
            ),
            "by_question_level": {
                level: _summary(
                    [row for row in predictions if str(row["level"]) == level]
                )
                for level in ("1", "2", "3")
            },
            "official": compute_official_rating(predictions),
        }
    )
    return metrics


def load_existing_predictions(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    return {str(row["index"]): row for row in rows if row.get("response") is not None}


def _write_outputs(output_dir: Path, predictions: list[dict]) -> dict:
    metrics = compute_metrics(predictions)
    (output_dir / "predictions.json").write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(args):
    return build_video_model(
        args.backend,
        args.model_path,
        num_frames=args.num_frames,
        sample_fps=args.sample_fps,
        max_soft_tokens=args.max_soft_tokens,
        max_new_tokens=args.max_new_tokens,
    )


def run_from_args(args) -> None:
    root = resolve_dataset_root(args.videommev2_root)
    samples = load_samples(root, args.annotation_file)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError(
            f"invalid shard {args.shard_index} for num_shards={args.num_shards}"
        )
    if args.num_shards > 1:
        samples = [
            sample
            for sample in samples
            if int.from_bytes(
                hashlib.sha256(sample["video"].encode("utf-8")).digest()[:8],
                "big",
            )
            % args.num_shards
            == args.shard_index
        ]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = (
        load_existing_predictions(output_dir / "predictions.json")
        if args.resume
        else {}
    )
    predictions: list[dict] = []
    seed_everything(args.seed)
    model = _build_model(args)

    for processed, sample in enumerate(tqdm(samples, total=len(samples)), start=1):
        sample_id = str(sample["index"])
        if sample_id in existing:
            predictions.append(existing[sample_id])
            continue
        video_path = root / sample["video_path"].removeprefix("./")
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        prompt = build_prompt(sample)
        response = None
        error = None
        started = time.perf_counter()
        try:
            response = model.generate(
                str(video_path),
                prompt,
                num_frames=args.num_frames,
                sample_fps=args.sample_fps,
                sampling=args.sampling,
                max_new_tokens=args.max_new_tokens,
                read_policy=args.read_policy,
                router_max_new_tokens=args.router_max_new_tokens,
                add_timestamp_overlay=args.add_timestamp_overlay,
            )
        except Exception as exc:
            logger.exception("failed sample %s (%s)", sample_id, video_path)
            error = f"{type(exc).__name__}: {exc}"
        prediction, parse_fallback = parse_multi_choice_response(response)
        predictions.append(
            {
                **sample,
                "video_path": str(video_path),
                "prompt": prompt,
                "response": response,
                "prediction": prediction,
                "ground_truth": sample["answer"],
                "correct": response is not None and prediction == sample["answer"],
                "parse_fallback": parse_fallback,
                "error": error,
                "elapsed_sec": time.perf_counter() - started,
                "actual_sampled_frames": getattr(model, "last_frame_count", None),
            }
        )
        if processed % args.save_every == 0:
            _write_outputs(output_dir, predictions)

    metrics = _write_outputs(output_dir, predictions)
    normalized = {
        "benchmark": "Video-MME-v2",
        "dataset": str(root / args.annotation_file),
        "model": {"backend": args.backend, "checkpoint": args.model_path},
        "sampling": args.sampling,
        "num_frames": args.num_frames,
        "sample_fps": args.sample_fps,
        "seed": args.seed,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "metrics": metrics,
    }
    (output_dir / "normalized.json").write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
