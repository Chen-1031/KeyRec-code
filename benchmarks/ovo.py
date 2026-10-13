"""OVO-Bench wrapper that uses Streaming VU model adapters."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from tqdm import tqdm

from models import build_video_model, ovo_score_model_name

DEFAULT_OVO_ROOT = "data/OVO-Bench"
DEFAULT_TASKS = [
    "EPM",
    "ASI",
    "HLD",
    "STU",
    "OJR",
    "ATR",
    "ACR",
    "OCR",
    "FPD",
]


def resolve_ovo_path(ovo_root: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(ovo_root / path)


def load_ovo_constants(ovo_root: Path):
    constant_path = ovo_root / "constant.py"
    spec = importlib.util.spec_from_file_location(
        "_streaming_vu_ovo_constant", constant_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load OVO constants from {constant_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ovo_annotations(anno_path: str, video_dir: str, tasks: list[str]) -> dict:
    with open(anno_path) as f:
        annotations = json.load(f)

    backward_tasks = {"EPM", "ASI", "HLD"}
    realtime_tasks = {"STU", "OJR", "ATR", "ACR", "OCR", "FPD"}
    forward_tasks = {"REC", "SSR", "CRR"}
    selected_tasks = set(tasks)

    splits = {"backward": [], "realtime": [], "forward": []}
    for item in annotations:
        if item["task"] not in selected_tasks:
            continue
        item = copy.deepcopy(item)
        item["video"] = os.path.join(video_dir, item["video"])
        if item["task"] in backward_tasks:
            splits["backward"].append(item)
        if item["task"] in realtime_tasks:
            splits["realtime"].append(item)
        if item["task"] in forward_tasks:
            splits["forward"].append(item)
    return splits


def limit_annotations(annotations: dict, max_samples: int | None) -> dict:
    """Keep at most max_samples annotations in stable split order."""
    if max_samples is None:
        return annotations
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    remaining = max_samples
    limited = {}
    for split in ("backward", "realtime", "forward"):
        rows = annotations.get(split, [])
        limited[split] = rows[:remaining]
        remaining -= len(limited[split])
    return limited


def build_prompt(constants, task: str, question, options, annotation, index):
    if task in ["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD"]:
        formatted_options = (
            "; ".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(options))
            + ";"
        )
        return constants.BR_PROMPT_TEMPLATE.format(question, formatted_options)
    if task == "REC":
        activity = annotation["activity"]
        question = "How many times did they " + activity + "?"
        return constants.REC_PROMPT_TEMPLATE.format(question)
    if task == "SSR":
        step = annotation["test_info"][index]["step"]
        return constants.SSR_PROMPT_TEMPLATE.format(step)
    if task == "CRR":
        question = annotation["question"]
        return constants.CRR_PROMPT_TEMPLATE.format(question)
    raise ValueError(f"Unsupported OVO task: {task}")


def build_ovo_runtime_args(args, ovo_root: Path, raw_result_dir: Path):
    return SimpleNamespace(
        anno_path=resolve_ovo_path(ovo_root, args.anno_path),
        video_dir=resolve_ovo_path(ovo_root, args.video_dir),
        chunked_dir=resolve_ovo_path(ovo_root, args.chunked_dir),
        result_dir=str(raw_result_dir),
        mode=args.mode,
        task=args.tasks,
        model=ovo_score_model_name(args.backend),
        model_path=args.model_path,
        save_results=True,
        save_tag=args.save_tag,
        read_policy=args.read_policy,
        sampling=args.sampling,
        router_max_new_tokens=args.router_max_new_tokens,
    )


def build_model_from_args(args):
    return build_video_model(
        args.backend,
        args.model_path,
        num_frames=args.num_frames,
        sample_fps=args.sample_fps,
        max_soft_tokens=args.max_soft_tokens,
        max_new_tokens=args.max_new_tokens,
    )


def run_inference(
    model, annotations: dict, task_list: list[str], mode: str, ovo_args, constants
) -> Path:
    backward_results = []
    for annotation in tqdm(annotations["backward"], desc="Backward Tasks"):
        sample_id = annotation["id"]
        prompt = build_prompt(
            constants,
            task=annotation["task"],
            question=annotation["question"],
            options=annotation["options"],
            annotation=None,
            index=None,
        )
        chunk_video_path = os.path.join(ovo_args.chunked_dir, f"{sample_id}.mp4")
        if not os.path.exists(chunk_video_path):
            raise FileNotFoundError(chunk_video_path)
        response = model.generate(
            chunk_video_path,
            prompt,
            read_policy=getattr(ovo_args, "read_policy", "llm_router"),
            sampling=getattr(ovo_args, "sampling", "step"),
            router_max_new_tokens=getattr(ovo_args, "router_max_new_tokens", 64),
        )
        backward_results.append(
            {
                "id": sample_id,
                "video": annotation["video"],
                "task": annotation["task"],
                "question": annotation["question"],
                "response": response,
                "ground_truth": chr(65 + annotation["gt"]),
            }
        )

    realtime_results = []
    for annotation in tqdm(annotations["realtime"], desc="Realtime Tasks"):
        sample_id = annotation["id"]
        prompt = build_prompt(
            constants,
            task=annotation["task"],
            question=annotation["question"],
            options=annotation["options"],
            annotation=None,
            index=None,
        )
        chunk_video_path = os.path.join(ovo_args.chunked_dir, f"{sample_id}.mp4")
        if not os.path.exists(chunk_video_path):
            raise FileNotFoundError(chunk_video_path)
        response = model.generate(
            chunk_video_path,
            prompt,
            read_policy=getattr(ovo_args, "read_policy", "llm_router"),
            sampling=getattr(ovo_args, "sampling", "step"),
            router_max_new_tokens=getattr(ovo_args, "router_max_new_tokens", 64),
        )
        realtime_results.append(
            {
                "id": sample_id,
                "video": annotation["video"],
                "task": annotation["task"],
                "question": annotation["question"],
                "response": response,
                "ground_truth": chr(65 + annotation["gt"]),
            }
        )

    forward_results = []
    for annotation in tqdm(annotations["forward"], desc="Forward Tasks"):
        forward_annotation = copy.deepcopy(annotation)
        sample_id = forward_annotation["id"]
        task = forward_annotation["task"]
        for i in range(len(forward_annotation["test_info"])):
            prompt = build_prompt(
                constants,
                task=task,
                question=None,
                options=None,
                annotation=forward_annotation,
                index=i,
            )
            chunk_video_path = os.path.join(
                ovo_args.chunked_dir, f"{sample_id}_{i}.mp4"
            )
            if not os.path.exists(chunk_video_path):
                raise FileNotFoundError(chunk_video_path)
            response = model.generate(
                chunk_video_path,
                prompt,
                read_policy=getattr(ovo_args, "read_policy", "llm_router"),
                sampling=getattr(ovo_args, "sampling", "step"),
                router_max_new_tokens=getattr(ovo_args, "router_max_new_tokens", 64),
            )
            forward_annotation["test_info"][i]["response"] = response
        forward_results.append(forward_annotation)

    if not ovo_args.save_results:
        raise ValueError("OVO wrapper currently expects save_results=True")

    model_variant = (
        Path(ovo_args.model_path).name if ovo_args.model_path else ovo_args.model
    )
    save_dir = Path(ovo_args.result_dir) / ovo_args.model
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = (
        save_dir
        / f"{model_variant}_{'_'.join(task_list)}_{mode}_{ovo_args.save_tag}.json"
    )
    save_path.write_text(
        json.dumps(
            {
                "backward": backward_results,
                "realtime": realtime_results,
                "forward": forward_results,
            },
            indent=4,
        )
    )
    return save_path


def validate_predictions(result_path: Path) -> None:
    data = json.loads(result_path.read_text())
    total = 0
    none_count = 0

    for split in ["backward", "realtime"]:
        for row in data.get(split, []):
            total += 1
            if row.get("response") is None:
                none_count += 1

    for row in data.get("forward", []):
        for test_item in row.get("test_info", []):
            total += 1
            if test_item.get("response") is None:
                none_count += 1

    if total == 0:
        raise RuntimeError(f"OVO prediction file contains no samples: {result_path}")
    if none_count == total:
        raise RuntimeError(
            f"All {total} OVO predictions are None. Check run.log for inference errors before scoring."
        )
    if none_count:
        print(
            f"WARNING: {none_count}/{total} OVO predictions are None; scoring partial results.",
            file=sys.stderr,
        )


def run_score(ovo_root: Path, ovo_args, output_dir: Path) -> dict:
    cmd = [
        sys.executable,
        "score.py",
        "--result_dir",
        ovo_args.result_dir,
        "--model",
        ovo_args.model,
        "--model_path",
        ovo_args.model_path,
        "--mode",
        ovo_args.mode,
        "--task",
        *ovo_args.task,
        "--save_tag",
        ovo_args.save_tag,
    ]
    proc = subprocess.run(cmd, cwd=ovo_root, capture_output=True, text=True)
    (output_dir / "score_stdout.txt").write_text(proc.stdout)
    (output_dir / "score_stderr.txt").write_text(proc.stderr)
    if proc.stdout:
        print("\n=== OVO-Bench Scoring ===")
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print("\n=== OVO-Bench Scoring stderr ===", file=sys.stderr)
        print(
            proc.stderr, end="" if proc.stderr.endswith("\n") else "\n", file=sys.stderr
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"OVO scoring failed with code {proc.returncode}: {proc.stderr[-2000:]}"
        )
    return parse_score_stdout(proc.stdout)


def parse_score_stdout(stdout: str) -> dict:
    task_acc = {}
    for task, acc in re.findall(r"Task: (\w+), Acc: ([\d.]+)", stdout):
        task_acc[task] = float(acc)

    metrics = {}
    avg_key_map = {
        "Backward": "backward_avg",
        "Realtime": "realtime_avg",
        "Forward": "forward_avg",
        "Total": "total_avg",
    }
    for name, acc in re.findall(r"([\w-]+) Avg\.: ([\d.]+)", stdout):
        metrics[avg_key_map.get(name, name.lower() + "_avg")] = float(acc)

    metrics["task_acc"] = task_acc
    return metrics


def normalized_ovo_choice(response: str | None) -> str:
    """Extract a single-choice answer from a model response for supplementary OVO metrics."""
    if not response:
        return ""
    text = str(response).strip()
    if not text:
        return ""
    leading = re.match(
        r"^\(?\s*([ABCD])(?:\s*\)|[.\s,:;-]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if leading:
        return leading.group(1).upper()
    answer_match = re.search(
        r"(?:answer|option|best\s+option|choice)\s*(?:is|:)?\s*\(?\s*([ABCD])\s*\)?",
        text,
        flags=re.IGNORECASE,
    )
    if answer_match:
        return answer_match.group(1).upper()
    standalone = re.search(r"\b([ABCD])\b", text, flags=re.IGNORECASE)
    if standalone:
        return standalone.group(1).upper()
    return ""


def compute_normalized_ovo_metrics(result_path: Path) -> dict:
    """Compute stricter answer-extraction metrics beside the official OVO scorer."""
    data = json.loads(result_path.read_text())
    metrics: dict[str, object] = {}
    split_task_scores: dict[str, dict[str, list[int]]] = {}

    for split in ["backward", "realtime"]:
        task_scores: dict[str, list[int]] = {}
        for row in data.get(split, []):
            task = row.get("task", "unknown")
            gt = str(row.get("ground_truth", "")).strip().upper()
            pred = normalized_ovo_choice(row.get("response"))
            task_scores.setdefault(task, []).append(int(pred == gt))
        split_task_scores[split] = task_scores
        if task_scores:
            split_avgs = [
                sum(scores) / len(scores) for scores in task_scores.values() if scores
            ]
            metrics[f"{split}_avg"] = 100.0 * sum(split_avgs) / len(split_avgs)

    task_acc = {}
    for task_scores in split_task_scores.values():
        for task, scores in task_scores.items():
            if scores:
                task_acc[task] = 100.0 * sum(scores) / len(scores)
    metrics["task_acc"] = task_acc

    present = [
        metrics[key] for key in ["backward_avg", "realtime_avg"] if key in metrics
    ]
    if present:
        metrics["total_avg"] = sum(present) / len(present)
    return metrics


def run_from_args(args) -> None:
    ovo_root = Path(args.ovo_root).resolve()
    output_dir = Path(args.output).resolve()
    raw_result_dir = output_dir / "raw_ovo_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_result_dir.mkdir(parents=True, exist_ok=True)

    constants = load_ovo_constants(ovo_root)
    ovo_args = build_ovo_runtime_args(args, ovo_root, raw_result_dir)
    annotations = load_ovo_annotations(
        ovo_args.anno_path, ovo_args.video_dir, args.tasks
    )
    annotations = limit_annotations(annotations, getattr(args, "max_samples", None))
    model = build_model_from_args(args)

    result_path = run_inference(
        model, annotations, args.tasks, args.mode, ovo_args, constants
    )
    validate_predictions(result_path)

    metrics = run_score(ovo_root, ovo_args, output_dir)
    metrics["normalized_metrics"] = compute_normalized_ovo_metrics(result_path)
    normalized = {
        "model": {
            "backend": args.backend,
            "model_path": args.model_path,
        },
        "benchmark": "ovo",
        "runner": "ovo",
        "dataset": {
            "ovo_root": str(ovo_root),
            "anno_path": ovo_args.anno_path,
            "video_dir": ovo_args.video_dir,
            "chunked_dir": ovo_args.chunked_dir,
            "tasks": args.tasks,
            "mode": args.mode,
        },
        "metrics": metrics,
        "artifacts": {
            "raw_results": str(raw_result_dir),
            "raw_result_file": str(result_path),
            "score_stdout": str(output_dir / "score_stdout.txt"),
            "score_stderr": str(output_dir / "score_stderr.txt"),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "normalized.json").write_text(json.dumps(normalized, indent=2))
