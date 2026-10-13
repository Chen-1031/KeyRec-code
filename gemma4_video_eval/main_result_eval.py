"""Deterministic execution plan for the default evaluation matrix."""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from gemma4_video_eval.main_result_config import MainResultConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/main_results.yaml"
DEFAULT_OUTPUT_ROOT = Path("results/reproduced/runs")
DEFAULT_DATA_ROOTS = {
    "ovo_bench": Path("data/OVO-Bench"),
    "streamingbench": Path("data/StreamingBench"),
    "longvideobench": Path("data/LongVideoBench"),
    "video_mme_v2": Path("data/Video-MME-v2"),
}
OVO_MAIN_TASKS = ("EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD")


@dataclass(frozen=True)
class MainResultJob:
    """One executable cell in the 3 x 2 x 4 evaluation matrix."""

    job_id: str
    model: str
    model_family: str
    checkpoint: str
    method: str
    benchmark: str
    profile: str
    environment_profile: str
    output_dir: str
    command: tuple[str, ...]

    def as_dict(self) -> dict:
        value = asdict(self)
        value["command"] = list(self.command)
        return value

    def shell_command(self) -> str:
        return shlex.join(self.command)


def backend_name(family: str, method: str) -> str:
    return {
        ("gemma", "vanilla"): "gemma",
        ("gemma", "keyrec"): "keyrec",
        ("neo", "vanilla"): "neo_ov",
        ("neo", "keyrec"): "keyrec_neo",
    }[(family, method)]


def _common_command(
    *,
    python: str,
    module: str,
    backend: str,
    checkpoint: str,
    output_dir: Path,
    frames: int,
    family: str,
    method: str,
) -> list[str]:
    command = [
        python,
        "-m",
        module,
        "--backend",
        backend,
        "--model-path",
        checkpoint,
        "--output",
        str(output_dir),
        "--num-frames",
        str(frames),
        "--sampling",
        "uniform" if family == "neo" else "linspace",
    ]
    if family == "gemma":
        command.extend(["--max-soft-tokens", "70"])
    if method == "keyrec":
        command.extend(["--read-policy", "llm_router"])
    return command


def build_main_result_jobs(
    config: MainResultConfig,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    data_roots: dict[str, Path] | None = None,
    model_paths: dict[str, Path] | None = None,
    gemma_python: str = ".venv/bin/python",
    neo_python: str = ".venv_neo/bin/python",
) -> tuple[MainResultJob, ...]:
    """Build all 24 jobs in stable model/method/benchmark order."""
    config.validate()
    roots = dict(DEFAULT_DATA_ROOTS)
    if data_roots:
        unknown = set(data_roots) - set(roots)
        if unknown:
            raise ValueError(f"unknown benchmark data roots: {sorted(unknown)}")
        roots.update({name: Path(path) for name, path in data_roots.items()})

    execution_model_paths: dict[str, Path] = {}
    if model_paths:
        unknown = set(model_paths) - set(config.models)
        if unknown:
            raise ValueError(f"unknown model paths: {sorted(unknown)}")
        execution_model_paths.update(
            {name: Path(path) for name, path in model_paths.items()}
        )

    jobs = []
    for model_name, model_value in config.models.items():
        family = str(model_value["family"])
        checkpoint = str(model_value["checkpoint"])
        python = gemma_python if family == "gemma" else neo_python
        for method in config.methods:
            backend = backend_name(family, method)
            for benchmark_name, benchmark_value in config.benchmarks.items():
                regime = str(benchmark_value["regime"])
                profile = config.profile(family, regime)
                job_id = f"{model_name}__{method}__{benchmark_name}"
                output_dir = output_root / job_id

                module = {
                    "ovo_bench": "gemma4_video_eval.ovo_eval",
                    "streamingbench": "gemma4_video_eval.streamingbench_eval",
                    "longvideobench": "gemma4_video_eval.longvideobench_eval",
                    "video_mme_v2": "gemma4_video_eval.videommev2_eval",
                }[benchmark_name]
                command = _common_command(
                    python=python,
                    module=module,
                    backend=backend,
                    checkpoint=str(execution_model_paths.get(model_name, checkpoint)),
                    output_dir=output_dir,
                    frames=profile.sampled_frames,
                    family=family,
                    method=method,
                )
                if benchmark_name == "ovo_bench":
                    command.extend(["--ovo-root", str(roots[benchmark_name])])
                    command.extend(["--tasks", *OVO_MAIN_TASKS])
                    command.extend(["--mode", "offline"])
                elif benchmark_name == "streamingbench":
                    command.extend(
                        ["--streamingbench-root", str(roots[benchmark_name])]
                    )
                elif benchmark_name == "longvideobench":
                    command.extend(
                        ["--longvideobench-root", str(roots[benchmark_name])]
                    )
                elif benchmark_name == "video_mme_v2":
                    command.extend(["--videommev2-root", str(roots[benchmark_name])])

                jobs.append(
                    MainResultJob(
                        job_id=job_id,
                        model=model_name,
                        model_family=family,
                        checkpoint=checkpoint,
                        method=method,
                        benchmark=benchmark_name,
                        profile=profile.name,
                        environment_profile=family,
                        output_dir=str(output_dir),
                        command=tuple(command),
                    )
                )
    return tuple(jobs)


def _parse_data_roots(values: Iterable[str]) -> dict[str, Path]:
    roots = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--data-root must use BENCHMARK=PATH")
        benchmark, path = value.split("=", 1)
        if not benchmark or not path:
            raise ValueError("--data-root must use non-empty BENCHMARK=PATH")
        if benchmark in roots:
            raise ValueError(f"duplicate --data-root for {benchmark!r}")
        roots[benchmark] = Path(path)
    return roots


def _parse_model_paths(values: Iterable[str]) -> dict[str, Path]:
    paths = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--model-path must use MODEL=PATH")
        model, path = value.split("=", 1)
        if not model or not path:
            raise ValueError("--model-path must use non-empty MODEL=PATH")
        if model in paths:
            raise ValueError(f"duplicate --model-path for {model!r}")
        paths[model] = Path(path)
    return paths


def _select_jobs(
    jobs: tuple[MainResultJob, ...], job_ids: list[str]
) -> list[MainResultJob]:
    if not job_ids:
        return list(jobs)
    by_id = {job.job_id: job for job in jobs}
    unknown = sorted(set(job_ids) - set(by_id))
    if unknown:
        raise ValueError(f"unknown job ids: {unknown}")
    return [by_id[job_id] for job_id in job_ids]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gemma-python", default=".venv/bin/python")
    parser.add_argument("--neo-python", default=".venv_neo/bin/python")
    parser.add_argument(
        "--data-root", action="append", default=[], metavar="BENCHMARK=PATH"
    )
    parser.add_argument(
        "--model-path", action="append", default=[], metavar="MODEL=PATH"
    )
    parser.add_argument("--job-id", action="append", default=[])
    parser.add_argument("--format", choices=["json", "shell"], default="json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config = MainResultConfig.from_yaml(args.config)
        jobs = build_main_result_jobs(
            config,
            output_root=args.output_root,
            data_roots=_parse_data_roots(args.data_root),
            model_paths=_parse_model_paths(args.model_path),
            gemma_python=args.gemma_python,
            neo_python=args.neo_python,
        )
        selected = _select_jobs(jobs, args.job_id)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.format == "shell":
        for job in selected:
            print(f"# {job.job_id} [{job.environment_profile}]")
            print(job.shell_command())
    else:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "config": str(args.config),
                    "job_count": len(selected),
                    "jobs": [job.as_dict() for job in selected],
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
