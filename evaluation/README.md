# Evaluation

Run commands from the source checkout with the environment appropriate for the
selected model. Model checkpoints and videos are downloaded separately.

## Dataset layouts

| Benchmark | Required files |
|---|---|
| OVO-Bench | `data/ovo_bench_new.json`, `data/chunked_videos/*.mp4`, and scorer files |
| StreamingBench | `src/data/questions_real.json`; annotation video paths must resolve |
| LongVideoBench | `lvb_val.json` and `videos/<annotation video_path>` |
| Video-MME-v2 | `Video-MME-v2.tsv` and `video/<video id>.mp4` |

For OVO-Bench, copy `constant.py`, `score.py`, and `utils/` from
`third_party/OVO-Bench/` into the dataset root next to `data/`.

Dataset sources:
[OVO-Bench](https://huggingface.co/datasets/JoeLeelyf/OVO-Bench),
[StreamingBench](https://huggingface.co/datasets/mjuicem/StreamingBench),
[LongVideoBench](https://huggingface.co/datasets/longvideobench/LongVideoBench), and
[Video-MME-v2](https://huggingface.co/datasets/MME-Benchmarks/Video-MME-v2).
The Video-MME-v2 adapter in the pinned VLMEvalKit checkout can prepare its
TSV/video layout from the public snapshot.

## Entry points

```bash
python -m gemma4_video_eval.ovo_eval --help
python -m gemma4_video_eval.streamingbench_eval --help
python -m gemma4_video_eval.longvideobench_eval --help
python -m gemma4_video_eval.videommev2_eval --help
```

Use `--backend keyrec` for Gemma E2B/E4B, or `--backend keyrec_neo` for NEO-ov.
Dense reference backends are `gemma` and `neo_ov`.
`--read-policy llm_router` is the default. Fixed policies are `all_recent`,
`heavy_recent`, `balanced`, `heavy_key_event`, and `all_key_event`.

Streaming profiles use 32 sampled frames and long-video profiles use 64.
Use `--sampling linspace` for Gemma and `--sampling uniform` for NEO-ov.
OVO-Bench defaults to its nine Backward/Realtime tasks.

For example:

```bash
python -m gemma4_video_eval.longvideobench_eval \
  --backend keyrec \
  --model-path /path/to/gemma-checkpoint \
  --longvideobench-root /path/to/LongVideoBench \
  --output /tmp/keyrec-longvideobench \
  --num-frames 64 \
  --sampling linspace \
  --max-samples 1
```

Remove `--max-samples 1` for a full run. Use a separate output directory for each
model/dataset setting. StreamingBench, LongVideoBench, and Video-MME-v2 write
`predictions.json`, `metrics.json`, and `normalized.json`. OVO-Bench preserves
raw results, scorer output, and normalized metrics. Scores from a restricted
sample count do not describe the full dataset.

## Command planning

```bash
python -m gemma4_video_eval.main_result_eval --format json
```

The planner prints 24 commands across three models, two methods, and four
benchmarks. It does not launch model inference. Use `--model-path MODEL=PATH`,
`--data-root BENCHMARK=PATH`, and `--job-id` to select local paths and jobs.

