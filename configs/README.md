# Default evaluation profiles

`main_results.yaml` defines the supported backbones, four benchmarks, and four
architecture/regime profiles. The command planner covers dense reference and
KeyRec adapters. Load the file through
`gemma4_video_eval.main_result_config.MainResultConfig`.

The loader validates memory budgets and router allocations. See
[method details](../docs/method.md) for their definitions.
