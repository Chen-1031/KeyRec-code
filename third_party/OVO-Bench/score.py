"""
    Calculate scores given inference results JSON
"""

import argparse
import os
import json
import os
from utils.OVOBenchScore import OVOBenchOfflineScore, OVOBenchOnlineScore

parser = argparse.ArgumentParser(description='Eval OVBench')
parser.add_argument("--result_dir", type=str, default="results", help="Root directory of results")
parser.add_argument("--model", type=str, required=True, help="Model to evaluate")
parser.add_argument("--model_path", type=str, required=False, default=None, help="Model path to evaluate a specific variant")
parser.add_argument("--mode", type=str, required=True, choices=["online", "offline"], help="Online of Offline model for testing")
parser.add_argument("--task", type=str, required=False, nargs="+",
                    choices=["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD", "REC", "SSR", "CRR"],
                    default=None,
                    help="Tasks to score")
parser.add_argument("--save_tag", type=str, required=False, default=None, help="Only score result files with this save tag")
args = parser.parse_args()

assert os.path.exists(os.path.join(args.result_dir, args.model))

results_paths = os.listdir(os.path.join(args.result_dir, args.model))

model_variant = args.model
if args.model_path:
    model_variant = args.model_path.split('/')[-1]

results = {
    "backward": [],
    "realtime": [],
    "forward": []
}
expected_result_name = None
if args.task is not None and args.save_tag is not None:
    expected_result_name = f"{model_variant}_{'_'.join(args.task)}_{args.mode}_{args.save_tag}.json"

for result_path in results_paths:
    if not result_path.startswith(model_variant):
        continue
    if expected_result_name is not None and result_path != expected_result_name:
        continue
    if expected_result_name is None and args.save_tag is not None and not result_path.endswith(f"_{args.save_tag}.json"):
        continue
    with open(os.path.join(args.result_dir, args.model, result_path), "r") as f:
        result = json.load(f)
        for split in ["backward", "realtime", "forward"]:
            split_results = result[split]
            if args.task is not None:
                split_results = [item for item in split_results if item["task"] in args.task]
            results[split] += split_results

if args.model in ["GPT", "Gemini", "InternVL2", "QWen2VL_7B", "QWen2VL_72B", "QWen2VL_7B_", 
"QWen2VL_72B_", "LongVU", "LLaVA_OneVision", "LLaVA_Video", "videollm_online", "FlashVStream", "MiniCPM_o", "Dispider", 
"Gemma4", "Qwen3VL", "Qwen3_5", "MiniCPMV_4_6", "DiffusionGemma", "KeyRecGemma"]:
    score_model = OVOBenchOfflineScore(args, results)
else:
    raise ValueError(f"Unsupported model: {args.model}. Please implement the model.")

score_model.score()
