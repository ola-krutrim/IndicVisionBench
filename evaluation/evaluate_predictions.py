import os
import sys
sys.path.append(".")
import json
import time
import argparse
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
from collections import defaultdict
from datasets import load_dataset

from utils import (
    compute_ocr_sample,
    compute_structured_sample,
    process_entry_for_mmt,
    build_prompt,
    get_llm_score,
    process_item_for_openended
)

# ======================================================
# ================ OCR EVALUATION ======================
# ======================================================

def evaluate_ocr(predictions_path, indv_scores_path, scores_report_path, num_processes):
    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    print(f"Evaluating {len(predictions)} OCR samples...")

    evaluated = []
    with Pool(processes=num_processes) as pool:
        for res in tqdm(pool.imap_unordered(compute_ocr_sample, predictions), total=len(predictions)):
            if res is not None:
                evaluated.append(res)

    os.makedirs(os.path.dirname(indv_scores_path), exist_ok=True)
    json.dump(evaluated, open(indv_scores_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Saved OCR per-sample results to {indv_scores_path}")

    lang_scores = defaultdict(lambda: {"anls_word": [], "anls_char": []})
    all_word, all_char = [], []

    for item in evaluated:
        language = item["language"]
        scores = item["results"]
        if scores:
            lang_scores[language]["anls_word"].append(scores["anls_word"])
            lang_scores[language]["anls_char"].append(scores["anls_char"])
            all_word.append(scores["anls_word"])
            all_char.append(scores["anls_char"])

    summary = {
        "overall": {
            "avg_anls_word": round(sum(all_word) / len(all_word), 3) if all_word else None,
            "avg_anls_char": round(sum(all_char) / len(all_char), 3) if all_char else None,
            "total_data_points": len(evaluated),
        },
        "per_language": {},
    }

    for language, vals in sorted(lang_scores.items()):
        summary["per_language"][language] = {
            "count": len(vals["anls_word"]),
            "avg_anls_word": round(sum(vals["anls_word"]) / len(vals["anls_word"]), 3) if vals["anls_word"] else None,
            "avg_anls_char": round(sum(vals["anls_char"]) / len(vals["anls_char"]), 3) if vals["anls_char"] else None,
        }

    os.makedirs(os.path.dirname(scores_report_path), exist_ok=True)
    json.dump(summary, open(scores_report_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Saved OCR summary report to {scores_report_path}")
    return summary


# ======================================================
# ================ OPEN-ENDED EVALUATION ===============
# ======================================================

def evaluate_openended(api_key, predictions_path, indv_scores_path, scores_report_path, num_processes):
    with open(predictions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Evaluating {len(data)} open-ended samples with GPT-4o judge...")

    results = []
    worker_fn = partial(process_item_for_openended, api_key=api_key)
    with Pool(processes=num_processes) as pool:
        for r in tqdm(pool.imap_unordered(worker_fn, data), total=len(data)):
            results.append(r)

    os.makedirs(os.path.dirname(indv_scores_path), exist_ok=True)
    json.dump(results, open(indv_scores_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Saved open-ended QA per-sample scores to {indv_scores_path}")

    lang_scores = defaultdict(lambda: defaultdict(list))
    for item in results:
        language = item["language"]
        for qtype, score in item["results"].items():
            if score is not None:
                lang_scores[language][qtype].append(score)

    summary = {"overall": {}, "per_language": {}}
    all_scores = []
    for language, qtypes in lang_scores.items():
        summary["per_language"][language] = {}
        for qtype, scores in qtypes.items():
            avg = round(sum(scores) / len(scores), 3) if scores else None
            summary["per_language"][language][qtype] = {"count": len(scores), "avg_score": avg}
            all_scores.extend(scores)

    summary["overall"]["avg_score"] = round(sum(all_scores) / len(all_scores), 3) if all_scores else None
    summary["overall"]["total_valid_scores"] = len(all_scores)
    summary["overall"]["total_data_points"] = len(data)

    os.makedirs(os.path.dirname(scores_report_path), exist_ok=True)
    json.dump(summary, open(scores_report_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Saved open-ended QA summary report to {scores_report_path}")
    return summary


# ======================================================
# ================ STRUCTURED EVALUATION ===============
# ======================================================

def evaluate_structured(predictions_path, indv_scores_path, scores_report_path, num_processes):
    with open(predictions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    with Pool(processes=num_processes) as pool:
        for r in tqdm(pool.imap_unordered(compute_structured_sample, data), total=len(data)):
            results.append(r)

    os.makedirs(os.path.dirname(indv_scores_path), exist_ok=True)
    json.dump(results, open(indv_scores_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Saved structured QA per-sample results to {indv_scores_path}")

    lang_scores = defaultdict(lambda: defaultdict(list))
    for item in results:
        language = item["language"]
        for qtype, score in item["results"].items():
            if score is not None:
                lang_scores[language][qtype].append(score)

    summary = {"overall": {}, "per_language": {}}
    all_scores = []
    for language, qtypes in lang_scores.items():
        summary["per_language"][language] = {}
        for qtype, scores in qtypes.items():
            avg_acc = round(sum(scores) / len(scores), 3) if scores else None
            summary["per_language"][language][qtype] = {"count": len(scores), "accuracy": avg_acc}
            all_scores.extend(scores)

    summary["overall"]["avg_accuracy"] = round(sum(all_scores) / len(all_scores), 3) if all_scores else None
    summary["overall"]["total_valid_scores"] = len(all_scores)
    summary["overall"]["total_data_points"] = len(data)

    os.makedirs(os.path.dirname(scores_report_path), exist_ok=True)
    json.dump(summary, open(scores_report_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Saved structured QA summary report to {scores_report_path}")
    return summary


# ======================================================
# ================ MMT (Bleu + Ribes) EVALUATION ===============
# ======================================================

def evaluate_mmt(predictions_path, indv_scores_path, scores_report_path, num_processes=20):
    with open(predictions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries for MMT evaluation...")

    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(process_entry_for_mmt, data), total=len(data)))

    os.makedirs(os.path.dirname(indv_scores_path), exist_ok=True)
    with open(indv_scores_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved MMT per-sample BLEU+RIBES scores to {indv_scores_path}")

    report = defaultdict(lambda: {"avg_bleu": 0.0, "avg_ribes": 0.0, "count": 0})
    for entry in results:
        language = entry["target_language"]
        if entry["bleu"] is not None and entry["ribes"] is not None:
            report[language]["avg_bleu"] += entry["bleu"]
            report[language]["avg_ribes"] += entry["ribes"]
            report[language]["count"] += 1

    for language, vals in report.items():
        if vals["count"] > 0:
            vals["avg_bleu"] /= vals["count"]
            vals["avg_ribes"] /= vals["count"]

    os.makedirs(os.path.dirname(scores_report_path), exist_ok=True)
    with open(scores_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved MMT summary report to {scores_report_path}")
    return report


# ======================================================
# ================ MAIN ENTRY ==========================
# ======================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified evaluation for IndicVisionBench")
    parser.add_argument("--task_type", required=True, choices=["ocr", "vqa_openended", "vqa_structured", "mmt"], help="Type of evaluation task.")
    parser.add_argument("--predictions_path", required=True, help="Path to predictions JSON")
    parser.add_argument("--indv_scores_path", required=True, help="Path to save individual scores JSON")
    parser.add_argument("--scores_report_path", required=True, help="Path to save scores' report JSON")
    parser.add_argument("--num_processes", type=int, default=cpu_count(), help="Number of CPU cores")
    parser.add_argument("--api_key", type=str, default=None, help="API key required for LLM-based evaluation")

    args = parser.parse_args()

    if args.task_type == "vqa_openended" and not args.api_key:
        parser.error("--api_key is required for vqa_openended evaluation.")

    if args.task_type == "ocr":
        evaluate_ocr(args.predictions_path, args.indv_scores_path, args.scores_report_path, args.num_processes)
    elif args.task_type == "vqa_openended":
        evaluate_openended(args.api_key, args.predictions_path, args.indv_scores_path, args.scores_report_path, args.num_processes)
    elif args.task_type == "vqa_structured":
        evaluate_structured(args.predictions_path, args.indv_scores_path, args.scores_report_path, args.num_processes)
    elif args.task_type == "mmt":
        evaluate_mmt(args.predictions_path, args.indv_scores_path, args.scores_report_path, args.num_processes)
