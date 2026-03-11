import os
import sys
import json
import argparse
import traceback
from tqdm import tqdm
from datasets import load_dataset

sys.path.append(".")
from generation.model_wrappers import run_model
from utils import truth_dict, build_vqa_prompt  # Used for VQA true/false mapping

# Optional: Hugging Face login for gated datasets
try:
    from huggingface_hub import login
    login(token="YOUR_HF_TOKEN")
except Exception:
    print("⚠️ Skipping Hugging Face login (token not available or invalid)")


# =========================================================
# --------------- OCR GENERATION -----------------
# =========================================================
def generate_ocr_predictions(model_name, output_path, num_samples=None, api_key=None):
    OCR_PROMPT = "Extract the exact text from this image using OCR. Respond with only the text."

    print("Loading OCR dataset: krutrim-ai-labs/IndicVisionBench - ocr")
    ds = load_dataset("krutrim-ai-labs/IndicVisionBench", "ocr")["test"]

    if num_samples:
        ds = ds.shuffle().select(range(min(num_samples, len(ds))))
        print(f"Using a random subset of {num_samples} samples.")

    results = []
    for sample in tqdm(ds, total=len(ds), desc=f"Running OCR with {model_name}"):
        image = sample["image"]
        id = sample["id"]
        language = sample["language"]
        try:
            pred = run_model(model_name, prompt=OCR_PROMPT, image=image, api_key=api_key)
        except Exception:
            pred = f"[ERROR: {traceback.format_exc()}]"
        results.append({
            "model": model_name,
            "id": id,
            "language": language,
            "page_url": sample["page_url"],
            "reference_text": sample["text"],
            "predicted_text": pred
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved OCR predictions to {output_path}")
    print(f"Total samples processed: {len(results)}")


# =========================================================
# --------------- VQA GENERATION -----------------
# =========================================================
def generate_vqa_predictions(model_name, output_path, language=None, split="vqa_indic", num_samples=None, api_key=None):

    print(f"Loading VQA dataset: krutrim-ai-labs/IndicVisionBench - {split}")

    ds = load_dataset("krutrim-ai-labs/IndicVisionBench", split)["test"]

    if language:
        ds = ds.filter(lambda x: x["language"].lower() == language.lower())
        print(f"Running for language: {language} ({len(ds)} samples)")
    else:
        print(f"Running for all languages ({len(ds)} samples)")

    if num_samples:
        ds = ds.shuffle().select(range(min(num_samples, len(ds))))
        print(f"Running on random subset of {num_samples} samples")

    results = []

    for sample in tqdm(ds, total=len(ds), desc=f"Generating VQA predictions with {model_name}"):
        image = sample["image"]
        id = sample["id"]
        language = sample["language"]
        state_ut = sample["State/UT"]
        topic = sample["topic"]
        source_url = sample["source_url"]

        qa_types = [
            ("short_q1", "short_a1"), ("short_q2", "short_a2"), ("mcq", "mcq_a"),
            ("true_false_q", "true_false_a"), ("long_q", "long_a"), ("adversarial_question", "adversarial_answer")
        ]

        preds = {}
        questions = {}
        for qa_type in qa_types:
            question = sample[qa_type[0]]
            if qa_type[0] == "mcq":
                questions[qa_type[0]] = f"{question}: {sample['mcq_opt1']} | {sample['mcq_opt2']} | {sample['mcq_opt3']} | {sample['mcq_opt4']}"
            else:
                questions[qa_type[0]] = question

            if not question:
                preds[qa_type[1]] = ""
                continue

            prompt = build_vqa_prompt(sample, qa_type[0])
            try:
                preds[qa_type[1]] = run_model(model_name, image=image, prompt=prompt, api_key=api_key)
            except Exception:
                preds[qa_type[1]] = f"[ERROR: {traceback.format_exc()}]"

        result = {
            "model": model_name,
            "id": id,
            "language": language,
            "State/UT": state_ut,
            "topic": topic,
            "source_url": source_url,
            "questions": questions,
            "predictions": preds,
            "references": {
                "short_a1": sample["short_a1"],
                "short_a2": sample["short_a2"],
                "mcq_a": sample["mcq_a"],
                "true_false_a": sample["true_false_a"],
                "long_a": sample["long_a"],
                "adversarial_answer": sample["adversarial_answer"]
            }
        }

        results.append(result)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved VQA predictions to {output_path}")
    print(f"Total samples processed: {len(results)}")


# =========================================================
# --------------- MMT GENERATION -----------------
# =========================================================
MMT_PROMPT_TEMPLATE = """{caption_en}

Translate the above caption for the given image to {target_language} language. 
Just respond with the exact translation. Do not provide any explanation or any other text from your side.
"""

def generate_mmt_predictions(model_name, output_path, target_languages=None, num_samples=None, api_key=None):
    print("Loading MMT dataset: krutrim-ai-labs/IndicVisionBench - mmt")
    ds = load_dataset("krutrim-ai-labs/IndicVisionBench", "mmt")["test"]

    if num_samples:
        ds = ds.shuffle().select(range(min(num_samples, len(ds))))
        print(f"Using a random subset of {num_samples} samples.")

    if not target_languages:
        target_languages = [
            "Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam",
            "Marathi", "Odia", "Punjabi", "Tamil", "Telugu"
        ]

    results = []

    for target_language in target_languages:
        print(f"\nGenerating predictions for {target_language}...")
        for sample in tqdm(ds, total=len(ds)):
            image = sample["image"]
            id = sample["id"]
            caption_en = sample["English"]

            prompt = MMT_PROMPT_TEMPLATE.format(caption_en=caption_en, target_language=target_language)

            try:
                pred = run_model(model_name, prompt=prompt, image=image, api_key=api_key)
            except Exception:
                pred = f"[ERROR: {traceback.format_exc()}]"

            results.append({
                "model": model_name,
                "id": id,
                "target_language": target_language,
                "source_url": sample["source_url"],
                "english_caption": caption_en,
                "reference_caption": sample[target_language],
                "predicted_caption": pred
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved MMT predictions to {output_path}")
    print(f"Total samples processed: {len(results)}")


# =========================================================
# --------------- MAIN ENTRY POINT -----------------
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified script for generating OCR, VQA, or MMT predictions.")
    parser.add_argument("--task", required=True, choices=["ocr", "vqa", "mmt"], help="Task type: 'ocr', 'vqa', or 'mmt'")
    parser.add_argument("--model", required=True, help="Model name to use for generation")
    parser.add_argument("--output_path", required=True, help="Output JSON file path")
    parser.add_argument("--api_key", type=str, default=None, help="API key for remote model inference (optional)")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit the number of samples (optional)")

    # VQA-specific
    parser.add_argument("--language", type=str, default=None, help="Language for VQA (optional)")
    parser.add_argument("--split", default="indic", help="Dataset split for VQA (parallel, en, indic)")

    # MMT-specific
    parser.add_argument(
        "--target_languages",
        nargs="+",
        default=None,
        help="List of target languages, e.g. --target_languages Hindi Tamil Bengali"
    )

    args = parser.parse_args()

    if args.task.lower() == "ocr":
        generate_ocr_predictions(
            model_name=args.model,
            output_path=args.output_path,
            num_samples=args.num_samples,
            api_key=args.api_key
        )

    elif args.task.lower() == "vqa":
        generate_vqa_predictions(
            model_name=args.model,
            output_path=args.output_path,
            language=args.language,
            split=args.split,
            num_samples=args.num_samples,
            api_key=args.api_key
        )

    elif args.task.lower() == "mmt":
        generate_mmt_predictions(
            model_name=args.model,
            output_path=args.output_path,
            target_languages=args.target_languages,
            num_samples=args.num_samples,
            api_key=args.api_key
        )
