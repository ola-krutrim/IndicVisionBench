import os
import re
import json
import argparse
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import traceback
import time
import requests
import evaluate as hf_evaluate
from RIBES import kendall
from functools import partial
from multiprocessing import Pool, cpu_count
from metrics import *


# ---------------- Normalization Helpers ---------------- #

def pre_process(text):
    return str(text).replace("\n", " ").strip()

def normalize_mcq(val, lang):
    """Normalize MCQ answers across languages to A/B/C/D"""

    if not isinstance(val, str) or not val.strip():
        return ""
    cleaned = val.strip()
    if cleaned.startswith("<") and cleaned.endswith(">") and len(cleaned) > 2:
        cleaned = cleaned[1:-1].strip()
    cleaned = re.sub(r"^[\s\.\।]+|[\s\.\।]+$", "", cleaned)
    cleaned = cleaned.replace("\u200c", "").replace("\u200d", "").replace("\xa0", "")

    mapping = lang_abcd_map[lang]
    first_token = re.split(r"[ .।]", cleaned)[0]
    if first_token in mapping:
        return mapping[first_token]
    if first_token.upper() in ["A", "B", "C", "D"]:
        return first_token.upper()

    return val.strip()

def normalize_true_false(val, lang):
    """Normalize True/False answers"""

    if not isinstance(val, str) or not val.strip():
        return ""
    cleaned = re.sub(r'^[\s\.\।]+|[\s\.\।]+$', '', val)

    mapping = lang_true_false_map.get(lang, {})
    if cleaned in mapping:
        return mapping[cleaned]

    return cleaned


# ------------- Computing Helpers ------------------#
def compute_ocr_sample(item):
    try:
        ref = pre_process(item["reference_text"])
        hyp = pre_process(item["predicted_text"])

        word_score = anls_word(ref, hyp)
        char_score = anls_char(ref, hyp)

        item["results"] = {
            "anls_word": round(word_score, 4),
            "anls_char": round(char_score, 4),
        }

        return item

    except Exception:
        traceback.print_exc()
        return None

def compute_structured_sample(item):
    lang = item["language"]
    preds, refs = item["predictions"], item["references"]
    results = {}

    ref_mcq = refs["mcq_a"].strip()
    pred_mcq = preds["mcq_a"].strip()
    if ref_mcq and pred_mcq:
        ref_clean = normalize_mcq(ref_mcq, lang)
        pred_clean = normalize_mcq(pred_mcq, lang)
        results["mcq"] = 1.0 if ref_clean.upper() == pred_clean.upper() else 0.0
    else:
        results["mcq"] = None

    ref_tf = refs["true_false_a"].strip()
    pred_tf = preds["true_false_a"].strip()
    if ref_tf and pred_tf:
        ref_clean = normalize_true_false(ref_tf, lang)
        pred_clean = normalize_true_false(pred_tf, lang)
        results["true_false_q"] = 1.0 if ref_clean.upper() == pred_clean.upper() else 0.0
    else:
        results["true_false_q"] = None

    item["results"] = results
    return item

def process_entry_for_mmt(entry):
    """Compute BLEU + RIBES for one JSON entry."""
    try:
        ref = entry["reference_caption"]
        hyp = entry["predicted_caption"]
        entry["bleu"] = bleu_score(ref, hyp)
        entry["ribes"] = ribes_score(ref, hyp)
    except Exception as e:
        traceback.print_exc()
        entry["bleu"] = None
        entry["ribes"] = None
    return entry

def process_item_for_openended(item, api_key):
    preds, refs, questions = item["predictions"], item["references"], item["questions"]
    results = {}
    for qtype in [("short_q1", "short_a1"), ("short_q2", "short_a2"), ("long_q", "long_a"), 
                ("adversarial_question", "adversarial_answer")]:

        question = questions[qtype[0]]
        response = preds[qtype[1]].strip()
        gt = refs[qtype[1]]

        if not response or not gt:
            results[qtype[0]] = None
            continue

        prompt_type = "short" if "short" in qtype[0] else ("adversarial" if "adversarial" in qtype[0] else "long")
        prompt = build_prompt(question, gt, response, prompt_type)
        results[qtype[0]] = get_llm_score(prompt, api_key)
    item["results"] = results
    return item

# ---- Structured evaluation utilities -------
truth_dict = { 
    "Bengali": {"True": "ঠিক", "False": "ভুল"},
    "Gujarati": {"True": "સાચું", "False": "ખોટું"},
    "Hindi": {"True": "सही", "False": "गलत"},
    "Kannada": {"True": "ಸರಿ", "False": "ತಪ್ಪು"},
    "Malayalam": {"True": "ശരി", "False": "തെറ്റ്"},
    "Marathi": {"True": "बरोबर", "False": "चूक"},
    "Odia": {"True": "ଠିକ୍", "False": "ଭୁଲ୍"},
    "Punjabi": {"True": "ਸਹੀ", "False": "ਗਲਤ"},
    "Tamil": {"True": "சரி", "False": "தவறு"},
    "Telugu": {"True": "ఒప్పు", "False": "తప్పు"},
    "English": {"True": "True", "False": "False"}
}

lang_abcd_map = {
    "English": {"A": "A", "B": "B", "C": "C", "D": "D"},
    "Hindi": {"ए": "A", "अ": "A","बी": "B", "ब": "B","सी": "C", "स": "C","डी": "D", "द": "D"},
    "Bengali": {"এ": "A", "বি": "B", "সি": "C", "ডি": "D"},
    "Gujarati": {"એ": "A", "બી": "B", "સી": "C", "ડી": "D"},
    "Kannada": {"ಏ": "A", "ಬಿ": "B", "ಸಿ": "C", "ಡಿ": "D"},
    "Malayalam": {"എ": "A", "ബി": "B", "സി": "C", "ഡി": "D"},
    "Marathi": {"ए": "A", "अ": "A", "बी": "B", "ब": "B", "सी": "C", "डी": "D"},
    "Odia": {"ଏ": "A", "ଅ": "A", "ବି": "B", "ସି": "C", "ଡି": "D"},
    "Punjabi": {"ਏ": "A", "ਅ": "A", "ਬੀ": "B", "ਸੀ": "C", "ਡੀ": "D"},
    "Tamil": {"ஏ": "A", "அ": "A", "பி": "B", "சி": "C", "டி": "D"},
    "Telugu": {"ఏ": "A", "ఎ": "A", "బి": "B", "సి": "C", "డి": "D"},
}

lang_true_false_map = {
    "Malayalam": {"ശരിയാണ്": "ശരി"},
    "Marathi": {"चुकीचे": "चूक"},
    "Odia": {"ଭୁଲ": "ଭୁଲ୍"},
    "Tamil": {"பொய்": "தவறு"}
}


# --------------------------
# PROMPT BUILDER
# --------------------------

def build_prompt(question, ground_truth, predicted_answer, qtype="long"):
    if qtype == "long":
        return (
            f"You are a judge evaluating how well a Vision-Language Model answers long-answer type questions. "
            f"Evaluate the model's response based on Consistency, Fluency, and Relevance based on the Ground Truth answer. "
            f"Assign a high score when the model's response matches closely with the ground truth and a low score when the model's response lacks knowledge, or is not related to the ground truth.\n\n"
            f"Question: {question}\n"
            f"Ground Truth: {ground_truth}\n"
            f"Model response: {predicted_answer}\n\n"
            f"Provide a single overall score from 0 to 10 based on the given criteria. "
            f"Strictly return only the numeric score, without any additional commentary."
        )
    elif qtype == "short":
        return (
            f"You are a judge evaluating how well a Vision-Language Model answers short-answer type questions. "
            f"Evaluate the model's response based on accuracy and correctness based on the Ground Truth answer. "
            f"Assign a high score when the model's response matches closely with the ground truth and a low score when the model's response lacks knowledge, or is not related to the ground truth.\n\n"
            f"Question: {question}\n"
            f"Ground Truth: {ground_truth}\n"
            f"Model response: {predicted_answer}\n\n"
            f"Provide a single overall score from 0 to 10 based on the given criteria. "
            f"Strictly return only the numeric score, without any additional commentary."
        )
    elif qtype == "adversarial":
        return f"""You are a judge evaluating how well a Vision-Language Model (VLM) answers adversarial questions. 
In adversarial questions, the question contains a false assumption about the image.

You are given:
- Question: {question}
- Correct Answer (Gold reference): {ground_truth}
- VLM Answer: {predicted_answer}

Your task:
Evaluate the VLM’s answer according to this rubric:

1. **Identifying the false assumption (0 or 5 points):**
   - Award **5 points** if the VLM explicitly identifies that the question contains a wrong assumption 
     OR expresses uncertainty/difficulty in confirming the assumption.  
   - Award **0 points** if the VLM ignores the wrong assumption and answers as if the question were correct.  

2. **Identifying what the image is actually about (0–5 points):**
   - Award **0–5 points** depending on how well the VLM correctly identifies the real content of the image.  
   - 0 = completely wrong or no attempt.  
   - 1–2 = vague or partially correct.  
   - 3–4 = mostly correct but incomplete.  
   - 5 = fully correct identification.  

**Final Score = Assumption Score (0 or 5) + Identification Score (0–5) → 0 to 10.**

Instructions:
- Only output the final score as a number between 0 and 10.
- Do not explain reasoning or repeat answers.
- Always respond with the final score. Do not return a blank response.
- Be fair but consistent: partial credit is encouraged for partial identification.

Now, provide the score.
"""

# ------LLM judge helper function --------#

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

def get_llm_score(prompt, api_key, max_retries=3, retry_delay=2):
    """
    Query GPT-4o to get a numeric score (0–10).
    Retries if the response cannot be parsed as a float.
    """

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "max_tokens": 50
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(OPENAI_API_URL, headers=headers, json=body)
            response.raise_for_status()

            response_data = response.json()
            score_text = response_data["choices"][0]["message"]["content"].strip()

            score = float(score_text)

            # Clamp score between 0 and 10
            return min(max(score, 0), 10)

        except ValueError:
            print(f"⚠️ Attempt {attempt}: Could not parse numeric score from response: {score_text}")
            if attempt < max_retries:
                print("🔁 Retrying...")
            else:
                print("❌ Max retries reached. Returning None.")
                return None

        except Exception as e:
            print(f"❌ Error getting score from GPT-4o API (attempt {attempt}): {e}")
            if attempt < max_retries:
                print(f"🔁 Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("❌ Max retries reached. Returning None.")
                traceback.print_exc()
                return None