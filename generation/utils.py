import os
import sys
import json
import argparse
import traceback
from tqdm import tqdm
from datasets import load_dataset

sys.path.append(".")

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

# =========================================================
# --------------- VQA PROMPT BUILDER -----------------
# =========================================================
def build_vqa_prompt(sample, qa_type):
    if qa_type in ["short_q1", "short_q2"]:
        return f"""{sample[qa_type].strip()} Please provide brief, clear responses in {sample['language']} language."""

    elif qa_type == "mcq":
        return f"""Strict Instruction: Respond with only one choice in the format <A>, <B>, <C>, or <D>
Do not include any explanation, reasoning, or extra text.
Question: {sample['mcq']}
Choices:
A. {sample['mcq_opt1']}
B. {sample['mcq_opt2']}
C. {sample['mcq_opt3']}
D. {sample['mcq_opt4']}"""

    elif qa_type == "true_false_q":
        return f"""Strict Instruction: Respond with only { (truth_dict[sample['language']])["True"] } or { (truth_dict[sample['language']])["False"] } 
Question: {sample['true_false_q'].strip()} 
Choices: { (truth_dict[sample['language']])["True"] } or { (truth_dict[sample['language']])["False"] }."""

    elif qa_type == "long_q":
        return f"""{sample['long_q'].strip()} Answer the question in detail in {sample['language']} language."""

    elif qa_type == "adversarial_question":
        return f"""{sample['adversarial_question'].strip()} Answer the question in detail in {sample['language']} language."""

    return ""