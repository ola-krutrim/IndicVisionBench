import os
import re
import json
import argparse
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import evaluate as hf_evaluate
from RIBES import kendall
from functools import partial
from multiprocessing import Pool, cpu_count

# === OCR METRIC FUNCTIONS ===
def levenshtein_distance_ocr_bench(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def anls_word(ref: str, hyp: str) -> float:
    ref_words = ref.split()
    pred_words = hyp.split()
    anls_word_dist = levenshtein_distance_ocr_bench(ref_words, pred_words)
    anls_word_value = anls_word_dist * 100 / max(len(ref_words), len(pred_words)) if max(len(ref_words), len(pred_words)) > 0 else float('inf')
    return anls_word_value


def anls_char(ref: str, hyp: str) -> float:
    anls_char_dist = levenshtein_distance_ocr_bench(ref, hyp)
    anls_char_value = anls_char_dist * 100 / max(len(ref), len(hyp)) if max(len(ref), len(hyp)) > 0 else float('inf')
    return anls_char_value

# ----------Bleu metric functions-----------

# Initialize BLEU evaluator globally (lazy load)
bleu_metric = hf_evaluate.load("sacrebleu")

def bleu_score(ref, hyp):
    """Compute sacreBLEU score between reference and hypothesis."""
    try:
        results = bleu_metric.compute(predictions=[str(hyp)], references=[[str(ref)]])
        return results["score"]
    except Exception as e:
        print(f"[BLEU ERROR] {e}")
        return None


def ribes_score(ref, hyp):
    """Compute RIBES score using kendall rank correlation."""
    ref_tokens = str(ref).lower().split()
    hyp_tokens = str(hyp).lower().split()
    try:
        nkt, precision, bp = kendall(ref_tokens, hyp_tokens)
        return nkt * (precision ** 0.25) * (bp ** 0.10)
    except Exception as e:
        print(f"[RIBES ERROR] {e}")
        return None
