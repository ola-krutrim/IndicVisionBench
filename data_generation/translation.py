import pandas as pd
import requests
import time
import traceback
import multiprocessing as mp
from tenacity import retry, stop_after_attempt, wait_fixed
import os
from datetime import datetime
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_CSV_PATH = "./state_wise_corpus_base.csv"
OUTPUT_CSV_PATH = "./translated_state_wise_corpus.csv"
TEMP_OUTPUT_CSV_PATH = "./translated_state_wise_corpus_partial.csv"
LOG_FILE = "./translation_state_wise_corpus_errors.log"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = "YOUR_OPENROUTER_API_KEY"
MODEL_NAME = "google/gemini-2.5-flash"
FIELDS_TO_TRANSLATE = [
    "short_q1", "short_a1", "short_q2", "short_a2",
    "mc_q1", "mc_a1", "mc_opt1_1", "mc_opt1_2", "mc_opt1_3", "mc_opt1_4",
    "true_false_q", "true_false_a",
    "long_q", "long_a",
    "adversarial_question", "adversarial_answer"
]
SAVE_EVERY = 50  # Save every 50 rows

# --- SETUP ERROR LOGGING ---
log_lock = mp.Lock()

def log_error(image_path, error_msg):
    with log_lock:
        with open(LOG_FILE, "a") as f:
            timestamp = datetime.now().isoformat()
            f.write(f"[{timestamp}] image_path: {image_path}, error: {error_msg}\n")

# --- TRANSLATION FUNCTION ---
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def translate_text(text, target_lang):
    if pd.isna(text) or not str(text).strip():
        return text

    prompt = f"Translate the following text to {target_lang}. Just respond with the translated text and nothing else. The text is:\n\n{text}"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(API_URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content'].strip()

# --- PER-ROW PROCESSING FUNCTION ---
def process_row(row_dict):
    try:
        row = row_dict.copy()
        target_lang = row.get("primary_language", "")
        image_path = row.get("image_path", "unknown")

        for field in FIELDS_TO_TRANSLATE:
            original_text = row.get(field, "")
            try:
                translated = translate_text(original_text, target_lang)
                row[field] = translated
            except Exception as e:
                row[field] = original_text  # Keep original if failed
                log_error(image_path, f"{field} - {str(e)}")

        return row
    except Exception as e:
        log_error(row_dict.get("image_path", "unknown"), f"row-level error - {traceback.format_exc()}")
        return row_dict  # Return original row even if failed

# --- MAIN ---
def main():
    df = pd.read_csv(INPUT_CSV_PATH)
    rows = df.to_dict(orient="records")
    total_rows = len(rows)

    translated_rows = []

    with mp.Pool(processes=7) as pool:
        for i, processed_row in enumerate(tqdm(pool.imap_unordered(process_row, rows), total=total_rows, desc="Translating rows")):
            translated_rows.append(processed_row)

            # Periodic saving
            if i % SAVE_EVERY == 0:
                pd.DataFrame(translated_rows).to_csv(TEMP_OUTPUT_CSV_PATH, index=False)
                print(f"✅ Saved {i} rows to {TEMP_OUTPUT_CSV_PATH}")

    # Final save
    pd.DataFrame(translated_rows).to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"All done! Final output saved to {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()
