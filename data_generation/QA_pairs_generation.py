import os
import base64
import json
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool

# =========================
# ==== Global Settings ====
# =========================

API_KEY = "YOUR_OPENROUTER_API_KEY"
MODEL = "google/gemini-2.5-flash"
MAX_RETRIES = 2
NUM_PROCESSES = 10


# =========================
# ==== Image Encoding =====
# =========================

def encode_image_to_base64(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        if width > 1080 or height > 1080:
            img.thumbnail((1080, 1080), Image.Resampling.LANCZOS)

        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


# =========================
# ==== Prompt Builders ====
# =========================

def build_standard_prompt(image_filename, caption, category):
    return f"""Here is an India-specific image and the image filename, caption and the category of the image I have on hand.
The image filename is this: {image_filename}
The caption is this: {caption}
The category is this: {category}
I'd like you to generate two short questions and answers, one multiple choice question and answer, one true/false question, and one long question and answer. Refer to the image filename, category and the caption for the context/hint. Take into account the cultural diversity of the category that this image falls under with respect to India.
Follow the following rules while designing questions and answers:
1. The question must be answerable only by looking at the image.
2. Ensure that the questions are culturally relevant to India and specific to the image.
3. Make the questions in such a way that someone who is not well aware of the Indian culture will find it difficult to answer the questions.
3. Provide answers that are concise, accurate, and directly related to the question.
4. You will also need to provide 1 correct option and 3 other incorrect options (distractors).
For the distractors, choose options that are relevant, not obvious wrong answers.
5. The question must be answerable even without the multiple-choice.
Example of the invalid question: (“What song is not performed by this musician” – not answerable if you don’t know the choices).
6. Make sure the questions are written fluently in English.
7. Be mindful of cultural sensitivities and avoid stereotyping or misrepresenting cultural aspects.
8. Ensure there are variations in your questions. Identity questions are fine, eg “What is this”, or “where is this”. But additionally, incorporate questions with multi-hop reasoning, referencing, ones that require local commonsense knowledge etc.
9. Just generate these in English.
10. Keep the answers short, around 1-2 sentences.
11. Make the questions distinct and unique from each other.
Give the answers in the following JSON format and make sure to only output a valid JSON,
{{ "short_questions": [ {{ "question": <question>, "answer": <answer> }}],
"multiple_choice_questions": [ {{ "question": <question>, "answer": <answer>, “options” <4 options> }}], "true_false_question": {{ "question": <question>, "answer": <answer> }},
"long_question": {{ "question": <question>, "answer": <answer> }}
"""


def build_adversarial_prompt(caption, category):
    return f"""
You are given an image from India along with its caption and the category it belongs to.

Your task is to create an **adversarial question** for the image — one that makes a **confident but subtly incorrect cultural assumption** about what is shown, especially with respect to **India’s diverse regional traditions, foods, festivals, clothing, rituals, art forms etc.**.

Here is the image's caption: {caption}
The category it belongs to is: {category}

Generate:
**Adversarial Question:** A misleading or culturally incorrect question that confidently makes a **specific, wrong assumption** which is **plausibly close to the truth** (e.g., mixing up similar Indian art styles, dishes, festivals, or traditions).  
The question must **not** reveal that it is making an assumption — just ask the question normally, as if the incorrect assumption is true.

⚠️ Avoid yes/no or speculative questions.
✅ Examples:
- *How was this Bikaneri folk painting prepared on canvas?* (image shows Kumaoni Aipan)
- *What are the main ingredients in this chicken tandoori dish?* (image shows paneer tikka)
- *Which Sikh gurdwara is being shown here?* (image shows a Hindu temple)
- *How is this Eid offering typically presented in Tamil Nadu?* (image shows a Pongal celebration)

❌ Avoid:
- *Is this a Bikaneri art piece?*
- *Are these Modaks made of coconut?*
- *Considering this is a South Indian Onam celebration...*

The adversarial question should be close enough to the actual content that it forces a model to **distinguish finely** between culturally similar options, and reject the incorrect assumption.
You also need to generate the answer for the question that you generate.
Do NOT include any introductions or explanations. 
Output ONLY the following two fields, in **exactly** this format:

Adversarial Question: <your question here>
Answer: <your answer here>
"""


# =========================
# ==== API Caller =========
# =========================

def call_api(prompt, image_data_url):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://openrouter.ai"
    }

    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ]
    }

    return requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data
    )


# =========================
# ==== Parsing Helpers ====
# =========================

def clean_json_block(text):
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def parse_standard_response(content):
    try:
        return json.loads(clean_json_block(content))
    except:
        return None


def parse_adversarial_response(content):
    question = ""
    answer = ""

    for line in content.strip().split("\n"):
        if line.startswith("Adversarial Question:"):
            question = line.replace("Adversarial Question:", "").strip()
        elif line.startswith("Answer:"):
            answer = line.replace("Answer:", "").strip()

    if question and answer:
        return question, answer
    return None, None


# =========================
# ==== Core Processing ====
# =========================

def process_row(row):
    image_path = row["image_path"]
    caption = row["caption"]
    image_filename = os.path.basename(image_path)
    category = os.path.basename(os.path.dirname(image_path))

    result = {
        **row,
        "short_q1": "", "short_a1": "",
        "short_q2": "", "short_a2": "",
        "mc_q1": "", "mc_a1": "",
        "mc_opt1_1": "", "mc_opt1_2": "",
        "mc_opt1_3": "", "mc_opt1_4": "",
        "true_false_q": "", "true_false_a": "",
        "long_q": "", "long_a": "",
        "adversarial_question": "",
        "adversarial_answer": "",
        "status": ""
    }

    encoded = encode_image_to_base64(image_path)

    mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
    image_data_url = f"data:{mime_type};base64,{encoded}"

    # ======================
    # Standard QA Generation
    # ======================

    standard_prompt = build_standard_prompt(image_filename, caption, category)

    for _ in range(MAX_RETRIES + 1):
        response = call_api(standard_prompt, image_data_url)
        if response.status_code != 200:
            continue

        content = response.json()["choices"][0]["message"]["content"]
        parsed = parse_standard_response(content)
        if parsed:
            break
    else:
        result["status"] = "Standard QA failed"
        return result

    # Extract standard QAs
    sq = parsed.get("short_questions", [])
    if len(sq) > 0:
        result["short_q1"] = sq[0].get("question", "")
        result["short_a1"] = sq[0].get("answer", "")
    if len(sq) > 1:
        result["short_q2"] = sq[1].get("question", "")
        result["short_a2"] = sq[1].get("answer", "")

    mcqs = parsed.get("multiple_choice_questions", [])
    if len(mcqs) > 0:
        opts = mcqs[0].get("options", ["", "", "", ""])
        if isinstance(opts, dict):
            opts = list(opts.values())
        opts = (opts + [""] * 4)[:4]

        result["mc_q1"] = mcqs[0].get("question", "")
        result["mc_a1"] = mcqs[0].get("answer", "")
        result["mc_opt1_1"], result["mc_opt1_2"], result["mc_opt1_3"], result["mc_opt1_4"] = opts

    tf = parsed.get("true_false_question", {})
    result["true_false_q"] = tf.get("question", "")
    result["true_false_a"] = tf.get("answer", "")

    longq = parsed.get("long_question", {})
    result["long_q"] = longq.get("question", "")
    result["long_a"] = longq.get("answer", "")

    # ==========================
    # Adversarial QA Generation
    # ==========================

    adversarial_prompt = build_adversarial_prompt(caption, category)

    for _ in range(MAX_RETRIES + 1):
        response = call_api(adversarial_prompt, image_data_url)
        if response.status_code != 200:
            continue

        content = response.json()["choices"][0]["message"]["content"]
        q, a = parse_adversarial_response(content)
        if q and a:
            result["adversarial_question"] = q
            result["adversarial_answer"] = a
            result["status"] = "success"
            return result

    result["status"] = "Adversarial QA failed"
    return result


# =========================
# ==== CSV Processing =====
# =========================

def process_csv(input_csv, output_csv, failed_csv):
    df = pd.read_csv(input_csv)
    rows = df.to_dict(orient="records")

    results = []
    with Pool(NUM_PROCESSES) as pool:
        for r in tqdm(pool.imap_unordered(process_row, rows), total=len(rows)):
            results.append(r)

    full_df = pd.DataFrame(results)

    df_success = full_df[full_df["status"] == "success"]
    df_success.to_csv(output_csv, index=False)
    df_failed = full_df[full_df["status"] != "success"]
    df_failed.to_csv(failed_csv, index=False)

    print(f"Successful saved to: {output_csv}")
    print(f"Failed saved to: {failed_csv}")


# =========================
# ==== Entry Point ========
# =========================

if __name__ == "__main__":
    states = os.listdir("./state_wise_images")   
    for state_name in states:
        print(f"Processing {state_name}")

        input_csv = f"./state_wise_captions_gemini/{state_name}_captions.csv"
        output_csv = f"./state_wise_full_QAs/{state_name}.csv"
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        failed_csv = f"./state_wise_full_QAs/{state_name}_failed.csv"

        process_csv(input_csv, output_csv, failed_csv)
