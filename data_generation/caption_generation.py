import pandas as pd
import os
from PIL import Image
from io import BytesIO
import base64
import requests
from tqdm import tqdm
import multiprocessing as mp

API_KEY = "YOUR_API_KEY"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://yourappname.com",  # replace with your app/site
    "X-Title": "indicvisionbench-caption-gen"
}

def encode_image(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")  # Ensure consistent mode
        width, height = img.size

        # Resize only if either dimension is greater than 1080
        if width > 1080 or height > 1080:
            img.thumbnail((1080, 1080), Image.Resampling.LANCZOS)

        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")

def build_image_path(row):
    filename = row['image_filename']
    category = row['category']

    return f"./state_wise_images/{state_name}/{category}/{filename}"

def process_row(args):
    row, state_name = args
    image_path = row['image_path']
    current_caption = row['caption']
    category = row['category']
    source_url = row['source_url']

    image_b64 = encode_image(image_path)

    prompt_text = f"""
Generate a high-quality, detailed caption specifically tailored for Vision-Language Model (VLM) training.

The provided image belongs to the context of an Indian State/Union Territory, and could capture elements such as cultural celebrations, traditional attire, local cuisine, religious sites, landscapes, urban and rural life, transport, occupations, public spaces, etc.

To be specific, it belongs to the State/UT - {state_name} and captures its {category}.
The current caption describing it is:
{current_caption.strip()}

### Guidelines:
- **Objectivity:** Describe only what is visible using the current caption provided for the context of the image.
- **Detail-Rich:** Include key elements like objects, people, actions, setting, colors, textures, and any visible text.
- **Contextual Awareness:** Recognize and mention famous Indian personalities if present in the image.
- **Strict Output Formatting:**
  - Do NOT add extra explanations—generate only the caption.
  - Do NOT include "Caption:" or any additional tags—only return the description.
  - Do NOT mix multiple languages in the caption.
  - Try to make the caption India-specific as much as possible.

### Output (Caption Only, No Additional Text):
"""

    body = {
        "model": "google/gemini-2.5-flash",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text.strip()},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.7
    }

    try:
        res = requests.post(OPENROUTER_URL, headers=HEADERS, json=body)
        res.raise_for_status()
        caption = res.json()["choices"][0]["message"]["content"].strip()
        return {
            "image_path": image_path,
            "caption": caption,
            "source_url": source_url,
            "status": "success"
        }
    except Exception as e:
        print(f"❌ API call error: {e}")
        return {
            "image_path": image_path,
            "caption": None,
            "source_url": source_url,
            "status": "api_failed"
        }

def parallel_process(df, state_name, num_workers=10):
    inputs = [(df.iloc[i], state_name) for i in range(len(df))]
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_row, inputs), total=len(inputs), desc="Processing"))
    return results

def process_with_retries(df, state_name, max_retries=5):
    final_results = []
    retry_df = df.copy()

    for attempt in range(max_retries):
        print(f"\n🔁 Retry Round {attempt+1} - {len(retry_df)} images")
        results = parallel_process(retry_df, state_name)

        # Collect successful
        successes = [r for r in results if r["status"] == "success"]
        final_results.extend(successes)

        # Prepare retry list (API-failed)
        failed_paths = [r["image_path"] for r in results if r["status"] == "api_failed"]
        retry_df = df[df["image_path"].isin(failed_paths)]

        if not failed_paths:
            break

    if not retry_df.empty:
        print(f"⚠️ Still failed after {max_retries} retries: {len(retry_df)} rows")

    final_results_df = pd.DataFrame(final_results)

    return final_results_df, retry_df

# ===== Main Loop =====
state_folders = os.listdir("./state_wise_images")

for state_name in state_folders:
    print(f"\n Processing state: {state_name}")
    OUTPUT_CSV = f"./state_wise_captions_gemini/{state_name}_captions.csv"
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    FAILED_CSV = f"./state_wise_captions_gemini/{state_name}_failed.csv"

    df = pd.read_csv(f"./state_wise_images/{state_name}/metadata.csv")
    df['image_path'] = df.apply(build_image_path, axis=1)

    df_success, df_failed = process_with_retries(df, state_name)

    df_success.to_csv(OUTPUT_CSV, index=False)
    df_failed.to_csv(FAILED_CSV, index=False)

    print(f"Captions saved: {len(df_success)} rows → {OUTPUT_CSV}")
    print(f"Failed rows saved: {len(df_failed)} rows → {FAILED_CSV}")
