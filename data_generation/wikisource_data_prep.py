import os
import json
import shutil
import logging
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import time

# ==============================
# CONFIG
# ==============================
INPUT_CSV = "./wikisource_ocr_docs/combined_indic_wikisource_urls.csv"
IMAGE_DOWNLOAD_DIR = "wikisource_all_images"
LANGWISE_DIR = "sampled_lang_wise_wikisource_images"
MAX_RETRIES = 1
RETRY_DELAY = 0  # seconds

VERIFIED_JSON = "all_verified_wikisource_docs.json"
FINAL_OUTPUT_JSON = "sampled_langwise_wikisource_docs.json"

LANGS = ['bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'or', 'pa', 'ta', 'te']
SAMPLE_PER_LANG = 90

os.makedirs(IMAGE_DOWNLOAD_DIR, exist_ok=True)
os.makedirs(LANGWISE_DIR, exist_ok=True)

# ==============================
# LOGGING
# ==============================
logging.basicConfig(
    filename="error_log.txt",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.ERROR
)
logger = logging.getLogger()

# ==============================
# SESSION
# ==============================
session = requests.Session()
session.headers.update({
    "User-Agent": "IndicVisionBenchResearchBot/1.0 (name@gmail.com)"
})

# ==============================
# HELPER FUNCTIONS
# ==============================
def save_verified_records(records):
    with open(VERIFIED_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def is_proofread(soup):
    quality_element = soup.find(class_="prp-page-qualityheader")
    if not quality_element:
        return False

    classes = quality_element.get("class", [])
    for class_name in classes:
        if class_name.startswith("quality"):
            quality_class_number = int(class_name.lstrip("quality"))
            if quality_class_number == 4:
                return True
    return False


def extract_text(soup):
    pagetext = soup.find(class_="pagetext")
    if not pagetext:
        return None
    return pagetext.get_text(separator="\n", strip=True)


def download_image(soup, output_filename):
    try:
        img_tag = soup.find(class_="prp-page-image").find("span").find("img")
        img_url = "https:" + img_tag["src"]

        response = session.get(img_url)
        if response.status_code == 200:
            with open(output_filename, "wb") as f:
                f.write(response.content)
        else:
            raise requests.HTTPError(f"Unexpected status code: {response.status_code}")
    except Exception as e:
        raise e


# ==============================
# STEP 1: SCRAPE + VERIFY
# ==============================
print("🔎 Step 1: Scraping and verifying pages...")

df_urls = pd.read_csv(INPUT_CSV).sample(n=50000)
urls_list = df_urls['url'].tolist()

verified_records = []
count = 0

for i, url in enumerate(urls_list):

    # Progress print
    if i % 10 == 0:
        print(f"Total urls checked: {i}")
        print(f"Total urls that are level-4 verified: {count}")

    # Periodic checkpoint save
    if i % 1000 == 0 and i != 0:
        print("💾 Checkpoint saving verified records...")
        save_verified_records(verified_records)

    # -------------------------
    # Fetch page
    # -------------------------
    soup = None
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url)
            if response.status_code != 200:
                raise requests.HTTPError(f"HTTP status {response.status_code}")
            soup = BeautifulSoup(response.text, "html.parser")
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"{url} - Error fetching page after retries: {e}")
                soup = None

    if soup is None:
        continue

    # -------------------------
    # Verify flag
    # -------------------------
    try:
        flag_verified = is_proofread(soup)
    except Exception as e:
        logger.error(f"{url} - Flag verification error: {e}")
        continue

    if not flag_verified:
        continue

    # -------------------------
    # Extract text
    # -------------------------
    try:
        text = extract_text(soup)
    except Exception as e:
        logger.error(f"{url} - Text extraction error: {e}")
        continue

    # -------------------------
    # Download image
    # -------------------------
    image_path = os.path.join(IMAGE_DOWNLOAD_DIR, f"image_{i}.jpg")
    download_success = False

    for attempt in range(MAX_RETRIES):
        try:
            download_image(soup, image_path)
            download_success = True
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"{url} - Image download error after retries: {e}")

    if not download_success:
        continue

    verified_records.append({
        "image_path": image_path,
        "text": text,
        "page_url": url
    })

    count += 1

# Final save
print("💾 Final save of verified records...")
save_verified_records(verified_records)

print(f"✅ Total verified pages: {count}")
print(f"Saved verified records to: {VERIFIED_JSON}")


# ==============================
# STEP 2: LANGUAGE SAMPLING
# ==============================
print("🌍 Step 2: Language-wise sampling...")

df_verified = pd.DataFrame(verified_records)
df_verified["lang"] = df_verified["image_url"].apply(
    lambda x: x.removeprefix("https://")[:2]
)

final_records = []

for lang in LANGS:
    print(f"Processing language: {lang}")

    lang_df = df_verified[df_verified['lang'] == lang]

    if len(lang_df) == 0:
        continue

    sampled_df = lang_df.sample(
        n=min(SAMPLE_PER_LANG, len(lang_df)),
        random_state=42
    )

    lang_dir = os.path.join(LANGWISE_DIR, lang)
    os.makedirs(lang_dir, exist_ok=True)

    for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df)):

        # Filter short texts (≤10 words)
        if len(row['text'].split()) <= 10:
            continue

        orig_image_path = row['image_path']
        filename = os.path.basename(orig_image_path)
        new_image_path = os.path.join(lang_dir, filename)

        try:
            shutil.copy(orig_image_path, new_image_path)
        except Exception as e:
            logger.error(f"Image copy error: {orig_image_path} - {str(e)}")
            continue

        final_records.append({
            "image_path": new_image_path,
            "text": row['text'],
            "page_url": row['page_url'],
            "lang": lang
        })

# ==============================
# STEP 3: SAVE FINAL JSON
# ==============================
with open(FINAL_OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(final_records, f, ensure_ascii=False, indent=2)

print("Done!")
print(f"Final dataset size: {len(final_records)}")
print(f"Saved to: {FINAL_OUTPUT_JSON}")