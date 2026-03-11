import bz2
import xml.etree.ElementTree as ET
import urllib.parse
import csv
import os


# Mapping language name -> subdomain code
LANGUAGES = {
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Malayalam": "ml",
    "Kannada": "kn",
    "Punjabi": "pa",
    "Oriya": "or"
}


def title_to_url(title, lang_code):
    return f"https://{lang_code}.wikisource.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"


def extract_urls_from_dump(dump_path, lang_name, lang_code, writer):
    count = 0
    print(f"\nProcessing {lang_name}...")

    with bz2.open(dump_path, 'rb') as f:
        context = ET.iterparse(f, events=('end',))

        for event, elem in context:
            if elem.tag.endswith('page'):
                title = elem.findtext('./{*}title')

                if title:
                    url = title_to_url(title, lang_code)
                    writer.writerow([lang_name, url])
                    count += 1

                    if count % 10000 == 0:
                        print(f"{lang_name}: {count} URLs extracted...")

                elem.clear()

        del context

    print(f"✅ {lang_name}: Total {count} URLs extracted.")


def extract_all_languages_combined(dump_directory, output_csv="combined_wikisource_urls.csv"):
    with open(output_csv, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["language", "url"])  # header

        for lang_name, lang_code in LANGUAGES.items():
            dump_filename = f"{lang_code}wikisource-latest-pages-articles-multistream.xml.bz2"
            dump_path = os.path.join(dump_directory, dump_filename)

            if not os.path.exists(dump_path):
                print(f"⚠️ Dump not found for {lang_name}: {dump_filename}")
                continue

            extract_urls_from_dump(dump_path, lang_name, lang_code, writer)

    print("\nAll languages processed!")
    print(f"Combined CSV saved as: {output_csv}")


# ====== RUN ======
if __name__ == "__main__":
    extract_all_languages_combined(
        dump_directory="./wikipedia_dumps",
        output_csv="combined_indic_wikisource_urls.csv"
    )