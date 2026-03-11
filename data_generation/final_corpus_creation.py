import os
import sys
import glob
import pandas as pd

input_folder = "./corrected_outputs_state_wise"
output_path = "./corrected_outputs_state_wise_corpus_translations/English_corrected.csv"

csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
dfs = [pd.read_csv(f) for f in csv_files]
combined_df = pd.concat(dfs, ignore_index=True)
combined_df.to_csv(output_path, index=False)

languages = ['Bengali', 'English', 'Gujarati', 'Hindi', 'Kannada', 'Malayalam', 'Marathi', 'Odia', 'Punjabi', 'Tamil', 'Telugu']

overall_df = pd.DataFrame()

for language in languages:
    input_csv = f"./corrected_outputs_state_wise_corpus_translations/{language}_corrected.csv"
    output_csv = f"./state_wise_corpus_final_csvs/{language}_corrected.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.read_csv(input_csv)

    df["language"] = len(df)*[language]
    df = df.drop(columns = ['status', 'caption', 'primary_language', 'state'], errors='ignore')
    df['State/UT'] = df['image_path'].apply(lambda x: x.split('/')[-3])
    df['topic'] = df['image_path'].apply(lambda x: x.split('/')[-2])

    df.to_csv(output_csv, index=False)
    overall_df = pd.concat([overall_df, df], ignore_index = True)
    print(f"Updated CSV saved to: {output_csv}")

overall_df.to_csv("./state_wise_corpus_final_csvs/all_languages_combined_corrected.csv", index = False)
