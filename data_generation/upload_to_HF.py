import pandas as pd
from datasets import Dataset, Features, Value, Image
from huggingface_hub import login

# -----------------------
# CONFIG
# -----------------------
csv_path = "./state_wise_corpus_final_csvs/all_languages_combined_corrected.csv"
repo_id = "YOUR_USERNAME/YOUR_DATASET_NAME"
login(token = "YOUR_HF_TOKEN") 

# -----------------------
# LOAD CSV
# -----------------------
df = pd.read_csv(csv_path)

# -----------------------
# REARRANGE COLUMNS
# -----------------------
column_order = [
    "image_path", "topic", "State/UT", "language", "short_q1", "short_a1", "short_q2", "short_a2", "mc_q1", 
    "mc_a1", "mc_opt1_1", "mc_opt1_2", "mc_opt1_3", "mc_opt1_4", "true_false_q", "true_false_a", "long_q",
    "long_a", "adversarial_question", "adversarial_answer", "source_url"
]

df = df[column_order]

# -----------------------
# RENAME image_path → image
# -----------------------
df = df.rename(columns={"image_path": "image", "mc_q1": "mcq", "mc_a1": "mcq_a", "mc_opt1_1": "mcq_opt1",
                        "mc_opt1_2": "mcq_opt2", "mc_opt1_3": "mcq_opt3", "mc_opt1_4": "mcq_opt4"})

# -----------------------
# DEFINE HF FEATURES
# -----------------------
features = Features({
    "image": Image(),
    "topic": Value("string"),
    "State/UT": Value("string"),
    "language": Value("string"),
    "short_q1": Value("string"),
    "short_a1": Value("string"),
    "short_q2": Value("string"),
    "short_a2": Value("string"),
    "mcq": Value("string"),
    "mcq_a": Value("string"),
    "mcq_opt1": Value("string"),
    "mcq_opt2": Value("string"),
    "mcq_opt3": Value("string"),
    "mcq_opt4": Value("string"),
    "true_false_q": Value("string"),
    "true_false_a": Value("string"),
    "long_q": Value("string"),
    "long_a": Value("string"),
    "adversarial_question": Value("string"),
    "adversarial_answer": Value("string"),
    "source_url": Value("string"),
})

# -----------------------
# CONVERT TO HF DATASET
# -----------------------
dataset = Dataset.from_pandas(df, features=features, preserve_index=False)


# -----------------------
# PUSH TO HUB
# -----------------------
dataset.push_to_hub(repo_id)

print("✅ Dataset uploaded successfully")