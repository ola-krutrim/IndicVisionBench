import pandas as pd
import os

primary_language_by_state_ut = {
    "Andhra Pradesh": "Telugu",
    "Arunachal Pradesh": "English",
    "Assam": "Assamese",
    "Bihar": "Hindi",
    "Chhattisgarh": "Hindi",
    "Goa": "Konkani",
    "Gujarat": "Gujarati",
    "Haryana": "Hindi",
    "Himachal Pradesh": "Hindi",
    "Jharkhand": "Hindi",
    "Karnataka": "Kannada",
    "Kerala": "Malayalam",
    "Madhya Pradesh": "Hindi",
    "Maharashtra": "Marathi",
    "Manipur": "Meitei",
    "Meghalaya": "English",
    "Mizoram": "Mizo",
    "Nagaland": "English",
    "Odhisa": "Odia",
    "Punjab": "Punjabi",
    "Rajasthan": "Hindi",
    "Sikkim": "Nepali",
    "Tamil Nadu": "Tamil",
    "Telangana": "Telugu",
    "Tripura": "Bengali",
    "Uttar Pradesh": "Hindi",
    "Uttarakhand": "Hindi",
    "West Bengal": "Bengali",
    "Andaman and Nicobar Islands": "Hindi",
    "Chandigarh": "Hindi",
    "Dadra and Nagar Haveli and Daman and Diu": "Gujarati",
    "Delhi": "Hindi",
    "Jammu and Kashmir": "Urdu",
    "Ladakh": "Ladakhi",
    "Lakshadweep": "Malayalam",
    "Puducherry": "Tamil"
}

target_languages = {"Hindi", "Gujarati", "Tamil", "Telugu", "Marathi", 
                    "Bengali", "Malayalam", "Odia", "Punjabi", "Kannada"}

base_path = "./corrected_outputs_state_wise"
files = os.listdir(base_path)
state_overall_df = pd.DataFrame()
for file in files:
    file_path = os.path.join(base_path, file)
    df = pd.read_csv(file_path)
    state_overall_df = pd.concat([state_overall_df, df], ignore_index = True)

state_overall_df["state"] = state_overall_df["image_path"].apply(lambda x: x.split("/")[-3])

state_overall_df["primary_language"] = state_overall_df["state"].map(primary_language_by_state_ut)

filtered_states = state_overall_df[state_overall_df["primary_language"].isin(target_languages)]

sampled_df = (
    filtered_states.groupby("state")
    .apply(lambda x: x.sample(n=min(30, len(x)), random_state=42))
    .reset_index(drop=True)
)

sampled_df.to_csv("./state_wise_corpus_base.csv", index = False)
