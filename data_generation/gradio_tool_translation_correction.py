import os
import pandas as pd
import gradio as gr
from PIL import Image

BASE_PATH = "."

def get_langs():
    langs = ["Telugu", "Gujarati", "Kannada", "Malayalam", "Marathi",
    "Odia", "Punjabi", "Tamil", "Bengali", "Hindi"]
    
    return langs

def load_original_data(lang):
    csv_path = os.path.join(BASE_PATH, "state_wise_corpus_base.csv")
    df = pd.read_csv(csv_path)
    lang_df = df[df["primary_language"] == lang]
    return lang_df

def load_translated_data(lang):
    csv_path = os.path.join(BASE_PATH, "translated_state_wise_corpus.csv")
    df = pd.read_csv(csv_path)
    lang_df = df[df["primary_language"] == lang]
    return lang_df

def show_data(index, lang):
    df = load_original_data(lang)
    translated_df = load_translated_data(lang)

    if index >= len(df):
        return [index, None] + [""] * (2*len(field_names)) + ["End of data"]
    
    if index >= len(translated_df):
        return [index, None] + [""] * (2*len(field_names)) + ["End of data"]

    row = df.iloc[index]
    for f in field_names:
        if f not in row:
            row[f] = ""
    
    matches = translated_df[translated_df["image_path"] == row["image_path"]]
    if not matches.empty:
        translated_row = matches.iloc[0]
    else:
        translated_row = row.copy()
    for f in field_names:
        if f not in translated_row:
            translated_row[f] = ""

    image_path = row['image_path']
    image = Image.open(image_path)
    image = image.convert("RGB")
    width, height = image.size
    if width > 1080 or height > 1080:
        image.thumbnail((1080, 1080), Image.Resampling.LANCZOS)

    corrected_path = os.path.join(f"{BASE_PATH}/corrected_outputs_state_wise_corpus_translations", f"{lang}_corrected.csv")
    os.makedirs(os.path.dirname(corrected_path), exist_ok=True)
    corrected_row = None
    if os.path.exists(corrected_path):
        corrected_df = pd.read_csv(corrected_path)
        matches = corrected_df[corrected_df["image_path"] == row["image_path"]]
        if not matches.empty:
            corrected_row = matches.iloc[-1]  # use the most recent correction

    if corrected_row is not None:
        status = "done"
    else:
        status = "Not done yet"

    outputs = [image]
    for f in field_names:
        original_val = row[f]
        translated_val = translated_row[f]
        updated_val = corrected_row[f] if corrected_row is not None and f in corrected_row else translated_val
        outputs.append(original_val)
        outputs.append(updated_val)

    return [index] + outputs + [status]

def save_and_next(index, lang, *updates):
    df = load_original_data(lang)
    corrected_path = os.path.join(f"{BASE_PATH}/corrected_outputs_state_wise_corpus_translations", f"{lang}_corrected.csv")

    row = df.iloc[index].copy()
    # Ensure all expected fields are present
    for f in field_names:
        if f not in row:
            row[f] = ""

    updates_index = 0
    for f in field_names:
        updated = updates[updates_index + 1]  # skip current value
        updates_index += 2  # move past current and updated
        if updated.strip():
            row[f] = str(updated)

    # Load or create corrected dataframe
    if os.path.exists(corrected_path):
        corrected_df = pd.read_csv(corrected_path)
        if "image_path" not in corrected_df.columns:
            corrected_df["image_path"] = ""
        mask = corrected_df["image_path"] == row["image_path"]
        if mask.any():
            for col in row.index:
                corrected_df.loc[mask, col] = row[col]

        else:
            corrected_df = pd.concat([corrected_df, pd.DataFrame([row])], ignore_index=True)
    else:
        corrected_df = pd.DataFrame([row])

    corrected_df.to_csv(corrected_path, index=False)
    return show_data(index + 1, lang)

def go_previous(index, lang):
    return show_data(index - 1, lang)

def go_to_index(index_input, lang):
    try:
        index = int(index_input)
    except:
        index = 0
    return show_data(index, lang)

# Fields (current + updated)
field_names = [
    "short_q1", "short_a1", "short_q2", "short_a2",
    "mc_q1", "mc_a1", "mc_opt1_1", "mc_opt1_2", "mc_opt1_3", "mc_opt1_4",
    "true_false_q", "true_false_a", "long_q", "long_a", "adversarial_question", "adversarial_answer"
]

with gr.Blocks() as demo:
    gr.Markdown("### Image Annotation Tool")

    lang_dropdown = gr.Dropdown(choices=get_langs(), label="Select Language")
    index_state = gr.State(0)

    with gr.Row():
        index_display = gr.Textbox(label="Current Index", interactive=False)
        prev_btn = gr.Button("← Previous")
        submit_btn = gr.Button("Submit + Next")
        next_btn = gr.Button("Next →")
        jump_index_input = gr.Textbox(label="Go to Index", scale=1)
        jump_btn = gr.Button("Go", scale=0)

    with gr.Row():
        image_output = gr.Image(label="Image", type="pil")

    input_outputs = []
    for f in field_names:
        with gr.Row():
            input_outputs.append(gr.Textbox(label=f"Original {f}", interactive=False))
            input_outputs.append(gr.Textbox(label=f"Translated {f}"))

    status_output = gr.Textbox(label="Status", interactive=False)

    lang_dropdown.change(
        fn=lambda f: show_data(0, f),
        inputs=lang_dropdown,
        outputs=[index_state, image_output] + input_outputs + [status_output]
    ).then(
        fn=lambda idx: str(idx),
        inputs=index_state,
        outputs=index_display
    )

    submit_btn.click(
        fn=save_and_next,
        inputs=[index_state, lang_dropdown] + input_outputs,
        outputs=[index_state, image_output] + input_outputs + [status_output]
    ).then(
        fn=lambda idx: str(idx),
        inputs=index_state,
        outputs=index_display
    )

    next_btn.click(
        fn=lambda idx, f: show_data(idx + 1, f),
        inputs=[index_state, lang_dropdown],
        outputs=[index_state, image_output] + input_outputs + [status_output]
    ).then(
        fn=lambda idx: str(idx),
        inputs=index_state,
        outputs=index_display
    )

    prev_btn.click(
        fn=go_previous,
        inputs=[index_state, lang_dropdown],
        outputs=[index_state, image_output] + input_outputs + [status_output]
    ).then(
        fn=lambda idx: str(idx),
        inputs=index_state,
        outputs=index_display
    )

    jump_btn.click(
        fn=go_to_index,
        inputs=[jump_index_input, lang_dropdown],
        outputs=[index_state, image_output] + input_outputs + [status_output]
    ).then(
        fn=lambda idx: str(idx),
        inputs=index_state,
        outputs=index_display
    )

demo.launch(server_name="0.0.0.0", server_port=7863)
