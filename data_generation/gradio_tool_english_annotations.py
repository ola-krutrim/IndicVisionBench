import os
import pandas as pd
import gradio as gr
from PIL import Image

BASE_PATH = "."

def get_folders():
    folders = os.listdir(os.path.join(BASE_PATH, "state_wise_images"))
    return folders

def load_data(folder):
    csv_path = os.path.join(BASE_PATH, f"state_wise_full_QAs/{folder}.csv")
    df = pd.read_csv(csv_path)
    return df

def show_data(index, folder):
    df = load_data(folder)
    if index >= len(df):
        return [index, None] + [""] * (len(field_names) * 2) + ["End of data"]

    row = df.iloc[index]
    for f in field_names:
        if f not in row:
            row[f] = ""

    image_path = row['image_path']
    image = Image.open(image_path)
    image = image.convert("RGB")
    width, height = image.size
    if width > 1080 or height > 1080:
        image.thumbnail((1080, 1080), Image.Resampling.LANCZOS)

    corrected_path = os.path.join(f"{BASE_PATH}/corrected_outputs_state_wise", f"{folder}_corrected.csv")
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
        current_val = row[f]
        updated_val = corrected_row[f] if corrected_row is not None and f in corrected_row else current_val
        outputs.append(current_val)
        outputs.append(updated_val)

    return [index] + outputs + [status]

def save_and_next(index, folder, *updates):
    df = load_data(folder)
    corrected_path = os.path.join(f"{BASE_PATH}/corrected_outputs_state_wise", f"{folder}_corrected.csv")

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
    return show_data(index + 1, folder)

def go_previous(index, folder):
    return show_data(index - 1, folder)

def go_to_index(index_input, folder):
    try:
        index = int(index_input)
    except:
        index = 0
    return show_data(index, folder)

# Fields (current + updated)
field_names = [
    "caption", "short_q1", "short_a1", "short_q2", "short_a2",
    "mc_q1", "mc_a1", "mc_opt1_1", "mc_opt1_2", "mc_opt1_3", "mc_opt1_4",
    "true_false_q", "true_false_a", "long_q", "long_a", "adversarial_question", "adversarial_answer"
]

with gr.Blocks() as demo:
    gr.Markdown("### Image Annotation Tool")
    
    folder_dropdown = gr.Dropdown(choices=get_folders(), label="Select Folder")
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
            input_outputs.append(gr.Textbox(label=f"Current {f}", interactive=False))
            input_outputs.append(gr.Textbox(label=f"Updated {f}"))

    status_output = gr.Textbox(label="Status", interactive=False)

    folder_dropdown.change(
        fn=lambda f: show_data(0, f),
        inputs=folder_dropdown,
        outputs=[index_state, image_output] + input_outputs + [status_output]
    ).then(
        fn=lambda idx: str(idx),
        inputs=index_state,
        outputs=index_display
    )

    submit_btn.click(
        fn=save_and_next,
        inputs=[index_state, folder_dropdown] + input_outputs,
        outputs=[index_state, image_output] + input_outputs + [status_output]
    ).then(
        fn=lambda idx: str(idx),
        inputs=index_state,
        outputs=index_display
    )

    next_btn.click(
        fn=lambda idx, f: show_data(idx + 1, f),
        inputs=[index_state, folder_dropdown],
        outputs=[index_state, image_output] + input_outputs + [status_output]
    ).then(
        fn=lambda idx: str(idx),
        inputs=index_state,
        outputs=index_display
    )

    prev_btn.click(
        fn=go_previous,
        inputs=[index_state, folder_dropdown],
        outputs=[index_state, image_output] + input_outputs + [status_output]
    ).then(
        fn=lambda idx: str(idx),
        inputs=index_state,
        outputs=index_display
    )

    jump_btn.click(
        fn=go_to_index,
        inputs=[jump_index_input, folder_dropdown],
        outputs=[index_state, image_output] + input_outputs + [status_output]
    ).then(
        fn=lambda idx: str(idx),
        inputs=index_state,
        outputs=index_display
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
