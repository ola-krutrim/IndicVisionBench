import os
import sys
import requests
import base64
import io
from PIL import Image
from functools import lru_cache
import logging
from io import BytesIO
import traceback
from openai import OpenAI
import torch

sys.path.append(".")
sys.path.append("./maya")
sys.path.append("./PALO")
sys.path.append("./Chitrarth")
sys.path.append("./Chitrapathak")
sys.path.append("./Chitranuvad")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ----------------------------------------------------------
# Generic utilities
# ----------------------------------------------------------

openrouter_API_URL = "https://openrouter.ai/api/v1/chat/completions"

openai_API_URL = "https://api.openai.com/v1/chat/completions"

def encode_image_to_base64_with_resize(image: Image.Image) -> str:
    try:
        image = image.convert("RGB")  # Ensure consistent mode
        width, height = image.size

        # Resize only if either dimension is greater than 1080
        if width > 1080 or height > 1080:
            image.thumbnail((1080, 1080), Image.Resampling.LANCZOS)

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode("utf-8")
    except Exception as e:
        print(f"❌ Error resizing and encoding image: {e}")
        return None

# ----------------------------------------------------------
# Model call functions
# ----------------------------------------------------------

def call_gpt(image, prompt, **kwargs):
    image_base64 = encode_image_to_base64_with_resize(image)

    headers = {
        "Authorization": f"Bearer {kwargs.get('api_key', '')}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]}
        ],
        "temperature": 0.7
    }

    response = requests.post(openai_API_URL, headers=headers, json=body)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def call_gemini(image, prompt, **kwargs):
    image_base64 = encode_image_to_base64_with_resize(image)

    headers = {
        "Authorization": f"Bearer {kwargs.get('api_key', '')}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "google/gemini-2.5-flash",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]}
        ],
        "temperature": 0.7
    }

    response = requests.post(openrouter_API_URL, headers=headers, json=body)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def call_llama(image, prompt, **kwargs):
    image_b64 = encode_image_to_base64_with_resize(image)
    if not image_b64:
        return "ERROR: Failed to encode image"
    
    image_data_url = f"data:image/jpeg;base64,{image_b64}"
    KRUTRIM_API_URL = "https://cloud.olakrutrim.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {kwargs.get('api_key', '')}"
    }

    body = {
        "model": "Llama-4-Maverick-17B-128E-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}}
                ]
            }
        ],
        "max_tokens": 2048,
        "stream": False
    }
    try:
        response = requests.post(KRUTRIM_API_URL, headers=headers, json=body)
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"].strip()
        return answer
    except Exception as e:
        print(f"❌ API call error: LLAMA  {e}")
        traceback.print_exc()
        return str(e)

def call_palo(image, prompt, **kwargs):

    from palo.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from palo.conversation import conv_templates, SeparatorStyle
    from palo.model.builder import load_pretrained_model
    from palo.utils import disable_torch_init
    from palo.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
    from transformers import StoppingCriteriaList

    # --- Lazy static variable: model only loads once ---
    if not hasattr(call_palo, "_loaded"):
        print("Loading PALO model for the first time...")
        disable_torch_init()
        model_path = "MBZUAI/PALO-7B"
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
        model = model.eval().cuda()
        # store in function attribute
        call_palo._loaded = (tokenizer, model, image_processor)
    else:
        tokenizer, model, image_processor = call_palo._loaded

    # --- Regular inference flow ---
    conv_mode = "vicuna_v1"

    if model.config.mm_use_im_start_end:
        prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
    else:
        prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image = image.convert("RGB")
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = StoppingCriteriaList([
        KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    ])

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=False,
            temperature=0,
            num_beams=1,
            max_new_tokens=2048,
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )

    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]

    return outputs.strip()


def call_chitrarth(image, prompt, **kwargs):

    from chitrarth.utils import disable_torch_init
    from chitrarth.model.builder import load_pretrained_model
    from chitrarth.inference import eval_model

    # --- Lazy static variable: load once ---
    if not hasattr(call_chitrarth, "_loaded"):
        print("Loading Chitrarth model for the first time...")
        disable_torch_init()

        model_path = "krutrim-ai-labs/chitrarth"

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, model_base=None, model_name="chitrarth"
        )

        model.eval().cuda()

        call_chitrarth._loaded = (tokenizer, model, image_processor, context_len)
    else:
        tokenizer, model, image_processor, context_len = call_chitrarth._loaded

    image = image.convert("RGB")
    image_path = "./tmp/temp_image.jpg"
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    image.save(image_path)

    try:
        return eval_model(tokenizer, model, image_processor, context_len, prompt, image_path)

    except Exception as e:
        tb = traceback.format_exc()
        return f"[ERROR] Exception during Chitrarth inference:\n{tb}"


def call_maya(image, prompt, **kwargs):
    # Save image to a temporary file
    image_path = "./tmp/temp_image.jpg"
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    image.save(image_path)

    from maya.llava.eval.talk2maya import run_vqa_model
    try:
        output = run_vqa_model(question=prompt, image_file=image_path)
    except Exception as e:
        print(f"❌ Error calling Maya model: {e}")
        output = str(e)
    return output


def call_pangea(image, prompt, **kwargs):

    from transformers import LlavaNextForConditionalGeneration, AutoProcessor

    # --- Lazy static variable: load model only once ---
    if not hasattr(call_pangea, "_loaded"):
        print("Loading Pangea-7B model for the first time...")
        model_name = "neulab/Pangea-7B-hf"

        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.float16
        ).to(0)

        processor = AutoProcessor.from_pretrained(model_name)
        model.resize_token_embeddings(len(processor.tokenizer))

        # store in function attribute for reuse
        call_pangea._loaded = (model, processor)
    else:
        model, processor = call_pangea._loaded

    # --- Prepare inputs ---
    image_input = image.convert("RGB")

    text_input = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<image>\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    model_inputs = processor(
        images=image_input,
        text=text_input,
        return_tensors='pt'
    ).to("cuda", torch.float16)

    # --- Generate output ---
    with torch.inference_mode():
        output = model.generate(
            **model_inputs,
            max_new_tokens=1024,
            min_new_tokens=32,
            temperature=1.0,
            top_p=0.9,
            do_sample=True
        )

    # --- Decode ---
    result = processor.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    result = result.replace(prompt, "")
    result = result.split("assistant")[2].strip()

    return result


def call_gemma(image, prompt, **kwargs):
    # --- API Config for Gemma ---
    KRUTRIM_API_URL = "https://cloud.olakrutrim.com/v1/chat/completions"
    HEADERS = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {kwargs.get('api_key', '')}"
    }

    try:
        messages = []

        # Construct content payload
        image_b64 = encode_image_to_base64_with_resize(image)
        image_data_url = f"data:image/jpeg;base64,{image_b64}"
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt.strip()},
                {"type": "image_url", "image_url": {"url": image_data_url}}
            ]
        })

        body = {
            "model": "Gemma-3-27B-IT",
            "messages": messages,
            "max_tokens": 1024,
            "stream": False
        }

        res = requests.post(KRUTRIM_API_URL, headers=HEADERS, json=body)
        res.raise_for_status()
        content = res.json()["choices"][0]["message"]["content"]
        return content.strip()

    except Exception as e:
        print(f"❌ Gemma API error: {e}")
        return str(e)


def call_chitrapathak(image, prompt, **kwargs):

    from chitrapathak.utils import disable_torch_init
    from chitrapathak.model.builder import load_pretrained_model
    from chitrapathak.inference import eval_model

    # --- Lazy static variable: load once ---
    if not hasattr(call_chitrapathak, "_loaded"):
        print("Loading Chitrapathak-1 model for the first time...")
        disable_torch_init()

        model_path = "krutrim-ai-labs/Chitrapathak-1"

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, model_base=None, model_name="chitrapathak"
        )

        model.eval().cuda()

        call_chitrapathak._loaded = (tokenizer, model, image_processor, context_len)
    else:
        tokenizer, model, image_processor, context_len = call_chitrapathak._loaded

    image = image.convert("RGB")

    try:
        return eval_model(tokenizer, model, image_processor, context_len, prompt, image)

    except Exception as e:
        tb = traceback.format_exc()
        return f"[ERROR] Exception during Chitrapathak-1 inference:\n{tb}"


def call_chitranuvad(image, prompt, **kwargs):

    from chitranuvad.utils import disable_torch_init
    from chitranuvad.model.builder import load_pretrained_model
    from chitranuvad.inference import eval_model

    # --- Lazy static variable: load once ---
    if not hasattr(call_chitranuvad, "_loaded"):
        print("Loading Chitranuvad model for the first time...")
        disable_torch_init()

        model_path = "krutrim-ai-labs/Chitranuvad"

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, model_base=None, model_name="chitranuvad"
        )

        model.eval().cuda()

        call_chitranuvad._loaded = (tokenizer, model, image_processor, context_len)
    else:
        tokenizer, model, image_processor, context_len = call_chitranuvad._loaded

    image = image.convert("RGB")
    image_path = "./tmp/temp_image.jpg"
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    image.save(image_path)

    try:
        return eval_model(tokenizer, model, image_processor, context_len, prompt, image_path)

    except Exception as e:
        tb = traceback.format_exc()
        return f"[ERROR] Exception during Chitranuvad inference:\n{tb}"


def call_surya(image, prompt, **kwargs):

    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    from surya.layout import LayoutPredictor
    from surya.settings import settings

    if not hasattr(call_surya, "_loaded"):
        # ---------- LOAD PREDICTORS ----------
        foundation_predictor = FoundationPredictor()
        recognition_predictor = RecognitionPredictor(foundation_predictor)
        detection_predictor = DetectionPredictor()
        layout_predictor = LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))

        call_surya._loaded = (foundation_predictor, recognition_predictor, detection_predictor, layout_predictor)

    else:
        foundation_predictor, recognition_predictor, detection_predictor, layout_predictor = call_surya._loaded
    
    # ---------- LAYOUT DETECTION ----------
    layout_predictions = layout_predictor([image])
    
    # ---------- OCR ON EACH TEXT BBOX ----------
    final_output = []
    for item in layout_predictions[0].bboxes:
            # Extract bbox
            x1, y1, x2, y2 = item.bbox
    
            # Crop the text region
            cropped_img = image.crop((x1, y1, x2, y2))
    
            # Run OCR on the cropped section
            predictions = recognition_predictor([cropped_img], det_predictor=detection_predictor)
    
            # Collect OCR text lines
            text_output = ""
            for line in predictions[0].text_lines:
                text_output += line.text + "\n"
    
            # Add cleaned text to final output
            if text_output.strip():
                final_output.append(text_output.strip())

    # ---------- COMBINE ALL OCR SECTIONS ----------
    final_text = "\n".join(final_output)
    final_text = final_text.replace("<br>", "\n")

    return final_text

# ----------------------------------------------------------
# Unified interface
# ----------------------------------------------------------

MODEL_DISPATCH = {
    "gemma": call_gemma,
    "llama": call_llama,
    "maya": call_maya,
    "palo": call_palo,
    "pangea": call_pangea,
    "chitrarth": call_chitrarth,
    "gemini": call_gemini,
    "gpt": call_gpt,
    "surya": call_surya,
    "chitrapathak": call_chitrapathak,
    "chitranuvad": call_chitranuvad
}

def run_model(model_name: str, image: Image.Image, prompt: str, **kwargs) -> str:
    model_name = model_name.lower()
    if model_name not in MODEL_DISPATCH:
        raise ValueError(f"Unknown model name: {model_name}")
    fn = MODEL_DISPATCH[model_name]
    logger.info(f"🔹 Running model: {model_name}")
    return fn(image, prompt, **kwargs)
