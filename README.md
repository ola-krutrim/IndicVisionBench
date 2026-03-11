# IndicVisionBench

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue)](https://openreview.net/forum?id=LmJoLn04iL)
[![Static Badge](https://img.shields.io/badge/Huggingface-IndicVisionBench-yellow?logo=huggingface)](https://huggingface.co/datasets/alifaraz/IndicVisionBench)
[![arXiv](https://img.shields.io/badge/arXiv-2511.04727-b31b1b.svg)](https://arxiv.org/abs/2511.04727)

This repository provides the inference and evaluation scripts for the **IndicVisionBench** benchmark.

**“IndicVisionBench: Benchmarking Cultural and Multilingual Understanding in VLMs”**  
📄 [arXiv:2511.04727](https://arxiv.org/abs/2511.04727)  
🏛️ Accepted at **ICLR 2026**  
🔗 OpenReview: https://openreview.net/forum?id=LmJoLn04iL

## 🚀 Steps for Evaluation

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate predictions
For the OCR track.
```bash
python generation/generate_predictions.py \
    --task ocr \
    --model llama \
    --output_path outputs/ocr_llama_preds.json \
    --num_samples 10 \
    --api_key YOUR_API_KEY
```
For the VQA track.
```bash
python generation/generate_predictions.py \
    --task vqa \
    --model llama \
    --language Hindi \
    --split vqa_indic \
    --output_path outputs/vqa_llama_preds.json \
    --num_samples 10 \
    --api_key YOUR_API_KEY
```
For the MMT track.
```bash
python generation/generate_predictions.py \
    --task mmt \
    --model llama \
    --output_path outputs/mmt_llama_preds.json \
    --target_languages Hindi Tamil Bengali \
    --num_samples 10 \
    --api_key YOUR_API_KEY
```

### Evaluate outputs
For the OCR track.
```bash
python evaluation/evaluate_predictions.py \
      --task_type ocr \
      --predictions_path outputs/ocr_llama_preds.json \
      --indv_scores_path scores/ocr_llama_scores.json \
      --scores_report_path reports/ocr_llama_report.json
```

For the structured questions i.e., MCQ and TF in the VQA track.
```bash
python evaluation/evaluate_predictions.py \
      --task_type vqa_structured \
      --predictions_path outputs/vqa_en_llama_preds.json \
      --indv_scores_path scores/vqa_en_structured_llama_scores.json \
      --scores_report_path reports/vqa_en_structured_llama_report.json
```

For the openended questions i.e., Short-answer type, Long-answer type and Adversarial questions in the VQA track.
```bash
python evaluation/evaluate_predictions.py \
      --task_type vqa_openended \
      --api_key YOUR_OPENAI_API_KEY \
      --predictions_path outputs/vqa_indic_llama_preds.json \
      --indv_scores_path scores/vqa_indic_openended_llama_scores.json \
      --scores_report_path reports/vqa_indic_openended_llama_report.json
```

For the MMT track.
```bash
python evaluation/evaluate_predictions.py \
      --task_type mmt \
      --predictions_path outputs/mmt_llama_preds.json \
      --indv_scores_path scores/mmt_llama_scores.json \
      --scores_report_path reports/mmt_llama_report.json
```

### Supported Models
- LLaMA-4 Maverick 17B
- Gemma-3 27B
- GPT-4o
- Gemini-2.5-Flash
- Chitrarth-1
- Maya
- PALO
- Pangea
- Surya OCR
- Chitrapathak-1
- Chitranuvad


For setting up the environment and other scripts for the Maya model, follow the instructions given here:
https://github.com/nahidalam/maya. Then move the maya directory to the current directory.
```bash
cp ~/maya ~/IndicVisionBench
```
After you do this, the function for inferring from maya will work. Also make sure that you activate the environment for maya before inferring from it. Also make sure that you install all the packages needed to run the benchmark in the model-specific environment.


Follow the same instructions as above for PALO (https://github.com/mbzuai-oryx/PALO), Chitrarth (https://github.com/ola-krutrim/Chitrarth), Surya (https://github.com/datalab-to/surya), Chitrapathak (https://github.com/ola-krutrim/Chitrapathak) and Chitranuvad (https://github.com/ola-krutrim/Chitranuvad).

For all the other models, the environment of this repo should be enough to carry out inference.

## Data Generation

### Image collection & organization

Before running the data generation scripts, arrange images and metadata in the following structure. Each state should have a folder (for example, `Tamil Nadu/`) containing category subfolders and a `metadata.csv` file.

#### Fields to put in `metadata.csv`
- image_filename
- source_url
- category

#### Example folder structure
/state_wise_images/
├── Kerala/
│   ├── Food/
│   │   ├── sadya.jpg
│   │   └── appam.jpg
│   ├── Architecture/
│   │   ├── padmanabhaswamy-temple.jpg
│   │   └── backwater-house.jpg
│   └── metadata.csv
│
├── Punjab/
│   ├── Music/
│   │   ├── bhangra-performance.jpg
│   │   └── dhol-player.jpg
│   ├── Religion/
│   │   ├── golden-temple.jpg
│   │   └── gurpurab.jpg
│   └── metadata.csv

The `data_generation/` folder contains scripts used to construct the benchmark dataset. The pipeline is as follows:

1. **Generate captions and QA pairs**  
   - `caption_generation.py`  
   - `QA_pairs_generation.py`  

2. **English QA correction (factual + cultural validation)**  
   - `gradio_tool_english_annotations.py`  

3. **Sampling for translation**  
   - `sampling_for_translation.py`  

4. **Translate sampled QA pairs**  
   - `translation.py`  

5. **Translation correction and validation**  
   - `gradio_tool_translation_correction.py`  

6. **Final corpus construction**  
   - `final_corpus_creation.py`

7. **Convert to HF dataset and upload** 
   - `upload_to_HF.py`

This general pipeline produces several corpus variants: **IVB-En** is the English benchmark created from crowdsourced images and properly licensed images (collected via Google Search), with QA pairs generated and validated in English; **IVB-Indic** is a translated subset of IVB-En in which QA pairs are translated into Indic languages and corrected using the same validation scripts; the **Parallel Corpus** is a smaller subset of images whose QA pairs have been translated into all ten Indic languages and manually corrected to form a fully parallel multilingual QA dataset; and the **MMT Corpus** uses that same subset but translates (and validates) the image captions into all ten Indic languages for multimodal machine-translation experiments.

To construct the IVB-OCR benchmark, follow the steps below:

1. **Download the Wikisource dumps**

   Run the following bash command to download the latest Wikisource dumps for all supported Indic languages:

   ```bash
   for lang in hi bn ta te mr gu ml kn pa or; do
     wget https://dumps.wikimedia.org/${lang}wikisource/latest/${lang}wikisource-latest-pages-articles-multistream.xml.bz2
   done

2. **Extract Wikisource URLs**

   Run the following script to generate a combined CSV file containing Wikisource URLs across all Indic languages:

   ```bash
   python data_generation/wikisource_urls_extraction.py
   ```

3. **Verification and Downloading**

   Run the script below to:
   - Verify if a particular wikisource page is level-4 verified or not
   - Download document images and extract the corresponding OCR ground-truth text for the verified ones 
   - Organize images into language-wise folders

   ```bash
   python data_generation/wikisource_data_prep.py
   ```

   To increase the success rate of fetching pages and downloading images, you can increase the retry parameters in:

   ```
   data_generation/wikisource_data_prep_with_retry.py
   ```

   Adjust the following variables:
   - MAX_RETRIES
   - RETRY_DELAY

   In practice, setting MAX_RETRIES = 1 is typically sufficient to collect enough data points for building the benchmark.

### License

This code repository is licensed under the [Krutrim Community License Agreement Version 1.0](LICENSE.md)

### Citation

If you use this repository, please cite:

```bibtex
@inproceedings{faraz2026indicvisionbench,
  title={IndicVisionBench: Benchmarking Cultural and Multilingual Understanding in VLMs},
  author={Ali Faraz and Akash and Shaharukh Khan and Raja Kolla and Akshat Patidar and Suranjan Goswami and Abhinav Ravi and Chandra Khatri and Shubham Agarwal},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026},
  url={https://openreview.net/forum?id=LmJoLn04iL}
}
