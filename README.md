# Image Captioning with Fine-Tuning and Custom Models

## Overview
This repository contains code for image captioning tasks using vision-language models. It includes:
- A Python script (`groq_flicker8k.py`) for fine-tuning Meta Llama-4 vision models on the Flickr8k dataset using the Groq API, followed by evaluation with metrics like BLEU, METEOR, ROUGE-L, and SPICE.
- A Jupyter Notebook (`CustomMobileNetv3_Blip_AoaNet.ipynb`) implementing a custom image captioning model with MobileNetV3 as the backbone, integrated with BLIP and AoA Net architectures. The notebook covers data mounting, dependency installation, data processing, model building, training, and inference, including text-to-speech for captions.

The project demonstrates end-to-end workflows for training and evaluating models on image description tasks.

## Features
- Fine-tuning of Llama-4 vision models (e.g., scout and maverick variants) on Flickr8k.
- Custom encoder-decoder architecture using MobileNetV3 for feature extraction, LSTM/Bidirectional LSTM for decoding, and attention mechanisms (AoA Net).
- Evaluation metrics for caption quality: BLEU (1-4), METEOR, ROUGE-L, SPICE.
- Data visualization (e.g., word clouds, sample images with captions).
- Text-to-speech integration for listening to generated captions.

## Dependencies
- Python 3.10+ (for the script) or compatible with Google Colab (for the notebook).
- Libraries (install via `pip`):
  - groq
  - datasets (Hugging Face)
  - nltk
  - rouge-score
  - pycocoevalcap (for SPICE)
  - tensorflow / keras (for the notebook)
  - matplotlib, seaborn, wordcloud
  - gtts (for text-to-speech)
  - Other: numpy, pandas, PIL, tqdm

For the script:
```bash
pip install groq datasets nltk rouge-score pycocoevalcap
```

For the notebook (run in Colab):
- Additional: tensorflow, keras, gtts

## Dataset
- **Flickr8k**: Used in both the script and notebook. Downloaded via Hugging Face Datasets (`atasoglu/flickr8k-dataset`).
- The script processes it into JSONL for fine-tuning.
- The notebook assumes data in Google Drive or local paths.

## Usage

### 1. Groq Fine-Tuning Script (`groq_flicker8k.py`)
- Set your Groq API key: `export GROQ_API_KEY=your_key_here` (replace the placeholder in the code).
- Run the script:
  ```bash
  python groq_flicker8k.py
  ```
- It will:
  - List available Llama-4 models.
  - Prepare and upload Flickr8k train data as JSONL.
  - Fine-tune models (polls until complete).
  - Evaluate on test set and print metrics.

Note: Fine-tuning requires Groq credits/API access. Models like "meta-llama/llama-4-scout-17b-16e-instruct" are used.

### 2. Custom Model Notebook (`CustomMobileNetv3_Blip_AoaNet.ipynb`)
- Open in Google Colab (mount Google Drive for data).
- Run cells sequentially:
  - Install dependencies (e.g., `!pip install gtts`).
  - Load and preprocess data (captions, images).
  - Build and train the model (encoder: MobileNetV3, decoder: LSTM with attention).
  - Generate captions and evaluate.
  - Visualize results and listen to captions via gTTS.

Model architecture:
- Encoder: MobileNetV3Large for image features.
- Decoder: Bidirectional LSTM with AoA (Attention over Attention) module.
- Training: Uses categorical cross-entropy, Adam optimizer.

## Evaluation
- Script: Computes BLEU-1/2/3/4, METEOR, ROUGE-L, SPICE on test set.
- Notebook: Includes BLEU scores and qualitative visualizations.

## Limitations
- Groq API access required for fine-tuning (costs apply).
- Notebook is Colab-specific; adapt paths for local runs.
- Evaluation assumes English captions; metrics may vary for other languages.
- SPICE requires Java (ensure setup for pycocoevalcap).
