# Linguistic-Agnostic SER: A Probing Approach to Acoustic Emotion Encoding

![Python](https://img.shields.io/badge/python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-Transformers-F9AB00?logo=huggingface&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-web%20framework-009688?logo=fastapi&logoColor=white)

## Project Overview

This repository contains an MLOps-ready framework for probing Speech Emotion Recognition (SER) transformers to detect how paralinguistic, acoustic knowledge is encoded across their layers. 

Modern SER systems (e.g., wav2vec 2.0, HuBERT, Whisper) rely on massive transformers that implicitly leverage linguistic cues. This project explicitly maps the acoustic hierarchy by extracting hidden states across all layers and using linear probes to evaluate layer-wise emotion retention.

## Features & Project Structure

The codebase has been refactored into a clean, modular MLOps architecture. 

```
linguistic-agnostic-ser/
├── notebooks/                  # Consolidated unified experiment notebook
├── src/
│   ├── data_ingestion/         # Loaders for 6 datasets: RAVDESS, EmoDB, IEMOCAP, SAVEE, AESDD, MESD
│   ├── preprocessing/          # Audio resampling, cropping, and dataset logic
│   ├── models/                 # Transformer wrapper and hidden state extractor 
│   ├── pipelines/              # Training logic (cross-val logreg) and plot generation
│   └── utils/                  # Helper math functions
├── scripts/                    # Entrypoints for running headless pipelines
├── server/                     # FastAPI backend and vanilla Javascript Premium UI
└── results/                    # Extracted embeddings, raw CSV outputs, and visual plots
```

### 1. Data Processing Pipeline (`src/data_ingestion/`, `src/preprocessing/`)
Dynamically parses multiple audio corpora using optimal metadata extraction and standardizes them to 16kHz for uniform Transformer ingest.

### 2. Transformer Representation Extractor (`src/models/feature_extractors.py`)
Intercepts and freezes representations across all layers of models like `facebook/wav2vec2-base` or `facebook/hubert-base-ls960`. Detaches memory dynamically to maintain low compute overhead.

### 3. Layer-Wise Statistical Probing (`src/pipelines/train.py`)
Fits lightweight logistic models across hidden states using rigorous 5-Fold Stratified Cross-Validation to guarantee generalized inference over different validation domains.

---

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/nikelroid/linguistic-agnostic-ser.git
cd linguistic-agnostic-ser
```

2. **Install Dependencies:**
The project relies on PyTorch, Transformers, Audio packages, and FastAPI.
```bash
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart  # If not in requirements
```

---

## Usage

### Method 1: Web Interface (FastAPI UI)

To explore predictions visually via a premium, dark-mode browser dashboard:
1. Start the server:
```bash
uvicorn server.main:app --reload
```
2. Navigate to `http://localhost:8000/` locally.
3. Drag & drop `.wav` files for analysis, or explore dynamically pre-loaded results.

### Method 2: Command Line Pipeline Execution

To extract features and cross-validate layers on raw datasets from the CLI:

```bash
python -m scripts.run_pipeline \
  --task classification \
  --data_dir /path/to/RAVDESS_folder \
  --dataset_name RAVDESS \
  --model_name facebook/wav2vec2-base \
  --batch_size 4
```

### Method 3: Notebook Playgrounds
For iterative R&D, open `notebooks/01_comprehensive_experiment.ipynb` to view our fully aggregated 18 experiments crossing 3 variants (w2v2, HuBERT, Whisper) against 6 domains.

---

## Comprehensive Results Summary

Our comprehensive evaluation on over **18 combinations** highlighted the behavior of layered embeddings:

| Model | Dataset | Best Layer | Best Accuracy | Best F1 |
|---|---|---|---|---|
| wav2vec2 | RAVDESS | Layer 5 | 0.9701 | 0.9701 |
| wav2vec2 | EmoDB | Layer 4 | 0.9720 | 0.9719 |
| wav2vec2 | IEMOCAP | Layer 1 | 0.6498 | 0.6498 |
| HuBERT | RAVDESS | Layer 8 | 0.9757 | 0.9757 |
| HuBERT | EmoDB | Layer 5 | 0.9776 | 0.9774 |
| Whisper | RAVDESS | Layer 6 | 0.9750 | 0.9751 |
| Whisper | EmoDB | Layer 5 | 0.9570 | 0.9566 |

*Check the extensive table inside notebooks documentation for complete numbers on SAVEE, AESDD, and MESD.*

## License
Open-source for Academic/Research purposes. Please attribute original authors if replicating the pipeline schema.
