# Linguistic-Agnostic Speech Emotion Recognition: A Probing Approach to Acoustic Emotion Encoding

![Python](https://img.shields.io/badge/python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-Transformers-F9AB00?logo=huggingface&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-web%20framework-009688?logo=fastapi&logoColor=white)

## Project Overview

Welcome to the **Linguistic-Agnostic SER** repository. Speech emotion recognition (SER) has advanced significantly in recent years, driven by pretrained transformer models. However, because models like **wav2vec 2.0**, **HuBERT**, and **Whisper** are pretrained on massive linguistic datasets, their capacity to recognize emotion often inadvertently relies on *what* is being said, rather than *how* it is said (the paralinguistic tone).

In this project, we completely freeze three prominent pretrained transformer architectures and probe their internal layers using a linear Logistic Regression classifier. By explicitly extracting the hidden states layer-by-layer across 6 datasets consisting of 4 wildly different languages (English, German, Greek, Spanish), we map out **where emotion-specific information resides within a neural network matrix**.

### Team Members
- **Nima Kelidari** (kelidari@usc.edu)
- **Minoo Ahmadi** (minooahm@usc.edu)
- **Chaitanya Parwatkar** (parwatka@usc.edu)
- **Xiangxu Lin** (xiangxul@usc.edu)

---

## 🔑 Key Findings

1. **Middle layers excel for acted speech.** Across datasets like RAVDESS and EmoDB, accuracy consistently peaks between Layers 2 and 5. Deeper layers, especially in wav2vec 2.0, catastrophically drop the acoustic information as the network focuses on linguistic structure.
2. **Spontaneous conversation peaks earlier.** On conversational datasets like IEMOCAP, the models peak immediately at Layer 1 or 2 as real-world emotion relies heavily on sudden fluctuations in immediate pitch and volume.
3. **HuBERT retains emotion better than wav2vec 2.0.** HuBERT’s offline clustering strategy preserves emotional signatures deep into Layer 12, whereas wav2vec 2.0’s contrastive task aggressively overwrites prosody.
4. **Emotional encoding is language-agnostic.** Despite pre-training exclusively on English, the layers reacted identically for Spanish (MESD), German (EmoDB), and Greek (AESDD). 

## 📂 Repository Architecture

We have constructed a full MLOps framework mapping a modular logic tree managed structurally by YAML representations:

```
linguistic-agnostic-ser/
├── config/                     # Core execution configs (config.yaml)
├── docs/                       # Theoretical and mechanical documentation map
├── notebooks/                  # Interactive experimentation playground
│   ├── 01_exploratory_probing_analysis.ipynb
│   └── 02_mass_experiments.ipynb
├── src/                        # Modular Core Logic
│   ├── data_ingestion/         # Parsers for RAVDESS, EmoDB, IEMOCAP, SAVEE, AESDD, MESD
│   ├── preprocessing/          # torchaudio & opensmile 16kHz processors
│   ├── models/                 # Transformer and hidden state extractor 
│   ├── pipelines/              # Layer-wise statistical probing execution
│   └── utils/                  # Universal YAML parameter fetchers
├── scripts/                    # Headless CLI endpoints
└── server/                     # FastAPI backend and vanilla Javascript Dashboard
```

---

## 🚀 Installation & Execution

### 1. Installation
Clone the repository and install standard requirements:
```bash
git clone https://github.com/nikelroid/linguistic-agnostic-ser.git
cd linguistic-agnostic-ser
pip install -r requirements.txt
```

### 2. Live Probing Dashboard (FastAPI)
The UI incorporates a dark-mode interactive testing lab to chart your layers dynamically and stream live stdout background pipes.
```bash
uvicorn server.main:app --reload
```
Navigate to `http://localhost:8000/`.

### 3. Headless Pipeline Runs
Drive the entire logic graph structurally from the command line on any dataset:
```bash
python -m scripts.run_pipeline \
  --task classification \
  --dataset_name RAVDESS \
  --data_dir /path/to/RAVDESS_folder \
  --model_name facebook/wav2vec2-base
```
*Note: Config defaults (like Batch Sizes and Random Seeds) are fetched from `config/config.yaml` automatically.*

---

## 📊 Performance Statistics

Our exhaustive evaluations returned the following peak layer characteristics across architectures:

| Model | Architecture Depth | Dataset | Best Layer | Accuracy |
|---|---|---|---|---|
| wav2vec2 | 13 representations | RAVDESS (English) | Layer 5 | 0.9701 |
| HuBERT | 13 representations | RAVDESS (English) | Layer 8 | 0.9757 |
| Whisper | 7 representations | RAVDESS (English) | Layer 6 | 0.9750 |
| wav2vec2 | 13 representations | IEMOCAP (Spontaneous) | Layer 1 | 0.6498 |
| Whisper | 7 representations | IEMOCAP (Spontaneous) | Layer 2 | 0.7009 |
| wav2vec2 | 13 representations | AESDD (Greek) | Layer 2 | 0.9039 |

*See `docs/project_documentation.md` for full cross-lingual and methodology derivations.*
