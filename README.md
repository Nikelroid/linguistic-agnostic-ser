# Linguistic-Agnostic Speech Emotion Recognition: A Probing Approach to Acoustic Emotion Encoding

![Python](https://img.shields.io/badge/python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-Transformers-F9AB00?logo=huggingface&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-web%20framework-009688?logo=fastapi&logoColor=white)

## Project Overview

Speech emotion recognition (SER) has advanced significantly in recent years, driven by pretrained transformer models. However, because models like **wav2vec 2.0**, **HuBERT**, and **Whisper** are pretrained on massive linguistic datasets, their capacity to recognize emotion often inadvertently relies on *what* is being said (linguistics), rather than *how* it is said (the paralinguistic tone).

In this project, we explicitly freeze three prominent pretrained transformer architectures and probe their internal layers using a linear Logistic Regression classifier. By extracting the hidden states layer-by-layer across 6 structurally diverse datasets encompassing 4 wildly different languages (English, German, Greek, Spanish), we map out **where emotion-specific acoustic information resides within the neural network**.

### Team Members
- **Nima Kelidari** (kelidari@usc.edu)
- **Minoo Ahmadi** (minooahm@usc.edu)
- **Chaitanya Parwatkar** (parwatka@usc.edu)
- **Xiangxu Lin** (xiangxul@usc.edu)

---

## 📊 Performance Statistics (All 18 Experiments)

Our evaluation spans 18 experimental combinations across 3 pre-training paradigms and 6 dataset configurations. This provides a comprehensive map of how emotion is retained layer-by-layer.

| Dataset (Language, Style) | Model | Best Layer | Accuracy | Weighted F1 |
|---|---|---|---|---|
| **RAVDESS** (English, Acted) | wav2vec 2.0 | Layer 5 | 0.9701 | 0.9701 |
| **RAVDESS** (English, Acted) | HuBERT | Layer 8 | 0.9757 | 0.9757 |
| **RAVDESS** (English, Acted) | Whisper | Layer 6 | 0.9750 | 0.9751 |
| **EmoDB** (German, Acted) | wav2vec 2.0 | Layer 4 | 0.9720 | 0.9719 |
| **EmoDB** (German, Acted) | HuBERT | Layer 5 | 0.9776 | 0.9774 |
| **EmoDB** (German, Acted) | Whisper | Layer 5 | 0.9570 | 0.9566 |
| **IEMOCAP** (English, Spontaneous) | wav2vec 2.0 | Layer 1 | 0.6498 | 0.6498 |
| **IEMOCAP** (English, Spontaneous) | HuBERT | Layer 1 | 0.6597 | 0.6594 |
| **IEMOCAP** (English, Spontaneous) | Whisper | Layer 2 | 0.7009 | 0.7009 |
| **SAVEE** (British English, Acted) | wav2vec 2.0 | Layer 5 | 0.8500 | 0.8473 |
| **SAVEE** (British English, Acted) | HuBERT | Layer 5 | 0.8750 | 0.8722 |
| **SAVEE** (British English, Acted) | Whisper | Layer 3 | 0.8208 | 0.8183 |
| **AESDD** (Greek, Acted) | wav2vec 2.0 | Layer 2 | 0.9039 | 0.9035 |
| **AESDD** (Greek, Acted) | HuBERT | Layer 5 | 0.9403 | 0.9399 |
| **AESDD** (Greek, Acted) | Whisper | Layer 3 | 0.9072 | 0.9068 |
| **MESD** (Spanish, Acted) | wav2vec 2.0 | Layer 2 | 0.8446 | 0.8443 |
| **MESD** (Spanish, Acted) | HuBERT | Layer 2 | 0.8597 | 0.8588 |
| **MESD** (Spanish, Acted) | Whisper | Layer 3 | 0.8109 | 0.8108 |

---

## 📈 Analysis & Key Findings

Our analysis of the extracted representation maps points us to the following conclusions:

![Layer-wise Architecture Comparison](docs/Midterm_Report/model_comparison_accuracy.png)
*(Above): Layer-wise probing accuracy broken down by architecture. Wav2vec 2.0 constantly collapses in deeper layers as it enforces linguistic understanding, whereas HuBERT retains emotion into the final layer block. Whisper maintains high baseline accuracy directly out of the box.*

1. **Middle layers excel for acted speech.** Across datasets like RAVDESS and EmoDB, accuracy consistently peaks between Layers 2 and 5. Early layers merely capture raw frequency bounds, while middle layers actively entwine representations of prosody (pitch and rhythm).
2. **Spontaneous conversation peaks earlier.** On raw, conversational datasets like IEMOCAP, the models peak immediately at Layer 1 or 2 as real-world emotion relies heavily on abrupt acoustic fluctuations, not highly structured theatrical contours.
3. **HuBERT retains emotion better than wav2vec 2.0.** HuBERT’s internal offline clustering preserves emotional signals deeply; wav2vec 2.0’s contrastive predictive learning structurally overwrites prosodic dynamics.
4. **Emotional encoding is language-agnostic.** Despite architectures pre-training exclusively on English logic maps, the feature extractions reliably behaved predictably over entirely unseen dialects like Spanish (MESD), German (EmoDB), and Greek (AESDD). 
5. **Whisper is highly robust for real-world dialogue.** Whisper strongly outperformed self-supervised options on spontaneous IEMOCAP interactions, capitalizing on its vast 680,000-hour noise tolerance footprint.

---

## 📉 Supplemental Visualizations

![Accuracy Mean Comparison](docs/Midterm_Report/comparison_accuracy.png)
*(Above): The average probing accuracy of all 6 datasets aggregated onto a singular path. This reinforces the "3-stage topological drop" commonly attributed to contrastive wav2vec encoding arrays compared to flattened HuBERT behavior.*

<img src="docs/Midterm_Report/confusion_matrix.png" width="50%" align="right">

**Per-Emotion Breakdown**
Fear (high-pitch, shaky amplitude) is consistently the easiest categorical target for the layer probes to decipher successfully since its physical signature is profoundly distinct. On the other hand, *Sadness* heavily confuses the acoustic classifiers—often misidentified as *Calm* due to the structurally slow, low-energy vocal bandwidth overlapping heavily between both states.

<br clear="both"/>

![Full 18 Experiments Matrix](docs/Midterm_Report/all_18_3panel.png)

---

## 🚀 Execution & Usage

### 1. Data MLOps & System Config (`config/config.yaml`)
Everything from Batch Sizes, Default Data paths, to active Architectures are configured safely without arbitrary hardcoding inside the YAML definitions file.

### 2. Live Dashboard (FastAPI Web App)
A deeply integrated React-esque Premium UI lets you stream training paths via background terminal rendering and view graphs dynamically.
```bash
uvicorn server.main:app --reload
```
Navigate to `http://localhost:8000/`.

### 3. Headless Pipeline Runs
Drive the entire logic script seamlessly behind the scenes:
```bash
python -m scripts.run_pipeline \
  --task classification \
  --dataset_name RAVDESS \
  --data_dir /path/to/RAVDESS_folder \
  --model_name facebook/wav2vec2-base
```
