# Linguistic-Agnostic SER - Developer Documentation

Welcome to the **Linguistic-Agnostic Speech Emotion Recognition (SER)** repository. This documentation is designed to act as a comprehensive guide for developers, researchers, and engineers who are diving into the codebase for the first time.

## 1. Theoretical Background

### Speech Emotion Recognition
Speech Emotion Recognition (SER) is a discipline of deep learning focused on identifying human emotions (e.g., happiness, anger, sadness) strictly from raw audio signals.

### What is "Layer-wise Probing"?
Historically, SER architectures were built by hand-crafting low-level acoustic features (such as pitch, jitter, and loudness contours) and feeding them into simple classifiers like SVMs.

With the rise of Self-Supervised Learning (SSL) models in audio—such as **wav2vec 2.0**, **HuBERT**, and **Whisper**—transformers can train massive networks on thousands of hours of speech data. However, transformers process language hierarchically:
- **Lower layers** usually capture acoustic representations (pitch, speed, amplitude).
- **Upper layers** often discard this acoustics to focus on abstract linguistic tokens (phonemes, words).

This project uses **layer-wise probing**. We extract the output vector (the "hidden states") from *every* layer of the frozen transformer and train very lightweight statistical models (Logistic Regression or Ridge Regression) specifically on those states. 
By plotting the accuracy of the probe layer-by-layer, we can visualize the "Acoustic Information Flow" and understand exactly where the transformer starts discarding emotion in favor of grammar.

---

## 2. Pipeline Architecture

To maintain a clean MLOps workflow, the logic is highly decoupled into dedicated services inside the `/src` folder.

### 2.1 Data Ingestion (`src/data_ingestion/`)
Since audio datasets are notoriously irregular, this module contains strict loaders (e.g., `load_ravdess()`, `load_emodb()`). It searches raw directory structures, maps unique filenames to generalized emotion targets, and loads the data into synchronized Numpy arrays using `librosa`.

### 2.2 Preprocessing (`src/preprocessing/`)
Handles audio standardization. Using `torchaudio` and `opensmile`, this ensures that every clip is resampled to exactly `16000Hz` and cropped/padded to uniform sequence lengths before hitting the neural network graph.

### 2.3 Feature Extractors (`src/models/`)
The `TransformerExtractor` wrapper takes a valid Hugging Face name string (e.g. `openai/whisper-base`) and pulls the full topological configuration. It freezes the memory parameters immediately (`param.requires_grad = False`), pushes the audio tensor through the model, and slices out all `N` hidden matrices representing the layer depths.

### 2.4 Training Pipeline (`src/pipelines/`)
The extracted layers (shape: `[num_samples, num_layers, hidden_dim]`) are handed to `probe_all_layers()`. It evaluates each matrix using Scikit-Learn tools wrapped in a 5-Fold Stratified Cross Validation to ensure generalized accuracy. It outputs `.csv` files summarizing execution.

---

## 3. Configuration Management

No massive variables should be hardcoded dynamically inside logic layers. All parameters are defined structurally inside the YAML file:
**Location:** `config/config.yaml`

This file determines batch limits, expected model paths across HuggingFace namespaces, supported dataset arrays, random seed variables, and learning definitions. Scripts like `scripts/run_pipeline.py` load this dynamically at execution via `src/utils/config_parser.py`.

---

## 4. Execution & API Guide

### Option 1: Command Line Interface (CLI)
You can directly invoke the modular graph headlessly:
```bash
python -m scripts.run_pipeline \
  --task classification \
  --dataset_name RAVDESS \
  --data_dir /data/RAVDESS_root_dir \
  --model_name facebook/wav2vec2-base 
```
Logs securely print to `stdout` and artifact `.csv` metrics are pushed to `results/csv/`.

### Option 2: Live Dashboard (FastAPI Web App)
You can launch an interactive testing lab to drag-and-drop live evaluations and stream training logs:
```bash
uvicorn server.main:app --reload
```
Navigate to `http://localhost:8000/`. By launching "Start Training" visually in the UI, an internal FastAPI POST router (`/api/train`) securely delegates the exact Command Line invocation using native Python `subprocess.Popen`, capturing its stdout via Server-Sent Events (SSE) back to your browser console window!
