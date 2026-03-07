# Linguistic-Agnostic SER: A Probing Approach to Acoustic Emotion Encoding

This repository contains the core implementation for probing Speech Emotion Recognition (SER) transformers to detect how paralinguistic, acoustic knowledge is encoded across their layers. The code maps the layer-stratified acoustic hierarchy, inspired by the reference paper *Probing Speech Emotion Recognition Transformers for Linguistic Knowledge*.

## Objective
Modern SER systems rely on large transformers (e.g., wav2vec 2.0, HuBERT) which often leverage implicit linguistic cues. This repository explicitly probes intermediate hidden states of these frozen models to predict handcrafted acoustic metrics (like pitch, jitter, and loudness) to evaluate *where* and *how* acoustic signals are retained or discarded.

## Directory Structure
- `data_loader.py`: Handles audio dataset loading, resampling, scaling, and precise cropping/padding. Also includes the OpenSMILE `FeatureExtractor` to pull baseline targets (eGeMAPS).
- `model_extractor.py`: Utility module wrapping Hugging Face `transformers` to isolate, freeze, and extract intermediate hidden states of a chosen pre-trained model.
- `probing.py`: The evaluation logic block implementing lightweight probes (e.g., Ridge Regression) with K-Fold Cross Validation.
- `main.py`: The main orchestrator connecting data loading, acoustic feature extraction, hidden state extraction, and layer-wise probing.
- `utils.py`: Analytical helpers, for example `calculate_information_increase()`.

## Setup & Requirements

1. **Python Environment**: Ensure you are using Python 3.8+
2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

### Required Toolkits (Included in requirements.txt):
- `torch`, `torchaudio`: For neural operations and processing audio tensors.
- `transformers`: For Hugging Face pre-trained self-supervised models.
- `opensmile`: For classical acoustic feature extraction.
- `scikit-learn`: For probes (Ridge Regression/Logistic Regression).
- `pandas`, `numpy`: Data manipulation.

## Data Preparation
Create a directory to place your `.wav` files. The data loader attempts to find all audio samples in that directory and aligns the outputs with the OpenSMILE extraction sequences. Common datasets to request or download:
- RAVDESS
- SAVEE
- IEMOCAP

## Running the Pipeline

Run the probing evaluation logic over your given data directory:

```bash
python main.py --data_dir data/my_audio_files --model_name facebook/wav2vec2-base --target_feature F0semitoneFrom27.5Hz_sma3nz_amean --batch_size 4
```

### Arguments:
- `--data_dir`: Directory containing your standard audio samples.
- `--model_name`: The HF pre-trained model path (e.g., `facebook/hubert-base-ls960`, `microsoft/wavlm-base`).
- `--target_feature`: The exact eGeMAPS acoustic feature to use as your objective reference.
- `--batch_size`: Control the memory use during the extraction process. Requires enough RAM/VRAM to buffer extracted tensors logic (which could be extensive for huge models).

## Expected Output
A `.csv` file detailing:
1. Model Hidden Layer Index (0th layer = raw input/embeddings)
2. Mean RMSE across the K-fold partitions.
3. Information Increase % (Relative RMSE drop from layer 0).

You can easily visualize these CSV trends across models to interpret linguistic or acoustic layer drops as stated in the baseline academic literature.
