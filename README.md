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

## Instruction: Phase 1 Immediate Steps (Midterm)

For the midterm presentation and report, your primary goal is to **analyze and replicate** what information lives inside the model's layers without reinventing the wheel. The focus is on layer-by-layer linear probing.

### 1. Extract Features Layer-by-Layer
Run datasets (like IEMOCAP) through provided models to extract acoustic features (eGeMAPS via OpenSMILE) and transformer layers. 
- **Wav2Vec2**: Use `--model_name facebook/wav2vec2-base` or `facebook/wav2vec2-large-robust`
- **Whisper Encoder**: Use `--model_name openai/whisper-base` (The pipeline automatically handles log-mel spectrogram pre-processing for you and probes only the encoder hidden states).

**Example Command (Wav2Vec2):**
```bash
python main.py --data_dir data/iemocap_audio --model_name facebook/wav2vec2-base --target_feature F0semitoneFrom27.5Hz_sma3nz_amean --batch_size 4
```

**Example Command (Whisper):**
```bash
python main.py --data_dir data/iemocap_audio --model_name openai/whisper-base --target_feature F0semitoneFrom27.5Hz_sma3nz_amean --batch_size 4
```

### 2. Compare Congruent vs. Incongruent Data
To test where the models capture language versus pure acoustics, you need to test congruent vs. incongruent splits where language perfectly matches emotion vs mismatches it. 
**Steps:**
1. Split your IEMOCAP dataset into two separate directories based on transcriptions:
   - `data/iemocap_congruent/` (e.g., Angry words + Angry emotion)
   - `data/iemocap_incongruent/` (e.g., Happy words + Angry emotion)
2. Run the `main.py` script independently on both folders.
3. Compare the generated CSV results! Check the `Information_Increase_Percent` across layers. If Wav2Vec2 layer 8 relies heavily on language, you will see a massive divergence in RMSE/Increase between the Congruent run and the Incongruent run.

### 3. Focus on the Report
Use the CSVs output by this pipeline to plot graphs for your Midterm. Plot the `Layer` index on the X-axis and `RMSE` on the Y-axis comparing congruent vs incongruent results. That graph explicitly answers the academic requirement!

## Expected Output
A `.csv` file detailing:
1. Model Hidden Layer Index (0th layer = raw input/embeddings)
2. Mean RMSE across the K-fold partitions.
3. Information Increase % (Relative RMSE drop from layer 0).
