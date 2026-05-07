"""
Run CREMA-D hidden state extraction for the missing models only (Whisper + MERT).

Standalone script for memory-efficient extraction. Each call runs the heavy work in
this single process, then exits so the OS reclaims everything cleanly.

Includes the two model-specific fixes that the notebook also has:
- Whisper needs padding='max_length' (3000 mel frames = 30 sec) instead of padding=True
- MERT was pretrained at 24kHz, so audio must be resampled from 16kHz to 24kHz first

Cached models (wav2vec2, HuBERT, WavLM, w2v-BERT) are skipped automatically.

Run with:
    cd /Users/chaitanyaparwatkar/Desktop/CSCI535-SER-Final/my_code
    .venv/bin/python cremad_finish.py
"""
import os
import gc
import glob
import traceback
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoConfig, AutoFeatureExtractor
import transformers.modeling_utils
from sklearn.preprocessing import LabelEncoder
from scipy.io import wavfile
from scipy.signal import resample
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

if hasattr(transformers.modeling_utils, 'check_torch_load_is_safe'):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None


# Paths
PROJECT_ROOT = '/Users/chaitanyaparwatkar/Desktop/CSCI535-SER-Final'
CREMAD_DIR = os.path.join(PROJECT_ROOT, 'my_code', 'data', 'CREMA-D', 'AudioWAV')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'my_code', 'results', 'cremad')
EMBEDDINGS_DIR = os.path.join(RESULTS_DIR, 'embeddings')
LOG_PATH = os.path.join(RESULTS_DIR, 'cremad_finish_log.txt')

# Models we still need to extract (cached ones are skipped via existence check)
MISSING_MODELS = {
    'Whisper': 'openai/whisper-medium',
    'MERT':    'm-a-p/MERT-v1-330M',
}

CREMAD_EMOTION_MAP = {
    'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear',
    'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad',
}

SAMPLE_RATE = 16000
MERT_SAMPLE_RATE = 24000
# Smaller batch for the heavy models. Whisper pads to 30s and MERT has a beefy frontend.
BATCH_SIZE_PER_MODEL = {
    'Whisper': 2,
    'MERT':    4,
}


# Device
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")


# Audio loader (mirrors cremad.ipynb)
def load_audio_file(filepath, target_sr=16000):
    sr, waveform = wavfile.read(filepath)
    if waveform.dtype == np.int16:
        waveform = waveform.astype(np.float32) / 32768.0
    elif waveform.dtype == np.int32:
        waveform = waveform.astype(np.float32) / 2147483648.0
    else:
        waveform = waveform.astype(np.float32)
        if np.max(np.abs(waveform)) > 1.0:
            waveform = waveform / np.max(np.abs(waveform))
    if len(waveform.shape) > 1 and waveform.shape[1] > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != target_sr:
        num_samples = int(len(waveform) * target_sr / sr)
        waveform = resample(waveform, num_samples)
    return waveform


def load_cremad(path):
    data = []
    files = sorted(glob.glob(os.path.join(path, '*.wav')))
    for fpath in tqdm(files, desc='CREMA-D'):
        fn = os.path.basename(fpath)
        parts = fn.replace('.wav', '').split('_')
        if len(parts) != 4:
            continue
        actor, sentence, ecode, intensity = parts
        if ecode not in CREMAD_EMOTION_MAP:
            continue
        try:
            audio = load_audio_file(fpath)
            data.append({
                'file': fn,
                'label': CREMAD_EMOTION_MAP[ecode],
                'audio': audio,
                'actor': actor,
                'sentence': sentence,
                'intensity': intensity,
            })
        except Exception:
            pass
    return data


class CREMADAudioDataset(Dataset):
    def __init__(self, audios):
        self.audios = audios
    def __len__(self):
        return len(self.audios)
    def __getitem__(self, idx):
        return self.audios[idx]


def collate_pad(batch):
    max_len = max(a.shape[0] for a in batch)
    padded = torch.zeros(len(batch), max_len)
    for i, a in enumerate(batch):
        padded[i, :a.shape[0]] = torch.FloatTensor(a)
    return padded


# Extraction (with the two bug fixes)
def extract_hidden_states(model_key, hf_name, audio_list, batch_size):
    is_whisper = 'whisper' in hf_name.lower()
    is_mert = 'MERT' in hf_name

    config = AutoConfig.from_pretrained(hf_name, trust_remote_code=True)
    config.output_hidden_states = True
    safe = 'MERT' not in hf_name
    model = AutoModel.from_pretrained(
        hf_name, config=config, use_safetensors=safe, trust_remote_code=True
    ).to(DEVICE)
    model.eval()
    fe = AutoFeatureExtractor.from_pretrained(hf_name, trust_remote_code=True)

    # MERT needs 24kHz audio; resample once up-front
    if is_mert:
        target_sr = MERT_SAMPLE_RATE
        audio_list_used = [resample(a, int(len(a) * MERT_SAMPLE_RATE / SAMPLE_RATE)) for a in audio_list]
    else:
        target_sr = SAMPLE_RATE
        audio_list_used = audio_list

    ds = CREMADAudioDataset(audio_list_used)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)

    all_layer_means = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'{model_key}'):
            wav_list = [w.cpu().numpy() for w in batch]

            # Whisper requires padding='max_length' (pads to 3000 mel frames = 30s);
            # padding=True only pads to longest-in-batch which Whisper rejects.
            if is_whisper:
                inputs = fe(wav_list, sampling_rate=target_sr, return_tensors='pt', padding='max_length')
            else:
                inputs = fe(wav_list, sampling_rate=target_sr, return_tensors='pt', padding=True)

            if is_whisper and 'input_features' in inputs:
                model_inputs = {'input_features': inputs['input_features'].to(DEVICE)}
                outputs = model.encoder(**model_inputs)
            elif 'input_values' in inputs:
                model_inputs = {'input_values': inputs['input_values'].to(DEVICE)}
                outputs = model(**model_inputs)
            else:
                model_inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                outputs = model(**model_inputs)

            hidden_states = outputs.hidden_states
            batch_mean = []
            for layer_hs in hidden_states:
                batch_mean.append(layer_hs.mean(dim=1).cpu().numpy())
            batch_stack = np.stack(batch_mean, axis=1)
            all_layer_means.append(batch_stack)

    del model
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    elif DEVICE.type == 'mps':
        torch.mps.empty_cache()

    return np.concatenate(all_layer_means, axis=0)


# Main
def main():
    log_lines = [f"=== CREMA-D finish run started at {datetime.now().isoformat()} ===", ""]

    def flush_log():
        with open(LOG_PATH, 'w') as f:
            f.write('\n'.join(log_lines))

    print(f"\nLoading CREMA-D from {CREMAD_DIR}...")
    data = load_cremad(CREMAD_DIR)
    print(f"Loaded {len(data)} samples")
    log_lines.append(f"Loaded {len(data)} samples")

    audios = [d['audio'] for d in data]
    le = LabelEncoder()
    labels = le.fit_transform([d['label'] for d in data])
    print(f"Classes: {list(le.classes_)}")

    succeeded, failed = [], []

    for model_key, hf_name in MISSING_MODELS.items():
        save_path = os.path.join(EMBEDDINGS_DIR, f'hidden_states_{model_key}_CREMA-D.npy')
        labels_path = os.path.join(EMBEDDINGS_DIR, f'labels_{model_key}_CREMA-D.npy')

        if os.path.exists(save_path) and os.path.exists(labels_path):
            print(f"\nCACHED {model_key}, skipping")
            succeeded.append(model_key)
            log_lines.append(f"CACHED {model_key}")
            continue

        bs = BATCH_SIZE_PER_MODEL.get(model_key, 4)
        print(f"\n{'='*60}")
        print(f"Extracting {model_key}  ({hf_name})  batch_size={bs}")
        print(f"{'='*60}")
        log_lines.append(f"\nSTART {model_key}  batch_size={bs}  at {datetime.now().isoformat()}")

        try:
            hs = extract_hidden_states(model_key, hf_name, audios, batch_size=bs)
            np.save(save_path, hs)
            np.save(labels_path, labels)
            print(f"  Saved: {save_path}  shape={hs.shape}")
            succeeded.append(model_key)
            log_lines.append(f"DONE {model_key}: shape={hs.shape}")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"\n  !!! FAILED {model_key}: {e}")
            traceback.print_exc()
            failed.append((model_key, str(e)))
            log_lines.append(f"FAILED {model_key}: {type(e).__name__}: {e}")
            log_lines.append(tb)

        gc.collect()
        if DEVICE.type == 'mps':
            torch.mps.empty_cache()
        elif DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        flush_log()

    log_lines.append(f"\n=== Run finished at {datetime.now().isoformat()} ===")
    log_lines.append(f"Succeeded: {succeeded}")
    log_lines.append(f"Failed: {[k for k, _ in failed]}")
    flush_log()

    print(f"\n{'='*60}")
    print(f"Succeeded: {succeeded}")
    if failed:
        print(f"Failed: {[k for k, _ in failed]}")
    print(f"Log written to: {LOG_PATH}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
