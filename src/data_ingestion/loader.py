import os
import glob
import json
import re
from scipy.io import wavfile
from scipy.signal import resample
import numpy as np
from tqdm import tqdm
import warnings
from scipy.io.wavfile import WavFileWarning

warnings.filterwarnings("ignore", category=WavFileWarning)

def load_audio_file(filepath, target_sr):
    # Pure SciPy implementation (Bulletproof! 0 external audio libraries needed)
    sr, waveform = wavfile.read(filepath)
    
    # Normalize to standard float32 between -1.0 and 1.0 (just like librosa)
    if waveform.dtype == np.int16:
        waveform = waveform.astype(np.float32) / 32768.0
    elif waveform.dtype == np.int32:
        waveform = waveform.astype(np.float32) / 2147483648.0
    else:
        waveform = waveform.astype(np.float32)
        if np.max(np.abs(waveform)) > 1.0:
            waveform = waveform / np.max(np.abs(waveform))
            
    # Downmix to mono if stereo
    if len(waveform.shape) > 1 and waveform.shape[1] > 1:
        waveform = np.mean(waveform, axis=1)
        
    # Resample if needed
    if sr != target_sr:
        num_samples = int(len(waveform) * target_sr / sr)
        waveform = resample(waveform, num_samples)
        
    return waveform, target_sr

def load_ravdess(path, sample_rate=16000, max_length=None):
    RAVDESS_EMOTION_MAP = {
        '01':'neutral','02':'calm','03':'happy','04':'sad',
        '05':'angry','06':'fearful','07':'disgust','08':'surprised'
    }
    data = []
    print("Loading RAVDESS...")
    for root, dirs, files in os.walk(path):
        for fn in files:
            if not fn.endswith('.wav'): continue
            parts = fn.replace('.wav','').split('-')
            if len(parts) < 3 or parts[2] not in RAVDESS_EMOTION_MAP: continue
            try:
                audio, sr = load_audio_file(os.path.join(root,fn), sample_rate)
                if max_length and len(audio) > max_length: audio = audio[:max_length]
                data.append({'file':fn, 'label':RAVDESS_EMOTION_MAP[parts[2]], 'audio':audio})
            except Exception as e:
                pass
    return data

def load_emodb(path, sample_rate=16000, max_length=None):
    EMODB_EMOTION_MAP = {'W':'happy','L':'bored','E':'disgust','A':'fearful','F':'happy','T':'sad','N':'neutral'}
    data = []
    print("Loading EmoDB...")
    for root, dirs, files in os.walk(path):
        for fn in files:
            if not fn.endswith('.wav'): continue
            try:
                ec = fn[5]
                if ec not in EMODB_EMOTION_MAP: continue
                audio, sr = load_audio_file(os.path.join(root,fn), sample_rate)
                if max_length and len(audio) > max_length: audio = audio[:max_length]
                data.append({'file':fn, 'label':EMODB_EMOTION_MAP[ec], 'audio':audio})
            except:
                pass
    return data

def load_iemocap(path, sample_rate=16000, max_length=None):
    # Depending on how it's extracted, we expect a meta_data.json
    data = []
    meta_path = os.path.join(path, 'meta_data', 'meta_data.json')
    if not os.path.exists(meta_path):
        meta_path = os.path.join(path, 'meta_data.json')
        if not os.path.exists(meta_path):
            print("IEMOCAP meta_data.json not found.")
            return data
        
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    print("Loading IEMOCAP Path Mapping...")
    # SUPER OPTIMIZATION: Scan directories once and map filenames to paths
    # instead of calling glob(recursive=True) inside the loop.
    path_map = {}
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith('.wav'):
                path_map[f] = os.path.join(root, f)

    print(f"Mapped {len(path_map)} files. Loading audio data...")
    for entry in tqdm(meta.get('meta_data', []), desc='IEMOCAP'):
        fname = os.path.basename(entry['path'])
        fpath = path_map.get(fname)
        if not fpath: continue
        
        try:
            audio, sr = load_audio_file(fpath, sample_rate)
            if max_length and len(audio) > max_length: audio = audio[:max_length]
            data.append({'file':fname, 'label':entry['label'], 'audio':audio})
        except Exception as e:
            # print(f"Error loading {fname}: {e}")
            pass
    return data

def load_savee(path, sample_rate=16000, max_length=None):
    SAVEE_MAP = {'a':'anger','d':'disgust','f':'fear','h':'happiness','n':'neutral','sa':'sadness','su':'surprise'}
    data = []
    print("Loading SAVEE...")
    for root, dirs, files in os.walk(path):
        for fn in files:
            if not fn.endswith('.wav'): continue
            m = re.match(r'^([A-Z]{2})_([a-z]+)(\d+)\.wav$', fn)
            if not m or m.group(2) not in SAVEE_MAP: continue
            try:
                audio, sr = load_audio_file(os.path.join(root,fn), sample_rate)
                if max_length and len(audio) > max_length: audio = audio[:max_length]
                data.append({'file':fn, 'label':SAVEE_MAP[m.group(2)], 'audio':audio})
            except:
                pass
    return data

def load_aesdd(path, sample_rate=16000, max_length=None):
    AESDD_EMOTIONS = ['anger','disgust','fear','happiness','sadness']
    data = []
    print("Loading AESDD...")
    valid_files = glob.glob(f'{path}/**/*.wav', recursive=True)
    for fpath in tqdm(valid_files, desc='AESDD'):
        parent = os.path.basename(os.path.dirname(fpath)).lower()
        emotion = next((e for e in AESDD_EMOTIONS if e in parent), None)
        if not emotion: continue
        try:
            audio, sr = load_audio_file(fpath, sample_rate)
            if max_length and len(audio) > max_length: audio = audio[:max_length]
            data.append({'file':os.path.basename(fpath), 'label':emotion, 'audio':audio})
        except:
            pass
    return data

def load_mesd(path, sample_rate=16000, max_length=None):
    MESD_EMOTIONS = ['anger','disgust','fear','happiness','neutral','sadness']
    data = []
    print("Loading MESD...")
    valid_files = glob.glob(f'{path}/**/*.wav', recursive=True)
    for fpath in tqdm(valid_files, desc='MESD'):
        parent = os.path.basename(os.path.dirname(fpath)).lower()
        fn_lower = os.path.basename(fpath).lower()
        emotion = next((e for e in MESD_EMOTIONS if e in parent or e in fn_lower), None)
        if not emotion: continue
        try:
            audio, sr = load_audio_file(fpath, sample_rate)
            if max_length and len(audio) > max_length: audio = audio[:max_length]
            data.append({'file':os.path.basename(fpath), 'label':emotion, 'audio':audio})
        except:
            pass
    return data

def load_msppodcast(path, sample_rate=16000, max_length=None, num_classes=8):
    import pandas as pd
    data = []
    labels_file = os.path.join(path, "v1", "labels_consensus.csv")
    if not os.path.exists(labels_file):
        print("MSP-Podcast labels_consensus.csv not found.")
        return data
    
    df = pd.read_csv(labels_file)
    # Use only Test1 partition for standard and quick evaluation
    df = df[df['Split_Set'] == 'Test1']
    
    if num_classes == 4:
        MSP_MAP = {
            'A':'anger', 
            'S':'sadness',
            'H':'happiness', 
            'N':'neutral'
        }
    else:
        MSP_MAP = {
            'A':'anger', 
            'S':'sadness',
            'H':'happiness', 
            'U':'surprise',
            'F':'fear',
            'D':'disgust',
            'C':'contempt',
            'N':'neutral'
        }
    
    df = df[df['EmoClass'].isin(MSP_MAP.keys())]
    
    print(f"Loading MSP-Podcast (Test1) with {num_classes} classes... Found {len(df)} files.")
    audio_dir = os.path.join(path, "v1", "Audio")
    for _, row in tqdm(df.iterrows(), total=len(df), desc='MSP-Podcast'):
        fname = row['FileName']
        fpath = os.path.join(audio_dir, fname)
        if not os.path.exists(fpath): continue
        
        try:
            audio, sr = load_audio_file(fpath, sample_rate)
            if max_length and len(audio) > max_length: audio = audio[:max_length]
            data.append({'file':fname, 'label':MSP_MAP[row['EmoClass']], 'audio':audio})
        except Exception as e:
            pass
    return data

def load_cremad(path, sample_rate=16000, max_length=None):
    """CREMA-D: 91 actors x 12 fixed sentences x 6 emotions (7,442 clips).

    Filenames are ``ActorID_SentenceCode_Emotion_Intensity.wav`` e.g.
    ``1001_DFA_ANG_XX.wav``. Because the 12 carrier sentences are fixed, the
    text channel is held constant across emotions — any recovered emotion signal
    is necessarily acoustic. Returns extra ``actor``/``sentence``/``intensity``
    metadata used by the lexicon-controlled SAE analysis.
    """
    CREMAD_EMOTION_MAP = {
        'ANG': 'anger', 'DIS': 'disgust', 'FEA': 'fear',
        'HAP': 'happiness', 'NEU': 'neutral', 'SAD': 'sadness',
    }
    data = []
    print("Loading CREMA-D...")
    audio_root = path
    # Canonical layout from CheyneyComputerScience/CREMA-D is <path>/AudioWAV.
    if os.path.isdir(os.path.join(path, 'AudioWAV')):
        audio_root = os.path.join(path, 'AudioWAV')
    for root, dirs, files in os.walk(audio_root):
        for fn in files:
            if not fn.endswith('.wav'):
                continue
            parts = fn.replace('.wav', '').split('_')
            if len(parts) < 4 or parts[2] not in CREMAD_EMOTION_MAP:
                continue
            actor, sentence, emo, intensity = parts[0], parts[1], parts[2], parts[3]
            try:
                audio, sr = load_audio_file(os.path.join(root, fn), sample_rate)
                if max_length and len(audio) > max_length:
                    audio = audio[:max_length]
                data.append({
                    'file': fn,
                    'label': CREMAD_EMOTION_MAP[emo],
                    'audio': audio,
                    'actor': actor,
                    'sentence': sentence,
                    'intensity': intensity,
                })
            except Exception:
                pass
    return data


def load_emis(path, sample_rate=16000, max_length=None):
    """EMIS: Emotionally Incongruent Synthetic Speech (Correa et al., 2025).

    1,248 TTS clips where the TEXT emotion and the AUDIO (prosodic) emotion are
    set independently. Filenames:
    ``{sentence-ID}_{text-emo}_{explicit|implicit}_{audio-emo}_{ESD-spk}_{TTS}.wav``
    e.g. ``01_angry_explicit_happy_0012_COSY.wav``. ``label`` is the AUDIO
    (prosodic) emotion (the SER target); ``text_label`` is the spoken-text
    emotion; ``explicit`` flags whether an emotion word appears in the text.
    """
    data = []
    print("Loading EMIS...")
    for root, dirs, files in os.walk(path):
        for fn in files:
            if not fn.endswith('.wav'):
                continue
            parts = os.path.splitext(fn)[0].split('_')
            if len(parts) < 5:
                continue
            sentence_id, text_emo = parts[0], parts[1]
            if parts[2] in ('explicit', 'implicit'):           # emotion-rich text carries this tag
                expl, audio_emo, esd_spk = parts[2], parts[3], parts[4]
                tts = '_'.join(parts[5:])
                explicit_flag = expl == 'explicit'
            else:                                              # neutral text: no explicit/implicit token
                audio_emo, esd_spk = parts[2], parts[3]
                tts = '_'.join(parts[4:])
                explicit_flag = False
            try:
                audio, sr = load_audio_file(os.path.join(root, fn), sample_rate)
                if max_length and len(audio) > max_length:
                    audio = audio[:max_length]
                data.append({
                    'file': fn,
                    'label': audio_emo.lower(),          # prosodic / audio emotion
                    'text_label': text_emo.lower(),      # spoken-text emotion
                    'audio': audio,
                    'actor': esd_spk,
                    'sentence': sentence_id,
                    'tts': tts,
                    'explicit': explicit_flag,
                    'congruent': text_emo.lower() == audio_emo.lower(),
                })
            except Exception:
                pass
    return data


def load_msppodcast_dimensional(path, target, sample_rate=16000, max_length=None):
    import pandas as pd
    data = []
    labels_file = os.path.join(path, "v1", "labels_consensus.csv")
    if not os.path.exists(labels_file):
        print("MSP-Podcast labels_consensus.csv not found.")
        return data
    
    df = pd.read_csv(labels_file)
    # Use only Test1 partition for evaluation (same as categorical loading)
    df = df[df['Split_Set'] == 'Test1']
    
    # Ensure the target column exists and contains valid numeric data
    if target not in df.columns:
        print(f"Target column {target} not found in MSP-Podcast labels.")
        return data
        
    df = df.dropna(subset=[target])
    
    print(f"Loading MSP-Podcast Dimensional (Test1) for target {target}... Found {len(df)} files.")
    audio_dir = os.path.join(path, "v1", "Audio")
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f'MSP-Podcast ({target})'):
        fname = row['FileName']
        fpath = os.path.join(audio_dir, fname)
        if not os.path.exists(fpath): continue
        
        try:
            audio, sr = load_audio_file(fpath, sample_rate)
            if max_length and len(audio) > max_length: audio = audio[:max_length]
            data.append({'file':fname, 'label':float(row[target]), 'audio':audio})
        except Exception as e:
            pass
    return data
