import os
import glob
import json
import re
import librosa
from tqdm import tqdm

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
                audio, sr = librosa.load(os.path.join(root,fn), sr=sample_rate)
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
                audio, sr = librosa.load(os.path.join(root,fn), sr=sample_rate)
                if max_length and len(audio) > max_length: audio = audio[:max_length]
                data.append({'file':fn, 'label':EMODB_EMOTION_MAP[ec], 'audio':audio})
            except:
                pass
    return data

def load_iemocap(path, sample_rate=16000, max_length=None):
    # Depending on how it's extracted, we expect a meta_data.json
    data = []
    meta_path = os.path.join(path, 'meta_data.json')
    if not os.path.exists(meta_path):
        print("IEMOCAP meta_data.json not found.")
        return data
        
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    print("Loading IEMOCAP...")
    for entry in tqdm(meta.get('meta_data', []), desc='IEMOCAP'):
        fname = os.path.basename(entry['path'])
        found = glob.glob(f'{path}/**/{fname}', recursive=True)
        if not found: continue
        try:
            audio, sr = librosa.load(found[0], sr=sample_rate)
            if max_length and len(audio) > max_length: audio = audio[:max_length]
            data.append({'file':fname, 'label':entry['label'], 'audio':audio})
        except:
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
                audio, sr = librosa.load(os.path.join(root,fn), sr=sample_rate)
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
            audio, sr = librosa.load(fpath, sr=sample_rate)
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
            audio, sr = librosa.load(fpath, sr=sample_rate)
            if max_length and len(audio) > max_length: audio = audio[:max_length]
            data.append({'file':os.path.basename(fpath), 'label':emotion, 'audio':audio})
        except:
            pass
    return data
