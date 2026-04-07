# ============================================================
# Aggregate SAVEE, AESDD, MESD datasets for the SER pipeline
# Downloads from Kaggle, extracts, validates, and prints summary
# Target: /scratch1/kelidari/ser_data/{SAVEE,AESDD,MESD}
# ============================================================
import os, re, glob

BASE_DIR = "/scratch1/kelidari/ser_data"
os.makedirs(BASE_DIR, exist_ok=True)

# Ensure kaggle CLI is available
!pip install -q kaggle 2>/dev/null

# --- Dataset definitions ---
DATASETS = {
    "SAVEE": {
        "kaggle_slug": "ejlok1/surrey-audiovisual-expressed-emotion-savee",
        "zip_name": "surrey-audiovisual-expressed-emotion-savee.zip",
    },
    "AESDD": {
        "kaggle_slug": "arie07/acted-emotional-speech-dynamic-database-aesdd",
        "zip_name": "acted-emotional-speech-dynamic-database-aesdd.zip",
    },
    "MESD": {
        "kaggle_slug": "saurabhshahane/mexican-emotional-speech-database-mesd",
        "zip_name": "mexican-emotional-speech-database-mesd.zip",
    },
}

for name, info in DATASETS.items():
    dest = os.path.join(BASE_DIR, name)
    wav_exists = glob.glob(f"{dest}/**/*.wav", recursive=True)
    if wav_exists:
        print(f"✅ {name} already exists ({len(wav_exists)} wav files). Skipping download.")
        continue
    print(f"⬇️  Downloading {name}...")
    zip_path = os.path.join(BASE_DIR, info['zip_name'])
    !kaggle datasets download -d {info['kaggle_slug']} -p {BASE_DIR}
    !unzip -q -o {zip_path} -d {dest}
    !rm -f {zip_path}
    n_wav = len(glob.glob(f"{dest}/**/*.wav", recursive=True))
    print(f"   Extracted {n_wav} wav files to {dest}")

# --- Validation: verify structure matches loader.py expectations ---
print("\n" + "="*60)
print("DATASET VALIDATION SUMMARY")
print("="*60)

# SAVEE: loader expects files like DC_a01.wav with regex ^[A-Z]{2}_[a-z]+\d+\.wav$
SAVEE_MAP = {'a':'anger','d':'disgust','f':'fear','h':'happiness',
             'n':'neutral','sa':'sadness','su':'surprise'}
savee_dir = os.path.join(BASE_DIR, 'SAVEE')
savee_emotions = set()
savee_count = 0
for root, dirs, files in os.walk(savee_dir):
    for fn in files:
        if not fn.endswith('.wav'): continue
        m = re.match(r'^([A-Z]{2})_([a-z]+)(\d+)\.wav$', fn)
        if m and m.group(2) in SAVEE_MAP:
            savee_emotions.add(SAVEE_MAP[m.group(2)])
            savee_count += 1
print(f"\nSAVEE   : {savee_count:5d} valid files, "
      f"{len(savee_emotions)} emotions: {sorted(savee_emotions)}")

# AESDD: loader expects subdirs named by emotion
AESDD_EMOTIONS = ['anger','disgust','fear','happiness','sadness']
aesdd_dir = os.path.join(BASE_DIR, 'AESDD')
aesdd_emotions = set()
aesdd_count = 0
for fpath in glob.glob(f"{aesdd_dir}/**/*.wav", recursive=True):
    parent = os.path.basename(os.path.dirname(fpath)).lower()
    emotion = next((e for e in AESDD_EMOTIONS if e in parent), None)
    if emotion:
        aesdd_emotions.add(emotion)
        aesdd_count += 1
print(f"AESDD   : {aesdd_count:5d} valid files, "
      f"{len(aesdd_emotions)} emotions: {sorted(aesdd_emotions)}")

# MESD: loader matches emotion from parent dir or filename
MESD_EMOTIONS = ['anger','disgust','fear','happiness','neutral','sadness']
mesd_dir = os.path.join(BASE_DIR, 'MESD')
mesd_emotions = set()
mesd_count = 0
for fpath in glob.glob(f"{mesd_dir}/**/*.wav", recursive=True):
    parent = os.path.basename(os.path.dirname(fpath)).lower()
    fn_lower = os.path.basename(fpath).lower()
    emotion = next((e for e in MESD_EMOTIONS if e in parent or e in fn_lower), None)
    if emotion:
        mesd_emotions.add(emotion)
        mesd_count += 1
print(f"MESD    : {mesd_count:5d} valid files, "
      f"{len(mesd_emotions)} emotions: {sorted(mesd_emotions)}")

# Final check
total = savee_count + aesdd_count + mesd_count
print(f"\n{'─'*60}")
print(f"Total   : {total} files across 3 datasets")
if total > 0:
    print("✅ All datasets ready at", BASE_DIR)
else:
    print("⚠️  No valid files found — check Kaggle credentials and paths!")
