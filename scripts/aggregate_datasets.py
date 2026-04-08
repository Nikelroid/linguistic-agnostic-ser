#!/usr/bin/env python3
"""
Aggregate SAVEE, AESDD, MESD, RAVDESS, EmoDB, and IEMOCAP 
for the SER pipeline. Standalone Python version.
"""
import os
import re
import glob
import json
import shutil
import subprocess
import argparse

def run_cmd(cmd):
    """Wait for and run a shell command."""
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    parser = argparse.ArgumentParser(description="SER Dataset Aggregator")
    current_user = os.environ.get('USER', 'user')
    default_base = f"/scratch1/{current_user}/ser_data"
    parser.add_argument("--base-dir", type=str, default=default_base,
                        help="Target directory for aggregation")
    parser.add_argument("--iemocap-source", type=str, default="/project2/msoleyma_1026/IEMOCAP_full_release",
                        help="Root IEMOCAP directory on cluster")
    args = parser.parse_args()

    BASE_DIR = args.base_dir
    IEMOCAP_SOURCE = args.iemocap_source
    os.makedirs(BASE_DIR, exist_ok=True)

    # --- Dataset definitions (Kaggle) ---
    DATASETS = {
        "SAVEE": {
            "slug": "ejlok1/surrey-audiovisual-expressed-emotion-savee",
            "zip": "surrey-audiovisual-expressed-emotion-savee.zip",
        },
        "AESDD": {
            "slug": "arie07/acted-emotional-speech-dynamic-database-aesdd",
            "zip": "acted-emotional-speech-dynamic-database-aesdd.zip",
        },
        "MESD": {
            "slug": "saurabhshahane/mexican-emotional-speech-database-mesd",
            "zip": "mexican-emotional-speech-database-mesd.zip",
        },
        "RAVDESS": {
            "slug": "uwrfkaggler/ravdess-emotional-speech-audio",
            "zip": "ravdess-emotional-speech-audio.zip",
        },
        "EmoDB": {
            "slug": "piyushagni5/berlin-database-of-emotional-speech-emodb",
            "zip": "berlin-database-of-emotional-speech-emodb.zip",
        },
    }

    # 1. Handle Kaggle Datasets
    for name, info in DATASETS.items():
        dest = os.path.join(BASE_DIR, name)
        if glob.glob(f"{dest}/**/*.wav", recursive=True):
            print(f"✅ {name} already exists. Skipping.")
            continue
        
        print(f"⬇️  Downloading {name}...")
        zip_path = os.path.join(BASE_DIR, info['zip'])
        try:
            run_cmd(f"kaggle datasets download -d {info['slug']} -p {BASE_DIR}")
            run_cmd(f"unzip -q -o {zip_path} -d {dest}")
            run_cmd(f"rm -f {zip_path}")
        except Exception as e:
            print(f"❌ Error downloading {name}: {e}")

    # 2. Handle IEMOCAP (Local Restructure)
    IEMOCAP_DEST = os.path.join(BASE_DIR, "IEMOCAP")
    if os.path.exists(os.path.join(IEMOCAP_DEST, "meta_data.json")):
        print(f"✅ IEMOCAP already exists in {IEMOCAP_DEST}. Skipping migration.")
    else:
        if os.path.exists(IEMOCAP_SOURCE):
            print(f"🔄 Migrating IEMOCAP from {IEMOCAP_SOURCE}...")
            os.makedirs(IEMOCAP_DEST, exist_ok=True)
            
            # Copy meta_data
            src_meta = os.path.join(IEMOCAP_SOURCE, "meta_data")
            dst_meta = os.path.join(IEMOCAP_DEST, "meta_data")
            if os.path.exists(src_meta):
                shutil.copytree(src_meta, dst_meta, dirs_exist_ok=True)
                if os.path.exists(os.path.join(src_meta, "meta_data.json")):
                    shutil.copy(os.path.join(src_meta, "meta_data.json"), os.path.join(IEMOCAP_DEST, "meta_data.json"))
            
            # Copy Sessions 1-5
            for i in range(1, 6):
                for sub, alias in [("dialog", "dialoge"), ("sentences", "sentence")]:
                    src_wav = os.path.join(IEMOCAP_SOURCE, f"Session{i}", sub, "wav")
                    dst_wav = os.path.join(IEMOCAP_DEST, f"wav-session{i}-{alias}")
                    if os.path.exists(src_wav):
                        shutil.copytree(src_wav, dst_wav, dirs_exist_ok=True)
            print(f"✅ IEMOCAP migration complete.")
        else:
            print(f"⚠️  IEMOCAP source not found at {IEMOCAP_SOURCE}. Skipping.")

    # 3. Final Validation Summary
    print("\n" + "="*60)
    print("DATASET VALIDATION SUMMARY")
    print("="*60)

    # Validation logic helper
    def count_valid(path, glob_pattern=None, regex=None, labels_map=None):
        count = 0
        emotions = set()
        if glob_pattern:
            files = glob.glob(os.path.join(path, glob_pattern), recursive=True)
            for fpath in files:
                m = None
                if regex:
                    m = re.match(regex, os.path.basename(fpath))
                
                # Check labels
                if labels_map and m:
                    emo = labels_map.get(m.group(2))
                    if emo:
                        emotions.add(emo)
                        count += 1
                elif labels_map is None: # Simple glob check
                    count += 1
        return count, emotions

    # SAVEE
    SAVEE_MAP = {'a':'anger','d':'disgust','f':'fear','h':'happiness','n':'neutral','sa':'sadness','su':'surprise'}
    c_savee, e_savee = count_valid(os.path.join(BASE_DIR, 'SAVEE'), "**/*.wav", r'^([A-Z]{2})_([a-z]+)(\d+)\.wav$', SAVEE_MAP)
    print(f"SAVEE   : {c_savee:5d} valid files, {len(e_savee)} emotions")

    # AESDD
    c_aesdd = len(glob.glob(os.path.join(BASE_DIR, 'AESDD', "**/*.wav"), recursive=True))
    print(f"AESDD   : {c_aesdd:5d} total wav files")

    # MESD
    c_mesd = len(glob.glob(os.path.join(BASE_DIR, 'MESD', "**/*.wav"), recursive=True))
    print(f"MESD    : {c_mesd:5d} total wav files")

    # RAVDESS 
    c_ravdess = len(glob.glob(os.path.join(BASE_DIR, 'RAVDESS', "**/*.wav"), recursive=True))
    print(f"RAVDESS : {c_ravdess:5d} total wav files")

    # EmoDB
    c_emodb = len(glob.glob(os.path.join(BASE_DIR, 'EmoDB', "**/*.wav"), recursive=True))
    print(f"EmoDB   : {c_emodb:5d} total wav files")

    # IEMOCAP
    iem_count = 0
    iem_meta = os.path.join(IEMOCAP_DEST, "meta_data.json")
    if os.path.exists(iem_meta):
        with open(iem_meta, 'r') as f:
            meta = json.load(f)
            iem_count = len(meta.get('meta_data', []))
    print(f"IEMOCAP : {iem_count:5d} files from metadata")

    total = c_savee + c_aesdd + c_mesd + c_ravdess + c_emodb + iem_count
    print(f"\nFinal Totals: {total} files across all datasets.")
    print(f"Data location: {BASE_DIR}")

if __name__ == "__main__":
    main()
