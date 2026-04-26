import nbformat

notebook_path = 'linguistic-agnostic-ser.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

exp10_code_agg = """import pandas as pd
from pathlib import Path
import re

EXP8 = 8
EXP10 = 10
# Define directories
input_dir_8 = Path(f"results/EXP{EXP8}/csv")
input_dir_10 = Path(f"results/EXP{EXP10}/csv")

output_dir = Path(f"results/EXP{EXP10}/summary/csv")
output_dir.mkdir(parents=True, exist_ok=True)

models = ['wav2vec2', 'HuBERT', 'Whisper', 'w2v-BERT','WavLM','MERT']
targets = ['EmoVal', 'EmoAct', 'EmoDom']

for model in models:
    files_8 = list(input_dir_8.glob(f"*_{model}_*.csv"))
    files_10 = list(input_dir_10.glob(f"*_{model}_*.csv"))
    all_files = files_8 + files_10
    
    if not all_files:
        continue
        
    df_list = []
    for f in all_files:
        parts = f.stem.split('_')
        
        target = None
        snr = None
        for t in targets:
            if t in parts:
                target = t
                break
                
        # Determine SNR
        if "clean" in parts:
            snr = "clean"
        else:
            for p in parts:
                if p.startswith("SNR"):
                    snr = p.replace("SNR", "")
                    break
                
        if target and snr:
            df_list.append(pd.read_csv(f).assign(Target=target, SNR=snr))
        
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        # Save the summarized model data
        combined_df.to_csv(output_dir / f"{model}_summary.csv", index=False)
        print(f"Saved {model}_summary.csv with {len(all_files)} files (including clean).")
"""

# Replace the aggregation block
for cell in nb.cells:
    if cell.cell_type == 'code':
        if "input_dir = Path(f\"results/EXP{EXP}/csv\")" in cell.source and "EXP = 10" in cell.source:
            cell.source = exp10_code_agg

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("EXP10 aggregation cell successfully replaced to include clean data from EXP8.")
