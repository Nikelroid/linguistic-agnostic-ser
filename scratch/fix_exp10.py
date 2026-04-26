import nbformat

notebook_path = 'linguistic-agnostic-ser.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

exp10_code_full = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup directories
EXP = 10
summary_dir = Path(f"results/EXP{EXP}/summary")
output_dir = summary_dir / "plots"
output_dir.mkdir(parents=True, exist_ok=True)

models = ['wav2vec2', 'HuBERT', 'Whisper', 'w2v-BERT', 'WavLM', 'MERT']
targets = ['EmoVal', 'EmoAct', 'EmoDom']

model_titles = {
    'wav2vec2': 'wav2vec 2.0', 
    'HuBERT': 'HuBERT', 
    'Whisper': 'Whisper',
    'w2v-BERT': 'w2v-BERT',
    'WavLM': 'WavLM',
    'MERT': 'MERT' 
}

# 1. Load all summary data
all_data = [pd.read_csv(summary_dir / f"csv/{m}_summary.csv").assign(Model=m) 
            for m in models if (summary_dir / f"csv/{m}_summary.csv").exists()]

if all_data:
    full_df = pd.concat(all_data, ignore_index=True)

    # Enforce correct chronological order for the layers and SNRs
    layer_order = ['CNN'] + [f'Layer {i}' for i in range(1, 25)]
    full_df['Layer'] = pd.Categorical(full_df['Layer'], categories=layer_order, ordered=True)
    
    # SNR order from clean/highest to lowest
    snr_order = ['clean', '20', '10', '5', '0']
    full_df['SNR'] = pd.Categorical(full_df['SNR'].astype(str), categories=snr_order, ordered=True)

    sns.set_theme(style="whitegrid")
    snr_palette = sns.color_palette("rocket_r", n_colors=len(snr_order))

    # ---------------------------------------------------------
    # Plot 1: 6-panel line plot per target (RMSE across layers by SNR)
    # ---------------------------------------------------------
    for target in targets:
        target_df = full_df[full_df['Target'] == target]
        if target_df.empty: continue
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharey=True)
        
        for ax, model in zip(axes.flatten(), models):
            model_data = target_df[target_df['Model'] == model]
            if model_data.empty: continue
            
            sns.lineplot(data=model_data, x='Layer', y='RMSE', hue='SNR', 
                         marker='o', linewidth=2, ax=ax, palette=snr_palette, hue_order=snr_order) 
            
            ax.set_title(f"{model_titles.get(model, model)} ({target})", fontsize=14, fontweight='bold')
            ax.set_xlabel("Layer", fontsize=12)
            if ax in axes[:, 0]: ax.set_ylabel("RMSE", fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.legend(title="SNR (dB)", loc='upper right', fontsize=9)

        plt.tight_layout()
        fig.savefig(output_dir / f"rmse_decay_{target}.png", dpi=300)
        plt.show()

    # ---------------------------------------------------------
    # Plot 2: Noise Degradation Slopes (Median RMSE vs SNR)
    # ---------------------------------------------------------
    fig2 = plt.figure(figsize=(10, 6))
    
    # Calculate median RMSE across all layers and targets for each Model and SNR
    median_degradation = full_df.groupby(['Model', 'SNR'], observed=True)['RMSE'].median().reset_index()
    
    sns.lineplot(data=median_degradation, x='SNR', y='RMSE', hue='Model', marker='s', linewidth=3, markersize=10)
    plt.title("Overall RMSE Degradation Across SNRs (All Targets)", fontsize=16, fontweight='bold')
    plt.xlabel("SNR Level (dB)", fontsize=14)
    plt.ylabel("Median RMSE", fontsize=14)
    plt.legend(title="Model", loc='upper left')
    
    # Reverse x-axis so it goes from clean -> 0dB
    ax = plt.gca()
    ax.invert_xaxis()
    
    plt.tight_layout()
    fig2.savefig(output_dir / "rmse_degradation_slopes.png", dpi=300)
    plt.show()

    # ---------------------------------------------------------
    # Plot 3: 0dB Robustness Bar Chart
    # ---------------------------------------------------------
    fig3 = plt.figure(figsize=(12, 6))
    
    noisy_df = full_df[full_df['SNR'] == '0']
    if not noisy_df.empty:
        sns.barplot(data=noisy_df, x='Model', y='RMSE', hue='Target', palette='viridis')
        plt.title("Model Robustness at Absolute Worst Noise (0dB)", fontsize=16, fontweight='bold')
        plt.xlabel("Model", fontsize=14)
        plt.ylabel("Mean RMSE at 0dB", fontsize=14)
        plt.legend(title="Target", loc='upper left')
        
        plt.tight_layout()
        fig3.savefig(output_dir / "robustness_at_0db.png", dpi=300)
        plt.show()

    # ---------------------------------------------------------
    # Plot 4: Heatmap of Model vs SNR Performance
    # ---------------------------------------------------------
    fig4 = plt.figure(figsize=(10, 8))
    
    heatmap_data = full_df.pivot_table(index='Model', columns='SNR', values='RMSE', aggfunc='median', observed=True)
    sns.heatmap(heatmap_data, cmap="rocket_r", annot=True, fmt=".3f", cbar_kws={'label': 'Median RMSE'})
    
    plt.title("Heatmap: Median RMSE across Models & SNR Levels", fontsize=16, fontweight='bold')
    plt.xlabel("SNR Level (dB)", fontsize=14)
    plt.ylabel("Model", fontsize=14)
    
    plt.tight_layout()
    fig4.savefig(output_dir / "model_snr_heatmap.png", dpi=300)
    plt.show()

    # ---------------------------------------------------------
    # Export Full Statistical Analysis to TXT
    # ---------------------------------------------------------
    txt_output_path = summary_dir / "statistical_analysis.txt"

    with open(txt_output_path, "w") as f:
        f.write("=== EXP10: NOISY DIMENSIONAL REGRESSION FULL ANALYSIS ===\\n\\n")
        
        # 1. Overall Model Performance (Mean, Std, Min for RMSE) grouped by Target and SNR
        f.write("--- 1. OVERALL PERFORMANCE BY SNR & TARGET (RMSE) ---\\n")
        snr_stats = full_df.groupby(['Target', 'SNR'], observed=True)[['RMSE']].agg(['mean', 'std', 'min']).round(4)
        f.write(snr_stats.to_string())
        f.write("\\n\\n")
        
        # 2. Peak Performance: Best Layer Per Model per Target at 0dB (Most severe noise)
        f.write("--- 2. BEST LAYER PER MODEL AT 0dB NOISE (Min RMSE) ---\\n")
        if not noisy_df.empty:
            best_rmse_idx = noisy_df.groupby(['Model', 'Target'])['RMSE'].idxmin()
            best_layers = noisy_df.loc[best_rmse_idx, ['Model', 'Target', 'Layer', 'RMSE']]
            f.write(best_layers.to_string(index=False))
        else:
            f.write("No 0dB results found yet.")
        f.write("\\n\\n")
        
        # 3. Overall Model Resilience Score (0dB vs Clean Delta)
        f.write("--- 3. NOISE RESILIENCE (Delta RMSE: 0dB - Clean) ---\\n")
        if 'clean' in full_df['SNR'].values and '0' in full_df['SNR'].values:
            clean_perf = full_df[full_df['SNR'] == 'clean'].groupby('Model')['RMSE'].median()
            noisy_perf = full_df[full_df['SNR'] == '0'].groupby('Model')['RMSE'].median()
            resilience = (noisy_perf - clean_perf).sort_values(ascending=True)
            resilience.name = "Delta_RMSE"
            f.write(resilience.to_string())
            f.write("\\n(Lower delta means the model is more resilient to noise)\\n")
        else:
            f.write("Requires both 'clean' and '0' SNR to compute resilience delta.\\n")
        
    print(f"Statistical analysis successfully saved to: {txt_output_path}")
else:
    print("No summary data found for EXP10.")
"""

for cell in nb.cells:
    if cell.cell_type == 'code':
        if "EXP = 10" in cell.source and "Generate one 6-panel plot" in cell.source:
            cell.source = exp10_code_full

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("EXP10 cells successfully replaced with full multi-plot analysis.")
