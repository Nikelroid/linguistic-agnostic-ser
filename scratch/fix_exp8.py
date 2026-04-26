import nbformat
import re

notebook_path = 'linguistic-agnostic-ser.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

exp8_code_full = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup directories
EXP = 8
summary_dir = Path(f"results/EXP{EXP}/summary")
output_dir = Path(f"results/EXP{EXP}/summary/plots")
output_dir.mkdir(parents=True, exist_ok=True)

models = ['wav2vec2', 'HuBERT', 'Whisper', 'w2v-BERT', 'WavLM', 'MERT']

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

    # Enforce correct chronological order for the layers
    layer_order = ['CNN'] + [f'Layer {i}' for i in range(1, 25)]
    full_df['Layer'] = pd.Categorical(full_df['Layer'], categories=layer_order, ordered=True)

    sns.set_theme(style="whitegrid")

    targets = full_df['Target'].unique()
    target_colors = dict(zip(targets, sns.color_palette("tab10", n_colors=len(targets))))

    # ---------------------------------------------------------
    # Plot 1: 3-Panel Line Plot (RMSE across Layers)
    # ---------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 12), sharey=True)

    for ax, model in zip(axes1.flatten(), models):
        model_data = full_df[full_df['Model'] == model]
        if model_data.empty: continue
        
        sns.lineplot(data=model_data, x='Layer', y='RMSE', hue='Target', 
                     marker='o', linewidth=2, ax=ax, 
                     palette=target_colors, hue_order=targets) 
        
        ax.set_title(model_titles.get(model, model), fontsize=14, fontweight='bold')
        ax.set_xlabel("Layer", fontsize=12)
        if ax in axes1[:, 0]:  # Only add y-label to the first column
            ax.set_ylabel("RMSE", fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title="Target", loc='upper right', fontsize=9)

    plt.tight_layout()
    fig1.savefig(output_dir / "all_models_rmse_3panel.png", dpi=300)
    plt.show()

    # ---------------------------------------------------------
    # Plot 2: Comprehensive Heatmap (All Models & Targets)
    # ---------------------------------------------------------
    fig2 = plt.figure(figsize=(12, 8))
    full_df['Model_Target'] = full_df['Target'] + " - " + full_df['Model']
    # If there are duplicate layers per Model_Target, we pivot using mean
    heatmap_data = full_df.pivot_table(index='Model_Target', columns='Layer', values='RMSE', aggfunc='mean')

    sns.heatmap(heatmap_data, cmap="rocket_r", annot=True, fmt=".3f", 
                cbar_kws={'label': 'RMSE'}, linewidths=.5,
                annot_kws={"size": 6}) 

    plt.title("Layer-wise RMSE Heatmap across Models and Targets", fontsize=14, fontweight='bold')
    plt.xlabel("Layer representation", fontsize=12)
    plt.ylabel("Model & Target", fontsize=12)
    plt.xticks(rotation=45)

    plt.tight_layout()
    fig2.savefig(output_dir / "rmse_heatmap.png", dpi=300)
    plt.show()

    # ---------------------------------------------------------
    # Plot 3: Median RMSE Comparison (All Models)
    # ---------------------------------------------------------
    fig3 = plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=full_df, 
        x='Layer', 
        y='RMSE', 
        hue='Model',
        estimator=np.median, 
        errorbar=None, 
        marker='o', 
        linewidth=2.5
    )

    plt.title("Median RMSE across Targets for Each Model", fontsize=14, fontweight='bold')
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Median RMSE", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title="Model", loc='upper right')

    plt.tight_layout()
    fig3.savefig(output_dir / "median_rmse_comparison.png", dpi=300)
    plt.show()

    # ---------------------------------------------------------
    # Export Full Statistical Analysis to TXT
    # ---------------------------------------------------------
    txt_output_path = summary_dir / "statistical_analysis.txt"

    with open(txt_output_path, "w") as f:
        f.write("=== EXP8: DIMENSIONAL REGRESSION ANALYSIS ===\\n\\n")
        
        # 1. General Overview
        f.write("--- 1. GENERAL OVERVIEW ---\\n")
        f.write(f"Total Records: {len(full_df)}\\n")
        f.write(f"Models Evaluated: {', '.join(full_df['Model'].unique())}\\n")
        f.write(f"Targets Evaluated: {', '.join(full_df['Target'].unique())}\\n\\n")
        
        # 2. Overall Model Performance (Mean, Std, Min for RMSE)
        f.write("--- 2. OVERALL MODEL STATISTICS (RMSE) ---\\n")
        model_stats = full_df.groupby('Model')[['RMSE']].agg(['mean', 'std', 'min', 'max']).round(4)
        f.write(model_stats.to_string())
        f.write("\\n\\n")
        
        # 3. Peak Performance: Best Layer Per Model (by min RMSE)
        f.write("--- 3. BEST LAYER PER MODEL (Min RMSE) ---\\n")
        best_rmse_idx = full_df.groupby('Model')['RMSE'].idxmin()
        best_layers = full_df.loc[best_rmse_idx, ['Model', 'Layer', 'Target', 'RMSE']]
        f.write(best_layers.to_string(index=False))
        f.write("\\n\\n")
        
        # 4. Target Benchmarks
        f.write("--- 4. TARGET PERFORMANCE (Averaged across all models) ---\\n")
        target_stats = full_df.groupby('Target')[['RMSE']].mean().sort_values(by='RMSE', ascending=True).round(4)
        f.write(target_stats.to_string())
        f.write("\\n\\n")

    print(f"Statistical analysis successfully saved to: {txt_output_path}")
else:
    print("No summary data found for EXP8.")
"""

# Replace the EXP8 and EXP10 cells to ensure R2 is stripped completely
for cell in nb.cells:
    if cell.cell_type == 'code':
        # Safely strip any lingering R2_Score text from EXP8 or EXP10
        cell.source = cell.source.replace("['RMSE', 'R2_Score']", "['RMSE']")
        cell.source = cell.source.replace(", 'R2_Score'", "")
        cell.source = cell.source.replace(" (RMSE & R2)", " (RMSE)")

        # Completely replace the EXP8 plotting cell with the massive multi-plot one
        if "Plot 1: 3-Panel Line Plot (RMSE across Layers)" in cell.source and "EXP = 8" in cell.source:
            cell.source = exp8_code_full

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("EXP8 cells successfully replaced with full multi-plot analysis and R2 stripped.")
