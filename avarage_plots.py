import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the compiled comparison data
df = pd.read_csv('probing_results_comparison.csv')

# Calculate the average across the 3 datasets for each model
wav2vec2_cols = [col for col in df.columns if 'wav2vec2' in col]
hubert_cols = [col for col in df.columns if 'HuBERT' in col]

df['wav2vec2_Avg'] = df[wav2vec2_cols].mean(axis=1)
df['HuBERT_Avg'] = df[hubert_cols].mean(axis=1)

# ==========================================
# Plot Option 1: Comparative Line Plot
# (Best for showing the layer-by-layer trend)
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(df['Layer'], df['wav2vec2_Avg'], label='wav2vec2 Average', color='#1f77b4', marker='o', linestyle='-', linewidth=2.5, markersize=8)
plt.plot(df['Layer'], df['HuBERT_Avg'], label='HuBERT Average', color='#ff7f0e', marker='s', linestyle='--', linewidth=2.5, markersize=8)

plt.xlabel('Transformer Layer', fontsize=12, fontweight='bold')
plt.ylabel('Average Accuracy', fontsize=12, fontweight='bold')
plt.title('Average Probing Performance: wav2vec2 vs HuBERT', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('model_averages_lineplot.png', dpi=300)

# ==========================================
# Plot Option 2: Grouped Bar Chart
# (Best for direct side-by-side magnitude comparison)
# ==========================================
plt.figure(figsize=(12, 6))
x = np.arange(len(df['Layer']))
width = 0.35

# Plot bars with edgecolors for crisp rendering in PDFs
plt.bar(x - width/2, df['wav2vec2_Avg'], width, label='wav2vec2', color='#1f77b4', edgecolor='black', alpha=0.8)
plt.bar(x + width/2, df['HuBERT_Avg'], width, label='HuBERT', color='#ff7f0e', edgecolor='black', alpha=0.8)

plt.xlabel('Transformer Layer', fontsize=12, fontweight='bold')
plt.ylabel('Average Accuracy', fontsize=12, fontweight='bold')
plt.title('Average Probing Performance per Layer: wav2vec2 vs HuBERT', fontsize=14, fontweight='bold')
plt.xticks(x, df['Layer'], rotation=45)
plt.legend(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Clip the y-axis at 0.60 to visually emphasize the differences between models
plt.ylim(0.60, max(df['wav2vec2_Avg'].max(), df['HuBERT_Avg'].max()) + 0.05)

plt.tight_layout()
plt.savefig('model_averages_barplot.png', dpi=300)