import pandas as pd
import matplotlib.pyplot as plt

# Load the compiled comparison data
df = pd.read_csv('probing_results_comparison.csv')

# ==========================================
# Plot 1: Accuracy across layers by Dataset/Model
# ==========================================
plt.figure(figsize=(10, 6))

# Dataset specific colors and model specific line styles
dataset_colors = {'RAVDESS': '#1f77b4', 'EmoDB': '#2ca02c', 'IEMOCAP': '#d62728'}
model_lines = {'wav2vec2': '-', 'HuBERT': '--'}

for col in df.columns[1:]:
    model, dataset = col.split(' + ')
    plt.plot(df['Layer'], df[col], 
             label=col, 
             color=dataset_colors[dataset], 
             linestyle=model_lines[model], 
             marker='o', markersize=6)

plt.xlabel('Transformer Layer', fontsize=12, fontweight='bold')
plt.ylabel('Probing Accuracy', fontsize=12, fontweight='bold')
plt.title('Probing Accuracy Across Layers by Model and Dataset', fontsize=14)
plt.xticks(rotation=45)
# Place legend outside the plot to avoid overlapping lines
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('accuracy_across_layers.png', dpi=300)

# ==========================================
# Plot 2: Average performance across all models/datasets
# ==========================================
# Calculate average across all experiments
df['Average'] = df.iloc[:, 1:].mean(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(df['Layer'], df['Average'], color='indigo', marker='s', linewidth=2.5, markersize=8)
plt.xlabel('Transformer Layer', fontsize=12, fontweight='bold')
plt.ylabel('Average Accuracy', fontsize=12, fontweight='bold')
plt.title('Average Probing Performance Across All Experiments', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)

# Annotate the absolute best layer
max_idx = df['Average'].idxmax()
plt.annotate(f"Best: {df['Layer'][max_idx]}\n({df['Average'][max_idx]:.4f})", 
             xy=(max_idx, df['Average'][max_idx]),
             xytext=(max_idx, df['Average'][max_idx] + 0.015),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('average_accuracy_across_layers.png', dpi=300)