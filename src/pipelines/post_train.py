import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_layer_results(results_df, model_name, dataset_name, save_dir='results/plots'):
    """
    Create a line plot showing accuracy and F1 score across layers.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = results_df['Layer_Num']
    
    if 'Accuracy' in results_df.columns:
        ax.plot(x, results_df['Accuracy'], marker='o', linewidth=2, label='Accuracy', color='#2196F3')
        if 'Weighted_F1' in results_df.columns:
            ax.plot(x, results_df['Weighted_F1'], marker='s', linewidth=2, label='Weighted F1', color='#FF5722')
            
        best_idx = results_df['Accuracy'].idxmax()
        best_acc = results_df.loc[best_idx, 'Accuracy']
        best_layer = results_df.loc[best_idx, 'Layer_Num']
        ax.annotate(f'Best: {best_acc:.3f}', xy=(best_layer, best_acc),
                    xytext=(best_layer + 0.5, best_acc + 0.03),
                    fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='black'))
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim([0, 1.0])
    elif 'RMSE' in results_df.columns:
        ax.plot(x, results_df['RMSE'], marker='o', linewidth=2, label='RMSE', color='#2196F3')
        ax.set_ylabel('RMSE', fontsize=12)
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_title(f'Emotion Recognition by Layer: {model_name} on {dataset_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['CNN'] + [str(i) for i in range(1, len(x))], fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'probing_{model_name.replace("/", "_")}_{dataset_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved to: {save_path}')

def plot_averages_across_models(df, save_dir='results/plots'):
    """
    Takes a combined dataframe of all results and plots the averages.
    Requires columns in format 'Model + Dataset' (e.g. 'wav2vec2 + RAVDESS')
    """
    os.makedirs(save_dir, exist_ok=True)
    
    wav2vec2_cols = [col for col in df.columns if 'wav2vec2' in col]
    hubert_cols = [col for col in df.columns if 'HuBERT' in col]
    
    if not wav2vec2_cols or not hubert_cols:
        print("Missing required columns for average plotting.")
        return
        
    df['wav2vec2_Avg'] = df[wav2vec2_cols].mean(axis=1)
    df['HuBERT_Avg'] = df[hubert_cols].mean(axis=1)

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
    
    save_path = os.path.join(save_dir, 'model_averages_lineplot.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
