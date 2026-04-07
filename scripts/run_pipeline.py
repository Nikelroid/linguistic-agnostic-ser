import os
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch

from src.data_ingestion.loader import load_ravdess, load_emodb, load_iemocap, load_savee, load_aesdd, load_mesd
from src.preprocessing.audio_processor import extract_features_opensmile
from src.models.feature_extractors import TransformerExtractor
from src.pipelines.train import probe_all_layers
from src.utils.helpers import calculate_information_increase
from src.utils.config_parser import load_config

def main(args):
    # Load standardized configuration
    config = load_config()
    
    # Resolve batch size or sample rate falling back to config if not provided explicitly
    batch_size = args.batch_size if args.batch_size else config['training']['batch_size']
    sample_rate = config['training']['sample_rate']
    
    print(f"Loaded config: Batch Size={batch_size}, Target Model={args.model_name}")

    if args.task == 'regression':
        audio_paths = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.wav')]
        if len(audio_paths) == 0:
            print("No .wav files found in the data directory.")
            return

        print(f"Found {len(audio_paths)} audio files. Extracting {args.target_feature} features...")
        features_df = extract_features_opensmile(audio_paths)
        
        if features_df.empty:
            print("Failed to extract acoustic features.")
            return

        target_feature = args.target_feature
        if target_feature not in features_df.columns:
            target_feature = features_df.select_dtypes(include=[np.number]).columns[0]
            print(f"Falling back to target feature: {target_feature}")

        y_target = features_df[target_feature].values

        if np.isnan(y_target).any():
            non_nan_idx = ~np.isnan(y_target)
            y_target = y_target[non_nan_idx]
            audio_paths = [p for i, p in enumerate(audio_paths) if non_nan_idx[i]]

        print(f"Extracting hidden states from {args.model_name}...")
        from src.preprocessing.audio_processor import AudioDataset
        dataset = AudioDataset(audio_paths)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        extractor = TransformerExtractor(model_name=args.model_name)
        
        hidden_states = []
        for batch in dataloader:
            if batch is None: continue
            h_states = extractor.extract_from_waveform(batch['waveform'], sample_rate=sample_rate)
            if not hidden_states:
                hidden_states = [[] for _ in range(len(h_states))]
            for i, h in enumerate(h_states):
                hidden_states[i].append(h)
                
        for i in range(len(hidden_states)):
            hidden_states[i] = np.concatenate(hidden_states[i], axis=0)
            
        hidden_states = np.stack(hidden_states, axis=1) 
        
        results_df = probe_all_layers(hidden_states, y_target, task_type='regression', random_state=config['training']['random_state'])
        
    elif args.task == 'classification':
        print(f"Loading dataset: {args.dataset_name}")
        # Utilize the dataset string matching safely
        if args.dataset_name == 'RAVDESS':
            data = load_ravdess(args.data_dir, sample_rate=sample_rate)
        elif args.dataset_name == 'EmoDB':
            data = load_emodb(args.data_dir, sample_rate=sample_rate)
        elif args.dataset_name == 'IEMOCAP':
            data = load_iemocap(args.data_dir, sample_rate=sample_rate)
        elif args.dataset_name == 'SAVEE':
            data = load_savee(args.data_dir, sample_rate=sample_rate)
        elif args.dataset_name == 'AESDD':
            data = load_aesdd(args.data_dir, sample_rate=sample_rate)
        elif args.dataset_name == 'MESD':
            data = load_mesd(args.data_dir, sample_rate=sample_rate)
        else:
            print(f"Dataset {args.dataset_name} not fully wired in script. Use RAVDESS, EmoDB, IEMOCAP, SAVEE, AESDD, or MESD.")
            return
            
        if not data:
            print("No data extracted. Please check dataset path.")
            return
            
        extractor = TransformerExtractor(model_name=args.model_name)
        audio_tensors = []
        labels = []
        
        print("Extracting features...")
        for row in data:
            # We take audio numpy arrays returned by loaders and convert them
            waveform = torch.from_numpy(row['audio']).unsqueeze(0)
            hidden = extractor.extract_from_waveform(waveform, sample_rate=sample_rate)
            audio_tensors.append([h[0] for h in hidden])
            labels.append(row['label'])
            
        hidden_states = np.array(audio_tensors)
        results_df = probe_all_layers(hidden_states, labels, task_type='classification', random_state=config['training']['random_state'])
        
    clean_model_name = args.model_name.split('/')[-1]
    exp_name = f"{clean_model_name}_{args.dataset_name}"
    
    # Save statistics directly into CSV routing
    csv_path = os.path.join("results/csv", f"probing_results_{exp_name}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    
    # Save tensor layers offline for custom dimensionality mapping (t-SNE/PCA)
    embed_dir = "results/embeddings"
    os.makedirs(embed_dir, exist_ok=True)
    embed_path = os.path.join(embed_dir, f"hidden_states_{exp_name}.npy")
    np.save(embed_path, hidden_states)
    
    # Reconstruct graphical plots natively using the standardized names
    from src.pipelines.post_train import plot_layer_results
    plot_layer_results(results_df, clean_model_name, args.dataset_name)
    
    print(f"[{exp_name}] Run fully committed across results/ (CSV, Plot, NPY).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=['classification', 'regression'], default='classification')
    parser.add_argument("--data_dir", type=str, required=True, help="Path to raw dataset")
    parser.add_argument("--dataset_name", type=str, default="RAVDESS")
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--target_feature", type=str, default="F0semitoneFrom27.5Hz_sma3nz_amean")
    parser.add_argument("--batch_size", type=int, default=None) # Falls back to config if None
    args = parser.parse_args()
    main(args)
