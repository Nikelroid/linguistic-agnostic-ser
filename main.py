import os
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from data_loader import AudioDataset, FeatureExtractor
from model_extractor import TransformerExtractor
from probing import LayerProber
from utils import calculate_information_increase

def main(args):
    # 1. Load Audio Data
    if not os.path.exists(args.data_dir):
        print(f"Data directory '{args.data_dir}' not found. Please provide valid audio files.")
        return
        
    audio_paths = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.wav')]
    if len(audio_paths) == 0:
        print("No .wav files found in the data directory.")
        return
        
    print(f"Found {len(audio_paths)} audio files. Extracting acoustic features...")
    
    # 2. Extract eGeMAPS features (Targets)
    feature_extractor = FeatureExtractor()
    features_df = feature_extractor.extract_batch(audio_paths)
    
    if features_df.empty:
        print("Failed to extract acoustic features.")
        return
        
    target_feature = args.target_feature
    if target_feature not in features_df.columns:
        print(f"Feature '{target_feature}' not found in extracted features.")
        # Fallback to the first available numeric column
        target_feature = features_df.select_dtypes(include=[np.number]).columns[0]
        print(f"Falling back to target feature: {target_feature}")
        
    y_target = features_df[target_feature].values
    
    # Fallback if there are missing values
    if np.isnan(y_target).any():
        non_nan_idx = ~np.isnan(y_target)
        y_target = y_target[non_nan_idx]
        audio_paths = [p for i, p in enumerate(audio_paths) if non_nan_idx[i]]
        print(f"Filtered out NaNs. Remaining valid samples: {len(audio_paths)}")
    
    if len(audio_paths) < 5: # Need enough samples for 5-fold CV
        print("Not enough valid samples to perform 5-fold cross validation. Please provide at least 5 audio files.")
        return

    # 3. Extract Transformer Hidden States
    print(f"Extracting hidden states from {args.model_name}...")
    dataset = AudioDataset(audio_paths)
    # Important: Do not shuffle here to maintain alignment with y_target
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    extractor = TransformerExtractor(model_name=args.model_name)
    num_layers = extractor.get_num_layers()
    
    layer_representations = {i: [] for i in range(num_layers)}
    
    for batch in dataloader:
        if batch is None:
            continue
        waveforms = batch['waveform']
        hidden_states = extractor.extract_from_waveform(waveforms)
        
        for i, h_state in enumerate(hidden_states):
            layer_representations[i].append(h_state)
            
    # Concatenate batches for each layer
    for i in range(num_layers):
        layer_representations[i] = np.concatenate(layer_representations[i], axis=0)
        
    # 4. Probe Layer-wise Information
    print("Training linear probes across layers...")
    prober = LayerProber(task_type='regression', alpha=1.0)
    
    layer_rmse_scores = []
    for i in range(num_layers):
        X = layer_representations[i]
        # Perform 5-fold CV to evaluate the prediction
        rmse = prober.evaluate(X, y_target, n_splits=5)
        layer_rmse_scores.append(rmse)
        print(f"Layer {i}: RMSE = {rmse:.4f}")
        
    # Calculate Information Increase
    info_increases = calculate_information_increase(layer_rmse_scores)
    
    # 5. Save Results
    results_df = pd.DataFrame({
        'Layer': range(num_layers),
        'RMSE': layer_rmse_scores,
        'Information_Increase_Percent': info_increases
    })
    
    out_path = f"probing_results_{args.model_name.replace('/', '_')}_{target_feature}.csv"
    results_df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linguistic-Agnostic SER Probing")
    parser.add_argument("--data_dir", type=str, default="data/sample_audio", help="Directory containing .wav files")
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base", help="HuggingFace model to probe")
    parser.add_argument("--target_feature", type=str, default="F0semitoneFrom27.5Hz_sma3nz_amean", help="eGeMAPS feature to predict (default: mean pitch)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for extracting representations")
    
    args = parser.parse_args()
    main(args)
