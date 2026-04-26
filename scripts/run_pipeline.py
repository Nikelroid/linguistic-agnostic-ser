import os
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch
import wandb
from tqdm import tqdm

from src.data_ingestion.loader import load_ravdess, load_emodb, load_iemocap, load_savee, load_aesdd, load_mesd
from src.preprocessing.audio_processor import extract_features_opensmile
from src.models.feature_extractors import TransformerExtractor
from src.pipelines.train import probe_all_layers
from src.utils.helpers import calculate_information_increase
from src.utils.config_parser import load_config

def main(args):
    # Load standardized configuration
    config = load_config()

    # Synchronize Model Name from CLI shorthand or full path
    base_model_key = args.model_name.lower().split('/')[-1].split('-')[0]
    if args.model_name in config['models']:
        resolved_model_path = config['models'][args.model_name]['path']
    elif base_model_key in config['models'] and '/' not in args.model_name:
        resolved_model_path = config['models'][base_model_key]['path']
    else:
        resolved_model_path = args.model_name

    # Initialize WandB
    clean_model_name_init = resolved_model_path.split('/')[-1]
    
    # Use provided args or fall back to config
    wandb_entity = args.wandb_entity if args.wandb_entity else config.get('wandb', {}).get('entity', 'AGSER')
    wandb_project = args.wandb_project if args.wandb_project else config.get('wandb', {}).get('project', 'linguistic-agnostic-ser')

    # Build SNR tag for naming
    snr_tag = f"_SNR{args.snr_db}" if args.snr_db != "clean" else ""
    
    print(f"--> Initializing W&B Run for {clean_model_name_init} on {args.dataset_name}{snr_tag}...")
    exp_id = args.exp_id if args.exp_id else config.get('wandb', {}).get('experiment_id', 4)
    wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        group=f"EXP{exp_id}",
        name=f"{clean_model_name_init}_{args.dataset_name}{snr_tag}",
        reinit=True,
        config={
            "args": vars(args),
            "config": config,
            "resolved_model": resolved_model_path,
            "snr_db": args.snr_db
        }
    )
    
    # Resolve batch size or sample rate falling back to config if not provided explicitly
    batch_size = args.batch_size if args.batch_size else config['training']['batch_size']
    sample_rate = config['training']['sample_rate']
    
    print(f"Loaded config: Batch Size={batch_size}, Target Model={resolved_model_path}, SNR={args.snr_db}")

    # --- Load noise pool if needed ---
    noise_pool = None
    noise_rng = None
    if args.snr_db != "clean":
        from src.preprocessing.noise_augmentor import load_esc50_noise_pool
        noise_seed = config.get('noise_augmentation', {}).get('seed', 42)
        noise_rng = np.random.RandomState(noise_seed)
        
        # Resolve noise directory
        if args.noise_dir:
            noise_dir = args.noise_dir
        else:
            noise_subdir = config.get('noise_augmentation', {}).get('noise_dir', 'ESC-50')
            noise_dir = os.path.join(f"/scratch1/{os.environ.get('USER', 'user')}/ser_data", noise_subdir)
        
        print(f"Loading noise pool from: {noise_dir}")
        noise_pool = load_esc50_noise_pool(noise_dir, sample_rate=sample_rate)
        print(f"Noise pool loaded: {len(noise_pool)} clips, target SNR={args.snr_db}dB")

    # --- Build results directory ---
    results_base = os.path.join("results", f"EXP{exp_id}")

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

        print(f"Extracting hidden states from {resolved_model_path}...")
        from src.preprocessing.audio_processor import AudioDataset
        dataset = AudioDataset(audio_paths)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        extractor = TransformerExtractor(model_name=resolved_model_path)
        
        hidden_states = []
        total_batches = len(dataloader)
        for i, batch in enumerate(tqdm(dataloader, desc=f"Extracting {args.target_feature}")):
            if batch is None: continue

            waveform = batch['waveform']
            # Apply noise augmentation if needed
            if noise_pool is not None:
                from src.preprocessing.noise_augmentor import apply_noise_to_batch
                waveform = apply_noise_to_batch(waveform, noise_pool, float(args.snr_db), noise_rng)

            h_states = extractor.extract_from_waveform(waveform, sample_rate=sample_rate)
            if not hidden_states:
                hidden_states = [[] for _ in range(len(h_states))]
            for j, h in enumerate(h_states):
                hidden_states[j].append(h)
            
            # Log extraction progress to WandB
            wandb.log({
                "extraction_progress_pct": round((i + 1) / total_batches * 100, 2),
                "batches_processed": i + 1
            })
                
        for i in range(len(hidden_states)):
            hidden_states[i] = np.concatenate(hidden_states[i], axis=0)
            
        hidden_states = np.stack(hidden_states, axis=1) 
        
        results_df = probe_all_layers(hidden_states, y_target, task_type='regression', random_state=config['training']['random_state'])
        labels_to_save = y_target
        filenames_to_save = [os.path.basename(p) for p in audio_paths]
        
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
        elif args.dataset_name == 'MSP-Podcast':
            from src.data_ingestion.loader import load_msppodcast
            msp_cfg = config.get('dataset_config', {}).get('msp_podcast', {})
            num_classes = msp_cfg.get('num_classes', 8)
            data = load_msppodcast(args.data_dir, sample_rate=sample_rate, num_classes=num_classes)
        else:
            print(f"Dataset {args.dataset_name} not fully wired in script. Use RAVDESS, EmoDB, IEMOCAP, SAVEE, AESDD, MESD, or MSP-Podcast.")
            return
            
        if not data:
            print("No data extracted. Please check dataset path.")
            return
            
        extractor = TransformerExtractor(model_name=resolved_model_path)

        # Batch processing for classification
        from torch.utils.data import Dataset, DataLoader

        class SimpleInMemoryDataset(Dataset):
            def __init__(self, data_list):
                self.data_list = data_list
            def __len__(self):
                return len(self.data_list)
            def __getitem__(self, idx):
                item = self.data_list[idx]
                return {
                    'waveform': torch.from_numpy(item['audio']),
                    'label': item['label'],
                    'file': item.get('file', f"sample_{idx}.wav")
                }

        def collate_fn(batch):
            # Pad waveforms to the same length in the batch if they differ
            waveforms = [item['waveform'] for item in batch]
            labels = [item['label'] for item in batch]
            files = [item['file'] for item in batch]
            
            # Find max length
            max_len = max(w.shape[0] for w in waveforms)
            padded_waveforms = []
            for w in waveforms:
                if w.shape[0] < max_len:
                    padding = max_len - w.shape[0]
                    w = torch.nn.functional.pad(w, (0, padding))
                padded_waveforms.append(w)
            
            return {
                'waveform': torch.stack(padded_waveforms),
                'labels': labels,
                'files': files
            }

        print(f"Extracting acoustic representations from {resolved_model_path} (Batch Size={batch_size})...")
        eval_dataset = SimpleInMemoryDataset(data)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        total_batches = len(eval_loader)
        audio_tensors = []
        labels = []
        filenames = []

        for i, batch in enumerate(tqdm(eval_loader, desc=f"Evaluating {args.dataset_name}")):
            waveform = batch['waveform']

            # Apply noise augmentation if needed
            if noise_pool is not None:
                from src.preprocessing.noise_augmentor import apply_noise_to_batch
                waveform = apply_noise_to_batch(waveform, noise_pool, float(args.snr_db), noise_rng)

            # Extract features for the entire batch at once
            hidden = extractor.extract_from_waveform(waveform, sample_rate=sample_rate)
            
            # hidden is a list of layers, each (B, hidden_dim)
            # We want to re-structure this to (B, L, hidden_dim) or similar
            # For now, matching the original format: list of (B, hidden_dim) per sample?
            # Actually, the original code did: audio_tensors.append([h[0] for h in hidden]) 
            # where h[0] is the first sample in batch (since batch_size was 1).
            
            # Correct batch logic:
            # hidden[layer_idx] is a numpy array of shape (batch_size, hidden_dim)
            batch_hidden = np.stack(hidden, axis=1) # (batch_size, num_layers, hidden_dim)
            
            audio_tensors.extend(batch_hidden)
            labels.extend(batch['labels'])
            filenames.extend(batch['files'])

            # Log extraction progress to WandB
            if (i + 1) % 5 == 0 or (i + 1) == total_batches:
                wandb.log({
                    "extraction_progress_pct": round((i + 1) / total_batches * 100, 2),
                    "batches_processed": i + 1,
                    "samples_processed": min((i + 1) * batch_size, len(data))
                })

        hidden_states = np.array(audio_tensors)
        results_df = probe_all_layers(hidden_states, labels, task_type='classification', random_state=config['training']['random_state'])
        labels_to_save = labels
        filenames_to_save = filenames
        
    elif args.task == 'dimensional_regression':
        print(f"Loading dataset for Dimensional Regression: {args.dataset_name} ({args.target_feature})")
        if args.dataset_name == 'MSP-Podcast':
            from src.data_ingestion.loader import load_msppodcast_dimensional
            data = load_msppodcast_dimensional(args.data_dir, target=args.target_feature, sample_rate=sample_rate)
        else:
            print(f"Dimensional regression not implemented for {args.dataset_name}")
            return
            
        if not data:
            print("No data extracted. Please check dataset path and labels.")
            return
            
        extractor = TransformerExtractor(model_name=resolved_model_path)
        
        # Reuse SimpleInMemoryDataset from classification block
        from torch.utils.data import Dataset, DataLoader

        class SimpleInMemoryDataset(Dataset):
            def __init__(self, data_list):
                self.data_list = data_list
            def __len__(self):
                return len(self.data_list)
            def __getitem__(self, idx):
                item = self.data_list[idx]
                return {
                    'waveform': torch.from_numpy(item['audio']),
                    'label': item['label'],
                    'file': item.get('file', f"sample_{idx}.wav")
                }

        def collate_fn(batch):
            waveforms = [item['waveform'] for item in batch]
            labels = [item['label'] for item in batch]
            files = [item['file'] for item in batch]
            
            max_len = max(w.shape[0] for w in waveforms)
            padded_waveforms = []
            for w in waveforms:
                if w.shape[0] < max_len:
                    padding = max_len - w.shape[0]
                    w = torch.nn.functional.pad(w, (0, padding))
                padded_waveforms.append(w)
            
            return {
                'waveform': torch.stack(padded_waveforms),
                'labels': labels,
                'files': files
            }

        print(f"Extracting acoustic representations from {resolved_model_path} (Batch Size={batch_size})...")
        eval_dataset = SimpleInMemoryDataset(data)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        total_batches = len(eval_loader)
        audio_tensors = []
        labels = []
        filenames = []

        for i, batch in enumerate(tqdm(eval_loader, desc=f"Evaluating {args.dataset_name} ({args.target_feature})")):
            waveform = batch['waveform']

            # Apply noise augmentation if needed
            if noise_pool is not None:
                from src.preprocessing.noise_augmentor import apply_noise_to_batch
                waveform = apply_noise_to_batch(waveform, noise_pool, float(args.snr_db), noise_rng)

            # Extract features for the entire batch at once
            hidden = extractor.extract_from_waveform(waveform, sample_rate=sample_rate)
            
            batch_hidden = np.stack(hidden, axis=1) # (batch_size, num_layers, hidden_dim)
            
            audio_tensors.extend(batch_hidden)
            labels.extend(batch['labels'])
            filenames.extend(batch['files'])

            # Log extraction progress to WandB
            if (i + 1) % 5 == 0 or (i + 1) == total_batches:
                wandb.log({
                    "extraction_progress_pct": round((i + 1) / total_batches * 100, 2),
                    "batches_processed": i + 1,
                    "samples_processed": min((i + 1) * batch_size, len(data))
                })

        hidden_states = np.array(audio_tensors)
        # We use regression task type for probe_all_layers to compute RMSE and R2
        results_df = probe_all_layers(hidden_states, labels, task_type='regression', random_state=config['training']['random_state'])
        labels_to_save = labels
        filenames_to_save = filenames

        


    lower_model_name = resolved_model_path.lower()
    if 'wav2vec2' in lower_model_name:
        clean_model_name = 'wav2vec2'
    elif 'hubert' in lower_model_name:
        clean_model_name = 'HuBERT'
    elif 'whisper' in lower_model_name:
        clean_model_name = 'Whisper'
    elif 'wavlm' in lower_model_name:
        clean_model_name = 'WavLM'
    elif 'mert' in lower_model_name:
        clean_model_name = 'MERT'
    elif 'w2v-bert' in lower_model_name or 'w2v_bert' in lower_model_name:
        clean_model_name = 'w2v-BERT'
    else:
        clean_model_name = clean_model_name_init

    if args.task == 'dimensional_regression':
        exp_name = f"{clean_model_name}_{args.dataset_name}_{args.target_feature}{snr_tag}"
        plot_dataset_str = f"{args.dataset_name}_{args.target_feature}{snr_tag}"
    else:
        exp_name = f"{clean_model_name}_{args.dataset_name}{snr_tag}"
        plot_dataset_str = f"{args.dataset_name}{snr_tag}"
    
    # Save statistics directly into CSV routing
    csv_path = os.path.join(results_base, "csv", f"results_{exp_name}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    
    # Save tensor layers offline for custom dimensionality mapping (t-SNE/PCA)
    embed_dir = os.path.join(results_base, "embeddings")
    os.makedirs(embed_dir, exist_ok=True)
    
    # Save hidden states
    embed_path = os.path.join(embed_dir, f"hidden_states_{exp_name}.npy")
    np.save(embed_path, hidden_states)
    
    # Save labels (matching 18exp.ipynb reference)
    labels_path = os.path.join(embed_dir, f"labels_{exp_name}.npy")
    np.save(labels_path, labels_to_save)
    
    # Save filenames for traceability
    files_path = os.path.join(embed_dir, f"filenames_{exp_name}.npy")
    np.save(files_path, np.array(filenames_to_save, dtype=object))
    
    print(f"💾 Embeddings, Labels, and Filenames saved to {embed_dir}")
    
    # Reconstruct graphical plots natively using the standardized names
    from src.pipelines.post_train import plot_layer_results
    plot_dir = os.path.join(results_base, "plots")
    plot_path = plot_layer_results(results_df, clean_model_name, plot_dataset_str, save_dir=plot_dir)
    
    # Log results to WandB
    if args.task == 'classification':
        best_layer = results_df.loc[results_df['Accuracy'].idxmax()]
        wandb.log({
            "best_accuracy": best_layer['Accuracy'],
            "best_f1": best_layer['Weighted_F1'],
            "best_layer": best_layer['Layer_Num'],
            "snr_db": args.snr_db
        })
    elif args.task == 'regression' or args.task == 'dimensional_regression':
        best_layer = results_df.loc[results_df['RMSE'].idxmin()]
        wandb.log({
            "best_rmse": best_layer['RMSE'],
            "best_layer": best_layer['Layer_Num'],
            "snr_db": args.snr_db
        })

    if plot_path and os.path.exists(plot_path):
        wandb.log({"performance_plot": wandb.Image(plot_path)})

    print(f"[{exp_name}] Run fully committed across {results_base}/ (CSV, Plot, NPY) and WandB.")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=['classification', 'regression', 'dimensional_regression'], default='classification')
    parser.add_argument("--data_dir", type=str, required=True, help="Path to raw dataset")
    parser.add_argument("--dataset_name", type=str, default="RAVDESS")
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-large-960h")
    parser.add_argument("--target_feature", type=str, default="F0semitoneFrom27.5Hz_sma3nz_amean")
    parser.add_argument("--batch_size", type=int, default=None) # Falls back to config if None
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--exp_id", type=str, default=None, help="Force specific experiment ID")
    parser.add_argument("--snr_db", type=str, default="clean", help="SNR level in dB (clean, 20, 10, 5, 0)")
    parser.add_argument("--noise_dir", type=str, default=None, help="Path to ESC-50 noise directory")
    args = parser.parse_args()
    main(args)
