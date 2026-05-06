import os
import torch
import torchaudio
import pandas as pd
import numpy as np

class AudioDataset(torch.utils.data.Dataset):
    """
    A generic audio dataset loader that handles cropping/padding to exact lengths.
    """
    def __init__(self, audio_paths, labels=None, target_sample_rate=16000, target_duration=8.0):
        self.audio_paths = audio_paths
        self.labels = labels
        self.target_sample_rate = target_sample_rate
        self.target_duration = target_duration
        self.target_length = int(target_sample_rate * target_duration)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        try:
            waveform, sample_rate = torchaudio.load(path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Resample if needed
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
                waveform = resampler(waveform)
                
            # Pad or crop
            curr_length = waveform.shape[1]
            if curr_length < self.target_length:
                padding = self.target_length - curr_length
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            elif curr_length > self.target_length:
                # Center crop or start crop. We do start crop for simplicity.
                waveform = waveform[:, :self.target_length]
                
            item = {'waveform': waveform.squeeze(0), 'path': path}
            if self.labels is not None:
                item['label'] = self.labels[idx]
                
            return item
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

def extract_features_opensmile(audio_paths):
    """
    Uses OpenSMILE to extract eGeMAPS features (e.g., average pitch, loudness, jitter, shimmer)
    for baseline targets.
    """
    try:
        import opensmile
    except ImportError:
        print("opensmile not installed. Run `pip install opensmile`")
        return pd.DataFrame()
        
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    
    df_list = []
    for path in audio_paths:
        try:
            df = smile.process_file(path)
            if df is not None:
                df = df.reset_index()
                df_list.append(df)
        except Exception as e:
            print(f"Error extracting features for {path}: {e}")
            
    if len(df_list) == 0:
        return pd.DataFrame()
        
    full_df = pd.concat(df_list, ignore_index=True)
    return full_df
