"""
Noise augmentation utilities for ESC-50 background noise mixing.
Mixes clean speech signals with environmental noise at controlled SNR levels.
"""
import os
import glob
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample as scipy_resample


def load_esc50_noise_pool(noise_dir, sample_rate=16000):
    """
    Load all ESC-50 .wav files into a list of numpy arrays (float32, mono).
    
    Args:
        noise_dir: Path to the ESC-50 dataset folder (expects .wav files inside)
        sample_rate: Target sample rate to resample noise clips to
    
    Returns:
        List of 1-D numpy arrays (float32, normalized to [-1, 1])
    """
    noise_pool = []
    wav_files = glob.glob(os.path.join(noise_dir, "**", "*.wav"), recursive=True)
    
    if len(wav_files) == 0:
        raise FileNotFoundError(f"No .wav files found in {noise_dir}")
    
    print(f"Loading ESC-50 noise pool: {len(wav_files)} files from {noise_dir}")
    
    for fpath in wav_files:
        try:
            sr, waveform = wavfile.read(fpath)
            
            # Normalize to float32 [-1, 1]
            if waveform.dtype == np.int16:
                waveform = waveform.astype(np.float32) / 32768.0
            elif waveform.dtype == np.int32:
                waveform = waveform.astype(np.float32) / 2147483648.0
            else:
                waveform = waveform.astype(np.float32)
                max_val = np.max(np.abs(waveform))
                if max_val > 1.0:
                    waveform = waveform / max_val
            
            # Downmix stereo to mono
            if len(waveform.shape) > 1 and waveform.shape[1] > 1:
                waveform = np.mean(waveform, axis=1)
            
            # Resample if needed
            if sr != sample_rate:
                num_samples = int(len(waveform) * sample_rate / sr)
                waveform = scipy_resample(waveform, num_samples)
            
            noise_pool.append(waveform)
        except Exception as e:
            # Skip corrupted files silently
            pass
    
    print(f"  Loaded {len(noise_pool)} noise clips into pool.")
    return noise_pool


def mix_signal_noise(signal, noise, snr_db):
    """
    Mix a clean signal with noise at the specified SNR level.
    
    Args:
        signal: 1-D numpy array (clean speech waveform)
        noise: 1-D numpy array (noise waveform, same length as signal)
        snr_db: Target signal-to-noise ratio in decibels
    
    Returns:
        Mixed waveform as 1-D numpy array
    """
    # Calculate power (RMS^2)
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0 or signal_power == 0:
        return signal
    
    # Calculate required noise scaling factor
    # SNR_dB = 10 * log10(P_signal / P_noise_scaled)
    # P_noise_scaled = P_signal / 10^(SNR_dB / 10)
    target_noise_power = signal_power / (10 ** (snr_db / 10.0))
    scaling_factor = np.sqrt(target_noise_power / noise_power)
    
    mixed = signal + scaling_factor * noise
    
    # Prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val
    
    return mixed


def apply_noise_to_batch(batch_waveforms, noise_pool, snr_db, rng=None):
    """
    Apply noise augmentation to an entire batch of waveforms.
    
    Args:
        batch_waveforms: Tensor of shape (batch_size, seq_length)
        noise_pool: List of 1-D numpy noise arrays (from load_esc50_noise_pool)
        snr_db: Target SNR in dB (int or float)
        rng: numpy RandomState for reproducibility
    
    Returns:
        Tensor of same shape with noise mixed in
    """
    import torch
    
    if rng is None:
        rng = np.random.RandomState(42)
    
    device = batch_waveforms.device
    batch_np = batch_waveforms.cpu().numpy()
    mixed_batch = np.zeros_like(batch_np)
    
    for i in range(batch_np.shape[0]):
        signal = batch_np[i]
        
        # Pick a random noise clip
        noise_clip = noise_pool[rng.randint(0, len(noise_pool))]
        
        # Tile or crop noise to match signal length
        sig_len = len(signal)
        if len(noise_clip) < sig_len:
            # Tile (repeat) noise to fill signal length
            repeats = (sig_len // len(noise_clip)) + 1
            noise_clip = np.tile(noise_clip, repeats)
        
        noise_clip = noise_clip[:sig_len]
        
        # Mix at target SNR
        mixed_batch[i] = mix_signal_noise(signal, noise_clip, snr_db)
    
    return torch.from_numpy(mixed_batch).to(device)
