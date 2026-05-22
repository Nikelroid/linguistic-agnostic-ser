#!/usr/bin/env python
"""Per-frame frozen-encoder activation extraction for the SAE study.

The project's cached embeddings (EXP4 etc.) are **mean-pooled per utterance**
``(N, 25, H)`` — too few samples to train an over-complete SAE. SAEs want
per-frame (per-token) activations, so this module re-runs the frozen encoders
and stores the **un-pooled** hidden states for the requested layer(s):

    frames_{ENCODER}_{DATASET}_L{layer}.npy   float32  (M_frames, H)
    frame_index_{ENCODER}_{DATASET}.npy        int32    (M_frames,)  -> utterance id
    utterances_{ENCODER}_{DATASET}.npz         label/file/actor/sentence per utterance
    meta_{ENCODER}_{DATASET}.json              shapes, frame rate, layers

Notes
-----
* Extraction is per-clip (batch=1) so non-Whisper encoders emit exactly the
  valid frame count (no padding). Whisper pads the mel to 30 s -> 1500 encoder
  frames regardless of length, so we keep only ``ceil(n_samples/320)`` frames.
* Layer index 0 is the CNN/embedding front-end; 1..24 are transformer layers.

Usage
-----
    python -m src.sae.extraction --encoder HuBERT --dataset CREMA-D \
        --data-dir /scratch1/kelidari/ser_data/CREMA-D --layers 13 \
        --out /scratch1/kelidari/ser-experiments/SAE_frames
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

# Make the repo importable when run as a script.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.models.feature_extractors import TransformerExtractor  # noqa: E402
from src.data_ingestion import loader as dload  # noqa: E402

DEFAULT_OUT = "/scratch1/kelidari/ser-experiments/SAE_frames"

# Cached-file display name -> config.yaml model key / HF path.
ENCODER_HF_PATHS = {
    "HuBERT": "facebook/hubert-large-ll60k",
    "WavLM": "microsoft/wavlm-large",
    "wav2vec2": "facebook/wav2vec2-large-960h",
    "w2v-BERT": "facebook/w2v-bert-2.0",
    "MERT": "m-a-p/MERT-v1-330M",
    "Whisper": "openai/whisper-medium",
}

# 16 kHz samples per output frame (50 Hz frame rate for all these encoders,
# including Whisper after its 2x conv downsample of the 100 Hz log-mel).
SAMPLES_PER_FRAME = 320

_LOADERS = {
    "CREMA-D": dload.load_cremad,
    "IEMOCAP": dload.load_iemocap,
    "RAVDESS": dload.load_ravdess,
    "EmoDB": dload.load_emodb,
    "SAVEE": dload.load_savee,
    "AESDD": dload.load_aesdd,
    "MESD": dload.load_mesd,
}


def load_dataset(dataset_name: str, data_dir: str, sample_rate: int = 16000):
    if dataset_name not in _LOADERS:
        raise ValueError(f"no loader wired for {dataset_name}; have {sorted(_LOADERS)}")
    return _LOADERS[dataset_name](data_dir, sample_rate=sample_rate)


class FrameExtractor:
    """Extract un-pooled per-frame hidden states from a frozen speech encoder."""

    def __init__(self, encoder: str, device=None):
        if encoder not in ENCODER_HF_PATHS:
            raise ValueError(f"unknown encoder {encoder}; have {sorted(ENCODER_HF_PATHS)}")
        self.encoder = encoder
        self._tx = TransformerExtractor(ENCODER_HF_PATHS[encoder], device=device)
        self.device = self._tx.device
        self.is_whisper = self._tx.is_whisper
        self.fe = self._tx.feature_extractor
        self.model = self._tx.model
        self.num_layers = self._tx.get_num_layers()

    @torch.no_grad()
    def extract(self, waveform_np: np.ndarray, sample_rate: int, layers: list[int]):
        """Return {layer: np.float32 (T_valid, H)} for one clip."""
        target_sr = getattr(self.fe, "sampling_rate", sample_rate)
        wav = np.asarray(waveform_np, dtype=np.float32)
        if sample_rate != target_sr:  # encoders here all use 16 kHz; guard anyway
            import torchaudio.functional as AF

            wav = AF.resample(torch.from_numpy(wav), sample_rate, target_sr).numpy()
            sample_rate = target_sr

        inputs = self.fe([wav], sampling_rate=sample_rate, return_tensors="pt")
        if self.is_whisper and "input_features" in inputs:
            model_inputs = {"input_features": inputs["input_features"].to(self.device)}
        elif "input_values" in inputs:
            model_inputs = {"input_values": inputs["input_values"].to(self.device)}
        else:
            model_inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self.is_whisper:
            outputs = self.model.encoder(**model_inputs)
        else:
            outputs = self.model(**model_inputs)
        hidden = outputs.hidden_states  # tuple length num_layers, each (1, T, H)

        # Valid-frame count: Whisper is padded to 1500; others are exact at bs=1.
        full_T = hidden[0].shape[1]
        if self.is_whisper:
            n_valid = int(min(full_T, np.ceil(len(wav) / SAMPLES_PER_FRAME)))
            n_valid = max(1, n_valid)
        else:
            n_valid = full_T

        out = {}
        for l in layers:
            state = hidden[l][0, :n_valid].detach().to(torch.float32).cpu().numpy().copy()
            out[l] = state
        del outputs, hidden, model_inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out


def extract_dataset_frames(
    encoder: str,
    dataset_name: str,
    data_dir: str,
    layers: list[int],
    out_dir: str = DEFAULT_OUT,
    sample_rate: int = 16000,
    limit: int | None = None,
    max_frames_per_clip: int | None = None,
    seed: int = 42,
):
    """Extract per-frame activations for a whole dataset and save to ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    data = load_dataset(dataset_name, data_dir, sample_rate)
    if not data:
        raise RuntimeError(f"no audio loaded for {dataset_name} from {data_dir}")
    if limit:
        data = data[:limit]
    print(f"[extract] {encoder}/{dataset_name}: {len(data)} clips, layers={layers}")

    fx = FrameExtractor(encoder)
    rng = np.random.default_rng(seed)
    per_layer: dict[int, list] = {l: [] for l in layers}
    frame_utt, labels, files, actors, sentences = [], [], [], [], []

    for ui, rec in enumerate(tqdm(data, desc=f"{encoder}/{dataset_name}")):
        try:
            out = fx.extract(rec["audio"], sample_rate, layers)
        except Exception as exc:  # keep going; log which clip failed
            print(f"  ! clip {ui} ({rec.get('file')}) failed: {exc}")
            continue
        T = out[layers[0]].shape[0]
        idx = np.arange(T)
        if max_frames_per_clip and T > max_frames_per_clip:
            idx = np.sort(rng.choice(T, size=max_frames_per_clip, replace=False))
        for l in layers:
            per_layer[l].append(out[l][idx])
        frame_utt.append(np.full(len(idx), ui, dtype=np.int32))
        labels.append(str(rec["label"]))
        files.append(str(rec["file"]))
        actors.append(str(rec.get("actor", "")))
        sentences.append(str(rec.get("sentence", "")))

    frame_index = np.concatenate(frame_utt)
    tag = f"{encoder}_{dataset_name}"
    meta = {
        "encoder": encoder,
        "dataset": dataset_name,
        "layers": layers,
        "n_utterances": len(labels),
        "n_frames": int(frame_index.shape[0]),
        "frame_rate_hz": 16000 / SAMPLES_PER_FRAME,
        "max_frames_per_clip": max_frames_per_clip,
    }
    for l in layers:
        arr = np.concatenate(per_layer[l], axis=0).astype(np.float32)
        meta[f"L{l}_shape"] = list(arr.shape)
        np.save(os.path.join(out_dir, f"frames_{tag}_L{l}.npy"), arr)
        print(f"  saved frames_{tag}_L{l}.npy {arr.shape}")
    np.save(os.path.join(out_dir, f"frame_index_{tag}.npy"), frame_index)
    np.savez(
        os.path.join(out_dir, f"utterances_{tag}.npz"),
        label=np.array(labels), filename=np.array(files),
        actor=np.array(actors), sentence=np.array(sentences),
    )
    with open(os.path.join(out_dir, f"meta_{tag}.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[extract] done: {meta['n_frames']} frames from {meta['n_utterances']} utts -> {out_dir}")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", required=True, choices=sorted(ENCODER_HF_PATHS))
    ap.add_argument("--dataset", required=True, choices=sorted(_LOADERS))
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--layers", default="13", help="comma-separated layer indices, e.g. 0,6,13,24")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None, help="cap #clips (debug)")
    ap.add_argument("--max-frames-per-clip", type=int, default=None,
                    help="randomly subsample frames per clip to bound dataset size")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    extract_dataset_frames(
        args.encoder, args.dataset, args.data_dir, layers,
        out_dir=args.out, limit=args.limit,
        max_frames_per_clip=args.max_frames_per_clip,
    )


if __name__ == "__main__":
    main()
