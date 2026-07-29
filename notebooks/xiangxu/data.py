from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data_ingestion.loader import load_audio_file
from src.models.feature_extractors import TransformerExtractor


LANGUAGE_BY_DATASET = {
    "RAVDESS": "en",
    "EmoDB": "de",
    "IEMOCAP": "en",
    "SAVEE": "en",
    "AESDD": "el",
    "MESD": "es",
}

CANONICAL_MAP: Dict[str, Dict[str, Optional[str]]] = {
    "RAVDESS": {
        "neutral": "neutral",
        "happy": "happy",
        "sad": "sad",
        "angry": "angry",
        "calm": None,
        "fearful": None,
        "disgust": None,
        "surprised": None,
    },
    "EmoDB": {
        "happy": "happy",
        "sad": "sad",
        "neutral": "neutral",
        "angry": "angry",
        "anger": "angry",
        "bored": None,
        "disgust": None,
        "fearful": None,
    },
    "IEMOCAP": {
        "ang": "angry",
        "hap": "happy",
        "exc": "happy",
        "sad": "sad",
        "neu": "neutral",
        "angry": "angry",
        "happy": "happy",
        "sadness": "sad",
        "neutral": "neutral",
    },
    "SAVEE": {
        "anger": "angry",
        "happiness": "happy",
        "sadness": "sad",
        "neutral": "neutral",
        "fear": None,
        "disgust": None,
        "surprise": None,
    },
    "AESDD": {
        "anger": "angry",
        "happiness": "happy",
        "sadness": "sad",
        "disgust": None,
        "fear": None,
    },
    "MESD": {
        "anger": "angry",
        "happiness": "happy",
        "sadness": "sad",
        "neutral": "neutral",
        "disgust": None,
        "fear": None,
    },
}

RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
SAVEE_MAP = {
    "a": "anger",
    "d": "disgust",
    "f": "fear",
    "h": "happiness",
    "n": "neutral",
    "sa": "sadness",
    "su": "surprise",
}
EMODB_EMOTION_MAP = {
    "W": "angry",
    "L": "bored",
    "E": "disgust",
    "A": "fearful",
    "F": "happy",
    "T": "sad",
    "N": "neutral",
}
AESDD_EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness"]
MESD_EMOTIONS = ["anger", "disgust", "fear", "happiness", "neutral", "sadness"]


@dataclass
class BenchmarkRecord:
    path: str
    file: str
    dataset: str
    language: str
    speaker_id: str
    raw_label: str
    canonical_label: str


class CompiledBenchmarkBuilder:
    def _canonicalize(self, dataset: str, raw_label: str) -> Optional[str]:
        return CANONICAL_MAP[dataset].get(raw_label)

    def _record(self, path: str, dataset: str, speaker_id: str, raw_label: str) -> Optional[BenchmarkRecord]:
        canonical = self._canonicalize(dataset, raw_label)
        if canonical is None:
            return None
        return BenchmarkRecord(
            path=os.path.abspath(path),
            file=os.path.basename(path),
            dataset=dataset,
            language=LANGUAGE_BY_DATASET[dataset],
            speaker_id=str(speaker_id),
            raw_label=raw_label,
            canonical_label=canonical,
        )

    def build_ravdess(self, root: str) -> List[BenchmarkRecord]:
        records = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if not fn.endswith(".wav"):
                    continue
                parts = fn.replace(".wav", "").split("-")
                if len(parts) < 7:
                    continue
                raw = RAVDESS_EMOTION_MAP.get(parts[2])
                speaker_id = parts[-1]
                if raw:
                    rec = self._record(os.path.join(dirpath, fn), "RAVDESS", speaker_id, raw)
                    if rec:
                        records.append(rec)
        return records

    def build_emodb(self, root: str) -> List[BenchmarkRecord]:
        records = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if not fn.endswith(".wav") or len(fn) < 6:
                    continue
                raw = EMODB_EMOTION_MAP.get(fn[5])
                if raw is None:
                    continue
                speaker_id = fn[:2]
                rec = self._record(os.path.join(dirpath, fn), "EmoDB", speaker_id, raw)
                if rec:
                    records.append(rec)
        return records

    def build_iemocap(self, root: str) -> List[BenchmarkRecord]:
        records = []
        meta_path = os.path.join(root, "meta_data", "meta_data.json")
        if not os.path.exists(meta_path):
            meta_path = os.path.join(root, "meta_data.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"IEMOCAP metadata not found under {root}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        path_map = {}
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith(".wav"):
                    path_map[fn] = os.path.join(dirpath, fn)

        for entry in meta.get("meta_data", []):
            raw = str(entry.get("label", "")).strip()
            fname = os.path.basename(entry.get("path", ""))
            abs_path = path_map.get(fname)
            if not abs_path:
                continue
            speaker_match = re.search(r"(Ses\d+[FM]_[^_]+)", fname)
            speaker_id = speaker_match.group(1) if speaker_match else fname.split("_")[0]
            rec = self._record(abs_path, "IEMOCAP", speaker_id, raw)
            if rec:
                records.append(rec)
        return records

    def build_savee(self, root: str) -> List[BenchmarkRecord]:
        records = []
        pattern = re.compile(r"^([A-Z]{2})_([a-z]+)(\d+)\.wav$")
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if not fn.endswith(".wav"):
                    continue
                m = pattern.match(fn)
                if not m:
                    continue
                speaker_id, label_code = m.group(1), m.group(2)
                raw = SAVEE_MAP.get(label_code)
                if raw:
                    rec = self._record(os.path.join(dirpath, fn), "SAVEE", speaker_id, raw)
                    if rec:
                        records.append(rec)
        return records

    def build_aesdd(self, root: str) -> List[BenchmarkRecord]:
        records = []
        for fpath in glob.glob(os.path.join(root, "**", "*.wav"), recursive=True):
            parent = os.path.basename(os.path.dirname(fpath)).lower()
            raw = next((e for e in AESDD_EMOTIONS if e in parent), None)
            if raw is None:
                continue
            speaker_match = re.search(r"(speaker[_-]?\d+|actor[_-]?\d+|s\d+)", fpath.lower())
            speaker_id = speaker_match.group(1) if speaker_match else Path(fpath).parts[-2]
            rec = self._record(fpath, "AESDD", speaker_id, raw)
            if rec:
                records.append(rec)
        return records

    def build_mesd(self, root: str) -> List[BenchmarkRecord]:
        records = []
        for fpath in glob.glob(os.path.join(root, "**", "*.wav"), recursive=True):
            parent = os.path.basename(os.path.dirname(fpath)).lower()
            fn_lower = os.path.basename(fpath).lower()
            raw = next((e for e in MESD_EMOTIONS if e in parent or e in fn_lower), None)
            if raw is None:
                continue
            speaker_match = re.search(r"(speaker[_-]?\d+|actor[_-]?\d+|s\d+)", fpath.lower())
            speaker_id = speaker_match.group(1) if speaker_match else Path(fpath).parts[-2]
            rec = self._record(fpath, "MESD", speaker_id, raw)
            if rec:
                records.append(rec)
        return records

    def build(self, dataset_roots: Dict[str, str]) -> List[BenchmarkRecord]:
        build_fns = {
            "RAVDESS": self.build_ravdess,
            "EmoDB": self.build_emodb,
            "IEMOCAP": self.build_iemocap,
            "SAVEE": self.build_savee,
            "AESDD": self.build_aesdd,
            "MESD": self.build_mesd,
        }
        records: List[BenchmarkRecord] = []
        for dataset_name, root in dataset_roots.items():
            records.extend(build_fns[dataset_name](root))
        return records

    @staticmethod
    def save_csv(records: Iterable[BenchmarkRecord], output_csv: str) -> None:
        rows = [asdict(r) for r in records]
        if not rows:
            raise ValueError("No rows to save.")
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


class ManifestAudioDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_sample_rate: int = 16000):
        self.df = df.reset_index(drop=True)
        self.target_sample_rate = target_sample_rate

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        try:
            audio, _ = load_audio_file(row["path"], self.target_sample_rate)
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim == 0:
                return None
            if audio.ndim > 1:
                audio = np.mean(audio, axis=-1)
            audio = np.squeeze(audio)
            if audio.ndim != 1 or audio.size == 0:
                return None
            return {"waveform": torch.from_numpy(audio).float(), "row_idx": idx}
        except Exception:
            return None


def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    waveforms = [x["waveform"] for x in batch]
    row_indices = [x["row_idx"] for x in batch]
    max_len = max(w.shape[0] for w in waveforms)
    padded = []
    for w in waveforms:
        if w.shape[0] < max_len:
            w = torch.nn.functional.pad(w, (0, max_len - w.shape[0]))
        padded.append(w)
    return {"waveform": torch.stack(padded), "row_indices": row_indices}


def ensure_label_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    labels = sorted(df["canonical_label"].unique().tolist())
    label_map = {lab: i for i, lab in enumerate(labels)}
    df["label_id"] = df["canonical_label"].map(label_map).astype(int)
    return df


def extract_and_cache(
    manifest_csv: str,
    model_name: str,
    cache_dir: str,
    batch_size: int = 8,
    sample_rate: int = 16000,
):
    df = pd.read_csv(manifest_csv)
    df = ensure_label_ids(df)

    os.makedirs(cache_dir, exist_ok=True)
    dataset = ManifestAudioDataset(df, target_sample_rate=sample_rate)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    extractor = TransformerExtractor(model_name=model_name)

    hidden_chunks = []
    kept_indices = []

    for batch in tqdm(loader, desc="Extracting hidden states"):
        if batch is None:
            continue
        hidden = extractor.extract_from_waveform(batch["waveform"], sample_rate=sample_rate)
        batch_hidden = np.stack(hidden, axis=1)
        hidden_chunks.append(batch_hidden)
        kept_indices.extend(batch["row_indices"])

    if not hidden_chunks:
        raise RuntimeError("No hidden states were extracted.")

    hidden_states = np.concatenate(hidden_chunks, axis=0)
    kept_df = df.iloc[kept_indices].reset_index(drop=True)
    labels = kept_df["label_id"].to_numpy(dtype=int)

    kept_df.to_csv(os.path.join(cache_dir, "manifest_with_labels.csv"), index=False)
    np.save(os.path.join(cache_dir, "hidden_states.npy"), hidden_states)
    np.save(os.path.join(cache_dir, "labels.npy"), labels)

    with open(os.path.join(cache_dir, "cache_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "num_samples": int(hidden_states.shape[0]),
                "num_layers": int(hidden_states.shape[1]),
                "hidden_dim": int(hidden_states.shape[2]),
            },
            f,
            indent=2,
        )

    print(f"Saved cache to {cache_dir}")
    print(f"Hidden states shape: {hidden_states.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["build_manifest", "extract_cache"], required=True)
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--manifest_csv", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    if args.mode == "build_manifest":
        if not args.base_dir or not args.manifest_csv:
            raise ValueError("--base_dir and --manifest_csv are required")
        dataset_roots = {
            "RAVDESS": os.path.join(args.base_dir, "RAVDESS"),
            "EmoDB": os.path.join(args.base_dir, "EmoDB"),
            "IEMOCAP": os.path.join(args.base_dir, "IEMOCAP"),
            "SAVEE": os.path.join(args.base_dir, "SAVEE"),
            "AESDD": os.path.join(args.base_dir, "AESDD"),
            "MESD": os.path.join(args.base_dir, "MESD"),
        }
        builder = CompiledBenchmarkBuilder()
        records = builder.build(dataset_roots)
        builder.save_csv(records, args.manifest_csv)
        print(f"Saved manifest to {args.manifest_csv}")
        print(f"Total rows: {len(records)}")

    elif args.mode == "extract_cache":
        if not args.manifest_csv or not args.model_name or not args.cache_dir:
            raise ValueError("--manifest_csv, --model_name, and --cache_dir are required")
        extract_and_cache(
            manifest_csv=args.manifest_csv,
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
