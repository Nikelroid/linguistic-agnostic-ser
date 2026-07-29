#!/usr/bin/env python
"""Inventory the cached per-layer embedding folders for the SAE study.

The SAE pipeline consumes the project's frozen-encoder hidden states, which are
cached under ``/scratch1/$USER/ser-experiments/EXP{N}/embeddings`` as triples:

    hidden_states_{ENCODER}_{DATASET}[_SNR{level}].npy   float32  (N, 25, H)
    labels_{ENCODER}_{DATASET}[_SNR{level}].npy          str/obj  (N,)
    filenames_{ENCODER}_{DATASET}[_SNR{level}].npy        obj      (N,)

The ``25`` axis is the per-utterance, mean-pooled hidden state of each of the
L0 (CNN/embedding front-end) + L1..L24 transformer layers. ``H`` is the encoder
hidden size (1024 for the *-large encoders; w2v-BERT/MERT differ).

This script does NOT assume the layout — it discovers what is actually on disk
and prints (and optionally writes) a coverage report. Run it before building any
SAE data loader, and keep its output as the committed inventory artifact.

Usage:
    python notebooks/sae/inventory_exp.py
    python notebooks/sae/inventory_exp.py --root /scratch1/$USER/ser-experiments \
        --markdown notebooks/sae/EXP_INVENTORY.md
"""
from __future__ import annotations

import argparse
import glob
import os

SER_SCRATCH = os.environ.get(
    "SER_SCRATCH", os.path.join("/scratch1", os.environ.get("USER", "unknown")))
import re
from collections import defaultdict

import numpy as np

DEFAULT_ROOT = os.path.join(SER_SCRATCH, "ser-experiments")
KINDS = ("hidden_states", "labels", "filenames")
_FNAME_RE = re.compile(r"^(hidden_states|labels|filenames)_(.+)\.npy$")


def parse_stub(stub: str):
    """Split '<ENCODER>_<DATASET>[_SNR<x>]' into (encoder, dataset, snr|None).

    Encoder is always the first underscore-token (HuBERT, WavLM, wav2vec2,
    w2v-BERT, MERT, Whisper). The SNR tag, if present, is the trailing token.
    Everything between is the dataset (may itself contain underscores, e.g.
    'MSP-Podcast_EmoVal').
    """
    parts = stub.split("_")
    encoder = parts[0]
    rest = parts[1:]
    snr = None
    if rest and rest[-1].startswith("SNR"):
        snr = rest[-1]
        rest = rest[:-1]
    dataset = "_".join(rest)
    return encoder, dataset, snr


def safe_load_meta(path: str):
    """Load a small label/filename array tolerating pickled object arrays."""
    try:
        return np.load(path, allow_pickle=True)
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"<load error: {exc}>"


def inventory_folder(exp_dir: str) -> dict:
    emb = os.path.join(exp_dir, "embeddings")
    info = {
        "path": exp_dir,
        "has_embeddings": os.path.isdir(emb),
        "subdirs": sorted(
            d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))
        )
        if os.path.isdir(exp_dir)
        else [],
        "combos": {},  # (encoder, dataset) -> {snrs, shape, dtype, n, layers, labels, example_fn}
    }
    if not info["has_embeddings"]:
        return info

    hs_files = sorted(glob.glob(os.path.join(emb, "hidden_states_*.npy")))
    for hs in hs_files:
        m = _FNAME_RE.match(os.path.basename(hs))
        if not m:
            continue
        encoder, dataset, snr = parse_stub(m.group(2))
        key = (encoder, dataset)
        rec = info["combos"].setdefault(
            key,
            {"snrs": set(), "shape": None, "dtype": None, "labels": None, "example_fn": None},
        )
        rec["snrs"].add(snr or "clean")
        # Inspect array shape only once per combo, preferring the clean variant.
        if rec["shape"] is None or snr is None:
            arr = np.load(hs, mmap_mode="r")
            rec["shape"] = tuple(arr.shape)
            rec["dtype"] = str(arr.dtype)
            lab_path = hs.replace("hidden_states_", "labels_")
            fn_path = hs.replace("hidden_states_", "filenames_")
            if os.path.exists(lab_path):
                lab = safe_load_meta(lab_path)
                if isinstance(lab, np.ndarray):
                    rec["labels"] = sorted({str(x) for x in lab.tolist()})
            if os.path.exists(fn_path):
                fn = safe_load_meta(fn_path)
                if isinstance(fn, np.ndarray) and len(fn):
                    rec["example_fn"] = str(fn[0])
    return info


def print_report(root: str, infos: list[dict]) -> str:
    """Print the inventory and return the same content as a markdown string."""
    lines: list[str] = []

    def emit(s: str = ""):
        print(s)
        lines.append(s)

    emit(f"# EXP embedding inventory\n\nRoot: `{root}`\n")
    all_encoders, all_datasets = set(), set()
    for info in infos:
        exp = os.path.basename(info["path"])
        if not info["has_embeddings"]:
            emit(f"## {exp}\n\n- no `embeddings/` dir; subdirs: {info['subdirs']}\n")
            continue
        combos = info["combos"]
        encoders = sorted({e for (e, _) in combos})
        datasets = sorted({d for (_, d) in combos})
        all_encoders.update(encoders)
        all_datasets.update(datasets)
        n_clean = sum(1 for r in combos.values() if "clean" in r["snrs"])
        emit(f"## {exp}\n")
        emit(f"- combos: {len(combos)} (clean: {n_clean})")
        emit(f"- encoders: {encoders}")
        emit(f"- datasets: {datasets}")
        # one representative shape
        any_rec = next(iter(combos.values()))
        emit(f"- example hidden_states shape: {any_rec['shape']} dtype={any_rec['dtype']}")
        emit("")
        emit("| encoder | dataset | shape | SNR variants | labels | example filename |")
        emit("|---|---|---|---|---|---|")
        for (enc, ds), rec in sorted(combos.items()):
            snrs = ",".join(sorted(rec["snrs"]))
            labs = ",".join(rec["labels"]) if rec["labels"] else "—"
            if len(labs) > 60:
                labs = labs[:57] + "..."
            emit(
                f"| {enc} | {ds} | {rec['shape']} | {snrs} | {labs} | "
                f"`{rec['example_fn']}` |"
            )
        emit("")

    emit("## Summary\n")
    emit(f"- all encoders seen: {sorted(all_encoders)}")
    emit(f"- all datasets seen: {sorted(all_datasets)}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="ser-experiments root")
    ap.add_argument("--markdown", default=None, help="optional path to write a markdown report")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        raise SystemExit(f"root not found: {args.root}")

    exp_dirs = sorted(
        (os.path.join(args.root, d) for d in os.listdir(args.root) if d.startswith("EXP")),
        key=lambda p: (len(os.path.basename(p)), os.path.basename(p)),
    )
    infos = [inventory_folder(d) for d in exp_dirs]
    md = print_report(args.root, infos)
    if args.markdown:
        with open(args.markdown, "w") as fh:
            fh.write(md)
        print(f"\n[wrote markdown report to {args.markdown}]")


if __name__ == "__main__":
    main()
