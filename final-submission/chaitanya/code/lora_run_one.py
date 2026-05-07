"""Run a single (model, dataset) LoRA fine-tune combo in an isolated process.

This exists because running all 30 combos in one notebook kernel exhausts the
16 GB RAM on this M-series Mac (PyTorch + MPS + HF models don't fully release
memory between iterations). By wrapping each combo in its own subprocess, the
OS reclaims everything cleanly between combos.

Usage:
    python lora_run_one.py --model HuBERT --dataset EmoDB

Reads cells 1, 3, 5, 7, 9 from lora_finetune.ipynb verbatim at startup so any
notebook edits flow through automatically.
"""
import argparse
import json
import os
import sys
import pathlib
import gc

NB_PATH = pathlib.Path(__file__).parent / 'lora_finetune.ipynb'


def exec_cell(ns, nb, idx):
    src = ''.join(nb['cells'][idx]['source'])
    # Strip jupyter line magics (e.g. %matplotlib inline) which aren't valid Python
    src = '\n'.join(ln for ln in src.split('\n') if not ln.lstrip().startswith('%'))
    exec(src, ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='Model key from MODEL_CONFIGS')
    ap.add_argument('--dataset', required=True, help='Dataset name')
    args = ap.parse_args()

    nb = json.loads(NB_PATH.read_text())
    ns = {'__name__': '__main__'}

    # Cell 1: imports
    exec_cell(ns, nb, 1)
    # Cell 3: config (defines MODEL_CONFIGS, AUDIO_PATHS, DATASETS, etc.)
    exec_cell(ns, nb, 3)
    # Cell 5: audio loader functions
    exec_cell(ns, nb, 5)
    # Cell 7: AudioDataset, SERLoRAModel
    exec_cell(ns, nb, 7)
    # Cell 9: finetune_and_extract()
    exec_cell(ns, nb, 9)

    MODEL_CONFIGS = ns['MODEL_CONFIGS']
    AUDIO_PATHS = ns['AUDIO_PATHS']
    LOADERS = ns['LOADERS']
    RESULTS_DIR = ns['RESULTS_DIR']
    finetune_and_extract = ns['finetune_and_extract']
    LabelEncoder = ns['LabelEncoder']
    np = ns['np']
    torch = ns['torch']

    if args.model not in MODEL_CONFIGS:
        print(f'ERROR: unknown model {args.model}. Known: {list(MODEL_CONFIGS.keys())}')
        sys.exit(2)
    if args.dataset not in AUDIO_PATHS:
        print(f'ERROR: unknown dataset {args.dataset}. Known: {list(AUDIO_PATHS.keys())}')
        sys.exit(2)

    save_hs = os.path.join(RESULTS_DIR, f'tuned_hidden_states_{args.model}_{args.dataset}.npy')
    save_lb = os.path.join(RESULTS_DIR, f'tuned_labels_{args.model}_{args.dataset}.npy')
    if os.path.exists(save_hs) and os.path.exists(save_lb):
        print(f'CACHED {args.model}/{args.dataset}, skipping')
        return 0

    hf_name = MODEL_CONFIGS[args.model]
    path = AUDIO_PATHS[args.dataset]
    loader_fn = LOADERS[args.dataset]

    print(f'Loading {args.dataset} audio from {path}...', flush=True)
    data = loader_fn(path)
    if not data:
        print(f'ERROR: no audio loaded for {args.dataset}')
        return 3
    print(f'  {len(data)} samples loaded', flush=True)

    le = LabelEncoder()
    le.fit([d['label'] for d in data])
    num_classes = len(le.classes_)

    print(f'\n{"="*60}\nFine-tuning {args.model} on {args.dataset} ({len(data)} samples, {num_classes} classes)\n{"="*60}', flush=True)

    try:
        result = finetune_and_extract(args.model, hf_name, args.dataset, data, num_classes, le)
    except Exception as e:
        print(f'FAILED {args.model}/{args.dataset}: {type(e).__name__}: {e}', flush=True)
        import traceback
        traceback.print_exc()
        return 4

    np.save(save_hs, result['tuned_hidden_states'])
    np.save(save_lb, result['labels'])
    print(f'\nSaved {save_hs}', flush=True)
    print(f'Saved {save_lb}', flush=True)
    print(f'Acc: {result["finetune_acc"]:.4f} ± {result["finetune_std"]:.4f}', flush=True)

    # Best-effort cleanup (but process exit will reclaim everything anyway)
    del data, result
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return 0


if __name__ == '__main__':
    sys.exit(main())
