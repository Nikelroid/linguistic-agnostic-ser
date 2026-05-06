"""Extract BERT (bert-base-uncased, mean-pool) embeddings for RAVDESS+SAVEE transcripts.

Output: per-clip BERT embedding aligned with hidden_states_HuBERT_RAVDESS/SAVEE order.
- RAVDESS: maps statement code (01 or 02) to known sentence
- SAVEE:   uses sentence code (a01..su15) as text — BERT tokenizes each distinctly.
           This is sufficient for CCA which only needs distinct text-side variation.
"""
import os
import json
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

SER_DATA = "/scratch1/minooahm/ser_data"

RAVDESS_SENTENCES = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}


def parse_ravdess(fn):
    parts = str(fn).replace(".wav", "").split("-")
    return RAVDESS_SENTENCES.get(parts[4], parts[4]) if len(parts) >= 5 else str(fn)


def parse_savee(fn):
    base = str(fn).replace(".wav", "")
    parts = base.split("_")
    return parts[-1] if len(parts) >= 2 else base


def mean_pool(last_hidden, attn_mask):
    mask = attn_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(1) / mask.sum(1)


def main():
    print("Loading BERT base uncased...")
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased").eval()

    # Use HuBERT filenames as canonical (all models share filename order at extraction time)
    fns_rav = np.load(f"{SER_DATA}/filenames_HuBERT_RAVDESS.npy", allow_pickle=True)
    fns_sav = np.load(f"{SER_DATA}/filenames_HuBERT_SAVEE.npy", allow_pickle=True)
    texts_rav = [parse_ravdess(fn) for fn in fns_rav]
    texts_sav = [parse_savee(fn) for fn in fns_sav]

    n_uniq_rav = len(set(texts_rav))
    n_uniq_sav = len(set(texts_sav))
    print(f"  RAVDESS clips: {len(texts_rav)} | unique transcripts: {n_uniq_rav}")
    print(f"  SAVEE clips:   {len(texts_sav)} | unique transcripts: {n_uniq_sav}")

    # Embed each unique text once for efficiency
    unique_texts = sorted(set(texts_rav) | set(texts_sav))
    print(f"  Total unique transcripts to embed: {len(unique_texts)}")

    text_to_emb = {}
    with torch.no_grad():
        for i, t in enumerate(unique_texts):
            inputs = tok(t, return_tensors="pt", truncation=True, max_length=64)
            out = bert(**inputs)
            pooled = mean_pool(out.last_hidden_state, inputs["attention_mask"])
            text_to_emb[t] = pooled.squeeze(0).numpy().astype(np.float32)

    # Build per-clip arrays
    emb_rav = np.stack([text_to_emb[t] for t in texts_rav])
    emb_sav = np.stack([text_to_emb[t] for t in texts_sav])
    np.save(f"{SER_DATA}/bert_embeddings_RAVDESS.npy", emb_rav)
    np.save(f"{SER_DATA}/bert_embeddings_SAVEE.npy", emb_sav)

    with open(f"{SER_DATA}/bert_text_mapping.json", "w") as f:
        json.dump({
            "ravdess_unique_transcripts": n_uniq_rav,
            "savee_unique_transcripts": n_uniq_sav,
            "ravdess_sentences": RAVDESS_SENTENCES,
            "savee_uses_codes_as_text": True,
            "savee_codes_sample": sorted(set(texts_sav))[:10],
            "bert_dim": int(emb_rav.shape[1]),
        }, f, indent=2)

    print(f"Saved BERT: RAVDESS {emb_rav.shape}, SAVEE {emb_sav.shape}")


if __name__ == "__main__":
    main()
