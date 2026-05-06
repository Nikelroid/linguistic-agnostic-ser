"""Extract BERT embeddings for IEMOCAP transcripts (5460 unique utterances).

Used as a richer CCA training corpus for Task 6 V3 (vs RAV+SAV's 122 unique).
"""
import os
import re
import glob
import json
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

SER_DATA = "/scratch1/minooahm/ser_data"
IEMOCAP_ROOT = "/project2/msoleyma_1026/IEMOCAP_full_release"

UTT_LINE_RE = re.compile(r"^(Ses\S+)\s*\[\d+\.\d+-\d+\.\d+\]:\s*(.+)$")


def parse_iemocap_transcripts(root):
    """Returns dict: utt_id (e.g. Ses01F_impro01_F000) -> transcript text."""
    utt2text = {}
    paths = sorted(glob.glob(os.path.join(root, "Session*", "dialog", "transcriptions", "*.txt")))
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = UTT_LINE_RE.match(line.strip())
                if m:
                    utt2text[m.group(1)] = m.group(2).strip()
    return utt2text


def mean_pool(last_hidden, attn_mask):
    mask = attn_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(1) / mask.sum(1)


def main():
    print("Parsing IEMOCAP transcripts...")
    utt2text = parse_iemocap_transcripts(IEMOCAP_ROOT)
    print(f"  Parsed {len(utt2text)} unique utterances")

    # Match to filenames in hidden_states_HuBERT_IEMOCAP.npy order
    fns = np.load(f"{SER_DATA}/filenames_HuBERT_IEMOCAP.npy", allow_pickle=True)
    texts = []
    missing = []
    for fn in fns:
        utt_id = str(fn).replace(".wav", "")
        if utt_id in utt2text:
            texts.append(utt2text[utt_id])
        else:
            texts.append("[MISSING]")
            missing.append(utt_id)
    print(f"  Matched: {len(fns) - len(missing)}/{len(fns)} ({100*(len(fns)-len(missing))/len(fns):.1f}%)")
    if missing:
        print(f"  Missing examples: {missing[:5]}")

    print("Loading BERT base uncased...")
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased").eval()

    # Embed unique texts once
    unique_texts = sorted(set(texts))
    print(f"  Unique transcripts (incl [MISSING] if any): {len(unique_texts)}")

    text_to_emb = {}
    with torch.no_grad():
        for i, t in enumerate(unique_texts):
            inputs = tok(t, return_tensors="pt", truncation=True, max_length=128)
            out = bert(**inputs)
            pooled = mean_pool(out.last_hidden_state, inputs["attention_mask"])
            text_to_emb[t] = pooled.squeeze(0).numpy().astype(np.float32)
            if (i + 1) % 1000 == 0:
                print(f"    {i+1}/{len(unique_texts)} embedded")

    emb_per_clip = np.stack([text_to_emb[t] for t in texts])
    np.save(f"{SER_DATA}/bert_embeddings_IEMOCAP.npy", emb_per_clip)

    with open(f"{SER_DATA}/bert_text_mapping_iemocap.json", "w") as f:
        json.dump({
            "n_clips": int(len(fns)),
            "n_matched": int(len(fns) - len(missing)),
            "n_unique_texts": int(len(unique_texts)),
            "bert_dim": int(emb_per_clip.shape[1]),
        }, f, indent=2)

    print(f"Saved BERT IEMOCAP: {emb_per_clip.shape}")


if __name__ == "__main__":
    main()
