#!/usr/bin/env python
"""Label SAE emotion features as acoustic vs lexical.

For each emotion feature we ask: is its (utterance-pooled) activation better
explained by **acoustic** descriptors (eGeMAPS, openSMILE) or by **lexical**
content (BERT embedding of the transcript)? We fit a cross-validated ridge
regression of the feature on each side and compare R²:

    r2_acoustic > r2_lexical  ->  "acoustic"
    r2_lexical  > r2_acoustic ->  "lexical"

On CREMA-D the lexicon is fixed (12 carrier sentences), so emotion features
*must* come out acoustic — the positive control that the labeling works. The
same machinery runs on IEMOCAP (real transcripts) for the Step 7 causal ablation.
"""
from __future__ import annotations

import numpy as np

# CREMA-D's 12 fixed carrier sentences (sentence code -> transcript).
CREMAD_SENTENCES = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}


def egemaps_functionals(records, sample_rate: int = 16000) -> np.ndarray:
    """eGeMAPSv02 functionals (88-dim) per utterance, aligned to ``records`` order."""
    import opensmile

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    feats = []
    for rec in records:
        audio = np.asarray(rec["audio"], dtype=np.float64)
        df = smile.process_signal(audio, sample_rate)
        feats.append(df.values[0])
    return np.asarray(feats, dtype=np.float32)


def bert_embed(texts: list[str], model_name: str = "bert-base-uncased", device=None) -> np.ndarray:
    """Mean-pooled BERT embeddings (n, 768) for a list of transcripts."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            hs = model(**enc).last_hidden_state              # (b, T, 768)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp_min(1)
            out.append(pooled.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def cv_r2(y: np.ndarray, X: np.ndarray, alpha: float = 10.0, cv: int = 5, seed: int = 42) -> float:
    """Cross-validated R² of ridge regression y ~ X (standardised features)."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if np.std(y) < 1e-9:
        return 0.0
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
    return float(np.mean(scores))


def label_features(utt_feats: np.ndarray, feature_ids, X_acoustic: np.ndarray,
                   X_lexical: np.ndarray, alpha: float = 10.0) -> "list[dict]":
    """Per-feature acoustic-vs-lexical R² comparison for the given feature ids."""
    rows = []
    for fid in feature_ids:
        y = utt_feats[:, fid]
        r2_ac = cv_r2(y, X_acoustic, alpha=alpha)
        r2_lex = cv_r2(y, X_lexical, alpha=alpha)
        rows.append({
            "feature": int(fid),
            "r2_acoustic": r2_ac,
            "r2_lexical": r2_lex,
            "margin": r2_ac - r2_lex,
            "label": "acoustic" if r2_ac >= r2_lex else "lexical",
        })
    return rows


def cremad_transcripts(sentence_codes) -> list[str]:
    """Map an array of CREMA-D sentence codes to their transcript strings."""
    return [CREMAD_SENTENCES.get(str(c), "") for c in sentence_codes]
