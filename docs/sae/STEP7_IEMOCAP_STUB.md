# Step 7 — IEMOCAP-transcript causal-ablation variant (BLOCKED, runnable stub)

**Status:** blocked on this cluster — IEMOCAP transcripts are not synced here.
`/scratch1/kelidari/ser_data/IEMOCAP` ships only `wav-session*/` audio and
`meta_data.json` (keys: `path`, `label`, `speaker` — **no transcript text**).
The full IEMOCAP release stores transcripts in
`IEMOCAP_full_release/Session{1..5}/dialog/transcriptions/*.txt`
(used by `notebooks/chaitanya/bert_alignment_transcripts.ipynb`, which read them
from a local copy). They are licensed, not public.

**What ran instead:** the causal-sufficiency ablation on CREMA-D
(`run_step7.py`), where the lexical axis is the fixed-sentence identity that
Step 3 validated. Result: emotion decodes from acoustic features far better than
from lexical/random ones → emotion is acoustic.

**To run the exact IEMOCAP variant once transcripts are available:**

1. Drop the transcripts under `/scratch1/kelidari/ser_data/IEMOCAP/transcriptions/`
   and parse `utterance_id -> text` (line format `Ses01F_impro01_F000 [t-t]: text`).
2. Extract per-frame HuBERT activations (best IEMOCAP layer — spontaneous IEMOCAP
   peaks early, ~L1–L4; sweep with `run_step4.py --dataset IEMOCAP`):
   `sbatch --export=ALL,ENCODERS_STR=HuBERT,DATASETS_STR=IEMOCAP,LAYERS_STR="<L>" slurm/sae_extract.sbatch`
3. Train the SAE: `python notebooks/sae/run_step1.py --dataset IEMOCAP --layer <L>`.
4. Select emotion features with `select_emotion_features(..., confounds={"speaker": speaker})`.
5. Label each acoustic vs lexical with `analysis.firing_auc(firing, eGeMAPS)` vs
   `analysis.firing_auc(firing, bert_embed(transcripts))`
   (`acoustic_lexical.bert_embed` already handles varying transcripts).
6. Sufficiency + necessity probes exactly as `run_step7.py`, but now "lexical"
   = real transcript content. Expectation under the project's C3 finding: SSL
   encoders should show emotion partly decodable from lexical features (the
   lexical shortcut), Whisper much less so.
