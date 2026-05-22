# SAE study — Sparse Autoencoder dissection of frozen-encoder emotion features

Mechanistic companion to the `linguistic-agnostic-ser` probing work (the "N-1"
direction in `SAE_CONTEXT.md`). Trains TopK Sparse Autoencoders on frozen
speech-encoder activations and asks, of the three project questions:
**(i)** at which depth does emotion *disentangle* (vs decode); **(ii)** does the
pattern hold across encoders / speakers / languages; **(iii)** is the recovered
signal acoustic or lexical.

## Environment
Use the spack Python module (NOT the repo's `ser_env` conda env, which lacks
sae-lens):
```bash
module load gcc/13.3.0 python/3.11.9
pip install --user -r notebooks/sae/requirements.txt   # sae-lens, opensmile, sklearn, matplotlib
```
Validated on a Tesla V100-32GB (torch 2.6.0+cu124). SLURM jobs `module load` the
same python — see `slurm/sae_envtest.sbatch`.

## Code
| file | role |
|------|------|
| `src/sae/extraction.py` | per-frame (un-pooled) frozen-encoder activation extraction; Whisper padding masked |
| `src/sae/sae_model.py` | self-contained TopK SAE (d_sae=8·d_in, k=32) + AuxK revival + train/encode/decode |
| `src/sae/analysis.py` | streaming encode→utterance-pool→stats, vectorized MI, **confound-controlled** `select_emotion_features`, `firing_auc`, monosemanticity, plots |
| `src/sae/acoustic_lexical.py` | eGeMAPS (openSMILE) + BERT helpers for acoustic-vs-lexical labeling |
| `notebooks/sae/inventory_exp.py` | discovers the cached EXP embedding layout (→ `EXP_INVENTORY.md`) |
| `notebooks/sae/run_step{1..7}.py` | one step each |
| `slurm/sae_extract.sbatch`, `slurm/sae_train.sbatch` | V100 array-job templates (encoder×dataset×layer) |

## Data
- **Cached pooled embeddings** (`/scratch1/kelidari/ser-experiments/EXP{N}`): used
  only for the inventory + probe references. EXP4 = clean categorical (6 enc × 6
  ds, mean-pooled `(N,25,1024)`); see `EXP_INVENTORY.md`.
- **Per-frame SAE inputs** (re-extracted, the actual SAE training data):
  `/scratch1/kelidari/ser-experiments/SAE_frames` → `SAE_ckpts` / `SAE_feats` / `SAE_aux`.
- **CREMA-D**: not cached + GitHub LFS budget exhausted → pulled from HuggingFace
  `confit/cremad` to `/scratch1/kelidari/ser_data/CREMA-D` (7,442 clips verified).
- Committed artifacts (plots/CSVs/metrics) live in `results/SAE/`. SAE `.pt`
  checkpoints stay on scratch (too big for git).

## Reproduce (HuBERT, CREMA-D, L13 unless noted)
```bash
module load gcc/13.3.0 python/3.11.9
# 0  inventory
python notebooks/sae/inventory_exp.py --markdown notebooks/sae/EXP_INVENTORY.md
# 0.5  CREMA-D frames (L13); for the depth sweep extract all layers via the array job
python -m src.sae.extraction --encoder HuBERT --dataset CREMA-D \
    --data-dir /scratch1/kelidari/ser_data/CREMA-D --layers 13
# 1  first SAE        2  emotion features   3  acoustic-vs-lexical control
python notebooks/sae/run_step1.py --encoder HuBERT --dataset CREMA-D --layer 13
python notebooks/sae/run_step2.py --encoder HuBERT --dataset CREMA-D --layer 13
python notebooks/sae/run_step3.py --encoder HuBERT --dataset CREMA-D --layer 13
# 4  depth sweep (needs all 25 layers' frames+SAEs — fan out on SLURM):
L=$(seq -s' ' 0 24)
sbatch --array=0-24%10 --export=ALL,ENCODERS_STR=HuBERT,DATASETS_STR=CREMA-D,LAYERS_STR="$L" slurm/sae_extract.sbatch
sbatch --array=0-24%10 --export=ALL,ENCODERS_STR=HuBERT,DATASETS_STR=CREMA-D,LAYERS_STR="$L" slurm/sae_train.sbatch
python notebooks/sae/run_step4.py --encoder HuBERT --dataset CREMA-D
# 5  cross-encoder (extract+train the other 5 at their best layers first), 6 cross-speaker, 7 ablation
python notebooks/sae/run_step5.py
python notebooks/sae/run_step6.py --encoder HuBERT --dataset CREMA-D --layer 13
python notebooks/sae/run_step7.py --encoder HuBERT --dataset CREMA-D --layer 13
```
Tip: run analysis steps with `OMP_NUM_THREADS=4` to avoid BLAS oversubscription.

## Deferred / blocked (runnable stubs)
- `STEP6_CROSSLINGUAL_STUB.md` — SpeechCraft audio not on cluster.
- `STEP7_IEMOCAP_STUB.md` — IEMOCAP transcripts not synced (only wav + meta).
- `STEP8_EMIS_STUB.md` — EMIS not available; two-regime audio-vs-text feature check.

See `FINDINGS.md` for results.
