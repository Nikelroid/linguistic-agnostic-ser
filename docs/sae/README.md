# SAE study — Sparse Autoencoder dissection of frozen-encoder emotion features

Mechanistic companion to the `linguistic-agnostic-ser` probing work (the "N-1"
direction in `SAE_CONTEXT.md`). Trains TopK Sparse Autoencoders on frozen
speech-encoder activations and answers four questions:
**(i)** at which depth does emotion *disentangle* (vs decode); **(ii)** does the
pattern hold across encoders / speakers; **(iii)** is the recovered signal
acoustic or lexical; **(iv)** where does the lexical shortcut live (EMIS).

## Environment
Use the spack Python module (NOT the repo's `ser_env` conda env, which lacks
sae-lens):
```bash
module load gcc/13.3.0 python/3.11.9
pip install --user -r docs/sae/requirements.txt   # sae-lens, opensmile, sklearn, matplotlib
```
Validated on a Tesla V100-32GB (torch 2.6.0+cu124). SLURM jobs `module load` the
same python — see `slurm/sae_envtest.sbatch`.

## Code layout (integrated under `src/sae/`)
| path | role |
|------|------|
| `src/sae/extraction.py` | per-frame (un-pooled) + `--pooled` activation extraction; Whisper padding masked |
| `src/sae/sae_model.py` | self-contained TopK SAE (d_sae=8·d_in, k=32) + AuxK revival + train/encode/decode |
| `src/sae/analysis.py` | streaming encode→pool→stats, vectorized MI, **confound-controlled** `select_emotion_features`, `firing_auc`, monosemanticity, plots |
| `src/sae/acoustic_lexical.py` | eGeMAPS (openSMILE) + BERT helpers for acoustic-vs-lexical labeling |
| `src/sae/steps/` | the pipeline steps as importable modules (one per step) |
| `src/sae/steps/__main__.py` | unified CLI dispatcher (`python -m src.sae.steps <cmd>`) |
| `slurm/sae_extract.sbatch`, `slurm/sae_train.sbatch` | V100 array-job templates (encoder×dataset×layer) |
| `docs/sae/` | this README, `FINDINGS.md`, `EXP_INVENTORY.md`, deferred-step stubs, `requirements.txt` |
| `results/SAE/summary/paper/` | the write-up (ACM LaTeX) |

CLI: `python -m src.sae.steps <command> [args]` where command ∈
`inventory, extract, step1..step8, step8b, summary` (run `… help` for the map).

## Data
- **Cached pooled embeddings** (`/scratch1/kelidari/ser-experiments/EXP{N}`): used
  only for the inventory + probe references. EXP4 = clean categorical (6 enc × 6
  ds, mean-pooled `(N,25,1024)`); see `EXP_INVENTORY.md`.
- **Per-frame SAE inputs** (re-extracted, the actual SAE training data):
  `/scratch1/kelidari/ser-experiments/SAE_frames` → `SAE_ckpts` / `SAE_feats` / `SAE_aux`.
- **CREMA-D**: pulled from HuggingFace `confit/cremad` to
  `/scratch1/kelidari/ser_data/CREMA-D` (7,442 clips; GitHub LFS was exhausted).
- **EMIS**: pulled from the open Zenodo mirror (record 19207001) to
  `/scratch1/kelidari/ser_data/EMIS` (1,248 clips; IEEE DataPort was paywalled).
- Committed artifacts (plots/CSVs/metrics) live in `results/SAE/`. SAE `.pt`
  checkpoints stay on scratch (too big for git).

## Reproduce (HuBERT, CREMA-D, L13 unless noted)
```bash
module load gcc/13.3.0 python/3.11.9
# 0  inventory
python -m src.sae.steps inventory --markdown docs/sae/EXP_INVENTORY.md
# 0.5  CREMA-D frames (L13); for the depth sweep extract all 25 layers via the array job
python -m src.sae.steps extract --encoder HuBERT --dataset CREMA-D \
    --data-dir /scratch1/kelidari/ser_data/CREMA-D --layers 13
# 1 first SAE   2 emotion features   3 acoustic-vs-lexical control
python -m src.sae.steps step1 --encoder HuBERT --dataset CREMA-D --layer 13
python -m src.sae.steps step2 --encoder HuBERT --dataset CREMA-D --layer 13
python -m src.sae.steps step3 --encoder HuBERT --dataset CREMA-D --layer 13
# 4 depth sweep (needs all 25 layers' frames+SAEs — fan out on SLURM):
L=$(seq -s' ' 0 24)
sbatch --array=0-24%10 --export=ALL,ENCODERS_STR=HuBERT,DATASETS_STR=CREMA-D,LAYERS_STR="$L" slurm/sae_extract.sbatch
sbatch --array=0-24%10 --export=ALL,ENCODERS_STR=HuBERT,DATASETS_STR=CREMA-D,LAYERS_STR="$L" slurm/sae_train.sbatch
python -m src.sae.steps step4 --encoder HuBERT --dataset CREMA-D
# 5 cross-encoder   6 cross-speaker   7 ablation
python -m src.sae.steps step5
python -m src.sae.steps step6 --encoder HuBERT --dataset CREMA-D --layer 13
python -m src.sae.steps step7 --encoder HuBERT --dataset CREMA-D --layer 13
# 8 EMIS text-bias (per-encoder) + 8b layer/TTS sweep ; then package results
python -m src.sae.steps step8
python -m src.sae.steps step8b
python -m src.sae.steps summary
```
Tip: run analysis steps with `OMP_NUM_THREADS=4` to avoid BLAS oversubscription.

## Status of deferred arms (stubs)
- `STEP6_CROSSLINGUAL_STUB.md` — SpeechCraft audio not on cluster (cross-lingual arm).
- `STEP7_IEMOCAP_STUB.md` — IEMOCAP transcripts not synced (real-transcript ablation).
- `STEP8_EMIS_STUB.md` — historical; **EMIS is now run** (Steps 8/8b), kept for provenance.

See `FINDINGS.md` for results and `results/SAE/summary/paper/` for the write-up.
