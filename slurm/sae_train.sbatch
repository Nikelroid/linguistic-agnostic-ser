#!/bin/bash -l
# Train one TopK SAE per (encoder, dataset, layer) as a SLURM array job, reusing
# notebooks/sae/run_step1.py (which trains, saves the checkpoint + utterance
# feature matrices + a dashboard). Requires the frames to be extracted first
# (slurm/sae_extract.sbatch).
#
# Submit (depth sweep: HuBERT CREMA-D, all 25 layers):
#   L=$(seq -s' ' 0 24)
#   sbatch --array=0-24 \
#     --export=ALL,ENCODERS_STR="HuBERT",DATASETS_STR="CREMA-D",LAYERS_STR="$L",EXPANSION=8,KSPARSE=32 \
#     slurm/sae_train.sbatch
#SBATCH --account=msoleyma_1026
#SBATCH --partition=gpu
#SBATCH --job-name=sae-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

set -e
mkdir -p slurm/logs
module load gcc/13.3.0 python/3.11.9 2>/dev/null || true
PY=$(command -v python); [ -z "$PY" ] && PY=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/python-3.11.9-x74mtjf/bin/python
cd /home1/$USER/linguistic-agnostic-ser
export PYTHONUNBUFFERED=1

ENCODERS=(${ENCODERS_STR:-HuBERT})
DATASETS=(${DATASETS_STR:-CREMA-D})
LAYERS=(${LAYERS_STR:-13})
nL=${#LAYERS[@]}; nD=${#DATASETS[@]}
i=$SLURM_ARRAY_TASK_ID
li=$(( i % nL )); i=$(( i / nL ))
di=$(( i % nD )); ei=$(( i / nD ))
ENC=${ENCODERS[$ei]}; DS=${DATASETS[$di]}; L=${LAYERS[$li]}

echo "[$(date)] task $SLURM_ARRAY_TASK_ID -> SAE train encoder=$ENC dataset=$DS layer=$L (node $SLURMD_NODENAME)"
"$PY" -u -m src.sae.steps.step1_train_sae \
  --encoder "$ENC" --dataset "$DS" --layer "$L" \
  --expansion "${EXPANSION:-8}" --k "${KSPARSE:-32}" --steps "${STEPS:-4000}"
echo "[$(date)] done task $SLURM_ARRAY_TASK_ID"
