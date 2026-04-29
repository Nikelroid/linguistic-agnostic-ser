#!/bin/bash
#SBATCH --account=msoleyma_1026
#SBATCH --partition=gpu
#SBATCH --job-name=speechcraft-exp20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err
#SBATCH --array=0-5

set -e

mkdir -p slurm/logs
echo "=========================================="
echo " Speechcraft EXP20 | Slurm Job: $SLURM_JOB_ID | Task: $SLURM_ARRAY_TASK_ID"
echo " Node: $SLURMD_NODENAME"
echo "=========================================="

source $HOME/miniconda/etc/profile.d/conda.sh
conda activate ser_env

cd /home1/$USER/linguistic-agnostic-ser

# Cache redirects (preserve home quota)
export HF_HOME="/scratch1/$USER/hf_cache"
export TRANSFORMERS_CACHE="/scratch1/$USER/hf_cache"
export TRITON_CACHE_DIR="/scratch1/$USER/triton_cache"
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR"

MODELS=(
  "facebook/wav2vec2-large-960h"
  "facebook/hubert-large-ll60k"
  "openai/whisper-medium"
  "microsoft/wavlm-large"
  "m-a-p/MERT-v1-330M"
  "facebook/w2v-bert-2.0"
)

if [ -z "$BATCH_SIZE" ]; then BATCH_SIZE=4; fi
if [ -z "$EXP_ID" ]; then EXP_ID=20; fi

MODEL_NAME="${MODELS[$SLURM_ARRAY_TASK_ID]}"
echo "--> Model: $MODEL_NAME"

python -u -m scripts.run_speechcraft \
  --model_name "$MODEL_NAME" \
  --batch_size $BATCH_SIZE \
  --exp_id $EXP_ID

echo "=========================================="
echo " Speechcraft task done"
echo "=========================================="
