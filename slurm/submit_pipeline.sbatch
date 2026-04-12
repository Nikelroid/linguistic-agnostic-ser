#!/bin/bash
#SBATCH --account=msoleyma_1026
#SBATCH --partition=gpu
#SBATCH --job-name=agnostic-ser-pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err
#SBATCH --array=0-35

set -e

echo "=========================================="
echo " Starting Slurm Job: $SLURM_JOB_ID"
echo " Node Assigned: $SLURMD_NODENAME"
mkdir -p slurm/logs
echo "=========================================="

echo "Initializing Miniconda Core..."
source $HOME/miniconda/etc/profile.d/conda.sh
conda activate ser_env

cd /home1/$USER/linguistic-agnostic-ser

BATCH_SIZE=4
DATASETS=("RAVDESS" "EmoDB" "IEMOCAP" "SAVEE" "AESDD" "MESD")
MODELS=("facebook/wav2vec2-large-960h" "facebook/hubert-large-ll60k" "openai/whisper-medium" "microsoft/wavlm-large" "m-a-p/MERT-v1-330M" "facebook/w2v-bert-2.0")

# Calculate Matrix Indices
N_DATASETS=${#DATASETS[@]}
ID_DATASET=$((SLURM_ARRAY_TASK_ID % N_DATASETS))
ID_MODEL=$((SLURM_ARRAY_TASK_ID / N_DATASETS))

DATASET_NAME="${DATASETS[$ID_DATASET]}"
MODEL_NAME="${MODELS[$ID_MODEL]}"
DATA_DIR="/scratch1/$USER/ser_data/${DATASET_NAME}"

echo "=========================================================="
echo "--> Task ID: $SLURM_ARRAY_TASK_ID | Experiment: $DATASET_NAME x $MODEL_NAME"
echo "=========================================================="

EXTRA_ARGS=""
if [ -n "$EXP_ID" ]; then
    EXTRA_ARGS="--exp_id $EXP_ID"
fi

python -u -m scripts.run_pipeline \
  --task classification \
  --data_dir "$DATA_DIR" \
  --dataset_name "$DATASET_NAME" \
  --model_name "$MODEL_NAME" \
  --batch_size $BATCH_SIZE \
  --wandb_entity AGSER \
  --wandb_project linguistic-agnostic-ser \
  $EXTRA_ARGS

echo "=========================================="
echo " Pipeline Finished Successfully"
echo "=========================================="
