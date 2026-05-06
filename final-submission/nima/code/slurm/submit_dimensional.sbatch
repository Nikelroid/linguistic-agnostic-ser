#!/bin/bash
#SBATCH --account=msoleyma_1026
#SBATCH --partition=gpu
#SBATCH --job-name=dim-regression
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err
#SBATCH --array=0-17

set -e

echo "=========================================="
echo " Starting MSP-Podcast Dimensional Regression Job: $SLURM_JOB_ID"
echo " Node Assigned: $SLURMD_NODENAME"
mkdir -p slurm/logs
echo "=========================================="

echo "Initializing Miniconda Core..."
source $HOME/miniconda/etc/profile.d/conda.sh
conda activate ser_env

cd /home1/$USER/linguistic-agnostic-ser

if [ -z "$BATCH_SIZE" ]; then BATCH_SIZE=4; fi
if [ -n "$MODELS_STR" ]; then MODELS=($MODELS_STR); else MODELS=("openai/whisper-medium" "facebook/hubert-large-ll60k" "microsoft/wavlm-large" "facebook/wav2vec2-large-960h" "m-a-p/MERT-v1-330M" "facebook/w2v-bert-2.0"); fi
TARGETS=("EmoVal" "EmoAct" "EmoDom")

# User requested clean SNR only for this experiment
SNR_DB="clean"
DATASET_NAME="MSP-Podcast"

# Calculate 2D Matrix Indices: model × target
N_TARGETS=${#TARGETS[@]}
N_MODELS=${#MODELS[@]}

ID_TARGET=$((SLURM_ARRAY_TASK_ID % N_TARGETS))
ID_MODEL=$((SLURM_ARRAY_TASK_ID / N_TARGETS))

MODEL_NAME="${MODELS[$ID_MODEL]}"
TARGET_FEATURE="${TARGETS[$ID_TARGET]}"

DATA_DIR="/scratch1/$USER/ser_data/${DATASET_NAME}"
NOISE_DIR="/scratch1/$USER/ser_data/ESC-50"

echo "=========================================================="
echo "--> Task ID: $SLURM_ARRAY_TASK_ID"
echo "--> Experiment: $DATASET_NAME x Target=${TARGET_FEATURE} x $MODEL_NAME"
echo "=========================================================="

# Force EXP_ID 8 as requested
EXP_ID=8

python -u -m scripts.run_pipeline \
  --task dimensional_regression \
  --data_dir "$DATA_DIR" \
  --dataset_name "$DATASET_NAME" \
  --target_feature "$TARGET_FEATURE" \
  --model_name "$MODEL_NAME" \
  --batch_size $BATCH_SIZE \
  --wandb_entity AGSER \
  --wandb_project linguistic-agnostic-ser \
  --snr_db "$SNR_DB" \
  --noise_dir "$NOISE_DIR" \
  --exp_id $EXP_ID

echo "=========================================="
echo " MSP-Podcast Dimensional Regression Pipeline Finished Successfully"
echo "=========================================="
