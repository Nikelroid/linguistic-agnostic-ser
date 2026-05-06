#!/bin/bash
#SBATCH --account=msoleyma_1026
#SBATCH --partition=gpu
#SBATCH --job-name=noisy-ser-pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err
#SBATCH --array=0-89

set -e

echo "=========================================="
echo " Starting Noisy Slurm Job: $SLURM_JOB_ID"
echo " Node Assigned: $SLURMD_NODENAME"
mkdir -p slurm/logs
echo "=========================================="

echo "Initializing Miniconda Core..."
source $HOME/miniconda/etc/profile.d/conda.sh
conda activate ser_env

cd /home1/$USER/linguistic-agnostic-ser

# Variables BATCH_SIZE, MODELS_STR, DATASETS_STR, SNR_STR should be passed via --export=ALL from setup_env.sh
# Convert space-separated strings to arrays, or use fallbacks
if [ -z "$BATCH_SIZE" ]; then BATCH_SIZE=4; fi
if [ -n "$MODELS_STR" ]; then MODELS=($MODELS_STR); else MODELS=("openai/whisper-medium" "facebook/hubert-large-ll60k" "microsoft/wavlm-large"); fi
if [ -n "$DATASETS_STR" ]; then DATASETS=($DATASETS_STR); else DATASETS=("IEMOCAP" "RAVDESS" "EmoDB" "SAVEE" "AESDD" "MESD"); fi
if [ -n "$SNR_STR" ]; then SNR_LEVELS=($SNR_STR); else SNR_LEVELS=("20" "10" "5" "0" "clean"); fi

# Calculate 3D Matrix Indices:  model × dataset × snr
# Total = 3 models × 6 datasets × 5 SNR = 90 jobs
N_DATASETS=${#DATASETS[@]}
N_SNR=${#SNR_LEVELS[@]}

ID_SNR=$((SLURM_ARRAY_TASK_ID % N_SNR))
ID_DATASET=$(((SLURM_ARRAY_TASK_ID / N_SNR) % N_DATASETS))
ID_MODEL=$((SLURM_ARRAY_TASK_ID / (N_SNR * N_DATASETS)))

DATASET_NAME="${DATASETS[$ID_DATASET]}"
MODEL_NAME="${MODELS[$ID_MODEL]}"
SNR_DB="${SNR_LEVELS[$ID_SNR]}"
DATA_DIR="/scratch1/$USER/ser_data/${DATASET_NAME}"
NOISE_DIR="/scratch1/$USER/ser_data/ESC-50"

echo "=========================================================="
echo "--> Task ID: $SLURM_ARRAY_TASK_ID"
echo "--> Experiment: $DATASET_NAME x $MODEL_NAME x SNR=${SNR_DB}dB"
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
  --snr_db "$SNR_DB" \
  --noise_dir "$NOISE_DIR" \
  $EXTRA_ARGS

echo "=========================================="
echo " Noisy Pipeline Finished Successfully"
echo "=========================================="
