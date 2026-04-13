#!/bin/bash
#SBATCH --account=msoleyma_1026
#SBATCH --partition=gpu
#SBATCH --job-name=msp-ser-pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err
#SBATCH --array=0-29

set -e

echo "=========================================="
echo " Starting MSP-Podcast Slurm Job: $SLURM_JOB_ID"
echo " Node Assigned: $SLURMD_NODENAME"
mkdir -p slurm/logs
echo "=========================================="

echo "Initializing Miniconda Core..."
source $HOME/miniconda/etc/profile.d/conda.sh
conda activate ser_env

cd /home1/$USER/linguistic-agnostic-ser

# Variables BATCH_SIZE, MODELS_STR, SNR_STR should be passed via --export=ALL from setup_env.sh
# Convert space-separated strings to arrays, or use fallbacks
if [ -z "$BATCH_SIZE" ]; then BATCH_SIZE=4; fi
if [ -n "$MODELS_STR" ]; then MODELS=($MODELS_STR); else MODELS=("openai/whisper-medium" "facebook/hubert-large-ll60k" "microsoft/wavlm-large"); fi
if [ -n "$SNR_STR" ]; then SNR_LEVELS=($SNR_STR); else SNR_LEVELS=("clean" "20" "10" "5" "0"); fi

DATASET_NAME="MSP-Podcast"

# Calculate 2D Matrix Indices: model × snr
# Total = 6 models × 5 SNR = 30 jobs
N_SNR=${#SNR_LEVELS[@]}
N_MODELS=${#MODELS[@]}

ID_SNR=$((SLURM_ARRAY_TASK_ID % N_SNR))
ID_MODEL=$((SLURM_ARRAY_TASK_ID / N_SNR))

MODEL_NAME="${MODELS[$ID_MODEL]}"
SNR_DB="${SNR_LEVELS[$ID_SNR]}"
DATA_DIR="/scratch1/$USER/ser_data/${DATASET_NAME}"
NOISE_DIR="/scratch1/$USER/ser_data/ESC-50"

echo "=========================================================="
echo "--> Task ID: $SLURM_ARRAY_TASK_ID"
echo "--> Experiment: $DATASET_NAME x $MODEL_NAME x SNR=${SNR_DB}dB"
echo "=========================================================="

# Force EXP_ID=6 for MSP-Podcast
EXP_ID=6

EXTRA_ARGS="--exp_id $EXP_ID"

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
echo " MSP-Podcast Pipeline Finished Successfully"
echo "=========================================="
