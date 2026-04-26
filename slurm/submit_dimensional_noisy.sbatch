#!/bin/bash
#SBATCH --account=msoleyma_1026
#SBATCH --partition=gpu
#SBATCH --job-name=dim-noisy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err
#SBATCH --array=0-71

set -e

echo "=========================================="
echo " Starting Noisy Dimensional Regression Job: $SLURM_JOB_ID"
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

# Filter out 'clean' from SNR_STR if it was passed, otherwise use default
if [ -n "$SNR_STR" ]; then
    # Create an array of SNRs, skipping 'clean'
    NOISY_SNRS=()
    for snr in $SNR_STR; do
        if [ "$snr" != "clean" ]; then
            NOISY_SNRS+=("$snr")
        fi
    done
else
    NOISY_SNRS=("20" "10" "5" "0")
fi

DATASET_NAME="MSP-Podcast"

# Calculate 3D Matrix Indices: model × target × snr
N_SNRS=${#NOISY_SNRS[@]}
N_TARGETS=${#TARGETS[@]}
N_MODELS=${#MODELS[@]}

ID_SNR=$((SLURM_ARRAY_TASK_ID % N_SNRS))
ID_TARGET=$(((SLURM_ARRAY_TASK_ID / N_SNRS) % N_TARGETS))
ID_MODEL=$((SLURM_ARRAY_TASK_ID / (N_SNRS * N_TARGETS)))

MODEL_NAME="${MODELS[$ID_MODEL]}"
TARGET_FEATURE="${TARGETS[$ID_TARGET]}"
SNR_DB="${NOISY_SNRS[$ID_SNR]}"

DATA_DIR="/scratch1/$USER/ser_data/${DATASET_NAME}"
NOISE_DIR="/scratch1/$USER/ser_data/ESC-50"

echo "=========================================================="
echo "--> Task ID: $SLURM_ARRAY_TASK_ID"
echo "--> Experiment: $DATASET_NAME x Target=${TARGET_FEATURE} x $MODEL_NAME x SNR=${SNR_DB}dB"
echo "=========================================================="

# Force EXP_ID 10 as requested
EXP_ID=10

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
echo " Noisy Dimensional Regression Pipeline Finished Successfully"
echo "=========================================="
