#!/bin/bash
#SBATCH --account=kelidari        # >> REPLACE THIS with your CARC Project ID <<
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1                 # Requested: 1 V100 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                 # Requested: 4 CPU Cores
#SBATCH --mem=16G                         # Requested: 16 GB RAM
#SBATCH --time=12:00:00                   # 12-hour queue limit
#SBATCH --output=slurm_logs/pipeline_%j.out
#SBATCH --error=slurm_logs/pipeline_%j.err

# Safe Exit on failure
set -e

echo "=========================================="
echo " Starting Slurm Job: $SLURM_JOB_ID"
echo " Node Assigned: $SLURMD_NODENAME"
echo "=========================================="

# Automatically connect conda exactly as the interactive shell does
echo "Initializing Miniconda Core..."
source $HOME/miniconda/etc/profile.d/conda.sh
conda activate ser_env

# Navigate directly to where the code rests on home1
cd /home1/$USER/linguistic-agnostic-ser

# Pipeline Matrices
BATCH_SIZE=4
DATASETS=("RAVDESS" "EmoDB" "IEMOCAP" "SAVEE" "AESDD" "MESD")
MODELS=("facebook/wav2vec2-base" "facebook/hubert-base-ls960" "openai/whisper-small")

echo "Commencing Aggregated Python Pipeline Execution Matrix..."

for MODEL_NAME in "${MODELS[@]}"; do
    for DATASET_NAME in "${DATASETS[@]}"; do
        
        DATA_DIR="/scratch1/$USER/ser_data/${DATASET_NAME}"
        
        echo "=========================================================="
        echo "--> Processing: $DATASET_NAME | Model: $MODEL_NAME"
        echo "=========================================================="
        
        # Execute run logic for each pair
        python -u -m scripts.run_pipeline \
          --task classification \
          --data_dir "$DATA_DIR" \
          --dataset_name "$DATASET_NAME" \
          --model_name "$MODEL_NAME" \
          --batch_size $BATCH_SIZE

    done
done

echo "=========================================="
echo " Pipeline Finished Successfully"
echo "=========================================="
