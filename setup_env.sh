#!/bin/bash
# setup_env.sh
# Automates repository syncing and Conda environment building for USC CARC partitions.

set -e

echo "=========================================="
echo " Starting CARC Environment Setup"
echo "=========================================="

# 1. GitHub Token Validation safely prompted if not in session memory
if [ -z "$GITHUB_TOKEN" ]; then
    echo "GitHub PAT not found in environment. Using default fallback..."
    export GITHUB_TOKEN=""
fi

# 2. WandB Configuration for automated tracking


REPO_URL="https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/nikelroid/linguistic-agnostic-ser.git"
REPO_NAME="linguistic-agnostic-ser"
WORK_DIR="/home1/$USER"

echo "Navigating to root workstation: $WORK_DIR"
cd $WORK_DIR

# 2. Seamless clone or update via git
if [ ! -d "$REPO_NAME" ]; then
    echo "Repository not found locally. Cloning..."
    git clone "$REPO_URL" "$REPO_NAME"
    echo "Successfully cloned!"
else
    echo "Repository found. Executing aggressive pull for latest changes..."
    cd $REPO_NAME
    git remote set-url origin "$REPO_URL"
    git pull
    cd ..
fi

echo "=========================================="
echo " Preparing Miniconda"
echo "=========================================="

export PATH="$HOME/miniconda/bin:$PATH"

# 3. Bypass Conda license prompts that freeze standard bash sessions.
echo "Preemptively accepting Conda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

cd $REPO_NAME

# 4. Environment Syncing mapping exactly to 'run.ipynb'

ENV_NAME="ser_env"
echo "Analyzing target environment footprint: $ENV_NAME"

if conda env list | grep -q "$ENV_NAME"; then
    echo "$ENV_NAME payload detected! Synchronizing missing packages..."
    conda env update -f environment.yml --prune
else
    echo "$ENV_NAME payload not found. Constructing fresh conda environment..."
    conda env create -f environment.yml
fi

echo "=========================================="
echo " 5. Pre-Caching Models"
echo "=========================================="
echo "Downloading models on login node to prevent Slurm timeouts..."
conda run -n $ENV_NAME python -c "
from transformers import AutoModel, AutoFeatureExtractor, AutoConfig
models = ['facebook/wav2vec2-large-960h', 'facebook/hubert-large-ll60k', 'openai/whisper-medium']
for m in models:
    print(f'Caching {m}...')
    AutoConfig.from_pretrained(m)
    AutoFeatureExtractor.from_pretrained(m)
    AutoModel.from_pretrained(m)
"

echo "=========================================="
echo " 6. Queueing Deployment"
echo "=========================================="

echo "Initializing cluster background job scheduler..."
# Sanitize script line-endings to avoid Slurm parsing errors
sed -i 's/\r$//' slurm/submit_pipeline.sbatch
# Explicitly pass all key parameters to bypass potential header parsing issues
sbatch --account=msoleyma_1026 --partition=gpu --array=0-17 slurm/submit_pipeline.sbatch

echo "=========================================="
echo " Initialization & Queue Complete!"
echo " Track your job's footprint via 'squeue -u $USER'"
echo "=========================================="
 