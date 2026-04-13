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

# 4.5 Dataset Aggregation & External Ingestion
echo "=========================================="
echo " 4.5. Dataset Aggregation"
echo "=========================================="

echo "Verifying data synchronization with /scratch1/$USER/ser_data..."

# Ensure kaggle CLI is available and credentials are set
conda run -n $ENV_NAME pip install -q kaggle 2>/dev/null || true
export KAGGLE_USERNAME=$KAGGLE_USERNAME
export KAGGLE_KEY=$KAGGLE_KEY

chmod +x scripts/aggregate_datasets.py
KAGGLE_USERNAME=$KAGGLE_USERNAME KAGGLE_KEY=$KAGGLE_KEY conda run -n $ENV_NAME python scripts/aggregate_datasets.py --base-dir "/scratch1/$USER/ser_data"

echo "=========================================="
echo " 5. Pre-Caching Models"
echo "=========================================="
echo "Downloading models on login node to prevent Slurm timeouts..."
conda run -n $ENV_NAME python -c "
from transformers import AutoModel, AutoFeatureExtractor, AutoConfig
import transformers.modeling_utils
# Bypass new CVE check which forces torch >= 2.6
if hasattr(transformers.modeling_utils, 'check_torch_load_is_safe'):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None
models = ['facebook/wav2vec2-large-960h', 'facebook/hubert-large-ll60k', 'openai/whisper-medium', 'microsoft/wavlm-large', 'm-a-p/MERT-v1-330M', 'facebook/w2v-bert-2.0']
for m in models:
    print(f'Caching {m}...')
    AutoConfig.from_pretrained(m, trust_remote_code=True)
    AutoFeatureExtractor.from_pretrained(m, trust_remote_code=True)
    safe = False if 'MERT' in m else True
    AutoModel.from_pretrained(m, use_safetensors=safe, trust_remote_code=True)
"

echo "=========================================="
echo " 6. Queueing Deployment"
echo "=========================================="

echo "Initializing cluster background job scheduler..."
# Sanitize script line-endings to avoid Slurm parsing errors
sed -i 's/\r$//' slurm/submit_pipeline.sbatch
sed -i 's/\r$//' slurm/submit_noisy_pipeline.sbatch
sed -i 's/\r$//' slurm/submit_msp_pipeline.sbatch

echo "Parsing variables from config.yaml..."
VARS=$(python -c "
import yaml
try:
    with open('config/config.yaml') as f:
        cfg = yaml.safe_load(f)
    
    exp_id = cfg.get('wandb', {}).get('experiment_id', 'unknown')
    batch_size = cfg.get('training', {}).get('batch_size', 4)
    
    models = [v['path'] for k, v in cfg.get('models', {}).items()]
    datasets = cfg.get('datasets', [])
    snr_levels = ['clean'] + [str(x) for x in cfg.get('noise_augmentation', {}).get('snr_levels', [20, 10, 5, 0])]
    
    print(f'EXP_ID={exp_id}')
    print(f'BATCH_SIZE={batch_size}')
    print(f'MODELS_STR=\"' + ' '.join(models) + '\"')
    print(f'DATASETS_STR=\"' + ' '.join(datasets) + '\"')
    print(f'SNR_STR=\"' + ' '.join(snr_levels) + '\"')
    
    print(f'CLEAN_NUM_JOBS={max(1, len(models) * len(datasets))}')
    print(f'NOISY_NUM_JOBS={max(1, len(models) * len(datasets) * len(snr_levels))}')
    print(f'MSP_NUM_JOBS={max(1, len(models) * len(snr_levels))}')
except Exception as e:
    print(f'echo \"Error parsing config.yaml: {e}\"')
")
eval "$VARS"

echo "Detected EXP_ID: $EXP_ID | Batch: $BATCH_SIZE"
echo "Models: $MODELS_STR"
echo "Datasets: $DATASETS_STR"
echo "SNRs: $SNR_STR"

CLEAN_ARRAY="0-$((CLEAN_NUM_JOBS - 1))"
NOISY_ARRAY="0-$((NOISY_NUM_JOBS - 1))"
MSP_ARRAY="0-$((MSP_NUM_JOBS - 1))"

# Submit clean pipeline (commented out by default)
# sbatch --account=msoleyma_1026 --partition=gpu --array=$CLEAN_ARRAY --export=ALL,EXP_ID=$EXP_ID,BATCH_SIZE=$BATCH_SIZE,MODELS_STR="$MODELS_STR",DATASETS_STR="$DATASETS_STR" slurm/submit_pipeline.sbatch

# Submit noisy pipeline (commented out by default)
# sbatch --account=msoleyma_1026 --partition=gpu --array=$NOISY_ARRAY --export=ALL,EXP_ID=$EXP_ID,BATCH_SIZE=$BATCH_SIZE,MODELS_STR="$MODELS_STR",DATASETS_STR="$DATASETS_STR",SNR_STR="$SNR_STR" slurm/submit_noisy_pipeline.sbatch

# Submit MSP-Podcast pipeline
sbatch --account=msoleyma_1026 --partition=gpu --array=$MSP_ARRAY --export=ALL,EXP_ID=$EXP_ID,BATCH_SIZE=$BATCH_SIZE,MODELS_STR="$MODELS_STR",SNR_STR="$SNR_STR" slurm/submit_msp_pipeline.sbatch

echo "=========================================="
echo " Initialization & Queue Complete!"
echo " Check 'squeue -u $USER' for queued jobs."
echo " Clean pipeline queued: $CLEAN_NUM_JOBS jobs."
echo " MSP pipeline queued: $MSP_NUM_JOBS jobs."
echo " Track your job's footprint via 'squeue -u $USER'"
echo "=========================================="

echo "DONT FORGOT: Increment WandB Experiment ID for next batch yourself."