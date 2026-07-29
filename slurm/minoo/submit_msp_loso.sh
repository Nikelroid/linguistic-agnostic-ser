#!/bin/bash
#SBATCH --job-name=loso-msp
#SBATCH --account=msoleyma_946
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=/home1/minooahm/logs/loso-msp-%A_%a.log

MODELS=(HuBERT wav2vec2 Whisper WavLM MERT w2v-BERT)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate ser_env
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /scratch1/$USER/linguistic-agnostic-ser

echo "============================================================"
echo "  Model: $MODEL  |  task: MSP-Podcast LOSO  |  min_samples=50"
echo "  Started: $(date)  on $(hostname)"
echo "============================================================"

python -u scripts/task4_loso.py \
  --model "$MODEL" \
  --dataset MSP-Podcast \
  --min_speaker_samples 50 \
  --data_dir /scratch1/$USER/ser_data

echo "============================================================"
echo "  Finished: $(date)"
echo "============================================================"
