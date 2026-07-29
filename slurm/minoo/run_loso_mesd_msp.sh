#!/bin/bash
cd /scratch1/$USER/linguistic-agnostic-ser
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate ser_env

for model in HuBERT wav2vec2 Whisper WavLM MERT w2v-BERT; do
  for dataset in MESD MSP-Podcast; do
    echo ""
    echo "============================================================"
    echo "  RUNNING: $model on $dataset"
    echo "============================================================"
    python -u scripts/task4_loso.py --model "$model" --dataset "$dataset" --data_dir /scratch1/$USER/ser_data
  done
done

echo ""
echo "============================================================"
echo "  ALL DONE at $(date)"
echo "============================================================"
