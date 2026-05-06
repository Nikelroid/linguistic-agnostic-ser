#!/bin/bash
cd /scratch1/minooahm/linguistic-agnostic-ser
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate ser_env
export PYTHONNOUSERSITE=1

for model in wav2vec2 Whisper WavLM MERT w2v-BERT; do
  echo ""
  echo "============================================================"
  echo "  MESD LOSO: $model  ($(date))"
  echo "============================================================"
  python -u scripts/task4_loso.py --model "$model" --dataset MESD --data_dir /scratch1/minooahm/ser_data
done

echo ""
echo "============================================================"
echo "  ALL MESD DONE at $(date)"
echo "============================================================"
