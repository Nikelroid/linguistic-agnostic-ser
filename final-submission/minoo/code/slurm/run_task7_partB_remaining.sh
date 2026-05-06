#!/bin/bash
cd /scratch1/minooahm/linguistic-agnostic-ser
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate ser_env
export PYTHONNOUSERSITE=1
export WANDB_SILENT=true
mkdir -p /scratch1/minooahm/wandb_tmp
export TMPDIR=/scratch1/minooahm/wandb_tmp

for model in wav2vec2 Whisper; do
  echo ""
  echo "============================================================"
  echo "  Task 7 Part B: $model on IEMOCAP  ($(date))"
  echo "============================================================"
  MODEL="$model" python -u scripts/task7_disagreement.py
done

echo ""
echo "============================================================"
echo "  TASK 7 PART B (wav2vec2+Whisper) DONE at $(date)"
echo "============================================================"
