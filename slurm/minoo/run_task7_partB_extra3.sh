#!/bin/bash
cd /scratch1/minooahm/linguistic-agnostic-ser
export PYTHONNOUSERSITE=1
export WANDB_SILENT=true
mkdir -p /scratch1/minooahm/wandb_tmp
export TMPDIR=/scratch1/minooahm/wandb_tmp

PYTHON=/scratch1/minooahm/miniconda/envs/ser_env/bin/python

for model in WavLM MERT w2v-BERT; do
  echo ""
  echo "=========================================="
  echo "  Task 7B: $model on IEMOCAP — start $(date)"
  echo "=========================================="
  MODEL="$model" $PYTHON -u scripts/task7_disagreement.py
  echo "  done $model at $(date)"
done

echo ""
echo "=========================================="
echo "  ALL 3 EXTRA MODELS DONE at $(date)"
echo "=========================================="
