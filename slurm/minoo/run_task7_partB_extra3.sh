#!/bin/bash
cd /scratch1/$USER/linguistic-agnostic-ser
export PYTHONNOUSERSITE=1
export WANDB_SILENT=true
mkdir -p /scratch1/$USER/wandb_tmp
export TMPDIR=/scratch1/$USER/wandb_tmp

PYTHON=/scratch1/$USER/miniconda/envs/ser_env/bin/python

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
