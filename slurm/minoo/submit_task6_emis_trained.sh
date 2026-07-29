#!/bin/bash
#SBATCH --job-name=task6-emistrain
#SBATCH --account=msoleyma_946
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home1/minooahm/logs/task6-emistrain-%j.log

set +u
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate ser_env

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p /scratch1/$USER/wandb_tmp
export TMPDIR=/scratch1/$USER/wandb_tmp

cd /scratch1/$USER/linguistic-agnostic-ser

echo "============================================================"
echo "  Task 6 — EMIS-trained variant (train on EMIS_congruent)"
echo "  Started: $(date)  on $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================================"

python -u scripts/task6_adversarial.py \
  --all \
  --data_dir /scratch1/$USER/ser_data \
  --train_datasets EMIS_congruent \
  --lambdas 0,0.01,0.1,1.0,10.0 \
  --n_seeds 5 \
  --n_epochs 50 \
  --device auto \
  --wandb \
  --run_suffix _emistrain

echo "============================================================"
echo "  Finished: $(date)"
echo "============================================================"
