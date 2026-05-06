#!/bin/bash
#SBATCH --job-name=task7A-emis
#SBATCH --account=msoleyma_946
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home1/minooahm/logs/task7A-emis-%j.log

set +u
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate ser_env

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p /scratch1/minooahm/wandb_tmp
export TMPDIR=/scratch1/minooahm/wandb_tmp

cd /scratch1/minooahm/linguistic-agnostic-ser

echo "============================================================"
echo "  Task 7 Part A — EMIS congruent vs incongruent"
echo "  Started: $(date)  on $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================================"

python -u scripts/task7_partA_EMIS.py

echo "============================================================"
echo "  Finished: $(date)"
echo "============================================================"
