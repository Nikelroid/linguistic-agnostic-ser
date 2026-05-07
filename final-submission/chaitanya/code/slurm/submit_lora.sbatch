#!/bin/bash
#SBATCH --job-name=lora_ser
#SBATCH --account=msoleyma_1026
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -e
mkdir -p slurm/logs

module load conda 2>/dev/null || module load anaconda3 2>/dev/null || true
source activate lora_ser

cd /project2/msoleyma_1026/parwatka/lora_run/my_code

bash lora_run_all.sh
