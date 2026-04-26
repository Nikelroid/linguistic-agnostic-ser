#!/bin/bash
#SBATCH --account=msoleyma_1026
#SBATCH --partition=gpu
#SBATCH --job-name=metadata-probing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -e

echo "=========================================="
echo " Starting MSP-Podcast Metadata Probing (EXP_ID 9): $SLURM_JOB_ID"
echo " Node Assigned: $SLURMD_NODENAME"
mkdir -p slurm/logs
echo "=========================================="

echo "Initializing Miniconda Core..."
source $HOME/miniconda/etc/profile.d/conda.sh
conda activate ser_env

cd /home1/$USER/linguistic-agnostic-ser

DATA_DIR="/scratch1/$USER/ser_data/MSP-Podcast"
EXP_ID=9

python -u -m scripts.probe_metadata \
  --data_dir "$DATA_DIR" \
  --exp_id $EXP_ID \
  --wandb_entity AGSER \
  --wandb_project linguistic-agnostic-ser

echo "=========================================="
echo " Metadata Probing Finished Successfully"
echo "=========================================="
