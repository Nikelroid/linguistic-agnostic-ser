#!/bin/bash -l
# Tiny job to validate that the SAE batch environment (module load python/3.11.9
# + ~/.local) resolves sae-lens/torch/CUDA on a compute node before launching
# the large array jobs. Safe to delete after it passes.
#SBATCH --account=msoleyma_1026
#SBATCH --partition=gpu
#SBATCH --job-name=sae-envtest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -e
mkdir -p slurm/logs
module load gcc/13.3.0 python/3.11.9 2>/dev/null || true
PY=$(command -v python); [ -z "$PY" ] && PY=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/python-3.11.9-x74mtjf/bin/python
echo "node=$SLURMD_NODENAME  PY=$PY"
"$PY" -c "import sae_lens, torch, opensmile, sklearn, matplotlib; print('ENV_OK sae_lens', sae_lens.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
