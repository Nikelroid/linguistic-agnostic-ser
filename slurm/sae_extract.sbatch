#!/bin/bash -l
# Per-frame activation extraction for the SAE study, as a SLURM array job.
# Each array task extracts ONE (encoder, dataset, layer) so the work fans out
# across V100s (one forward pass per layer; bounded RAM ~4 GB/task).
#
# Submit (example: HuBERT depth sweep, all 25 layers on CREMA-D):
#   L=$(seq -s' ' 0 24)
#   sbatch --array=0-24 \
#     --export=ALL,ENCODERS_STR="HuBERT",DATASETS_STR="CREMA-D",LAYERS_STR="$L" \
#     slurm/sae_extract.sbatch
#
# Submit (example: 6 encoders x CREMA-D at L13):
#   sbatch --array=0-5 \
#     --export=ALL,ENCODERS_STR="HuBERT WavLM wav2vec2 w2v-BERT MERT Whisper",DATASETS_STR="CREMA-D",LAYERS_STR="13" \
#     slurm/sae_extract.sbatch
#SBATCH --account=msoleyma_1026
#SBATCH --partition=gpu
#SBATCH --job-name=sae-extract
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

set -e
mkdir -p slurm/logs
module load gcc/13.3.0 python/3.11.9 2>/dev/null || true
PY=$(command -v python); [ -z "$PY" ] && PY=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/python-3.11.9-x74mtjf/bin/python
cd /home1/$USER/linguistic-agnostic-ser
export PYTHONUNBUFFERED=1

ENCODERS=(${ENCODERS_STR:-HuBERT})
DATASETS=(${DATASETS_STR:-CREMA-D})
LAYERS=(${LAYERS_STR:-13})
nL=${#LAYERS[@]}; nD=${#DATASETS[@]}
i=$SLURM_ARRAY_TASK_ID
li=$(( i % nL )); i=$(( i / nL ))
di=$(( i % nD )); ei=$(( i / nD ))
ENC=${ENCODERS[$ei]}; DS=${DATASETS[$di]}; L=${LAYERS[$li]}
DATA_DIR="/scratch1/$USER/ser_data/${DS}"

echo "[$(date)] task $SLURM_ARRAY_TASK_ID -> encoder=$ENC dataset=$DS layer=$L (node $SLURMD_NODENAME)"
"$PY" -c "import sae_lens, torch; print('env ok: sae_lens', sae_lens.__version__, 'cuda', torch.cuda.is_available())"
"$PY" -u -m src.sae.extraction --encoder "$ENC" --dataset "$DS" --data-dir "$DATA_DIR" --layers "$L" ${POOLED:+--pooled}
echo "[$(date)] done task $SLURM_ARRAY_TASK_ID"
