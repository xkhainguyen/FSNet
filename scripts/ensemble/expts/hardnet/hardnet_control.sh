#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 02:00:00
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

# Args via env: EPOCHS (default 1500), CONVEX (0/1, default 0)
EPOCHS="${EPOCHS:-1500}"
CONVEX="${CONVEX:-0}"

echo "host: $(hostname)  date: $(date)  EPOCHS=$EPOCHS CONVEX=$CONVEX"
nvidia-smi --query-gpu=name --format=csv,noheader

source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

CONVEX_FLAG=""
if [ "$CONVEX" = "1" ]; then CONVEX_FLAG="--convex_obj"; fi

set -e
python /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/hardnet/hardnet_verify.py \
    --epochs "$EPOCHS" --seed 42 $CONVEX_FLAG

echo "=== done $(date) ==="
