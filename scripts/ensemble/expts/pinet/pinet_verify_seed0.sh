#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 01:00:00
#SBATCH -J pinet-verify-s0
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

set -u

echo "host: $(hostname)  date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PINET=/orcd/scratch/orcd/008/khain/FSNet/third-party/pinet
cd "$PINET"
source .venv/bin/activate

export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

set -e
python /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/pinet/pinet_verify.py \
    --n_epochs 50 --seed 0

echo "=== done $(date) ==="
