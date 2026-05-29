#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 01:00:00
#SBATCH -J pinet-verify
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

set -u

echo "host: $(hostname)  date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PINET=/orcd/scratch/orcd/008/khain/FSNet/third-party/pinet
cd "$PINET"
source .venv/bin/activate

# Headless wandb (run_qp.py / setup_model don't gate on this but Logger may try)
export WANDB_MODE=disabled

set -e
python /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/pinet/pinet_verify.py \
    --n_epochs 50 --seed 42

echo "=== done $(date) ==="
