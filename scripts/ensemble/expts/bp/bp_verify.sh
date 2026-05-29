#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 00:30:00
#SBATCH -J bp-verify
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

set -u
echo "host: $(hostname)  date: $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1

set -e
python /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/bp/bp_verify.py

echo "=== done $(date) ==="
