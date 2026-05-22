#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 03:00:00
#SBATCH -J van-fsnet-rtx
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

set -e
source ~/.bashrc
conda activate ml4opt
cd /orcd/scratch/orcd/008/khain/FSNet
echo "=== Vanilla ens5 FSNet on RTX Pro 6000 $(date) on $SLURM_NODELIST ==="
nvidia-smi -L

python main.py \
  --method FSNet \
  --prob_type nonsmooth_nonconvex \
  --prob_name socp \
  --hidden_dim 1024 \
  --num_layers 4 \
  --ensemble_size 5 \
  --ensemble_mode vanilla \
  --seed 0

echo "=== done $(date) ==="
