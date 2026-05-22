#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 00:20:00
#SBATCH -J mhe-smoke
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

set -e
source ~/.bashrc
conda activate ml4opt
cd /orcd/scratch/orcd/008/khain/FSNet
mkdir -p logs

echo "=== smoke $(date) job=$SLURM_JOB_ID node=$SLURM_NODELIST ==="
nvidia-smi -L

python main.py \
  --method penalty \
  --prob_type nonsmooth_nonconvex \
  --prob_name socp \
  --network MultiHeadMLP \
  --mhe_num_heads 5 \
  --num_epochs 5 \
  --train_size 500 \
  --seed 0
echo "=== smoke done $(date) ==="
