#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 04:00:00
#SBATCH -J mhe-train-fsnet
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

set -e
source ~/.bashrc
conda activate ml4opt
cd /orcd/scratch/orcd/008/khain/FSNet

SEED=${SEED:-0}
NUM_HEADS=${NUM_HEADS:-5}
HIDDEN_DIM=${HIDDEN_DIM:-1024}
echo "=== MHE FSNet seed=$SEED M=$NUM_HEADS hdim=$HIDDEN_DIM $(date) on $SLURM_NODELIST ==="

python main.py \
  --method FSNet \
  --prob_type nonsmooth_nonconvex \
  --prob_name socp \
  --network MultiHeadMLP \
  --mhe_num_heads $NUM_HEADS \
  --hidden_dim $HIDDEN_DIM \
  --num_layers 4 \
  --seed $SEED

echo "=== done $(date) ==="
