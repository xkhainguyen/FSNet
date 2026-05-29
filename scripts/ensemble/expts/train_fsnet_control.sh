#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 02:00:00
#SBATCH -J train-fsnet-ctrl
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

source /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/_common.sh

PS=${PER_SAMPLE:-0}
echo "=== Control FSNet train per_sample_lbfgs=$PS on $SLURM_NODELIST $(date) ==="
python main.py --method FSNet --prob_type nonsmooth_nonconvex --prob_name socp \
  --hidden_dim 1024 --num_layers 4 --seed 0 \
  --per_sample_lbfgs $PS
echo "=== done $(date) ==="
