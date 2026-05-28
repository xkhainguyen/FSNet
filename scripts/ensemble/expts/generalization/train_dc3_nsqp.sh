#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 02:00:00
#SBATCH -J train-dc3-nsqp
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

source /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/_common.sh
echo "=== train DC3 on nonsmooth_nonconvex QP $(date) on $SLURM_NODELIST ==="
python main.py --method DC3 --prob_type nonsmooth_nonconvex --prob_name qp \
  --hidden_dim 1024 --num_layers 4 --seed 0
echo "=== done $(date) ==="
