#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 02:00:00
#SBATCH -J fsnet-ncqcqp-s1
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err
source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /orcd/scratch/orcd/008/khain/FSNet
set -e
echo "=== FSNet nonconvex QCQP seed=1 $(date) ==="
python main.py --method FSNet --prob_type nonconvex --prob_name qcqp \
  --hidden_dim 1024 --num_layers 4 --seed 1
echo "=== done $(date) ==="
