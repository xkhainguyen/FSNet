#!/bin/bash
#SBATCH -t 06:00:00
#SBATCH -p mit_normal_gpu,mit_preemptable
#SBATCH --gres=gpu:l40s:1
#SBATCH -J ml4opt
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

set -euo pipefail

# ----------------------------------------
# Environment setup
# ----------------------------------------
source ~/.bashrc
conda activate ml4opt

cd /home/khain/orcd/scratch/FSNet


# for seed in 0 1 2 3; do
#     echo "=============================================="
#     echo " Job started at: $(date '+%Y-%m-%d %H:%M:%S')"
#     echo " Job ID: $SLURM_JOB_ID"
#     echo " Node: $SLURM_NODELIST"
#     echo "=============================================="

#     python main.py \
#     --seed $seed \
#     --method FSNet \
#     --prob_type nonsmooth_nonconvex \
#     --prob_name socp \
#     --network LocalContextMLPv2 \
#     --hidden_dim 1024

# done

for seed in 1 2 3; do
    echo "=============================================="
    echo " Job started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo " Job ID: $SLURM_JOB_ID"
    echo " Node: $SLURM_NODELIST"
    echo "=============================================="

    python main.py \
    --seed $seed \
    --method FSNet \
    --prob_type nonsmooth_nonconvex \
    --prob_name socp \
    --network MLP \
    --hidden_dim 512

done

