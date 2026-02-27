#!/bin/bash
#SBATCH -t 06:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH -J ens-fsnet
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

source ~/.bashrc
conda activate ml4opt
cd ~/orcd/scratch/FSNet
mkdir -p logs

echo "=== [ens-fsnet] $(date '+%Y-%m-%d %H:%M:%S') Job=$SLURM_JOB_ID Node=$SLURM_NODELIST ==="

# python main.py \
#     --method FSNet \
#     --prob_type nonsmooth_nonconvex \
#     --prob_name socp \
#     --ensemble_size 10 \
#     --seed 0 \
#     --ensemble_mode fge \
#     --wandb \
#     --num_epochs 300 

python main.py \
    --method penalty \
    --prob_type nonsmooth_nonconvex \
    --prob_name socp \
    --ensemble_size 20 \
    --ensemble_mode vanilla \
    --wandb

echo "=== [ens-fsnet] done $(date '+%Y-%m-%d %H:%M:%S') ==="
