#!/bin/bash
#SBATCH -t 06:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH -J ml4opf
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

# ----------------------------------------
# Environment setup
# ----------------------------------------

# Load conda properly in non-interactive shells
source ~/.bashrc        # ensures conda is available
conda activate ml4opt

cd ~/FSNet

# ----------------------------------------
# Run your job
# ----------------------------------------

echo "=============================================="
echo " Job started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Job ID: $SLURM_JOB_ID"
echo " Node: $SLURM_NODELIST"
echo "=============================================="

for seed in 0 1; do
    python main.py \
        --method semi \
        --prob_type nonsmooth_nonconvex \
        --prob_name socp \
        --lr 0.0001  \
        --seed $seed \
        --num_epochs 300 \
        --en_subopt 3 \
        --subopt_ratio 0.5 
done
