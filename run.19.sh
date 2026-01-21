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
# source ~/.bashrc        # ensures conda is available
# conda activate ml4opt

# cd ~/FSNet

# ----------------------------------------
# Run your job
# ----------------------------------------

echo "=============================================="
echo " Job started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Job ID: $SLURM_JOB_ID"
echo " Node: $SLURM_NODELIST"
echo "=============================================="

for seed in 2; do
    # python main.py \
    #     --method adaptive_penalty \
    #     --prob_type nonsmooth_nonconvex \
    #     --prob_name socp \
    #     --lr 0.0001  \
    #     --seed 3 \
    #     --num_epochs 1000
    python main.py \
        --method adaptive_penalty \
        --prob_type nonsmooth_nonconvex \
        --prob_name socp \
        --lr 0.00013  \
        --seed $seed \
        --num_epochs 1000 \
        --checkpoint results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260116-022657_MLP_sup_pen_seed0_nepochs1000_lr0.0001_trainsize800_subopt_3_0.5/model_200.pt
    # python main.py \
    #     --method adaptive_penalty \
    #     --prob_type nonsmooth_nonconvex \
    #     --prob_name socp \
    #     --lr 0.0001  \
    #     --seed 3 \
    #     --num_epochs 1000 \
    #     --checkpoint results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260116-150125_MLP_sup_pen_seed2_nepochs1000_lr0.0001_trainsize800_subopt_3_0.5/model.pt
done
