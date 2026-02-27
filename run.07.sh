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

for seed in 0 1 2 3; do
    python main.py \
        --method FSNet \
        --prob_type nonsmooth_nonconvex \
        --prob_name socp \
        --lr 0.0001  \
        --seed $seed \
        --num_epochs 300 \
        --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260106-202959_MLP_sup_seed1_dropout0.05_subopt_2_-10.0/model_60.pt"
    
    echo "============================================================"
    echo "==========         =====================         ==========="
    echo "==========   $seed      =====================         ==========="
    echo "==========         =====================         ==========="
    echo "============================================================"
done

for seed in 0 1 2 3; do
    python main.py \
        --method FSNet \
        --prob_type nonsmooth_nonconvex \
        --prob_name socp \
        --lr 0.0001  \
        --seed $seed \
        --num_epochs 300 \
        --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260106-202959_MLP_sup_seed1_dropout0.05_subopt_2_-10.0/model_400.pt"

    echo "============================================================"
    echo "==========         =====================         ==========="
    echo "==========   $seed      =====================         ==========="
    echo "==========         =====================         ==========="
    echo "============================================================"
done

# for seed in 0 1 2 3; do
#     python main.py \
#         --method FSNet \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --lr 0.0002  \
#         --seed $seed \
#         --num_epochs 300 \
#         --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260108-031455_MLP_sup_pen_seed1_nepochs500_lr0.0002_trainsize1000_subopt_2_1.0/model_470.pt"
    
#     echo "============================================================"
#     echo "==========         =====================         ==========="
#     echo "==========   $seed      =====================         ==========="
#     echo "==========         =====================         ==========="
#     echo "============================================================"
# done

# for seed in 0 1 2 3; do
#     python main.py \
#         --method FSNet \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --lr 0.0002  \
#         --seed $seed \
#         --num_epochs 300 \
#         --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260108-031328_MLP_sup_seed1_nepochs500_lr0.0005_trainsize1000_subopt_2_1.0/model_280.pt"

#     echo "============================================================"
#     echo "==========         =====================         ==========="
#     echo "==========   $seed      =====================         ==========="
#     echo "==========         =====================         ==========="
#     echo "============================================================"
# done

# for seed in 0 1 2 3; do
#     python main.py \
#         --method FSNet \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --lr 0.0002  \
#         --seed $seed \
#         --num_epochs 300 \
#         --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260108-014039_MLP_sup_pen_seed1_nepochs500_lr0.0002_trainsize7000_subopt_2_1.0/model_230.pt"
    
#     echo "============================================================"
#     echo "==========         =====================         ==========="
#     echo "==========   $seed      =====================         ==========="
#     echo "==========         =====================         ==========="
#     echo "============================================================"
# done

# for seed in 0 1 2 3; do
#     python main.py \
#         --method FSNet \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --lr 0.0002  \
#         --seed $seed \
#         --num_epochs 300 \
#         --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260106-202336_MLP_sup_seed1_dropout0.05_subopt_2_1.0/model_400.pt"

#     echo "============================================================"
#     echo "==========         =====================         ==========="
#     echo "==========   $seed      =====================         ==========="
#     echo "==========         =====================         ==========="
#     echo "============================================================"
# done

# python main.py \
#     --method FSNet \
#     --prob_type nonsmooth_nonconvex \
#     --prob_name socp \
#     --lr 0.0002  \
#     --seed 0 \
#     --num_epochs 300 \
#     --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260106-202336_MLP_sup_seed1_dropout0.05_subopt_2_1.0/model_60.pt"
