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

# for seed in 0 1 2 3; do
#     python main.py \
#         --method adaptive_penalty \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --lr 0.0001  \
#         --seed $seed \
#         --num_epochs 1000
#     python main.py \
#         --method adaptive_penalty \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --lr 0.0001  \
#         --seed 3 \
#         --num_epochs 1000 \
#         --checkpoint results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260116-022657_MLP_sup_pen_seed0_nepochs1000_lr0.0001_trainsize800_subopt_3_0.5/model_940.pt
# done

# for seed in 0 1 2; do
#     python main.py \
#         --method DC3 \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --lr 0.0001  \
#         --seed $seed \
#         --num_epochs 1000
#     # python main.py \
#     #     --method DC3 \
#     #     --prob_type nonsmooth_nonconvex \
#     #     --prob_name socp \
#     #     --lr 0.00005  \
#     #     --seed $seed\
#     #     --num_epochs 1000 \
#     #     --checkpoint results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260118-184700_MLP_sup_partial_seed0_nepochs1000_lr0.0005_trainsize800_subopt_3_0.5/model_230.pt
#     # python main.py \
#     #     --method DC3 \
#     #     --prob_type nonsmooth_nonconvex \
#     #     --prob_name socp \
#     #     --lr 0.0001  \
#     #     --seed 3 \
#     #     --num_epochs 1000 \
#     #     --checkpoint results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260118-234011_MLP_sup_partial_seed0_nepochs1000_lr0.0005_trainsize3000_subopt_3_0.5/model_300.pt
# done

for seed in 0; do
    python main.py \
        --method sup_partial \
        --prob_type nonsmooth_nonconvex \
        --prob_name socp \
        --train_size 3000 \
        --num_epochs 1000 \
        --lr 0.0005 \
        --seed $seed \
        --en_subopt 3 \
        --subopt_ratio 0.5 \
        --save_intermediate True
done