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

for seed in 2 3; do
    python main.py \
        --method FSNet \
        --prob_type nonsmooth_nonconvex \
        --prob_name socp \
        --lr 0.0001  \
        --seed $seed \
        --num_epochs 300 \
        --checkpoint results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260118-032632_MLP_sup_pen_seed0_nepochs2000_lr0.0002_trainsize7000_subopt_3_3.0/model_1990.pt
    python main.py \
        --method FSNet \
        --prob_type nonsmooth_nonconvex \
        --prob_name socp \
        --lr 0.0001  \
        --seed $seed \
        --num_epochs 300 \
        --checkpoint results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260118-032632_MLP_sup_pen_seed0_nepochs2000_lr0.0002_trainsize7000_subopt_3_3.0/model_120.pt
done

# for seed in 0 1 2 3; do
#     python main.py \
#         --method sup \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --train_size 7000 \
#         --num_epochs 2000 \
#         --lr 0.0002 \
#         --seed $seed \
#         --en_subopt 3 \
#         --subopt_ratio 10.0 
# done

# python main.py \
#     --method sup_pen \
#     --prob_type nonsmooth_nonconvex \
#     --prob_name socp \
#     --train_size 7000 \
#     --num_epochs 2000 \
#     --lr 0.0002 \
#     --seed 0 \
#     --en_subopt 3 \
#     --subopt_ratio 3.0 \
#     --save_intermediate True

# for seed in 2 3; do
#     # python main.py \
#     #     --method FSNet \
#     #     --prob_type nonsmooth_nonconvex \
#     #     --prob_name socp \
#     #     --lr 0.0001  \
#     #     --seed $seed \
#     #     --num_epochs 300 \
#     #     --checkpoint results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260115-184556_MLP_sup_pen_seed0_nepochs1000_lr0.0001_trainsize7000_subopt_3_4.0/model_990.pt
#     # python main.py \
#     #     --method FSNet \
#     #     --prob_type nonsmooth_nonconvex \
#     #     --prob_name socp \
#     #     --lr 0.0001  \
#     #     --seed $seed \
#     #     --num_epochs 300 \
#     #     --checkpoint results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260115-184556_MLP_sup_pen_seed0_nepochs1000_lr0.0001_trainsize7000_subopt_3_3.0/model_990.pt
#     # python main.py \
#     #     --method FSNet \
#     #     --prob_type nonsmooth_nonconvex \
#     #     --prob_name socp \
#     #     --lr 0.0001  \
#     #     --seed $seed \
#     #     --num_epochs 300 \
#     #     --checkpoint results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260115-184954_MLP_sup_pen_seed0_nepochs1000_lr0.0001_trainsize7000_subopt_3_2.0/model_990.pt
#     python main.py \
#         --method FSNet \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --lr 0.0001  \
#         --seed $seed \
#         --num_epochs 300 \
#         --checkpoint results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260116-173042_MLP_sup_pen_seed0_nepochs1000_lr0.0001_trainsize7000_subopt_3_10.0/model_990.pt
# done