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



# for seed in 2 3; do
#     python main.py \
#         --method FSNet \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --lr 0.0002  \
#         --seed $seed \
#         --num_epochs 300 \
#         --checkpoint results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260116-022657_MLP_sup_pen_seed0_nepochs1000_lr0.0001_trainsize50_subopt_3_0.5/model_970.pt
#     python main.py \
#         --method FSNet \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --lr 0.0002  \
#         --seed $seed \
#         --num_epochs 300 \
#         --checkpoint results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260116-022757_MLP_sup_pen_seed0_nepochs1000_lr0.0001_trainsize200_subopt_3_0.5/model_780.pt
#     # python main.py \
#     #     --method FSNet \
#     #     --prob_type nonsmooth_nonconvex \
#     #     --prob_name socp \
#     #     --lr 0.0002  \
#     #     --seed $seed \
#     #     --num_epochs 300 \
#     #     --checkpoint results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260116-022657_MLP_sup_pen_seed0_nepochs1000_lr0.0001_trainsize800_subopt_3_0.5/model_940.pt
#     # python main.py \
#     #     --method FSNet \
#     #     --prob_type nonsmooth_nonconvex \
#     #     --prob_name socp \
#     #     --lr 0.0002  \
#     #     --seed $seed \
#     #     --num_epochs 300 \
#     #     --checkpoint results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260116-022847_MLP_sup_pen_seed0_nepochs1000_lr0.0001_trainsize3000_subopt_3_0.5/model_920.pt
# done



# for size in 800 3000; do
#     python main.py \
#         --method sup_pen \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --train_size $size \
#         --seed 0 \
#         --en_subopt 3 \
#         --subopt_ratio 0.5 \
#         --save_intermediate True
#     python main.py \
#         --method sup \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --train_size $size \
#         --seed 0 \
#         --en_subopt 3 \
#         --subopt_ratio 0.5 \
#         --save_intermediate True
# done


# for seed in 0 1; do
#     python main.py \
#         --method FSNet \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --lr 0.0002  \
#         --seed $seed \
#         --num_epochs 300 \
#         --checkpoint results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260116-060756_MLP_sup_pen_seed0_nepochs3000_lr0.0001_trainsize7000_subopt_3_0.5/model_2170.pt
#     python main.py \
#         --method FSNet \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --lr 0.0002  \
#         --seed $seed \
#         --num_epochs 300 \
#         --checkpoint results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260116-051658_MLP_sup_pen_seed0_nepochs5000_lr2e-05_trainsize7000_subopt_3_0.5/model_2990.pt
# done

# for seed in 0 1 2 3; do
#     # python main.py \
#     #     --method FSNet \
#     #     --prob_type nonsmooth_nonconvex \
#     #     --prob_name socp \
#     #     --lr 0.0002  \
#     #     --seed $seed \
#     #     --num_epochs 300 \
#     #     --checkpoint results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260116-151545_MLP_sup_pen_seed0_nepochs5000_lr5e-05_trainsize800_subopt_3_0.5/model_980.pt
#     python main.py \
#         --method FSNet \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --lr 0.0002  \
#         --seed $seed \
#         --num_epochs 300 \
#         --checkpoint results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260116-151545_MLP_sup_pen_seed0_nepochs5000_lr5e-05_trainsize800_subopt_3_0.5/model_4990.pt
# done


python main.py \
    --method sup_pen \
    --prob_type nonsmooth_nonconvex \
    --prob_name socp \
    --train_size 7000 \
    --num_epochs 1000 \
    --lr 0.00005 \
    --seed 0 \
    --en_subopt 3 \
    --subopt_ratio 0.5 \
    --save_intermediate True

# for seed in 0; do
#     python main.py \
#         --method sup_pen \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --train_size 7000 \
#         --num_epochs 1000 \
#         --lr 0.0001 \
#         --seed $seed \
#         --en_subopt 3 \
#         --subopt_ratio 10.0 \
#         --save_intermediate True
# done