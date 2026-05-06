#!/bin/bash
#SBATCH -t 06:00:00
# SBATCH -p mit_normal_gpu,mit_preemptable
# SBATCH --gres=gpu:l40s:1
#SBATCH -J ml4opf
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

# ----------------------------------------
# Environment setup
# ----------------------------------------

# Load conda properly in non-interactive shells
source ~/.bashrc        # ensures conda is available
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

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
    # python main.py \
    #     --method FSNet \
    #     --prob_type nonsmooth_nonconvex \
    #     --prob_name socp \
    #     --lr 0.0001  \
    #     --seed $seed \
    #     --num_epochs 300 \
    #     --checkpoint results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260327-094852_MLP_sup_pen_seed0_nepochs1000_lr0.0001_trainsize7000_subopt_3_2.0/model_100.pt
    python main.py \
        --method FSNet \
        --prob_type nonsmooth_nonconvex \
        --prob_name socp \
        --lr 0.0001  \
        --seed $seed \
        --num_epochs 300 \
        --checkpoint results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260327-094852_MLP_sup_pen_seed0_nepochs1000_lr0.0001_trainsize7000_subopt_3_2.0/model_350.pt
    python main.py \
        --method FSNet \
        --prob_type nonsmooth_nonconvex \
        --prob_name socp \
        --lr 0.0001  \
        --seed $seed \
        --num_epochs 300 \
        --checkpoint results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260327-094852_MLP_sup_pen_seed0_nepochs1000_lr0.0001_trainsize7000_subopt_3_2.0/model_400.pt
done

# python main.py \
#     --method sup_pen \
#     --prob_type nonsmooth_nonconvex \
#     --prob_name socp \
#     --train_size 7000 \
#     --num_epochs 1000 \
#     --lr 0.0001 \
#     --seed 0 \
#     --en_subopt 3 \
#     --subopt_ratio 0.0 \
#     --save_intermediate True

# python main.py \
#     --method sup_pen \
#     --prob_type nonsmooth_nonconvex \
#     --prob_name socp \
#     --train_size 800 \
#     --num_epochs 1000 \
#     --lr 0.0001 \
#     --seed 0 \
#     --en_subopt 3 \
#     --subopt_ratio 0.0 \
#     --save_intermediate True

# python main.py \
#     --method sup_partial \
#     --prob_type nonsmooth_nonconvex \
#     --prob_name socp \
#     --train_size 7000 \
#     --num_epochs 1000 \
#     --lr 0.0005 \
#     --seed 0 \
#     --en_subopt 3 \
#     --subopt_ratio 0.0 \
#     --save_intermediate True

