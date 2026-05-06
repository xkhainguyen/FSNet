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

SL_CKPT_DIR="results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260327-092838_MLP_sup_pen_seed0_nepochs1000_lr0.0001_trainsize7000_subopt_3_3.0"
# SL_CKPT_DIR="results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260327-094852_MLP_sup_pen_seed0_nepochs1000_lr0.0001_trainsize7000_subopt_3_2.0"
LR=0.0001
EARLY_STOP_PATIENCE=10
for seed in 3; do
    # python main.py \
    #     --method FSNet \
    #     --prob_type nonsmooth_nonconvex \
    #     --prob_name socp \
    #     --lr $LR  \
    #     --seed $seed \
    #     --num_epochs 300 \
    #     --eval_step 5 \
    #     --early_stop_patience $EARLY_STOP_PATIENCE \
    #     --checkpoint $SL_CKPT_DIR/model_100.pt
    python main.py \
        --method FSNet \
        --prob_type nonsmooth_nonconvex \
        --prob_name socp \
        --lr $LR  \
        --seed $seed \
        --num_epochs 300 \
        --eval_step 5 \
        --early_stop_patience $EARLY_STOP_PATIENCE \
        --checkpoint $SL_CKPT_DIR/model_200.pt
    # python main.py \
    #     --method FSNet \
    #     --prob_type nonsmooth_nonconvex \
    #     --prob_name socp \
    #     --lr $LR  \
    #     --seed $seed \
    #     --num_epochs 300 \
    #     --eval_step 5 \
    #     --early_stop_patience $EARLY_STOP_PATIENCE \
    #     --checkpoint $SL_CKPT_DIR/model_300.pt
    python main.py \
        --method FSNet \
        --prob_type nonsmooth_nonconvex \
        --prob_name socp \
        --lr $LR  \
        --seed $seed \
        --num_epochs 300 \
        --eval_step 5 \
        --early_stop_patience $EARLY_STOP_PATIENCE \
        --checkpoint $SL_CKPT_DIR/model_400.pt
    # python main.py \
    #     --method FSNet \
    #     --prob_type nonsmooth_nonconvex \
    #     --prob_name socp \
    #     --lr $LR  \
    #     --seed $seed \
    #     --num_epochs 300 \
    #     --eval_step 5 \
    #     --early_stop_patience $EARLY_STOP_PATIENCE \
    #     --checkpoint $SL_CKPT_DIR/model_500.pt
    python main.py \
        --method FSNet \
        --prob_type nonsmooth_nonconvex \
        --prob_name socp \
        --lr $LR  \
        --seed $seed \
        --num_epochs 300 \
        --eval_step 5 \
        --early_stop_patience $EARLY_STOP_PATIENCE \
        --checkpoint $SL_CKPT_DIR/model_600.pt
    # python main.py \
    #     --method FSNet \
    #     --prob_type nonsmooth_nonconvex \
    #     --prob_name socp \
    #     --lr $LR  \
    #     --seed $seed \
    #     --num_epochs 300 \
    #     --eval_step 5 \
    #     --early_stop_patience $EARLY_STOP_PATIENCE \
    #     --checkpoint $SL_CKPT_DIR/model_700.pt
    python main.py \
        --method FSNet \
        --prob_type nonsmooth_nonconvex \
        --prob_name socp \
        --lr $LR  \
        --seed $seed \
        --num_epochs 300 \
        --eval_step 5 \
        --early_stop_patience $EARLY_STOP_PATIENCE \
        --checkpoint $SL_CKPT_DIR/model_800.pt
    # python main.py \
    #     --method FSNet \
    #     --prob_type nonsmooth_nonconvex \
    #     --prob_name socp \
    #     --lr $LR  \
    #     --seed $seed \
    #     --num_epochs 300 \
    #     --eval_step 5 \
    #     --early_stop_patience $EARLY_STOP_PATIENCE \
    #     --checkpoint $SL_CKPT_DIR/model_900.pt
    python main.py \
        --method FSNet \
        --prob_type nonsmooth_nonconvex \
        --prob_name socp \
        --lr $LR  \
        --seed $seed \
        --num_epochs 300 \
        --eval_step 5 \
        --early_stop_patience $EARLY_STOP_PATIENCE \
        --checkpoint $SL_CKPT_DIR/model_990.pt
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
