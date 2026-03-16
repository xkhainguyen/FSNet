#!/bin/bash
#SBATCH -t 06:00:00
#SBATCH -p mit_normal_gpu,mit_preemptable
#SBATCH --gres=gpu:l40s:1
#SBATCH -J ml4opt
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

# ----------------------------------------
# Environment setup
# ----------------------------------------

# Load conda properly in non-interactive shells
source ~/.bashrc        # ensures conda is available
conda activate ml4opt

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
        --lr 0.00005  \
        --seed $seed \
        --hidden_dim 1024 \
        --num_epochs 300 \
        --network MLP \
        --wandb
done

# for seed in 0 1 2 3; do
#     python main.py \
#         --method penalty \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --lr 0.0001  \
#         --seed $seed \
#         --hidden_dim 1024 \
#         --network MLP \
#         --wandb
# done

# python main.py \
#     --method FSNet \
#     --prob_type nonsmooth_nonconvex \
#     --prob_name socp \
#     --lr 0.0001  \
#     --seed 2025 \
#     --hidden_dim 512 \
#     --num_epochs 300 \
#     --network MoE \
#     --moe_num_experts 4 --moe_top_k 2 \
#     --moe_aux_loss_weight 0.005 \
#     --moe_warmup_epochs 30 \
#     --moe_start_temp 2.0 --moe_final_temp 1.0 --moe_temp_decay_epochs 150 \
#     --moe_gate_noise_std 0.05 --moe_gate_noise_final 0.0 \
#     --wandb