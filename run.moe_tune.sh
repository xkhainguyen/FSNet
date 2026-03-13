#!/bin/bash
#SBATCH -t 03:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH -J moe-tune
#SBATCH -o logs/%x-%A.out
#SBATCH -e logs/%x-%A.err

# set -euo pipefail

# source ~/.bashrc
# conda activate ml4opt

# cd ~/orcd/scratch/FSNet

echo "=============================================="
echo " Job started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Job ID: $SLURM_JOB_ID"
echo " Node: $SLURM_NODELIST"
echo "=============================================="

COMMON_ARGS=(
  --method FSNet
  --prob_type nonsmooth_nonconvex
  --prob_name socp
  --seed 0
  --num_epochs 300
  --lr 1e-4
  --train_size 7000
)

run_case () {
  local name="$1"
  shift
  echo "[CASE] $name"
  python main.py "${COMMON_ARGS[@]}" "$@"
}

# Baseline
# run_case "MLP" --network MLP

# Existing MoE-like setup
run_case "MoE-top2-no-warmup" \
  --network MoE \
  --moe_num_experts 4 --moe_top_k 2 \
  --moe_aux_loss_weight 0.01 \
  --moe_warmup_epochs 0 \
  --moe_gate_temperature 1.0 \
  --moe_gate_noise_std 0.0

# Stabilized sparse routing
run_case "MoE-top2-warmup-temp" \
  --network MoE \
  --moe_num_experts 4 --moe_top_k 2 \
  --moe_aux_loss_weight 0.005 \
  --moe_warmup_epochs 30 \
  --moe_start_temp 2.0 --moe_final_temp 1.0 --moe_temp_decay_epochs 150 \
  --moe_gate_noise_std 0.05 --moe_gate_noise_final 0.0

# Dense MoE (tests if sparsity is hurting)
run_case "MoE-dense" \
  --network MoE \
  --moe_num_experts 4 --moe_top_k 0 \
  --moe_aux_loss_weight 0.002 \
  --moe_warmup_epochs 0 \
  --moe_gate_temperature 1.0 \
  --moe_gate_noise_std 0.0

echo "All cases completed at $(date '+%Y-%m-%d %H:%M:%S')"
