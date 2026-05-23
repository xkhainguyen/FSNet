#!/bin/bash
#SBATCH -t 06:00:00
#SBATCH -p mit_normal_gpu,mit_preemptable
#SBATCH --gres=gpu:l40s:1
#SBATCH -J ml4opt
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

set -euo pipefail

# ----------------------------------------
# Environment setup
# ----------------------------------------
source ~/.bashrc
conda activate ml4opt

cd /home/khain/orcd/scratch/FSNet

# ----------------------------------------
# Baseline FSNet MLP(1024) over 4 seeds
# ----------------------------------------
LR=${LR:-5e-5}
DROPOUT=${DROPOUT:-0.10}
DIST_WEIGHT=${DIST_WEIGHT:-5.0}
HIDDEN_DIM=${HIDDEN_DIM:-1024}
NUM_LAYERS=${NUM_LAYERS:-4}
EPOCHS=${EPOCHS:-300}

SEEDS=(0 1 2 3)

echo "=============================================="
echo " Job started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Job ID: $SLURM_JOB_ID"
echo " Node: $SLURM_NODELIST"
echo " Config: lr=$LR dropout=$DROPOUT dist_weight=$DIST_WEIGHT hidden_dim=$HIDDEN_DIM layers=$NUM_LAYERS epochs=$EPOCHS"
echo "=============================================="

for seed in "${SEEDS[@]}"; do
    echo "[Run] seed=$seed"
    python main.py \
        --method FSNet \
        --prob_type nonsmooth_nonconvex \
        --prob_name socp \
        --network MLP \
        --hidden_dim "$HIDDEN_DIM" \
        --num_layers "$NUM_LAYERS" \
        --num_epochs "$EPOCHS" \
        --seed "$seed" \
        --lr "$LR" \
        --dropout "$DROPOUT" \
        --dist_weight "$DIST_WEIGHT" \
        --wandb \
        --wandb_tags fsnet-mlp1024 baseline

done
