#!/bin/bash
#SBATCH -J vanFreeze
#SBATCH -p mit_normal_gpu,mit_preemptable
#SBATCH -G l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 01:30:00
#SBATCH --requeue
#SBATCH --exclude=node4104
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-7%4

# Vanilla @2667 with cosine horizon frozen after epoch 1000
# (same LR schedule as paper @1000, then hold LR for remaining budget).
# 2 methods × 4 seeds

source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
mkdir -p logs

METHODS=(penalty adaptive_penalty)
IDX=$SLURM_ARRAY_TASK_ID
METHOD=${METHODS[$((IDX / 4))]}
SEED=$((IDX % 4))
EPOCHS=2667
FREEZE=1000
LR=0.0001

echo "=============================================="
echo " started $(date '+%Y-%m-%d %H:%M:%S')"
echo " array=$IDX node=$SLURM_NODELIST"
echo " method=$METHOD seed=$SEED epochs=$EPOCHS freeze_lr_after_epoch=$FREEZE lr=$LR"
echo "=============================================="

python main.py \
    --method "$METHOD" \
    --prob_type nonsmooth_nonconvex \
    --prob_name socp \
    --lr "$LR" \
    --seed "$SEED" \
    --num_epochs "$EPOCHS" \
    --train_size 7000 \
    --freeze_lr_after_epoch "$FREEZE"

echo " finished $(date '+%Y-%m-%d %H:%M:%S')"
