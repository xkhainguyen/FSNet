#!/bin/bash
#SBATCH -J vanContLR
#SBATCH -p mit_normal_gpu,mit_preemptable
#SBATCH -G l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH --requeue
#SBATCH --exclude=node4104
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-7%4

# Continue paper Vanilla@1000 final ckpt with CONSTANT lr for remaining budget.
# Already spent ~180s (@1000); remaining ~300s → 1667 epochs (0.18 s/ep).
# Total wall ≈ 480s matched budget.
# 2 methods × 4 seeds

source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
mkdir -p logs

BASE=results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000

# paper vanilla @1000 dirs (seed-ordered)
PEN_CKPTS=(
  "$BASE/20260118-174600_MLP_penalty_seed0_nepochs1000_lr0.0001_trainsize7000/model.pt"
  "$BASE/20260118-174916_MLP_penalty_seed1_nepochs1000_lr0.0001_trainsize7000/model.pt"
  "$BASE/20260118-172439_MLP_penalty_seed2_nepochs1000_lr0.0001_trainsize7000/model.pt"
  "$BASE/20260118-172757_MLP_penalty_seed3_nepochs1000_lr0.0001_trainsize7000/model.pt"
)
ADA_CKPTS=(
  "$BASE/20260118-175812_MLP_adaptive_penalty_seed0_nepochs1000_lr0.0001_trainsize7000/model.pt"
  "$BASE/20260118-180127_MLP_adaptive_penalty_seed1_nepochs1000_lr0.0001_trainsize7000/model.pt"
  "$BASE/20260118-180441_MLP_adaptive_penalty_seed2_nepochs1000_lr0.0001_trainsize7000/model.pt"
  "$BASE/20260118-180755_MLP_adaptive_penalty_seed3_nepochs1000_lr0.0001_trainsize7000/model.pt"
)

IDX=$SLURM_ARRAY_TASK_ID
METHOD_I=$((IDX / 4))
SEED=$((IDX % 4))
METHODS=(penalty adaptive_penalty)
METHOD=${METHODS[$METHOD_I]}

if [[ "$METHOD" == "penalty" ]]; then
  CKPT=${PEN_CKPTS[$SEED]}
else
  CKPT=${ADA_CKPTS[$SEED]}
fi

# remaining epochs for ~300s SSL after the original 1000-ep / ~180s run
EXTRA_EP=1667
LR=0.0001

echo "=============================================="
echo " started $(date '+%Y-%m-%d %H:%M:%S')"
echo " array=$IDX node=$SLURM_NODELIST"
echo " method=$METHOD seed=$SEED extra_epochs=$EXTRA_EP constant_lr lr=$LR"
echo " ckpt=$CKPT"
echo "=============================================="

test -f "$CKPT" || { echo "MISSING $CKPT"; exit 1; }

python main.py \
    --method "$METHOD" \
    --prob_type nonsmooth_nonconvex \
    --prob_name socp \
    --lr "$LR" \
    --seed "$SEED" \
    --num_epochs "$EXTRA_EP" \
    --train_size 7000 \
    --checkpoint "$CKPT" \
    --constant_lr

echo " finished $(date '+%Y-%m-%d %H:%M:%S')"
