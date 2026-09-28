#!/bin/bash
#SBATCH -J acopfRec
#SBATCH -p mit_normal_gpu,mit_preemptable
#SBATCH -G l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH --exclude=node4104
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-9%4

# ACOPF-style recipe on SOCP (keep schedCmp running):
#  - Longer WS SSL 3000 ep, lr=5e-4, stretched cosine eta_min=1e-6
#  - save_intermediate every eval_step=10 for hindsight
#  - Vanilla continue from paper@1000 for +4500 ep (matched ~980s wall)
#
# Fair wall (table rates): WS = 400 + (ckpt/250)*40 + 0.18*ep
#                          Van = 0.18 * ssl_ep
# Hindsight later: best val merit among WS ckpts with wall_WS <= wall_Van
#
# 0-5: Ours seed0  (2 methods × ckpt 50/100/200)
# 6-9: Van continue seed0/1 for pen+ada (2 methods × 2 seeds) — expand later

source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
mkdir -p logs

BASE=results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000
SL0="$BASE/20260116-022657_MLP_sup_pen_seed0_nepochs1000_lr0.0001_trainsize800_subopt_3_0.5"

PEN1000=(
  "$BASE/20260118-174600_MLP_penalty_seed0_nepochs1000_lr0.0001_trainsize7000/model.pt"
  "$BASE/20260118-174916_MLP_penalty_seed1_nepochs1000_lr0.0001_trainsize7000/model.pt"
)
ADA1000=(
  "$BASE/20260118-175812_MLP_adaptive_penalty_seed0_nepochs1000_lr0.0001_trainsize7000/model.pt"
  "$BASE/20260118-180127_MLP_adaptive_penalty_seed1_nepochs1000_lr0.0001_trainsize7000/model.pt"
)

IDX=$SLURM_ARRAY_TASK_ID
METHODS=(penalty adaptive_penalty)
CKPTS=(50 100 200)
LR=0.0005

if (( IDX < 6 )); then
  KIND=ours
  METH_I=$((IDX / 3))
  CKPT_I=$((IDX % 3))
  METHOD=${METHODS[$METH_I]}
  CKPT=${CKPTS[$CKPT_I]}
  SEED=0
  EPOCHS=3000
  EXTRA=(--checkpoint "$SL0/model_${CKPT}.pt" --save_intermediate True)
  TAG="ours_ckpt${CKPT}"
else
  KIND=van_cont
  LOCAL=$((IDX - 6))
  METH_I=$((LOCAL / 2))
  SEED=$((LOCAL % 2))
  METHOD=${METHODS[$METH_I]}
  # +4500 ep after paper@1000 ≈ 5500 total SSL; wall≈990s matches WS@3000
  EPOCHS=4500
  if [[ "$METHOD" == "penalty" ]]; then
    CKPT_PATH=${PEN1000[$SEED]}
  else
    CKPT_PATH=${ADA1000[$SEED]}
  fi
  EXTRA=(--checkpoint "$CKPT_PATH" --save_intermediate True)
  TAG="van_cont_seed${SEED}"
fi

echo "=============================================="
echo " started $(date '+%Y-%m-%d %H:%M:%S')"
echo " array=$IDX node=$SLURM_NODELIST"
echo " kind=$KIND tag=$TAG method=$METHOD seed=$SEED epochs=$EPOCHS lr=$LR"
echo " cosine stretched eta_min=1e-6 save_intermediate"
echo "=============================================="

python main.py \
    --method "$METHOD" \
    --prob_type nonsmooth_nonconvex \
    --prob_name socp \
    --lr "$LR" \
    --seed "$SEED" \
    --num_epochs "$EPOCHS" \
    --train_size 7000 \
    --lr_schedule cosine \
    --eta_min 1e-6 \
    --eval_step 10 \
    "${EXTRA[@]}"

echo " finished $(date '+%Y-%m-%d %H:%M:%S')"
