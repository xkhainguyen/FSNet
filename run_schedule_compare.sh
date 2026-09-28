#!/bin/bash
#SBATCH -J schedCmp
#SBATCH -p mit_normal_gpu,mit_preemptable
#SBATCH -G l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 01:30:00
#SBATCH --requeue
#SBATCH --exclude=node4104
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-19%4

# Compare LR schedules (same settings on Van and fair Ours):
#   A) cosine + eta_min=1e-6
#   B) StepLR decay 0.9 / 100 epochs
# Jobs 0-15: Vanilla@2667  (2 sched × 2 method × 4 seeds)
# Jobs 16-19: fair Ours seed0 ckpt100 (2 sched × 2 method)

source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
mkdir -p logs

IDX=$SLURM_ARRAY_TASK_ID
METHODS=(penalty adaptive_penalty)
CKPT_BASE="results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260116-022657_MLP_sup_pen_seed0_nepochs1000_lr0.0001_trainsize800_subopt_3_0.5"

if (( IDX < 16 )); then
  # Vanilla@2667
  SCHED_I=$((IDX / 8))          # 0 cosine, 1 step
  REST=$((IDX % 8))
  METH_I=$((REST / 4))          # 0 pen, 1 ada
  SEED=$((REST % 4))
  METHOD=${METHODS[$METH_I]}
  EPOCHS=2667
  EXTRA=()
  KIND=van
else
  # Fair Ours seed0 ckpt100 → SSL epochs = floor((80-16)/0.18)=355
  LOCAL=$((IDX - 16))
  SCHED_I=$((LOCAL / 2))
  METH_I=$((LOCAL % 2))
  METHOD=${METHODS[$METH_I]}
  SEED=0
  EPOCHS=355
  CKPT="${CKPT_BASE}/model_100.pt"
  EXTRA=(--checkpoint "$CKPT")
  KIND=ours
fi

LR=0.0001
if (( SCHED_I == 0 )); then
  SCHED_ARGS=(--lr_schedule cosine --eta_min 1e-6)
  SCHED_TAG=cosine_etamin1e-6
else
  SCHED_ARGS=(--lr_schedule step --lr_decay 0.9 --lr_decay_step 100)
  SCHED_TAG=step_d0.9_s100
fi

echo "=============================================="
echo " started $(date '+%Y-%m-%d %H:%M:%S')"
echo " array=$IDX node=$SLURM_NODELIST"
echo " kind=$KIND method=$METHOD seed=$SEED epochs=$EPOCHS sched=$SCHED_TAG"
echo "=============================================="

python main.py \
    --method "$METHOD" \
    --prob_type nonsmooth_nonconvex \
    --prob_name socp \
    --lr "$LR" \
    --seed "$SEED" \
    --num_epochs "$EPOCHS" \
    --train_size 7000 \
    "${SCHED_ARGS[@]}" \
    "${EXTRA[@]}"

echo " finished $(date '+%Y-%m-%d %H:%M:%S')"
