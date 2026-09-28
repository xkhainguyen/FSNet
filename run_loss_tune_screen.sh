#!/bin/bash
#SBATCH -J lossTune0
#SBATCH -p mit_normal_gpu,mit_preemptable
#SBATCH -G l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 01:30:00
#SBATCH --requeue
#SBATCH --exclude=node4104
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-35%4

# Seed-0 screen: fair 480s budget, tune shared loss weights.
# Jobs 0-23: Ours warmstart (ckpt × method × loss)
# Jobs 24-35: Vanilla@2667 matched loss (method × loss)  — fair baseline
#
# Fair: Gen400 + SL(ckpt) + SSL = 480; SSL_ep = floor((80 - SL)/0.18)
# Vanilla gets full 480s SSL ≈ 2667 epochs (same loss as paired Ours).

source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
mkdir -p logs

CKPTS=(50 100 150)
METHODS=(penalty adaptive_penalty)
# (eq, ineq, obj) — default paper is 10/10/1
EQS=(10 50 100 200)
INEQS=(10 10 10 10)   # keep ineq=10 first wave; eq is the bottleneck
OBJS=(1.0 1.0 1.0 1.0)
# also try stronger ineq / weaker obj in a compact extra set via CFG below

# Compact explicit config list (24 ours + 12 vanilla = 36)
# format: KIND METHOD CKPT EQ INEQ OBJ LR
# KIND: ours | van
CONFIGS=(
  # ---- Ours: ckpt50/100/150 × penalty × 4 eq weights ----
  "ours penalty 50 10 10 1.0 0.0001"
  "ours penalty 50 50 10 1.0 0.0001"
  "ours penalty 50 100 10 1.0 0.0001"
  "ours penalty 50 200 10 1.0 0.0001"
  "ours penalty 100 10 10 1.0 0.0001"
  "ours penalty 100 50 10 1.0 0.0001"
  "ours penalty 100 100 10 1.0 0.0001"
  "ours penalty 100 200 10 1.0 0.0001"
  "ours penalty 150 50 10 1.0 0.0001"
  "ours penalty 150 100 10 1.0 0.0001"
  "ours penalty 150 200 50 1.0 0.0001"
  "ours penalty 50 100 50 0.5 0.0001"
  # ---- Ours: adaptive ----
  "ours adaptive_penalty 50 10 10 1.0 0.0001"
  "ours adaptive_penalty 50 50 10 1.0 0.0001"
  "ours adaptive_penalty 50 100 10 1.0 0.0001"
  "ours adaptive_penalty 50 200 10 1.0 0.0001"
  "ours adaptive_penalty 100 50 10 1.0 0.0001"
  "ours adaptive_penalty 100 100 10 1.0 0.0001"
  "ours adaptive_penalty 100 200 10 1.0 0.0001"
  "ours adaptive_penalty 150 50 10 1.0 0.0001"
  "ours adaptive_penalty 150 100 10 1.0 0.0001"
  "ours adaptive_penalty 50 100 50 0.5 0.0001"
  "ours adaptive_penalty 100 100 50 0.5 0.00013"
  "ours adaptive_penalty 50 200 50 0.5 0.00013"
  # ---- Vanilla@2667 matched loss (no checkpoint) ----
  "van penalty 0 10 10 1.0 0.0001"
  "van penalty 0 50 10 1.0 0.0001"
  "van penalty 0 100 10 1.0 0.0001"
  "van penalty 0 200 10 1.0 0.0001"
  "van penalty 0 100 50 0.5 0.0001"
  "van penalty 0 200 50 1.0 0.0001"
  "van adaptive_penalty 0 10 10 1.0 0.0001"
  "van adaptive_penalty 0 50 10 1.0 0.0001"
  "van adaptive_penalty 0 100 10 1.0 0.0001"
  "van adaptive_penalty 0 200 10 1.0 0.0001"
  "van adaptive_penalty 0 100 50 0.5 0.00013"
  "van adaptive_penalty 0 200 50 0.5 0.00013"
)

IDX=$SLURM_ARRAY_TASK_ID
read -r KIND METHOD CKPT EQ INEQ OBJ LR <<< "${CONFIGS[$IDX]}"

if [[ "$KIND" == "ours" ]]; then
  SSL_EP=$(python - <<PY
ckpt=$CKPT
sl = ckpt/250*40
ssl_budget = 80 - sl
assert ssl_budget > 0
print(int(ssl_budget / 0.18))
PY
)
  CKPT_PATH="results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260116-022657_MLP_sup_pen_seed0_nepochs1000_lr0.0001_trainsize800_subopt_3_0.5/model_${CKPT}.pt"
  EXTRA=(--checkpoint "$CKPT_PATH")
  TAG="ours_ckpt${CKPT}"
else
  SSL_EP=2667
  EXTRA=()
  TAG="van"
fi

echo "=============================================="
echo " started $(date '+%Y-%m-%d %H:%M:%S')"
echo " array=$IDX node=$SLURM_NODELIST"
echo " kind=$KIND method=$METHOD seed=0 ckpt=$CKPT ssl_epochs=$SSL_EP"
echo " loss: obj=$OBJ eq=$EQ ineq=$INEQ lr=$LR"
echo "=============================================="

python main.py \
    --method "$METHOD" \
    --prob_type nonsmooth_nonconvex \
    --prob_name socp \
    --lr "$LR" \
    --seed 0 \
    --num_epochs "$SSL_EP" \
    --train_size 7000 \
    --obj_weight "$OBJ" \
    --eq_pen_weight "$EQ" \
    --ineq_pen_weight "$INEQ" \
    "${EXTRA[@]}"

echo " finished $(date '+%Y-%m-%d %H:%M:%S') tag=$TAG"
