#!/bin/bash
#SBATCH -J fairWS0
#SBATCH -p mit_normal_gpu,mit_preemptable
#SBATCH -G l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH --requeue
#SBATCH --exclude=node4104
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-27%4

# Fair total budget = 480s = Gen(400) + SL + SSL
# SL cost = (ckpt/250)*40s ; SSL epochs = floor((80 - SL_cost)/0.18)
# Screen on seed=0; expand winners to seeds 1-3 later.

source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
mkdir -p logs

CKPTS=(50 100 150 200 250 300 400)
METHODS=(penalty penalty adaptive_penalty adaptive_penalty)
LRS=(0.0001 0.0002 0.0001 0.00013)

# 7 ckpts × 4 (method,lr) = 28
IDX=$SLURM_ARRAY_TASK_ID
CKPT_I=$((IDX / 4))
CFG_I=$((IDX % 4))
CKPT=${CKPTS[$CKPT_I]}
METHOD=${METHODS[$CFG_I]}
LR=${LRS[$CFG_I]}

# budget math
# SL_COST = CKPT/250 * 40
# SSL_BUDGET = 80 - SL_COST
# SSL_EP = int(SSL_BUDGET / 0.18)
SSL_EP=$(python - <<PY
ckpt=$CKPT
sl = ckpt/250*40
ssl_budget = 80 - sl
assert ssl_budget > 0
print(int(ssl_budget / 0.18))
PY
)

CKPT_PATH="results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260116-022657_MLP_sup_pen_seed0_nepochs1000_lr0.0001_trainsize800_subopt_3_0.5/model_${CKPT}.pt"

echo "=============================================="
echo " started $(date '+%Y-%m-%d %H:%M:%S')"
echo " array=$IDX node=$SLURM_NODELIST"
echo " method=$METHOD lr=$LR seed=0 ckpt=$CKPT ssl_epochs=$SSL_EP"
echo " fair: Gen=400 SL=$(( CKPT * 40 / 250 ))s SSL~$(python -c "print(round($SSL_EP*0.18,1))")s total=480"
echo " ckpt=$CKPT_PATH"
echo "=============================================="

test -f "$CKPT_PATH" || { echo "MISSING $CKPT_PATH"; exit 1; }

python main.py \
    --method "$METHOD" \
    --prob_type nonsmooth_nonconvex \
    --prob_name socp \
    --lr "$LR" \
    --seed 0 \
    --num_epochs "$SSL_EP" \
    --train_size 7000 \
    --checkpoint "$CKPT_PATH"

echo " finished $(date '+%Y-%m-%d %H:%M:%S')"
