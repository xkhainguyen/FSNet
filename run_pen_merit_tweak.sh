#!/bin/bash
#SBATCH -J penTweak
#SBATCH -p pi_donti_gpu,mit_preemptable,mit_normal_gpu
#SBATCH -G rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH --exclude=node4104
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-11%8
# Prefer donti RTX PRO 6000 (node5101); fall back via partition list.
# Fair timing still TABLE rates: Gen=400, SL=(ckpt/250)*40, SSL=0.18s/ep.

# Push Pen WS over Van on mean Merit1e5.
# Fair TIMING uses TABLE RATES (not measured GPU wall):
#   wall_WS  = 400 + (ckpt/250)*40 + 0.18*ep
#   wall_Van = 0.18 * total_ssl_ep   (continue: 0.18*(1000+cont_ep))
# Any GPU OK; rates stay table-based for matched-budget picks.
#
# 0-7:  Ours Pen seed0  (ckpt 100/200 × lr 1e-4/2e-4 × eqW 10/50)
# 8-11: Van Pen continue seed0/1 × lr 1e-4/2e-4

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

IDX=$SLURM_ARRAY_TASK_ID

if (( IDX < 8 )); then
  CKPTS=(100 200)
  LRS=(0.0001 0.0002)
  EQS=(10 50)
  CKPT=${CKPTS[$((IDX / 4))]}
  REST=$((IDX % 4))
  LR=${LRS[$((REST / 2))]}
  EQ=${EQS[$((REST % 2))]}
  SEED=0
  EPOCHS=3000
  EXTRA=(--checkpoint "$SL0/model_${CKPT}.pt" --save_intermediate True
         --eq_pen_weight "$EQ" --ineq_pen_weight 10)
  KIND=ours
else
  LOCAL=$((IDX - 8))
  LRS=(0.0001 0.0002)
  LR=${LRS[$((LOCAL / 2))]}
  SEED=$((LOCAL % 2))
  EPOCHS=4500
  EXTRA=(--checkpoint "${PEN1000[$SEED]}" --save_intermediate True
         --eq_pen_weight 10 --ineq_pen_weight 10)
  CKPT=final
  EQ=10
  KIND=van_cont
fi

echo "=============================================="
echo " started $(date '+%Y-%m-%d %H:%M:%S') node=$SLURM_NODELIST"
echo " kind=$KIND method=penalty seed=$SEED ckpt=$CKPT lr=$LR eqW=$EQ epochs=$EPOCHS"
echo " TABLE timing: SSL_RATE=0.18s/ep Gen=400 SL=(ckpt/250)*40"
echo "=============================================="

python main.py \
    --method penalty \
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
