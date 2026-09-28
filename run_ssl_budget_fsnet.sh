#!/bin/bash
#SBATCH -J sslFSN
#SBATCH -p pi_donti_gpu
#SBATCH -G h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH --requeue
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-6%4

# FSNet SSL under a fixed total offline budget B = 2800s = Gen + SL + SSL.
# Seed-0 screen; expand to seeds 1-3 once the ordering looks right.
#
# TIMING CONVENTION: epoch counts below are derived from the reference s/ep
# rates measured on the original machine, NOT from H200 wall clock. Runs
# execute on pi_donti_gpu H200s but the charged budget is held fixed so the
# accounting stays comparable across all arms and regimes. Do not recompute
# epoch counts from H200 timings.
#
# B=2800 is bounded below by Gen(q=2.0, 800 inst) = 1618s, so a smaller budget
# is infeasible for the medium regime.
#
# Measured gen cost (sum of solve_time_sec over first 800 instances):
#   q=0.5 (low)     396.3s
#   q=2.0 (medium) 1618.1s
# SL charged at a flat 0.16 s/ep; FSNet SSL at 3.30 s/ep.
#
# Arms: vanilla (no labels, all budget to SSL) vs warm-start from three
# checkpoint-selection rules. k* is the argmin of validation stop_merit over
# the 20-checkpoint SL pool, selected post hoc but charged only k*x0.16s.
#   seed 0:  k* = 900 (q=0.5), 200 (q=2.0)
# 'early' (k=100) and 'conv' (k=950) are the fixed-checkpoint baselines that
# should each fail in one regime.

source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
mkdir -p logs

SEED=0
METHOD=FSNet
RES=results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000
SL05="$RES/20260726-035102_MLP_sup_pen_seed0_nepochs1000_lr0.0001_trainsize800_subopt_3_0.5_obj0.1_eq5.0_ineq5.0_lrschedcosine_etamin1e-06"
SL20="$RES/20260726-035722_MLP_sup_pen_seed0_nepochs1000_lr0.0001_trainsize800_subopt_3_2.0_obj0.1_eq5.0_ineq5.0_lrschedcosine_etamin1e-06"

#        0        1          2          3         4          5          6
LABELS=(vanilla q0.5merit q0.5early q0.5conv q2.0merit q2.0early q2.0conv)
EPOCHS=(    848       685       724      682       348       353      312)
CKPTS=(  ""  "$SL05/model_900.pt" "$SL05/model_100.pt" "$SL05/model_950.pt" \
             "$SL20/model_200.pt" "$SL20/model_100.pt" "$SL20/model_950.pt")

IDX=$SLURM_ARRAY_TASK_ID
LABEL=${LABELS[$IDX]}
NUM_EPOCHS=${EPOCHS[$IDX]}
CKPT=${CKPTS[$IDX]}

echo "=============================================="
echo " Job started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Job ID: $SLURM_JOB_ID  Array: $IDX"
echo " Node: $SLURM_NODELIST  GPU: ${CUDA_VISIBLE_DEVICES:-unknown}"
echo " method=$METHOD seed=$SEED arm=$LABEL epochs=$NUM_EPOCHS"
echo " budget: B=2800s total (gen+SL+SSL)"
echo " ckpt=${CKPT:-<none, from scratch>}"
echo "=============================================="

ARGS=(
    --method "$METHOD"
    --prob_type nonsmooth_nonconvex
    --prob_name socp
    --lr 0.0001
    --seed "$SEED"
    --num_epochs "$NUM_EPOCHS"
    --train_size 7000
    --eval_step 50
)

# SCHED=const removes the LR-schedule confound: with cosine, T_max=num_epochs,
# so each arm gets a DIFFERENT LR trajectory (the 1277-ep arm decays to eta_min
# quickly, the 11111-ep arm stays near base lr for thousands of epochs). Arms
# would then differ in schedule shape as well as checkpoint and budget.
# Constant LR makes epochs the only budget axis, and avoids re-decaying from
# full lr over a pretrained checkpoint by a per-arm-different amount.
SCHED=${SCHED:-cosine}
if [ "$SCHED" = "const" ]; then
    ARGS+=(--constant_lr)
else
    ARGS+=(--lr_schedule cosine --eta_min 1e-6)
fi
echo " lr schedule: $SCHED"

if [ -n "$CKPT" ]; then
    test -f "$CKPT" || { echo "MISSING CKPT $CKPT"; exit 1; }
    ARGS+=(--checkpoint "$CKPT")
fi

python main.py "${ARGS[@]}"

echo "=============================================="
echo " Job finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
