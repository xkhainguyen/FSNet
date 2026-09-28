#!/bin/bash
#SBATCH -J slPre800
#SBATCH -p mit_normal_gpu,mit_preemptable
#SBATCH -G l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 00:30:00
#SBATCH --requeue
#SBATCH --exclude=node4104
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-7%4

# SL pretraining pool for the fixed-offline-budget study.
#
# 2 label-quality regimes x 4 seeds = 8 runs, 800 training instances each.
#   low    = maxt0.5  (gen 396s / 800 instances)
#   medium = maxt2.0  (gen 1618s / 800 instances)
#
# Runs the full 1000 epochs with NO early stopping and saves a checkpoint every
# 50 epochs, giving a 20-checkpoint pool per (regime, seed). The merit rule and
# the fixed-checkpoint baselines all select k* from this same pool post hoc;
# budget accounting later charges only k* * 0.16s of SL time.
#
# These checkpoints warm-start BOTH penalty and FSNet SSL, so 8 runs cover
# every downstream arm.
#
# Rerun (rather than reuse the 20260116 trainsize800 checkpoints) because those
# predate the eta_min fix and were trained with a rising LR schedule.
#
# Cost: ~160s train + validation overhead per run; 30 min walltime is generous.

source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
mkdir -p logs

# QUAL_LIST selects which maxt tiers to build pools for; 4 seeds each.
# Array index maps as: quality = QUAL_LIST[idx / 4], seed = idx % 4.
# Default reproduces the original two-regime pools (array 0-7).
#
# To map how much merit-based selection buys you as a function of label
# inexactness, build pools across tiers and compare, per pool, the task merit at
# the SL-loss-selected checkpoint against the merit-selected one. That analysis
# needs no SSL runs at all.
QUAL_LIST=${QUAL_LIST:-"0.5 2.0"}
read -r -a QARR <<< "$QUAL_LIST"

# SEED_BASE shifts the seed block, so extra replicates can be added later
# without disturbing the original pools (SEED_BASE=4 gives seeds 4-7).
IDX=$SLURM_ARRAY_TASK_ID
QUALITY=${QARR[$((IDX / 4))]}
SEED=$(( ${SEED_BASE:-0} + IDX % 4 ))

DATASET="datasets/nonsmooth_nonconvex/socp/random2025_socp_dataset_var100_ineq50_eq50_ex10000_maxt${QUALITY}_ready"

echo "=============================================="
echo " Job started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Job ID: $SLURM_JOB_ID  Array: $IDX"
echo " Node: $SLURM_NODELIST  GPU: ${CUDA_VISIBLE_DEVICES:-unknown}"
echo " method=sup_pen quality=maxt${QUALITY} seed=$SEED train_size=${TRAIN_SIZE:-800}"
echo " 1000 epochs, ckpt every 50, no early stopping, eta_min=1e-6"
echo " dataset=$DATASET"
echo "=============================================="

test -f "$DATASET" || { echo "MISSING DATASET $DATASET"; exit 1; }

# Pre-flight quota guard. trainer.py wraps every save in try/except and prints
# "x Error saving ..." then exits 0, so a quota-exhausted run otherwise looks
# COMPLETED with an empty result dir. Abort before burning GPU time instead.
QUOTA_LINE=$(quota -s 2>/dev/null | tail -1)
USED_G=$(echo "$QUOTA_LINE" | awk '{gsub(/[GT]/,"",$1); print $1+0}')
HARD_G=$(echo "$QUOTA_LINE" | awk '{gsub(/[GT]/,"",$3); print $3+0}')
if [ "$HARD_G" -gt 0 ] 2>/dev/null; then
    REMAIN_G=$((HARD_G - USED_G))
    echo " quota: ${USED_G}G used / ${HARD_G}G hard  (${REMAIN_G}G free)"
    if [ "$REMAIN_G" -lt 2 ]; then
        echo "ABORT: under 2G of quota headroom; saves would fail silently."
        exit 92
    fi
fi

RUNLOG=$(mktemp)
python main.py \
    --method sup_pen \
    --prob_type nonsmooth_nonconvex \
    --prob_name socp \
    --lr 0.0001 \
    --seed "$SEED" \
    --num_epochs 1000 \
    --train_size ${TRAIN_SIZE:-800} \
    --en_subopt 3 \
    --subopt_ratio "$QUALITY" \
    --eval_step 50 \
    --save_intermediate True \
    --lr_schedule cosine \
    --eta_min 1e-6 2>&1 | tee "$RUNLOG"
RC=${PIPESTATUS[0]}

# Fail loudly: propagate python's exit code, and treat a swallowed save error
# as a failure even though python itself returned 0.
if [ "$RC" -ne 0 ]; then
    echo "FAILED: python exited $RC"
    rm -f "$RUNLOG"; exit "$RC"
fi
if grep -q "Error saving" "$RUNLOG"; then
    echo "FAILED: save error (quota?) -- results were NOT written"
    rm -f "$RUNLOG"; exit 90
fi
if ! grep -q "Detailed results saved" "$RUNLOG"; then
    echo "FAILED: no results.pkl was written"
    rm -f "$RUNLOG"; exit 91
fi
rm -f "$RUNLOG"

echo "=============================================="
echo " Job finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
