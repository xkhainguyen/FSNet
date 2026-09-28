#!/bin/bash
#SBATCH -J pen05
#SBATCH -p pi_donti_gpu,mit_normal_gpu,mit_preemptable
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH --requeue
#SBATCH --exclude=node4104
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

# penalty (plain, not adaptive) fixed-offline-budget test at maxt0.5, B=1100s.
# Manifest: budget_manifest_pen05.tsv  (gen with gen_manifest_penalty05.py)
# Columns: idx  B  seed  q  arm  k  epochs  ckpt
#
# TIMING CONVENTION: epoch counts come from the reference rates measured on the
# original machine (SL 0.16, penalty SSL 0.18 s/ep, gen 396.3s for 800 @ maxt0.5),
# NOT from wall clock here. The charged budget is held fixed so every arm stays
# comparable across machines.
#
# RECIPE: lr=1e-4 with eq_pen_weight=50 and a full cosine anneal to eta_min=1e-6.
# lr=5e-4 (the adaptive_penalty setting) makes plain penalty's eq term diverge.
# Every arm, vanilla included, uses the identical lr/schedule/weights: the only
# differences are the epoch count each arm can afford and its init checkpoint.
#
# Submit:
#   sbatch --array=0-15%6 run_penalty_budget_pen05.sh

source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
mkdir -p logs

MANIFEST=${MANIFEST:-budget_manifest_pen05.tsv}
test -f "$MANIFEST" || { echo "MISSING $MANIFEST"; exit 1; }

LINE=$(awk -F'\t' -v i="$SLURM_ARRAY_TASK_ID" '$1==i' "$MANIFEST")
test -n "$LINE" || { echo "no manifest row for idx $SLURM_ARRAY_TASK_ID"; exit 1; }
IFS=$'\t' read -r IDX B SEED Q ARM K NUM_EPOCHS CKPT LR_ROW <<< "$LINE"

METHOD=${METHOD:-penalty}
LR=${LR_ROW:-${LR:-1e-4}}
EQW=${EQW:-50}
INEQW=${INEQW:-10}
# Scale the adaptive caps with the initial weight (defaults are 50x eq, 10x ineq)
# so raising eqW does not just pin the ramp at its ceiling.
EQMAX=${EQMAX:-}
INEQMAX=${INEQMAX:-}
[ -n "$EPOCH_OVERRIDE" ] && NUM_EPOCHS=$EPOCH_OVERRIDE

echo "=============================================="
echo " started $(date '+%F %T')  job $SLURM_JOB_ID array $SLURM_ARRAY_TASK_ID"
echo " node $SLURM_NODELIST  gpu ${CUDA_VISIBLE_DEVICES:-unknown}"
echo " method=$METHOD B=$B seed=$SEED q=$Q arm=$ARM k=$K epochs=$NUM_EPOCHS"
echo " lr=$LR eq_pen_weight=$EQW ineq_pen_weight=$INEQW sched=cosine eta_min=1e-6"
echo " ckpt=$CKPT"
echo "=============================================="

ARGS=(
    --method "$METHOD"
    --prob_type nonsmooth_nonconvex
    --prob_name socp
    --lr "$LR"
    --eq_pen_weight "$EQW"
    --ineq_pen_weight "$INEQW"
    --lr_schedule cosine
    --eta_min 1e-6
    --seed "$SEED"
    --num_epochs "$NUM_EPOCHS"
    --train_size 7000
    --eval_step 50
)

[ -n "$EQMAX" ]   && ARGS+=(--eq_pen_weight_max "$EQMAX")
[ -n "$INEQMAX" ] && ARGS+=(--ineq_pen_weight_max "$INEQMAX")

if [ "$CKPT" != "-" ]; then
    test -f "$CKPT" || { echo "MISSING CKPT $CKPT"; exit 1; }
    ARGS+=(--checkpoint "$CKPT")
fi

# Pre-flight space guard on the filesystem that actually holds results/ (a
# symlink to a separate pool, not the home quota). trainer.py wraps saves in
# try/except and still exits 0, so a full pool otherwise yields an empty result
# dir reported as COMPLETED.
RESPATH=$(readlink -f results)
AVAIL_G=$(df -BG --output=avail "$RESPATH" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$AVAIL_G" ]; then
    echo " results pool free: ${AVAIL_G}G  ($RESPATH)"
    [ "$AVAIL_G" -lt 5 ] && { echo "ABORT: <5G free; saves would fail silently."; exit 92; }
fi

RUNLOG=$(mktemp)
python main.py "${ARGS[@]}" 2>&1 | tee "$RUNLOG"
RC=${PIPESTATUS[0]}
[ "$RC" -ne 0 ] && { echo "FAILED: python exited $RC"; rm -f "$RUNLOG"; exit "$RC"; }
grep -q "Error saving" "$RUNLOG" && { echo "FAILED: save error"; rm -f "$RUNLOG"; exit 90; }
grep -q "Detailed results saved" "$RUNLOG" || { echo "FAILED: no results.pkl"; rm -f "$RUNLOG"; exit 91; }
rm -f "$RUNLOG"

echo "=============================================="
echo " finished $(date '+%F %T')"
echo "=============================================="
