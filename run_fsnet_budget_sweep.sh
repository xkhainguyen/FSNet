#!/bin/bash
#SBATCH -J fsnSweep
#SBATCH -p pi_donti_gpu,mit_normal_gpu,mit_preemptable
#SBATCH -G rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH --exclude=node4104
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Table-rate epochs; any GPU OK. Prefer donti RTX PRO 6000.

# FSNet fixed-offline-budget sweep, driven by budget_manifest.tsv
# (regenerate with: python gen_budget_manifest.py)
#
# Manifest columns: idx  B  seed  q  arm  k  epochs  ckpt
#
# TIMING CONVENTION: epoch counts come from the reference s/ep rates measured
# on the original machine (SL 0.16, FSNet SSL 3.30 s/ep; gen 396.3s / 1618.1s),
# NOT from wall clock on the GPU this lands on. Do not recompute from H200 or
# RTX PRO 6000 timings -- the charged budget is held fixed so every arm and
# both regimes stay comparable.
#
# Submit as (manifest has 56 rows: B=2000 idx 0-27, B=2800 idx 28-55):
#   sbatch --array=0-55%6 --export=ALL,SCHED=const run_fsnet_budget_sweep.sh
# Constant LR: arms differ only in epochs/ckpt, not cosine T_max shape.

source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
mkdir -p logs

MANIFEST=${MANIFEST:-budget_manifest.tsv}
test -f "$MANIFEST" || { echo "MISSING $MANIFEST"; exit 1; }

LINE=$(awk -F'\t' -v i="$SLURM_ARRAY_TASK_ID" '$1==i' "$MANIFEST")
test -n "$LINE" || { echo "no manifest row for idx $SLURM_ARRAY_TASK_ID"; exit 1; }

IFS=$'\t' read -r IDX B SEED Q ARM K NUM_EPOCHS CKPT <<< "$LINE"
SCHED=${SCHED:-const}

echo "=============================================="
echo " Job started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Job ID: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
echo " Node: $SLURM_NODELIST  GPU: ${CUDA_VISIBLE_DEVICES:-unknown}"
echo " B=$B seed=$SEED regime=q$Q arm=$ARM k=$K epochs=$NUM_EPOCHS sched=$SCHED"
echo " ckpt=$CKPT"
echo "=============================================="

ARGS=(
    --method ${METHOD:-FSNet}
    --prob_type nonsmooth_nonconvex
    --prob_name socp
    --lr 0.0001
    --seed "$SEED"
    --num_epochs "$NUM_EPOCHS"
    --train_size 7000
    --eval_step 50
)

if [ "$SCHED" = "const" ]; then
    ARGS+=(--constant_lr)
else
    ARGS+=(--lr_schedule cosine --eta_min 1e-6)
fi

if [ "$CKPT" != "-" ]; then
    test -f "$CKPT" || { echo "MISSING CKPT $CKPT"; exit 1; }
    ARGS+=(--checkpoint "$CKPT")
fi

# Pre-flight space guard on the filesystem that actually holds results/
# (a symlink to a separate pool, NOT the home quota). trainer.py wraps saves in
# try/except and exits 0, so without this a full pool yields an empty result
# dir reported as COMPLETED -- that silently cost 1.8 GPU-h earlier.
RESPATH=$(readlink -f results)
AVAIL_G=$(df -BG --output=avail "$RESPATH" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$AVAIL_G" ]; then
    echo " results pool free: ${AVAIL_G}G  ($RESPATH)"
    if [ "$AVAIL_G" -lt 5 ]; then
        echo "ABORT: under 5G free on results pool; saves would fail silently."
        exit 92
    fi
fi

RUNLOG=$(mktemp)
python main.py "${ARGS[@]}" 2>&1 | tee "$RUNLOG"
RC=${PIPESTATUS[0]}
if [ "$RC" -ne 0 ]; then
    echo "FAILED: python exited $RC"; rm -f "$RUNLOG"; exit "$RC"
fi
if grep -q "Error saving" "$RUNLOG"; then
    echo "FAILED: save error (disk full?) -- results were NOT written"
    rm -f "$RUNLOG"; exit 90
fi
if ! grep -q "Detailed results saved" "$RUNLOG"; then
    echo "FAILED: no results.pkl was written"; rm -f "$RUNLOG"; exit 91
fi
rm -f "$RUNLOG"

echo "=============================================="
echo " Job finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
