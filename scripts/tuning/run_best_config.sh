#!/bin/bash
# run_best_config.sh — Final confirmation: run the best-found config on seeds 0-3.
#
# Usage:
#   Set the vars below to whatever Wave-2 (or Wave-1) identified as best, then run:
#     sbatch scripts/tuning/run_best_config.sh
#   OR source it directly on a node (no sbatch):
#     bash scripts/tuning/run_best_config.sh
#
#SBATCH -t 08:00:00
#SBATCH -p mit_normal_gpu,mit_preemptable
#SBATCH --gres=gpu:l40s:1
#SBATCH -J fsnet-m1024-best
#SBATCH --array=0-3
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

# ── Fill these in from the leaderboard before submitting ──────────────────────
: "${BEST_LR:=5e-5}"
: "${BEST_DROPOUT:=0.10}"
: "${BEST_DIST_WEIGHT:=5.0}"
: "${BEST_VAL_TOL:=1e-7}"
: "${BEST_DECAY_TOL_STEP:=100}"
: "${BEST_MEMORY_SIZE:=30}"
: "${BEST_MAX_DIFF_ITER:=30}"
: "${BEST_CFG_ID:=best}"
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

source ~/.bashrc
conda activate ml4opt

cd /home/khain/orcd/scratch/FSNet

seed=${SLURM_ARRAY_TASK_ID:-0}

mkdir -p logs/fsnet_tuning_final
run_log="logs/fsnet_tuning_final/${BEST_CFG_ID}_seed${seed}_job${SLURM_JOB_ID:-local}_${seed}.log"
manifest="logs/fsnet_tuning_final/final_manifest.tsv"

echo "==============================================" | tee "$run_log"
echo "Final confirmation run start: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$run_log"
echo "Config: $BEST_CFG_ID seed=$seed" | tee -a "$run_log"
echo "  lr=$BEST_LR dropout=$BEST_DROPOUT dist_weight=$BEST_DIST_WEIGHT" | tee -a "$run_log"
echo "  val_tol=$BEST_VAL_TOL decay_tol_step=$BEST_DECAY_TOL_STEP" | tee -a "$run_log"
echo "  memory_size=$BEST_MEMORY_SIZE max_diff_iter=$BEST_MAX_DIFF_ITER" | tee -a "$run_log"
echo "==============================================" | tee -a "$run_log"

python main.py \
  --method FSNet \
  --prob_type nonsmooth_nonconvex \
  --prob_name socp \
  --network MLP \
  --hidden_dim 1024 \
  --num_layers 4 \
  --num_epochs 300 \
  --seed "$seed" \
  --lr "$BEST_LR" \
  --dropout "$BEST_DROPOUT" \
  --dist_weight "$BEST_DIST_WEIGHT" \
  --val_tol "$BEST_VAL_TOL" \
  --decay_tol_step "$BEST_DECAY_TOL_STEP" \
  --memory_size "$BEST_MEMORY_SIZE" \
  --max_diff_iter "$BEST_MAX_DIFF_ITER" \
  --wandb \
  --wandb_tags fsnet-mlp1024 final "$BEST_CFG_ID" \
  2>&1 | tee -a "$run_log"

save_dir=$(grep -oE 'save_dir: .*' "$run_log" | tail -n1 | sed 's/save_dir: //')

if [[ -n "${save_dir:-}" ]]; then
  if [[ ! -f "$manifest" ]]; then
    echo -e "cfg_id\tseed\tlr\tdropout\tdist_weight\tval_tol\tdecay_tol_step\tmemory_size\tmax_diff_iter\tsave_dir\trun_log" > "$manifest"
  fi
  echo -e "${BEST_CFG_ID}\t${seed}\t${BEST_LR}\t${BEST_DROPOUT}\t${BEST_DIST_WEIGHT}\t${BEST_VAL_TOL}\t${BEST_DECAY_TOL_STEP}\t${BEST_MEMORY_SIZE}\t${BEST_MAX_DIFF_ITER}\t${save_dir}\t${run_log}" >> "$manifest"
  echo "Recorded manifest entry for $BEST_CFG_ID seed=$seed -> $save_dir"
else
  echo "WARNING: could not parse save_dir from log"
fi

echo "Final confirmation run end: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$run_log"
