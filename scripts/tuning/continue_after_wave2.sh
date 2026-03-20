#!/bin/bash
# Continue the tuning pipeline from an already-submitted Wave-2 job.
#
# Usage:
#   bash scripts/tuning/continue_after_wave2.sh 10579381
#
# It will:
#   1) wait for wave2_manifest.tsv to reach 36 runs,
#   2) build wave2_leaderboard.csv,
#   3) submit final 4-seed confirmation,
#   4) wait for final_manifest.tsv to reach 4 runs,
#   5) build final_leaderboard.csv,
#   6) append a summary to scripts/tuning/TUNING_LOG.md.

set -euo pipefail

W2_JOB_ID=${1:-}
if [[ -z "$W2_JOB_ID" ]]; then
  echo "Usage: bash scripts/tuning/continue_after_wave2.sh <wave2_job_id>"
  exit 2
fi

source ~/.bashrc
conda activate ml4opt

ROOT="/home/khain/orcd/scratch/FSNet"
cd "$ROOT"

LOG_MD="scripts/tuning/TUNING_LOG.md"
POLL=120

w2_manifest="logs/fsnet_tuning_wave2/wave2_manifest.tsv"
w2_csv="logs/fsnet_tuning_wave2/wave2_leaderboard.csv"
final_manifest="logs/fsnet_tuning_final/final_manifest.tsv"
final_csv="logs/fsnet_tuning_final/final_leaderboard.csv"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
count_rows() { [[ -f "$1" ]] && awk 'NR>1{c++}END{print c+0}' "$1" || echo 0; }

log "Continuation monitor started for Wave-2 job $W2_JOB_ID"

# Phase A: wait for wave2 manifest completion
log "Waiting for Wave-2 manifest (36 runs)..."
while true; do
  n=$(count_rows "$w2_manifest")
  log "Wave-2 manifest rows: $n / 36"
  if [[ "$n" -ge 36 ]]; then
    break
  fi

  # If the job is no longer in queue and manifest is still incomplete, fail loudly.
  if ! squeue -j "$W2_JOB_ID" --noheader 2>/dev/null | grep -q "$W2_JOB_ID"; then
    log "ERROR: Wave-2 job $W2_JOB_ID left queue before 36 completed rows."
    exit 3
  fi
  sleep "$POLL"
done

log "Wave-2 complete. Building leaderboard..."
python3 scripts/tuning/fsnet_mlp1024_leaderboard.py \
  --manifest "$w2_manifest" \
  --batch-size 256 \
  --out-csv "$w2_csv"

# Parse top-1 from CSV
read -r best_cfg best_lr best_drop best_dist best_vt best_dt best_mem best_diff < <(awk -F',' 'NR==2{print $1, $2, $3, $4, $5, $6, $7, $8}' "$w2_csv")

log "Wave-2 winner: cfg=$best_cfg lr=$best_lr drop=$best_drop dist=$best_dist vt=$best_vt dt=$best_dt mem=$best_mem diff=$best_diff"

# Append wave-2 result to markdown log
{
  echo ""
  echo "### Wave-2 Auto-Update ($(date '+%Y-%m-%d %H:%M:%S'))"
  echo "- Status: COMPLETED"
  echo "- Winner: $best_cfg"
  echo "- Params: lr=$best_lr, dropout=$best_drop, dist_weight=$best_dist, val_tol=$best_vt, decay_tol_step=$best_dt, memory_size=$best_mem, max_diff_iter=$best_diff"
  echo "- Leaderboard: logs/fsnet_tuning_wave2/wave2_leaderboard.csv"
} >> "$LOG_MD"

# Phase B: submit final confirmation
log "Submitting final confirmation (4 seeds)..."
final_job=$(BEST_LR="$best_lr" BEST_DROPOUT="$best_drop" BEST_DIST_WEIGHT="$best_dist" \
  BEST_VAL_TOL="$best_vt" BEST_DECAY_TOL_STEP="$best_dt" \
  BEST_MEMORY_SIZE="$best_mem" BEST_MAX_DIFF_ITER="$best_diff" \
  BEST_CFG_ID="${best_cfg}_final" \
  sbatch --parsable scripts/tuning/run_best_config.sh)
final_job=$(echo "$final_job" | tr -d '[:space:]')
log "Final job submitted: $final_job"

{
  echo "- Final confirmation job: $final_job"
} >> "$LOG_MD"

# Phase C: wait for final manifest completion
log "Waiting for final manifest (4 runs)..."
while true; do
  m=$(count_rows "$final_manifest")
  log "Final manifest rows: $m / 4"
  if [[ "$m" -ge 4 ]]; then
    break
  fi

  if ! squeue -j "$final_job" --noheader 2>/dev/null | grep -q "$final_job"; then
    log "ERROR: Final job $final_job left queue before 4 completed rows."
    exit 4
  fi
  sleep "$POLL"
done

log "Final runs complete. Building final leaderboard..."
python3 scripts/tuning/fsnet_mlp1024_leaderboard.py \
  --manifest "$final_manifest" \
  --batch-size 256 \
  --out-csv "$final_csv"

{
  echo ""
  echo "### Final Auto-Update ($(date '+%Y-%m-%d %H:%M:%S'))"
  echo "- Status: COMPLETED"
  echo "- Final leaderboard: logs/fsnet_tuning_final/final_leaderboard.csv"
  echo ""
  echo "Final leaderboard snapshot:"
  echo ""
  echo "cfg_id,lr,dropout,dist_weight,val_tol,decay_tol_step,memory_size,max_diff_iter,n_seeds,opt_gap_mean_mean,opt_gap_mean_std"
  awk -F',' 'NR==2{print $1","$2","$3","$4","$5","$6","$7","$8","$9","$12","$13}' "$final_csv"
} >> "$LOG_MD"

log "End-to-end continuation complete. Updated $LOG_MD"
