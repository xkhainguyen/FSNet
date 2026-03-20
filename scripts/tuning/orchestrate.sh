#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH -p mit_normal_gpu,mit_preemptable
#SBATCH --mem=4G
#SBATCH -c 1
#SBATCH -J fsnet-orch
#SBATCH -o logs/orchestrate-%j.out
#SBATCH -e logs/orchestrate-%j.err
#
# End-to-end tuning orchestrator for FSNet MLP-1024.
#
# Lifecycle:
#   [Wave-1 already submitted]
#     → Wait for all 32 manifest rows
#     → Run leaderboard, pick top-1
#     → Submit Wave-2 with top-1 base config
#     → Wait for all 36 manifest rows
#     → Run leaderboard, pick best knob config
#     → Submit final confirmation (4 seeds)
#     → Wait for final runs
#     → Update TUNING_LOG.md with results
#
# Usage:
#   sbatch scripts/tuning/orchestrate.sh         # preferred
#   bash   scripts/tuning/orchestrate.sh         # interactive (screen/tmux)

set -euo pipefail

source ~/.bashrc
conda activate ml4opt

ROOT="/home/khain/orcd/scratch/FSNet"
cd "$ROOT"

POLL_INTERVAL=120   # seconds between checks
LOG_MD="scripts/tuning/TUNING_LOG.md"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── helpers ───────────────────────────────────────────────────────────────────

manifest_count() {
  local f="$1"
  [[ -f "$f" ]] && awk 'NR>1{c++}END{print c+0}' "$f" || echo 0
}

# Wait until a Slurm job array is fully gone from the queue (RUNNING or PENDING).
wait_for_job() {
  local job_id="$1"
  local desc="$2"
  log "Waiting for job $job_id ($desc) to finish ..."
  while squeue -j "$job_id" --noheader 2>/dev/null | grep -q "$job_id"; do
    sleep "$POLL_INTERVAL"
    local remaining
    remaining=$(squeue -j "$job_id" --noheader 2>/dev/null | wc -l)
    log "  job $job_id: $remaining tasks still in queue"
  done
  log "Job $job_id done."
}

# Wait until a manifest has at least N complete rows.
wait_for_manifest() {
  local manifest="$1"
  local needed="$2"
  local desc="$3"
  log "Waiting for $needed rows in $manifest ($desc) ..."
  while true; do
    local n
    n=$(manifest_count "$manifest")
    log "  manifest rows: $n / $needed"
    [[ "$n" -ge "$needed" ]] && break
    sleep "$POLL_INTERVAL"
  done
  log "$desc manifest complete ($needed rows)."
}

# Run the leaderboard script and return the top-1 row as space-separated fields.
run_leaderboard() {
  local manifest="$1"
  local csv_out="$2"
  local batch_size="${3:-256}"
  log "Running leaderboard: $manifest → $csv_out"
  python3 scripts/tuning/fsnet_mlp1024_leaderboard.py \
    --manifest "$manifest" \
    --batch-size "$batch_size" \
    --out-csv "$csv_out"
  log "Leaderboard written: $csv_out"
}

# Parse the top-1 row of a leaderboard CSV.
# Prints: cfg_id lr dropout dist_weight val_tol decay_tol_step memory_size max_diff_iter
top1_config() {
  local csv="$1"
  # Skip header, print second line (rank 1).
  awk -F',' 'NR==2{print $1, $2, $3, $4, $5, $6, $7, $8}' "$csv"
}

# Patch a key=value line in TUNING_LOG.md (used for status updates).
update_log_status() {
  local section="$1"  # grep anchor text
  local new_status="$2"
  sed -i "s|\(${section}.*\)Status\`: .*|\1Status\`: ${new_status}|g" "$LOG_MD" 2>/dev/null || true
}

# ── Main flow ─────────────────────────────────────────────────────────────────

log "Orchestrator starting. PID=$$"
log "ROOT=$ROOT"

# ── Phase 1: Wait for Wave-1 ──────────────────────────────────────────────────
W1_MANIFEST="logs/fsnet_tuning_wave1/wave1_manifest.tsv"
W1_CSV="logs/fsnet_tuning_wave1/wave1_leaderboard.csv"
W1_TOTAL=32

log "=== Phase 1: waiting for Wave-1 ($W1_TOTAL runs) ==="
wait_for_manifest "$W1_MANIFEST" "$W1_TOTAL" "Wave-1"

# ── Phase 2: Build Wave-1 leaderboard ────────────────────────────────────────
log "=== Phase 2: building Wave-1 leaderboard ==="
run_leaderboard "$W1_MANIFEST" "$W1_CSV"

# Extract top-1
read -r w1_cfg w1_lr w1_drop w1_dist w1_vt w1_dt w1_mem w1_diff <<< "$(top1_config "$W1_CSV")"
log "Wave-1 winner: cfg=$w1_cfg lr=$w1_lr dropout=$w1_drop dist_weight=$w1_dist"
log "  val_tol=$w1_vt decay_tol_step=$w1_dt memory_size=$w1_mem max_diff_iter=$w1_diff"

# Update TUNING_LOG.md with top-1 info
{
  echo ""
  echo "<!-- Wave-1 results populated by orchestrator $(date '+%Y-%m-%d %H:%M:%S') -->"
  echo "**Wave-1 Winner**: \`$w1_cfg\`  lr=$w1_lr  dropout=$w1_drop  dist_weight=$w1_dist"
  echo "See full leaderboard: \`$W1_CSV\`"
} >> "$LOG_MD"

# ── Phase 3: Submit Wave-2 ────────────────────────────────────────────────────
log "=== Phase 3: submitting Wave-2 ==="
W2_JOB_SUBMIT=$(W2_LR="$w1_lr" W2_DROPOUT="$w1_drop" W2_DIST_WEIGHT="$w1_dist" \
  sbatch --parsable scripts/tuning/fsnet_mlp1024_wave2.slurm.sh)
W2_JOB_ID=$(echo "$W2_JOB_SUBMIT" | tr -d '[:space:]')
log "Wave-2 submitted: job $W2_JOB_ID"
{
  echo "**Wave-2 Job**: \`$W2_JOB_ID\`  base: lr=$w1_lr dropout=$w1_drop dist_weight=$w1_dist"
} >> "$LOG_MD"

# ── Phase 4: Wait for Wave-2 ─────────────────────────────────────────────────
W2_MANIFEST="logs/fsnet_tuning_wave2/wave2_manifest.tsv"
W2_CSV="logs/fsnet_tuning_wave2/wave2_leaderboard.csv"
W2_TOTAL=36

log "=== Phase 4: waiting for Wave-2 ($W2_TOTAL runs) ==="
# Wait for Wave-2 tasks to fully appear in the queue first (brief grace period)
sleep 30
wait_for_manifest "$W2_MANIFEST" "$W2_TOTAL" "Wave-2"

# ── Phase 5: Build Wave-2 leaderboard ────────────────────────────────────────
log "=== Phase 5: building Wave-2 leaderboard ==="
run_leaderboard "$W2_MANIFEST" "$W2_CSV"

read -r w2_cfg w2_lr w2_drop w2_dist w2_vt w2_dt w2_mem w2_diff <<< "$(top1_config "$W2_CSV")"
log "Wave-2 winner: cfg=$w2_cfg val_tol=$w2_vt decay_tol_step=$w2_dt memory_size=$w2_mem max_diff_iter=$w2_diff"
{
  echo ""
  echo "<!-- Wave-2 results populated by orchestrator $(date '+%Y-%m-%d %H:%M:%S') -->"
  echo "**Wave-2 Winner**: \`$w2_cfg\`  val_tol=$w2_vt  decay_tol_step=$w2_dt  memory_size=$w2_mem  max_diff_iter=$w2_diff"
  echo "See full leaderboard: \`$W2_CSV\`"
} >> "$LOG_MD"

# ── Phase 6: Submit final confirmation runs ───────────────────────────────────
log "=== Phase 6: submitting final confirmation runs (4 seeds) ==="
FINAL_JOB_SUBMIT=$(
  BEST_LR="$w2_lr" BEST_DROPOUT="$w2_drop" BEST_DIST_WEIGHT="$w2_dist" \
  BEST_VAL_TOL="$w2_vt" BEST_DECAY_TOL_STEP="$w2_dt" \
  BEST_MEMORY_SIZE="$w2_mem" BEST_MAX_DIFF_ITER="$w2_diff" \
  BEST_CFG_ID="${w2_cfg}_final" \
  sbatch --parsable scripts/tuning/run_best_config.sh
)
FINAL_JOB_ID=$(echo "$FINAL_JOB_SUBMIT" | tr -d '[:space:]')
log "Final runs submitted: job $FINAL_JOB_ID"

# ── Phase 7: Wait for final runs ─────────────────────────────────────────────
FINAL_MANIFEST="logs/fsnet_tuning_final/final_manifest.tsv"
FINAL_TOTAL=4

log "=== Phase 7: waiting for final confirmation runs ==="
sleep 30
wait_for_manifest "$FINAL_MANIFEST" "$FINAL_TOTAL" "Final"

# ── Phase 8: Aggregate final results ─────────────────────────────────────────
log "=== Phase 8: aggregating final results ==="
run_leaderboard "$FINAL_MANIFEST" "logs/fsnet_tuning_final/final_leaderboard.csv"

# Print summary
log "=== ALL DONE ==="
cat "logs/fsnet_tuning_final/final_leaderboard.csv" | column -t -s','

{
  echo ""
  echo "## Final Summary (auto-generated $(date '+%Y-%m-%d %H:%M:%S'))"
  echo ""
  echo "**Best config**: \`${w2_cfg}_final\`"
  echo "  lr=$w2_lr  dropout=$w2_drop  dist_weight=$w2_dist"
  echo "  val_tol=$w2_vt  decay_tol_step=$w2_dt  memory_size=$w2_mem  max_diff_iter=$w2_diff"
  echo ""
  echo "\`\`\`"
  cat "logs/fsnet_tuning_final/final_leaderboard.csv" | column -t -s','
  echo "\`\`\`"
} >> "$LOG_MD"

log "TUNING_LOG.md updated: $LOG_MD"
log "Orchestration complete."
