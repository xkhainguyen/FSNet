#!/bin/bash
# check_progress.sh — quick status snapshot for the FSNet MLP1024 tuning campaign
# Usage: bash scripts/tuning/check_progress.sh [wave1|wave2|all]

WAVE=${1:-all}
ROOT="/home/khain/orcd/scratch/FSNet"

# ── helpers ───────────────────────────────────────────────────────────────────

section() { echo; echo "=== $* ==="; }

count_done() {
  local manifest="$1"
  [[ -f "$manifest" ]] && awk 'NR>1{c++}END{print c+0}' "$manifest" || echo 0
}

# ── Slurm jobs ────────────────────────────────────────────────────────────────
section "Slurm queue (tuning jobs)"
squeue -u "$USER" --noheader -o '%18i %8T %10M %R' 2>/dev/null | grep -E 'fsnet|10572332|wave' || echo "(none)"

# ── Wave-1 ────────────────────────────────────────────────────────────────────
if [[ "$WAVE" == "all" || "$WAVE" == "wave1" ]]; then
  section "Wave-1 progress"

  W1_DIR="$ROOT/logs/fsnet_tuning_wave1"
  W1_MANIFEST="$W1_DIR/wave1_manifest.tsv"
  W1_LEADER="$W1_DIR/wave1_leaderboard.csv"

  DONE=$(count_done "$W1_MANIFEST")
  echo "Completed runs recorded in manifest: $DONE / 32"

  # Show currently running task log tail
  RUNNING_LOG=$(ls -t "$W1_DIR"/*.log 2>/dev/null | head -1)
  if [[ -n "$RUNNING_LOG" ]]; then
    echo ""
    echo "Latest log: $(basename "$RUNNING_LOG")"
    echo "--- last 6 lines ---"
    tail -6 "$RUNNING_LOG"
  else
    echo "(no logs yet)"
  fi

  # Show leaderboard if it exists
  if [[ -f "$W1_LEADER" ]]; then
    section "Wave-1 leaderboard (from CSV)"
    column -t -s',' "$W1_LEADER" | head -12
  else
    echo ""
    echo "Leaderboard not yet generated."
    echo "Run when all 32 manifest rows exist:"
    echo "  python3 $ROOT/scripts/tuning/fsnet_mlp1024_leaderboard.py \\"
    echo "    --manifest $W1_MANIFEST --batch-size 256 \\"
    echo "    --out-csv $W1_LEADER"
  fi
fi

# ── Wave-2 ────────────────────────────────────────────────────────────────────
if [[ "$WAVE" == "all" || "$WAVE" == "wave2" ]]; then
  section "Wave-2 progress"

  W2_DIR="$ROOT/logs/fsnet_tuning_wave2"
  W2_MANIFEST="$W2_DIR/wave2_manifest.tsv"
  W2_LEADER="$W2_DIR/wave2_leaderboard.csv"

  if [[ ! -d "$W2_DIR" ]]; then
    echo "Wave-2 not yet started."
  else
    DONE=$(count_done "$W2_MANIFEST")
    echo "Completed runs recorded in manifest: $DONE / 36"

    RUNNING_LOG=$(ls -t "$W2_DIR"/*.log 2>/dev/null | head -1)
    if [[ -n "$RUNNING_LOG" ]]; then
      echo "Latest log: $(basename "$RUNNING_LOG")"
      echo "--- last 4 lines ---"
      tail -4 "$RUNNING_LOG"
    fi

    if [[ -f "$W2_LEADER" ]]; then
      section "Wave-2 leaderboard (from CSV)"
      column -t -s',' "$W2_LEADER" | head -12
    fi
  fi
fi
