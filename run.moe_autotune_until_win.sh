#!/bin/bash
#SBATCH -t 06:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH -J moe-autotune
#SBATCH -o logs/%x-%A.out
#SBATCH -e logs/%x-%A.err

set -euo pipefail

source ~/.bashrc
conda activate ml4opt

cd ~/orcd/scratch/FSNet

RESULT_ROOT="results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000"
EPOCHS=200
SEED=2025

COMMON_ARGS=(
  --method FSNet
  --prob_type nonsmooth_nonconvex
  --prob_name socp
  --seed ${SEED}
  --num_epochs ${EPOCHS}
  --lr 1e-4
  --train_size 7000
  --wandb 
)

echo "=============================================="
echo " Job started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Job ID: $SLURM_JOB_ID"
echo " Node: $SLURM_NODELIST"
echo "=============================================="

latest_dir() {
  ls -td "${RESULT_ROOT}"/* 2>/dev/null | head -n1 || true
}

run_case() {
  local label="$1"
  shift
  echo "[CASE] ${label}"
  local before after
  before="$(latest_dir)"
  python main.py "${COMMON_ARGS[@]}" "$@"
  after="$(latest_dir)"
  if [[ -z "$after" || "$after" == "$before" ]]; then
    echo "[ERROR] Could not locate new run dir for ${label}" >&2
    return 1
  fi
  echo "[RUN_DIR] ${label} ${after}"
}

extract_metric() {
  local run_dir="$1"
  local key="$2"
  # Use grep+awk to avoid yaml.safe_load failing on torch-specific YAML tags
  # The file has the key at 4-space indentation; take the first occurrence (smallest batch size)
  grep "    ${key}:" "${run_dir}/test_summary.yaml" | head -n1 | awk '{print $2}'
}

# 1) Baseline MLP — skip if RESUME_FROM_DIR is provided
RESUME_FROM_DIR ="results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260313-020128_FSNet_seed2025_e300_lr1e-04_n7000"
if [[ -n "${RESUME_FROM_DIR:-}" ]]; then
  MLP_DIR="${RESUME_FROM_DIR}"
  echo "[CASE] MLP_BASELINE (SKIPPED - using ${MLP_DIR})"
  echo "[RUN_DIR] MLP_BASELINE ${MLP_DIR}"
else
  run_case "MLP_BASELINE" --network MLP
  MLP_DIR=$(grep -oP '^\[RUN_DIR\] MLP_BASELINE \K.*' <(tail -n 50 logs/moe-autotune-${SLURM_JOB_ID}.out) | tail -n1)
  if [[ -z "${MLP_DIR}" ]]; then
    MLP_DIR="$(latest_dir)"
  fi
fi
BASE_OPTGAP=$(extract_metric "${MLP_DIR}" "opt_gap_mean")
BASE_EQ=$(extract_metric "${MLP_DIR}" "eq_violation_l1_mean")
BASE_INEQ=$(extract_metric "${MLP_DIR}" "ineq_violation_l1_mean")

echo "[BASELINE] dir=${MLP_DIR} opt_gap_mean=${BASE_OPTGAP} eq_l1=${BASE_EQ} ineq_l1=${BASE_INEQ}"

# Candidate rounds: each round tries multiple MoE settings.
# Success criterion: opt_gap_mean < baseline_opt_gap_mean.
declare -a CANDIDATES=(
  "moe_top2_nowarm:--network MoE --moe_num_experts 4 --moe_top_k 2 --moe_aux_loss_weight 0.01 --moe_warmup_epochs 0 --moe_gate_temperature 1.0 --moe_gate_noise_std 0.0"
  "moe_top2_warm:--network MoE --moe_num_experts 4 --moe_top_k 2 --moe_aux_loss_weight 0.005 --moe_warmup_epochs 30 --moe_start_temp 2.0 --moe_final_temp 1.0 --moe_temp_decay_epochs 150 --moe_gate_noise_std 0.05 --moe_gate_noise_final 0.0"
  "moe_dense4:--network MoE --moe_num_experts 4 --moe_top_k 0 --moe_aux_loss_weight 0.002 --moe_warmup_epochs 0 --moe_gate_temperature 1.0 --moe_gate_noise_std 0.0"
  "moe_top1_4:--network MoE --moe_num_experts 4 --moe_top_k 1 --moe_aux_loss_weight 0.002 --moe_warmup_epochs 40 --moe_start_temp 2.5 --moe_final_temp 1.0 --moe_temp_decay_epochs 200 --moe_gate_noise_std 0.08 --moe_gate_noise_final 0.0"
  "moe_top2_8:--network MoE --moe_num_experts 8 --moe_top_k 2 --moe_aux_loss_weight 0.001 --moe_warmup_epochs 50 --moe_start_temp 3.0 --moe_final_temp 1.0 --moe_temp_decay_epochs 220 --moe_gate_noise_std 0.1 --moe_gate_noise_final 0.0"
  "moe_top4_8:--network MoE --moe_num_experts 8 --moe_top_k 4 --moe_aux_loss_weight 0.001 --moe_warmup_epochs 40 --moe_start_temp 2.0 --moe_final_temp 1.0 --moe_temp_decay_epochs 200 --moe_gate_noise_std 0.05 --moe_gate_noise_final 0.0"
)

WIN=0
BEST_LABEL=""
BEST_DIR=""
BEST_OPTGAP="999999"

for item in "${CANDIDATES[@]}"; do
  label="${item%%:*}"
  args="${item#*:}"

  run_case "${label}" ${args}
  DIR=$(grep -oP "^\[RUN_DIR\] ${label} \\K.*" <(tail -n 200 logs/moe-autotune-${SLURM_JOB_ID}.out) | tail -n1)
  if [[ -z "${DIR}" ]]; then
    DIR="$(latest_dir)"
  fi

  OPTGAP=$(extract_metric "${DIR}" "opt_gap_mean")
  EQ=$(extract_metric "${DIR}" "eq_violation_l1_mean")
  INEQ=$(extract_metric "${DIR}" "ineq_violation_l1_mean")

  echo "[RESULT] ${label} dir=${DIR} opt_gap_mean=${OPTGAP} eq_l1=${EQ} ineq_l1=${INEQ}"

  python - <<'PY' "$OPTGAP" "$BEST_OPTGAP"
import sys
cur=float(sys.argv[1]); best=float(sys.argv[2])
print("1" if cur < best else "0")
PY
  is_better=$(python - <<'PY' "$OPTGAP" "$BEST_OPTGAP"
import sys
cur=float(sys.argv[1]); best=float(sys.argv[2])
print(1 if cur < best else 0)
PY
)

  if [[ "$is_better" == "1" ]]; then
    BEST_OPTGAP="$OPTGAP"
    BEST_LABEL="$label"
    BEST_DIR="$DIR"
  fi

  win=$(python - <<'PY' "$OPTGAP" "$BASE_OPTGAP"
import sys
cur=float(sys.argv[1]); base=float(sys.argv[2])
print(1 if cur < base else 0)
PY
)
  if [[ "$win" == "1" ]]; then
    WIN=1
    echo "[WIN] ${label} beats baseline on opt_gap_mean: ${OPTGAP} < ${BASE_OPTGAP}"
    break
  fi

done

echo "=============================================="
if [[ "$WIN" == "1" ]]; then
  echo "AUTOTUNE SUCCESS: ${BEST_LABEL} at ${BEST_DIR}"
  echo "Best opt_gap_mean=${BEST_OPTGAP} vs baseline=${BASE_OPTGAP}"
else
  echo "AUTOTUNE NO-WIN IN THIS ROUND"
  echo "Best candidate: ${BEST_LABEL} at ${BEST_DIR}"
  echo "Best opt_gap_mean=${BEST_OPTGAP} vs baseline=${BASE_OPTGAP}"
fi
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
