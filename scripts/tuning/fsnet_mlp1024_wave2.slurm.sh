#!/bin/bash
#SBATCH -t 08:00:00
#SBATCH -p mit_normal_gpu,mit_preemptable
#SBATCH --gres=gpu:l40s:1
#SBATCH -J fsnet-m1024-w2
#SBATCH --array=0-35
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

# ── Wave-2: fine-tune feasibility-solver knobs ─────────────────────────────
#
# Prerequisites:
#   1.  Wave-1 is complete and wave1_leaderboard.csv exists.
#   2.  Set W2_LR / W2_DROPOUT / W2_DIST_WEIGHT to the best config values
#       found in Wave-1 before submitting:
#
#         W2_LR=3e-5 W2_DROPOUT=0.05 W2_DIST_WEIGHT=3.0 \
#           sbatch scripts/tuning/fsnet_mlp1024_wave2.slurm.sh
#
# If the env vars are unset the defaults below are used (same as default.yaml).
# ── defaults ──────────────────────────────────────────────────────────────────
: "${W2_LR:=5e-5}"
: "${W2_DROPOUT:=0.10}"
: "${W2_DIST_WEIGHT:=5.0}"
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

source ~/.bashrc
conda activate ml4opt

cd /home/khain/orcd/scratch/FSNet

SEEDS=(0 1 2 3)

# 9 configs × 4 seeds = 36 tasks
# Format: cfg_id  val_tol  decay_tol_step  memory_size  max_diff_iter
CANDIDATES=(
  "w2_baseline 1e-7 100 30 30"
  "w2_vt_loose 1e-6 100 30 30"
  "w2_vt_tight 1e-8 100 30 30"
  "w2_dt_fast  1e-7  50 30 30"
  "w2_dt_slow  1e-7 150 30 30"
  "w2_mem_sm   1e-7 100 20 30"
  "w2_mem_lg   1e-7 100 40 30"
  "w2_diff_sm  1e-7 100 30 20"
  "w2_diff_lg  1e-7 100 30 40"
)

num_cfg=${#CANDIDATES[@]}
num_seed=${#SEEDS[@]}
expected=$((num_cfg * num_seed))

if [[ "$SLURM_ARRAY_TASK_COUNT" -ne "$expected" ]]; then
  echo "ERROR: array count ($SLURM_ARRAY_TASK_COUNT) != expected ($expected)"
  exit 2
fi

task_id=${SLURM_ARRAY_TASK_ID}
cfg_idx=$((task_id / num_seed))
seed_idx=$((task_id % num_seed))
seed=${SEEDS[$seed_idx]}

read -r cfg_id val_tol decay_tol_step memory_size max_diff_iter <<<"${CANDIDATES[$cfg_idx]}"

mkdir -p logs/fsnet_tuning_wave2
run_log="logs/fsnet_tuning_wave2/${cfg_id}_seed${seed}_job${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
manifest="logs/fsnet_tuning_wave2/wave2_manifest.tsv"

echo "==============================================" | tee "$run_log"
echo "Wave2 task start: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$run_log"
echo "Job: $SLURM_JOB_ID ArrayTask: $SLURM_ARRAY_TASK_ID Node: $SLURM_NODELIST" | tee -a "$run_log"
echo "Base config: lr=$W2_LR dropout=$W2_DROPOUT dist_weight=$W2_DIST_WEIGHT" | tee -a "$run_log"
echo "Knobs: $cfg_id val_tol=$val_tol decay_tol_step=$decay_tol_step memory_size=$memory_size max_diff_iter=$max_diff_iter seed=$seed" | tee -a "$run_log"
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
  --lr "$W2_LR" \
  --dropout "$W2_DROPOUT" \
  --dist_weight "$W2_DIST_WEIGHT" \
  --val_tol "$val_tol" \
  --decay_tol_step "$decay_tol_step" \
  --memory_size "$memory_size" \
  --max_diff_iter "$max_diff_iter" \
  --wandb \
  --wandb_tags fsnet-mlp1024 wave2 "$cfg_id" \
  2>&1 | tee -a "$run_log"

save_dir=$(grep -oE 'save_dir: .*' "$run_log" | tail -n1 | sed 's/save_dir: //')

if [[ -n "${save_dir:-}" ]]; then
  if [[ ! -f "$manifest" ]]; then
    echo -e "cfg_id\tseed\tlr\tdropout\tdist_weight\tval_tol\tdecay_tol_step\tmemory_size\tmax_diff_iter\tsave_dir\trun_log" > "$manifest"
  fi
  echo -e "${cfg_id}\t${seed}\t${W2_LR}\t${W2_DROPOUT}\t${W2_DIST_WEIGHT}\t${val_tol}\t${decay_tol_step}\t${memory_size}\t${max_diff_iter}\t${save_dir}\t${run_log}" >> "$manifest"
  echo "Recorded manifest entry for $cfg_id seed=$seed"
else
  echo "WARNING: could not parse save_dir from log: $run_log"
fi

echo "Wave2 task end: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$run_log"
