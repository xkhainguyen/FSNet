#!/bin/bash
#SBATCH -t 08:00:00
#SBATCH -p mit_normal_gpu,mit_preemptable
#SBATCH --gres=gpu:l40s:1
#SBATCH -J fsnet-m1024-w1
#SBATCH --array=0-31
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

set -euo pipefail

source ~/.bashrc
conda activate ml4opt

cd /home/khain/orcd/scratch/FSNet

# Seeds are fixed by requirement.
SEEDS=(0 1 2 3)

# Medium-budget Wave 1: 8 configs x 4 seeds = 32 tasks.
# Format: cfg_id lr dropout dist_weight
CANDIDATES=(
  "w1_c00 3e-5 0.05 3.0"
  "w1_c01 3e-5 0.10 5.0"
  "w1_c02 5e-5 0.05 5.0"
  "w1_c03 5e-5 0.15 7.0"
  "w1_c04 7e-5 0.10 3.0"
  "w1_c05 7e-5 0.15 5.0"
  "w1_c06 1e-4 0.05 7.0"
  "w1_c07 1e-4 0.10 5.0"
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

read -r cfg_id lr dropout dist_weight <<<"${CANDIDATES[$cfg_idx]}"

mkdir -p logs/fsnet_tuning_wave1
run_log="logs/fsnet_tuning_wave1/${cfg_id}_seed${seed}_job${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
manifest="logs/fsnet_tuning_wave1/wave1_manifest.tsv"

echo "==============================================" | tee "$run_log"
echo "Wave1 task start: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$run_log"
echo "Job: $SLURM_JOB_ID ArrayTask: $SLURM_ARRAY_TASK_ID Node: $SLURM_NODELIST" | tee -a "$run_log"
echo "Config: $cfg_id lr=$lr dropout=$dropout dist_weight=$dist_weight seed=$seed" | tee -a "$run_log"
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
  --lr "$lr" \
  --dropout "$dropout" \
  --dist_weight "$dist_weight" \
  --wandb \
  --wandb_tags fsnet-mlp1024 wave1 "$cfg_id" \
  2>&1 | tee -a "$run_log"

save_dir=$(grep -oE 'save_dir: .*' "$run_log" | tail -n1 | sed 's/save_dir: //')

if [[ -n "${save_dir:-}" ]]; then
  if [[ ! -f "$manifest" ]]; then
    echo -e "cfg_id\tseed\tlr\tdropout\tdist_weight\tsave_dir\trun_log" > "$manifest"
  fi
  echo -e "${cfg_id}\t${seed}\t${lr}\t${dropout}\t${dist_weight}\t${save_dir}\t${run_log}" >> "$manifest"
  echo "Recorded manifest entry for $cfg_id seed=$seed"
else
  echo "WARNING: could not parse save_dir from log: $run_log"
fi

echo "Wave1 task end: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$run_log"
