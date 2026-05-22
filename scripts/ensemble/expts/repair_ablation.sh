#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 02:00:00
#SBATCH -J repair-ablation
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

source /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/_common.sh

# Idea 3: how much does the repair layer help?
# - skip_repair=True : raw NN output (pre-repair)
# - repair_max_iter sweep : 0 (=skip), 5, 10, 20, 50 (default)
# Compare across single-model AND ensemble checkpoints.

eval_no_repair() {
  local NAME=$1 DIR=$2
  echo "------ NO-REPAIR $NAME ------"
  python eval.py --run_dir "$DIR" --skip_repair --test_batch_sizes 256 2>&1 | tail -25
}

eval_repair_iter() {
  local NAME=$1 DIR=$2 IT=$3
  echo "------ REPAIR-IT=$IT $NAME ------"
  python eval.py --run_dir "$DIR" --repair_max_iter $IT --test_batch_sizes 256 2>&1 | tail -25
}

for cfg in "PEN_SINGLE $PEN_SINGLE" "FSNET_SINGLE $FSNET_SINGLE" ; do
  NAME=$(echo $cfg | cut -d' ' -f1)
  DIR=$(echo $cfg | cut -d' ' -f2)
  echo "######## $NAME ########"
  eval_no_repair "$NAME" "$DIR"
  for IT in 1 5 10 20 50 ; do
    eval_repair_iter "$NAME" "$DIR" $IT
  done
done

# Also: vanilla ens5 with no repair (best_merit), to see if ensemble alone (no repair) does anything.
echo "######## ENS5_VAN no-repair best_merit ########"
python eval.py --run_dir "$PEN_ENS5_VAN" --skip_repair --ensemble_post post --ensemble_agg best_merit --test_batch_sizes 256 2>&1 | tail -25
python eval.py --run_dir "$FSNET_ENS5_VAN" --skip_repair --ensemble_post post --ensemble_agg best_merit --test_batch_sizes 256 2>&1 | tail -25

echo "=== repair-ablation done $(date) ==="
