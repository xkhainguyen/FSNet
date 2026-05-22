#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 03:00:00
#SBATCH -J audit-baselines
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

source /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/_common.sh

run_eval() {
  local NAME=$1 DIR=$2 POST=$3 AGG=$4 EXTRA="$5"
  echo "------ AUDIT $NAME post=$POST agg=$AGG ${EXTRA} ------"
  python eval.py --run_dir "$DIR" --ensemble_post "$POST" --ensemble_agg "$AGG" --test_batch_sizes 256 $EXTRA 2>&1 | tail -25
}

# === Single-model baselines (no ensemble args needed) ===
echo "######## SINGLE-MODEL BASELINES ########"
python eval.py --run_dir "$PEN_SINGLE"   --test_batch_sizes 256 2>&1 | tail -20
python eval.py --run_dir "$FSNET_SINGLE" --test_batch_sizes 256 2>&1 | tail -20

# === ENSEMBLE GRIDS ===
for cfg in "PEN_ENS5_VAN $PEN_ENS5_VAN" "FSNET_ENS5_VAN $FSNET_ENS5_VAN" "FSNET_ENS5_FGE $FSNET_ENS5_FGE" "PEN_ENS20_VAN $PEN_ENS20_VAN" ; do
  NAME=$(echo $cfg | cut -d' ' -f1)
  DIR=$(echo $cfg | cut -d' ' -f2)
  echo "######## $NAME ########"
  for POST in pre post ; do
    for AGG in mean median best_obj best_merit ; do
      run_eval "$NAME" "$DIR" "$POST" "$AGG"
    done
  done
done

echo "=== audit-baselines done $(date) ==="
