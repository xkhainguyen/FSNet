#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 02:00:00
#SBATCH -J eval-mhe
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

source /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/_common.sh

# Eval all MHE checkpoints trained tonight.
for MHE_DIR in $(ls -d $P_NSC/202605*_mhe*_seed* 2>/dev/null) ; do
  if [ ! -f "$MHE_DIR/model.pt" ] ; then
    echo "SKIP $MHE_DIR (no model.pt)" ; continue
  fi
  if [[ "$MHE_DIR" == *_e5_* ]]; then
    echo "SKIP $MHE_DIR (smoke run)" ; continue
  fi
  RUN=$(basename $MHE_DIR)
  echo "######## $RUN ########"
  for POST in pre post ; do
    for AGG in mean median best_obj best_merit ; do
      echo "------ MHE $RUN post=$POST agg=$AGG ------"
      python eval.py --run_dir "$MHE_DIR" --ensemble_post "$POST" --ensemble_agg "$AGG" --test_batch_sizes 256 2>&1 | tail -20
    done
  done
done

echo "=== eval-mhe done $(date) ==="
