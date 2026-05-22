#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 00:30:00
#SBATCH -J perturb-iter
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

source /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/_common.sh

# Compose: perturbation × reduced repair iterations.
# Hypothesis: K=20 × max_iter=20 may give same quality at 0.4× inference cost
# vs K=20 × max_iter=50.

for K in 10 20 ; do
  for IT in 10 20 50 ; do
    echo "------ FSNET single (hdim=2048) K=$K eps=0.1 max_iter=$IT ------"
    python eval.py --run_dir "$FSNET_SINGLE" \
       --inference_perturb_k $K --inference_perturb_eps 0.1 \
       --ensemble_agg best_merit --repair_max_iter $IT \
       --test_batch_sizes 256 2>&1 | tail -20
  done
done

echo "=== perturb-iter-combo done $(date) ==="
