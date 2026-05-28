#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 00:30:00
#SBATCH -J verify-vec
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

source /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/_common.sh

# Correctness + speed check: same K perturbation eval, vectorised vs sequential.
# Random perturbations differ run-to-run; expect Merit within ~1 unit of each other.

for K in 20 50 ; do
  for FLAG in "" "--no_vectorize_repair" ; do
    LABEL=$([ -z "$FLAG" ] && echo "VECTORISED" || echo "SEQUENTIAL")
    echo "------ K=$K $LABEL ------"
    /usr/bin/time -f "wall=%e s, peak_mem=%M KB" \
      python eval.py --run_dir "$FSNET_SINGLE" \
        --inference_perturb_k $K --inference_perturb_eps 0.1 \
        --ensemble_agg best_merit --test_batch_sizes 256 $FLAG 2>&1 | tail -25
  done
done

# Also smoke the post-ensemble path
echo "######## VE M=5 FSNet post + best_merit (both modes) ########"
for FLAG in "" "--no_vectorize_repair" ; do
  LABEL=$([ -z "$FLAG" ] && echo "VECTORISED" || echo "SEQUENTIAL")
  echo "------ ens5 $LABEL ------"
  /usr/bin/time -f "wall=%e s, peak_mem=%M KB" \
    python eval.py --run_dir "$FSNET_ENS5_VAN" --ensemble_post post --ensemble_agg best_merit \
      --test_batch_sizes 256 $FLAG 2>&1 | tail -20
done

echo "=== verify-vectorize done $(date) ==="
