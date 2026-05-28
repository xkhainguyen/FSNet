#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 02:00:00
#SBATCH -J verify-b1-clean
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

source /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/_common.sh

# Test if B=1 (homogeneous K-candidate batch of size K only) lets vectorisation
# work without the line-search heterogeneity issue.
# Now running on the NEW per-sample-convergence L-BFGS.

echo "######## B=1 (deployment scenario, fresh L-BFGS) ########"
for FLAG in "" "--vectorize_repair" ; do
  LABEL=$([ -z "$FLAG" ] && echo "SEQUENTIAL" || echo "VECTORISED")
  echo "------ B=1 K=20 $LABEL ------"
  /usr/bin/time -f "wall=%e s" \
    python eval.py --run_dir "$FSNET_SINGLE" \
      --inference_perturb_k 20 --inference_perturb_eps 0.1 \
      --ensemble_agg best_merit --test_batch_sizes 1 $FLAG 2>&1 | tail -25
done

echo "=== done $(date) ==="
