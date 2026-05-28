#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 00:30:00
#SBATCH -J verify-b1
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

source /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/_common.sh

# Hypothesis: at B=1, the K candidates are all small perturbations of ONE NN
# output → they converge with similar L-BFGS trajectories → global line search
# is fine → vectorisation works.

echo "######## B=256 baseline ########"
for FLAG in "" "--vectorize_repair" ; do
  LABEL=$([ -z "$FLAG" ] && echo "SEQUENTIAL (default)" || echo "VECTORISED   ")
  echo "------ B=256 K=20 $LABEL ------"
  /usr/bin/time -f "wall=%e s" \
    python eval.py --run_dir "$FSNET_SINGLE" \
      --inference_perturb_k 20 --inference_perturb_eps 0.1 \
      --ensemble_agg best_merit --test_batch_sizes 256 $FLAG 2>&1 | tail -15
done

echo "######## B=1 (deployment scenario) ########"
for FLAG in "" "--vectorize_repair" ; do
  LABEL=$([ -z "$FLAG" ] && echo "SEQUENTIAL (default)" || echo "VECTORISED   ")
  echo "------ B=1 K=20 $LABEL ------"
  /usr/bin/time -f "wall=%e s" \
    python eval.py --run_dir "$FSNET_SINGLE" \
      --inference_perturb_k 20 --inference_perturb_eps 0.1 \
      --ensemble_agg best_merit --test_batch_sizes 1 $FLAG 2>&1 | tail -15
done

# Also try a moderate B=16 — partial homogeneity
echo "######## B=16 ########"
for FLAG in "" "--vectorize_repair" ; do
  LABEL=$([ -z "$FLAG" ] && echo "SEQUENTIAL" || echo "VECTORISED")
  echo "------ B=16 K=20 $LABEL ------"
  /usr/bin/time -f "wall=%e s" \
    python eval.py --run_dir "$FSNET_SINGLE" \
      --inference_perturb_k 20 --inference_perturb_eps 0.1 \
      --ensemble_agg best_merit --test_batch_sizes 16 $FLAG 2>&1 | tail -15
done

echo "=== verify-b1 done $(date) ==="
