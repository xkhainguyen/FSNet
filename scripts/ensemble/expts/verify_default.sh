#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 00:15:00
#SBATCH -J verify-default
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

source /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/_common.sh

# Default behaviour (no flag) should match prior K=20 ε=0.1 → Merit ~23.26
echo "------ K=20 ε=0.1 DEFAULT (no vectorize) ------"
/usr/bin/time -f "wall=%e s" \
  python eval.py --run_dir "$FSNET_SINGLE" \
    --inference_perturb_k 20 --inference_perturb_eps 0.1 \
    --ensemble_agg best_merit --test_batch_sizes 256 2>&1 | tail -20

echo "------ K=20 ε=0.1 OPT-IN VECTORIZE (slower-quality, faster-time) ------"
/usr/bin/time -f "wall=%e s" \
  python eval.py --run_dir "$FSNET_SINGLE" \
    --inference_perturb_k 20 --inference_perturb_eps 0.1 \
    --ensemble_agg best_merit --test_batch_sizes 256 \
    --vectorize_repair 2>&1 | tail -20

echo "=== done $(date) ==="
