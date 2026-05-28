#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 00:30:00
#SBATCH -J verify-lbfgs
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

source /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/_common.sh

echo "######## Sanity baselines (existing eval paths — should be unchanged) ########"
echo "------ Single FSNet hdim=2048 (no perturb) — expect ~85.05 ------"
python eval.py --run_dir "$FSNET_SINGLE" --test_batch_sizes 256 2>&1 | tail -12

echo "------ VE M=5 FSNet post+best_merit — expect ~30.27 ------"
python eval.py --run_dir "$FSNET_ENS5_VAN" --ensemble_post post --ensemble_agg best_merit \
   --test_batch_sizes 256 2>&1 | tail -12

echo "######## Perturbation paths ########"
echo "------ K=20 ε=0.1 SEQUENTIAL (default) — expect ~23.5 ------"
/usr/bin/time -f "wall=%e s" \
  python eval.py --run_dir "$FSNET_SINGLE" \
    --inference_perturb_k 20 --inference_perturb_eps 0.1 \
    --ensemble_agg best_merit --test_batch_sizes 256 2>&1 | tail -12

echo "------ K=20 ε=0.1 VECTORIZED (was 33, target ~23-25 now) ------"
/usr/bin/time -f "wall=%e s" \
  python eval.py --run_dir "$FSNET_SINGLE" \
    --inference_perturb_k 20 --inference_perturb_eps 0.1 \
    --ensemble_agg best_merit --test_batch_sizes 256 \
    --vectorize_repair 2>&1 | tail -12

echo "------ K=50 ε=0.1 VECTORIZED — expect close to sequential ~22 ------"
/usr/bin/time -f "wall=%e s" \
  python eval.py --run_dir "$FSNET_SINGLE" \
    --inference_perturb_k 50 --inference_perturb_eps 0.1 \
    --ensemble_agg best_merit --test_batch_sizes 256 \
    --vectorize_repair 2>&1 | tail -12

echo "=== verify-lbfgs done $(date) ==="
