#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 00:20:00
#SBATCH -J verify-flag-def
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

source /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/_common.sh
echo "######## per_sample default off — should match LEGACY numbers ########"
echo "------ Single FSNet hdim=2048 (no perturb) — expect ~85.05 (LEGACY) ------"
python eval.py --run_dir "$FSNET_SINGLE" --test_batch_sizes 256 2>&1 | tail -12

echo "------ K=20 ε=0.1 SEQ — expect ~23.5 (LEGACY) ------"
python eval.py --run_dir "$FSNET_SINGLE" \
  --inference_perturb_k 20 --inference_perturb_eps 0.1 \
  --ensemble_agg best_merit --test_batch_sizes 256 2>&1 | tail -12

echo "######## per_sample on — should match new numbers ########"
echo "------ Single FSNet (no perturb) per_sample=1 — expect ~25 ------"
python eval.py --run_dir "$FSNET_SINGLE" --test_batch_sizes 256 --per_sample_lbfgs 1 2>&1 | tail -12

echo "------ K=20 ε=0.1 per_sample=1 — expect ~18 ------"
python eval.py --run_dir "$FSNET_SINGLE" \
  --inference_perturb_k 20 --inference_perturb_eps 0.1 \
  --ensemble_agg best_merit --test_batch_sizes 256 --per_sample_lbfgs 1 2>&1 | tail -12

echo "=== done $(date) ==="
