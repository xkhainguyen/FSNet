#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 00:10:00
#SBATCH -J eval-fresh
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

source /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/_common.sh
NEW="results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260528-195753_FSNet_e300_lr1e-04_n7000_hdim1024_seed0"

echo "######## Freshly trained baseline (per_sample=0) ########"
echo "------ Eval per_sample=0 (default, legacy) ------"
python eval.py --run_dir "$NEW" --test_batch_sizes 256 2>&1 | tail -10

echo "------ Eval per_sample=1 (new) ------"
python eval.py --run_dir "$NEW" --test_batch_sizes 256 --per_sample_lbfgs 1 2>&1 | tail -10

echo "------ Eval per_sample=1 + K=20 ε=0.1 perturb ------"
python eval.py --run_dir "$NEW" --test_batch_sizes 256 --per_sample_lbfgs 1 \
  --inference_perturb_k 20 --inference_perturb_eps 0.1 --ensemble_agg best_merit 2>&1 | tail -10
echo "=== done $(date) ==="
