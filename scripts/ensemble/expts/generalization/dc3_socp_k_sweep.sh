#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 00:20:00
#SBATCH -J dc3-k-sweep
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

source /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/_common.sh

DC3_SOCP="results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260528-171119_DC3_e1000_lr5e-05_n7000_hdim1024_seed0"

echo "######## DC3 + nonsmooth SOCP: K sweep at ε=0.01 ########"
for K in 5 10 20 50 100 ; do
  echo "------ K=$K eps=0.01 ------"
  python eval.py --run_dir "$DC3_SOCP" \
    --inference_perturb_k $K --inference_perturb_eps 0.01 \
    --ensemble_agg best_merit --test_batch_sizes 256 2>&1 | tail -12
done

# Also try even smaller ε to see if there's a better sweet spot
echo "######## DC3 + nonsmooth SOCP: finer ε sweep at K=100 ########"
for EPS in 0.001 0.005 0.01 0.02 0.03 ; do
  echo "------ K=100 eps=$EPS ------"
  python eval.py --run_dir "$DC3_SOCP" \
    --inference_perturb_k 100 --inference_perturb_eps $EPS \
    --ensemble_agg best_merit --test_batch_sizes 256 2>&1 | tail -12
done

echo "=== done $(date) ==="
