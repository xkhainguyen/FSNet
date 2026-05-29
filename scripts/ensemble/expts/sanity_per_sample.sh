#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 00:30:00
#SBATCH -J sanity-ps
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

source /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/_common.sh

echo "######## Under per_sample=1: does perturbation still help anywhere? ########"
echo
echo "------ Single FSNet hdim=2048: K sweep at ε=0.1 per_sample=1 ------"
for K in 1 5 20 100 ; do
  python eval.py --run_dir "$FSNET_SINGLE" --per_sample_lbfgs 1 \
    $([ "$K" -gt 1 ] && echo "--inference_perturb_k $K --inference_perturb_eps 0.1 --ensemble_agg best_merit") \
    --test_batch_sizes 256 2>&1 | grep -E "Merit:|^Eq Vio l1:|^Ineq Vio l1:" | head -3
  echo "  ↑ K=$K"
done

echo
echo "------ Single FSNet hdim=2048: ε sweep at K=20 per_sample=1 ------"
for EPS in 0.01 0.05 0.1 0.5 1.0 ; do
  python eval.py --run_dir "$FSNET_SINGLE" --per_sample_lbfgs 1 \
    --inference_perturb_k 20 --inference_perturb_eps $EPS \
    --ensemble_agg best_merit --test_batch_sizes 256 2>&1 | grep -E "Merit:" | head -1
  echo "  ↑ ε=$EPS"
done

echo
echo "------ VE M=5 post+best_merit per_sample=1 vs 0 ------"
for PS in 0 1 ; do
  python eval.py --run_dir "$FSNET_ENS5_VAN" --ensemble_post post --ensemble_agg best_merit \
    --per_sample_lbfgs $PS --test_batch_sizes 256 2>&1 | grep -E "Merit:" | head -1
  echo "  ↑ per_sample=$PS"
done

echo
echo "------ FGE ens5 per_sample=1 vs 0 ------"
for PS in 0 1 ; do
  python eval.py --run_dir "$FSNET_ENS5_FGE" --ensemble_post post --ensemble_agg best_merit \
    --per_sample_lbfgs $PS --test_batch_sizes 256 2>&1 | grep -E "Merit:" | head -1
  echo "  ↑ per_sample=$PS"
done

echo "=== done $(date) ==="
