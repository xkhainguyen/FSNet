#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 00:30:00
#SBATCH -J qp-eps-sweep
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

source /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/_common.sh

FSNET_QP="results/nonsmooth_nonconvex/qp/QPProblem-100-50-50-10000/20260528-171119_FSNet_e300_lr1e-04_n7000_hdim1024_seed0"
DC3_QP="results/nonsmooth_nonconvex/qp/QPProblem-100-50-50-10000/20260528-171119_DC3_e1000_lr5e-05_n7000_hdim1024_seed0"

echo "######## FSNet + nonsmooth QP: ε sweep at K=100 ########"
for EPS in 0.005 0.01 0.02 0.05 0.1 0.2 ; do
  echo "------ K=100 eps=$EPS ------"
  python eval.py --run_dir "$FSNET_QP" \
    --inference_perturb_k 100 --inference_perturb_eps $EPS \
    --ensemble_agg best_merit --test_batch_sizes 256 2>&1 | tail -12
done

echo "######## DC3 + nonsmooth QP: ε sweep at K=100 ########"
for EPS in 0.005 0.01 0.02 0.05 0.1 0.2 ; do
  echo "------ K=100 eps=$EPS ------"
  python eval.py --run_dir "$DC3_QP" \
    --inference_perturb_k 100 --inference_perturb_eps $EPS \
    --ensemble_agg best_merit --test_batch_sizes 256 2>&1 | tail -12
done

echo "=== done $(date) ==="
