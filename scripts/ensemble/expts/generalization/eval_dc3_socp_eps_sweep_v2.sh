#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 00:30:00
#SBATCH -J dc3-eps-v2
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

source /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/_common.sh

DC3_SOCP="results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260528-171119_DC3_e1000_lr5e-05_n7000_hdim1024_seed0"
[ ! -f "$DC3_SOCP/model.pt" ] && { echo "ABORT: no checkpoint at $DC3_SOCP" ; exit 1 ; }

echo "######## DC3 + nonsmooth SOCP: ε sweep at K=20 ########"
for EPS in 0.01 0.05 0.1 0.5 1.0 2.0 5.0 10.0 ; do
  echo "------ K=20 eps=$EPS ------"
  python eval.py --run_dir "$DC3_SOCP" \
    --inference_perturb_k 20 --inference_perturb_eps $EPS \
    --ensemble_agg best_merit --test_batch_sizes 256 2>&1 | tail -12
done

echo "######## Skip-repair (raw NN output) ########"
python eval.py --run_dir "$DC3_SOCP" --skip_repair --test_batch_sizes 256 2>&1 | tail -12

echo "######## ε=2.0 at K=100 ########"
python eval.py --run_dir "$DC3_SOCP" \
  --inference_perturb_k 100 --inference_perturb_eps 2.0 \
  --ensemble_agg best_merit --test_batch_sizes 256 2>&1 | tail -12

echo "=== done $(date) ==="
