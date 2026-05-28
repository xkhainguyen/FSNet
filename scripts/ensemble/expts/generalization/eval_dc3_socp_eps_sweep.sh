#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 00:30:00
#SBATCH -J dc3-eps-sweep
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

source /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/_common.sh

DC3_SOCP=$(ls -dt $P_NSC/*_DC3_*_seed0 2>/dev/null | head -1)
echo "Using $DC3_SOCP"

echo "######## DC3 + nonsmooth SOCP: ε sweep at K=20 ########"
for EPS in 0.01 0.05 0.1 0.5 1.0 2.0 5.0 10.0 ; do
  echo "------ K=20 eps=$EPS ------"
  python eval.py --run_dir "$DC3_SOCP" \
    --inference_perturb_k 20 --inference_perturb_eps $EPS \
    --ensemble_agg best_merit --test_batch_sizes 256 2>&1 | tail -15
done

# Best K=100 if any eps shows an effect — pick the one with largest IneqVio
# improvement and rerun at K=100. Or just sweep K=100 at promising eps.
echo "######## ε=1.0 at K=100 (if K=20 shows movement) ########"
python eval.py --run_dir "$DC3_SOCP" \
  --inference_perturb_k 100 --inference_perturb_eps 1.0 \
  --ensemble_agg best_merit --test_batch_sizes 256 2>&1 | tail -15

# Sanity: what does the RAW NN output look like? Skip-repair on DC3
echo "######## DC3 + nonsmooth SOCP: --skip_repair (raw NN output) ########"
python eval.py --run_dir "$DC3_SOCP" --skip_repair --test_batch_sizes 256 2>&1 | tail -15

echo "=== done $(date) ==="
