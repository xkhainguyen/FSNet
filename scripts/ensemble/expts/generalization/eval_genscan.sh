#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 01:00:00
#SBATCH -J eval-genscan
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

source /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/_common.sh

# Eval grid: each new (method, problem) combo at default + K=20 + K=100 perturbation
P_NSQP="results/nonsmooth_nonconvex/qp/QPProblem-100-50-50-10000"

eval_combo() {
  local LABEL="$1" DIR="$2"
  [ ! -d "$DIR" ] && { echo "SKIP $LABEL (no dir)" ; return ; }
  [ ! -f "$DIR/model.pt" ] && { echo "SKIP $LABEL (no model.pt)" ; return ; }
  echo "######## $LABEL ########"
  echo "------ no perturb (baseline) ------"
  python eval.py --run_dir "$DIR" --test_batch_sizes 256 2>&1 | tail -15
  echo "------ K=20 eps=0.1 ------"
  python eval.py --run_dir "$DIR" \
    --inference_perturb_k 20 --inference_perturb_eps 0.1 \
    --ensemble_agg best_merit --test_batch_sizes 256 2>&1 | tail -15
  echo "------ K=100 eps=0.1 ------"
  python eval.py --run_dir "$DIR" \
    --inference_perturb_k 100 --inference_perturb_eps 0.1 \
    --ensemble_agg best_merit --test_batch_sizes 256 2>&1 | tail -15
}

# Find the most recent checkpoint for each combo
FSNET_NSQP=$(ls -dt $P_NSQP/*_FSNet_*_seed0 2>/dev/null | head -1)
DC3_NSSOCP=$(ls -dt $P_NSC/*_DC3_*_seed0 2>/dev/null | head -1)
DC3_NSQP=$(ls -dt $P_NSQP/*_DC3_*_seed0 2>/dev/null | head -1)

eval_combo "FSNet on nonsmooth_nonconvex QP" "$FSNET_NSQP"
eval_combo "DC3 on nonsmooth_nonconvex SOCP" "$DC3_NSSOCP"
eval_combo "DC3 on nonsmooth_nonconvex QP"   "$DC3_NSQP"

echo "=== eval-genscan done $(date) ==="
