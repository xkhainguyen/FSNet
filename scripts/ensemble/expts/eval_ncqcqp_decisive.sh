#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 00:40:00
#SBATCH -J eval-ncqcqp-dec
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err
source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
cd /orcd/scratch/orcd/008/khain/FSNet
set -e
RUN="results/nonconvex/qcqp/QCQPProblem-100-50-50-10000/20260529-125053_FSNet_e300_lr1e-04_n7000_hdim1024_seed0"
# DECISIVE: converged repair (per_sample=1) + perturbation. Does K=100 beat K=1=2.03?
for K in 1 20 100; do
  for EPS in 0.0 0.05 0.1 0.3; do
    if [ "$K" = "1" ] && [ "$EPS" != "0.0" ]; then continue; fi
    echo ">>> per_sample=1 max_iter=50 K=$K eps=$EPS"
    python eval.py --run_dir "$RUN" --ensemble_post post --ensemble_agg best_merit \
      --per_sample_lbfgs 1 --repair_max_iter 50 --vectorize_repair \
      --inference_perturb_k $K --inference_perturb_eps $EPS 2>&1 | grep -E "Obj:|Merit:" || true
  done
done
echo "=== done $(date) ==="
