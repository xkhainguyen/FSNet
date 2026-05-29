#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 00:40:00
#SBATCH -J eval-ncqcqp-s1
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err
source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
cd /orcd/scratch/orcd/008/khain/FSNet
set -e
RUN="results/nonconvex/qcqp/QCQPProblem-100-50-50-10000/20260529-133057_FSNet_e300_lr1e-04_n7000_hdim1024_seed1"
echo "=== seed1 multimodal confirmation $(date) ==="
for PS in 0 1; do
  for K in 1 100; do
    for EPS in 0.0 0.1; do
      if [ "$K" = "1" ] && [ "$EPS" != "0.0" ]; then continue; fi
      echo ">>> per_sample=$PS K=$K eps=$EPS"
      python eval.py --run_dir "$RUN" --ensemble_post post --ensemble_agg best_merit         --per_sample_lbfgs $PS --repair_max_iter 50 --vectorize_repair         --inference_perturb_k $K --inference_perturb_eps $EPS 2>&1 | grep -E "Merit:" || true
    done
  done
done
echo "=== done $(date) ==="
