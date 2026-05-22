#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 00:20:00
#SBATCH -J mhe-eval-smoke
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

set -e
source ~/.bashrc
conda activate ml4opt
cd /orcd/scratch/orcd/008/khain/FSNet

MHE_DIR="results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260522-032108_penalty_e5_lr1e-04_n500_hdim1024_mhe5_seed0"
PEN_SINGLE="results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260417-114439_penalty_seed0_e1000_lr1e-04_n7000"

echo "=== 1) Eval MHE checkpoint (post + best_merit) ==="
python eval.py --run_dir "$MHE_DIR" --ensemble_post post --ensemble_agg best_merit --test_batch_sizes 256

echo "=== 2) Eval penalty single + skip_repair ==="
python eval.py --run_dir "$PEN_SINGLE" --skip_repair --test_batch_sizes 256

echo "=== 3) Eval penalty single + perturb K=5 eps=0.05 best_merit ==="
python eval.py --run_dir "$PEN_SINGLE" --inference_perturb_k 5 --inference_perturb_eps 0.05 --ensemble_agg best_merit --test_batch_sizes 256

echo "=== eval-smoke done $(date) ==="
