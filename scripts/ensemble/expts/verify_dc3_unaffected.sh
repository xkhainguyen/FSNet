#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 00:15:00
#SBATCH -J dc3-unaff
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

source /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/_common.sh

DC3_SOCP="results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260528-171119_DC3_e1000_lr5e-05_n7000_hdim1024_seed0"
DC3_QP="results/nonsmooth_nonconvex/qp/QPProblem-100-50-50-10000/20260528-171119_DC3_e1000_lr5e-05_n7000_hdim1024_seed0"

echo "######## DC3 + SOCP: should be UNCHANGED (no L-BFGS) ########"
echo "------ per_sample=0 no perturb (expect 624) ------"
python eval.py --run_dir "$DC3_SOCP" --test_batch_sizes 256 2>&1 | grep -E "Merit:|Eq Vio|Ineq Vio" | head -3
echo "------ per_sample=1 no perturb (expect same 624 — no L-BFGS path) ------"
python eval.py --run_dir "$DC3_SOCP" --test_batch_sizes 256 --per_sample_lbfgs 1 2>&1 | grep -E "Merit:|Eq Vio|Ineq Vio" | head -3
echo "------ per_sample=0 + K=100 ε=0.01 (expect ~24, prior finding) ------"
python eval.py --run_dir "$DC3_SOCP" --inference_perturb_k 100 --inference_perturb_eps 0.01 \
   --ensemble_agg best_merit --test_batch_sizes 256 2>&1 | grep -E "Merit:|Eq Vio|Ineq Vio" | head -3
echo "------ per_sample=1 + K=100 ε=0.01 (expect ~24 — same as above) ------"
python eval.py --run_dir "$DC3_SOCP" --inference_perturb_k 100 --inference_perturb_eps 0.01 \
   --ensemble_agg best_merit --per_sample_lbfgs 1 --test_batch_sizes 256 2>&1 | grep -E "Merit:|Eq Vio|Ineq Vio" | head -3

echo
echo "######## DC3 + QP: same ########"
echo "------ per_sample=0 no perturb (expect 23725) ------"
python eval.py --run_dir "$DC3_QP" --test_batch_sizes 256 2>&1 | grep -E "Merit:|Eq Vio|Ineq Vio" | head -3
echo "------ per_sample=0 K=100 ε=0.05 (expect ~463, prior finding) ------"
python eval.py --run_dir "$DC3_QP" --inference_perturb_k 100 --inference_perturb_eps 0.05 \
   --ensemble_agg best_merit --test_batch_sizes 256 2>&1 | grep -E "Merit:|Eq Vio|Ineq Vio" | head -3
echo "------ per_sample=1 K=100 ε=0.05 (expect ~463) ------"
python eval.py --run_dir "$DC3_QP" --inference_perturb_k 100 --inference_perturb_eps 0.05 \
   --ensemble_agg best_merit --per_sample_lbfgs 1 --test_batch_sizes 256 2>&1 | grep -E "Merit:|Eq Vio|Ineq Vio" | head -3

echo "=== done $(date) ==="
