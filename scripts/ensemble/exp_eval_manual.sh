#!/bin/bash
#SBATCH -t 06:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH -J ens-eval
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# source ~/.bashrc
# conda activate ml4opt
# cd ~/orcd/scratch/FSNet
# mkdir -p logs

PROB_STR="SOCPProblem-100-50-50-10000"

echo "=== [ens-eval] $(date '+%Y-%m-%d %H:%M:%S') Job=$SLURM_JOB_ID ==="

# ── Exp 3: agg mode sweep (reuse vanilla penalty training) ──
# VANILLA_DIR="results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260227-012033_penalty_seed2025_e1000_lr1e-04_n7000_ens5_vanilla_pre"

# echo "Vanilla dir: $VANILLA_DIR"

# if [ -n "$VANILLA_DIR" ]; then
#     for agg in mean median best_obj best_merit; do
#         echo "--- eval agg=$agg ---"
#         python eval.py --run_dir "$VANILLA_DIR" --ensemble_agg $agg
#     done
# else
#     echo "WARNING: vanilla penalty run dir not found, skipping agg evals"
# fi

# VANILLA_DIR="results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260227-013946_penalty_seed2025_e1000_lr1e-04_n7000_ens5_fge_pre"

# echo "Vanilla dir: $VANILLA_DIR"

# if [ -n "$VANILLA_DIR" ]; then
#     for agg in mean median best_obj best_merit; do
#         echo "--- eval agg=$agg ---"
#         python eval.py --run_dir "$VANILLA_DIR" --ensemble_agg $agg
#     done
# else
#     echo "WARNING: vanilla penalty run dir not found, skipping agg evals"
# fi

# ── Exp 2: pre vs post (reuse FSNet training, which used pre) ──
# FSNET_DIR="results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260227-015231_FSNet_seed2025_e300_lr1e-04_n7000_ens5_vanilla_pre"
# echo "FSNet dir: $FSNET_DIR"

# post="pre"
# agg="best_merit"

# if [ -n "$FSNET_DIR" ]; then
#     for agg in mean median best_obj best_merit; do
#         echo "--- eval ensemble_post=$post ---"
#         python eval.py --run_dir "$FSNET_DIR" --ensemble_post $post --ensemble_agg $agg
#     done
# else
#     echo "WARNING: FSNet run dir not found, skipping post eval"
# fi


# FSNET_DIR="results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260227-025927_FSNet_seed2025_e300_lr1e-04_n7000_ens5_fge_pre"
# echo "FSNet dir: $FSNET_DIR"

# if [ -n "$FSNET_DIR" ]; then
#     for agg in mean median best_obj best_merit; do
#         echo "--- eval ensemble_post=$post ---"
#         python eval.py --run_dir "$FSNET_DIR" --ensemble_post $post --ensemble_agg $agg
#     done
# else
#     echo "WARNING: FSNet run dir not found, skipping post eval"
# fi

# # ── Exp 2: pre vs post (reuse FSNet training, which used pre) ──
# FSNET_DIR="results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260227-015231_FSNet_seed2025_e300_lr1e-04_n7000_ens5_vanilla_pre"
# echo "FSNet dir: $FSNET_DIR"

# post="pre"
# agg="best_merit"

# if [ -n "$FSNET_DIR" ]; then
#     for agg in mean median best_obj best_merit; do
#         echo "--- eval ensemble_post=$post ---"
#         python eval.py --run_dir "$FSNET_DIR" --ensemble_post $post --ensemble_agg $agg
#     done
# else
#     echo "WARNING: FSNet run dir not found, skipping post eval"
# fi


# FSNET_DIR="results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260227-025927_FSNet_seed2025_e300_lr1e-04_n7000_ens5_fge_pre"
# echo "FSNet dir: $FSNET_DIR"

# if [ -n "$FSNET_DIR" ]; then
#     for agg in mean median best_obj best_merit; do
#         echo "--- eval ensemble_post=$post ---"
#         python eval.py --run_dir "$FSNET_DIR" --ensemble_post $post --ensemble_agg $agg
#     done
# else
#     echo "WARNING: FSNet run dir not found, skipping post eval"
# fi


# ── Exp 3: agg mode sweep (reuse vanilla penalty training) ──

agg="best_merit"
size=10
VANILLA_DIR="results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260227-153237_penalty_seed2025_e1000_lr1e-04_n7000_ens20_vanilla_pre"

echo "Vanilla dir: $VANILLA_DIR"

if [ -n "$VANILLA_DIR" ]; then
    for size in 5 10 15 20; do
        echo "--- eval size=$size ---"
        python eval.py --run_dir "$VANILLA_DIR" --ensemble_agg $agg --ensemble_size $size
    done
else
    echo "WARNING: vanilla penalty run dir not found, skipping agg evals"
fi

# # ── Exp 3: agg mode sweep (reuse vanilla penalty training) ──

# agg="best_merit"
# post="post"

# FSNET_DIR="results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260227-025927_FSNet_seed0_e300_lr1e-04_n7000_ens5_vanilla_pre"

# echo "FSNet dir: $FSNET_DIR"

# if [ -n "$FSNET_DIR" ]; then
#         echo "--- eval size=$size ---"
#         python eval.py --run_dir "$FSNET_DIR" --ensemble_agg $agg --ensemble_post $post
# else
#     echo "WARNING: FSNet penalty run dir not found, skipping agg evals"
# fi

# FSNET_DIR="results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260227-031523_FSNet_seed0_e300_lr1e-04_n7000_ens10_fge_pre"

# echo "FSNet dir: $FSNET_DIR"

# if [ -n "$FSNET_DIR" ]; then
#         echo "--- eval size=$size ---"
#         python eval.py --run_dir "$FSNET_DIR" --ensemble_agg $agg --ensemble_post $post
# else
#     echo "WARNING: FSNet penalty run dir not found, skipping agg evals"
# fi


echo "=== [ens-eval] done $(date '+%Y-%m-%d %H:%M:%S') ==="
