#!/bin/bash
#SBATCH -t 06:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH -J ens-eval
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

source ~/.bashrc
conda activate ml4opt
cd ~/orcd/scratch/FSNet
mkdir -p logs

PROB_STR="SOCPProblem-100-50-50-10000"

latest_run() {
    local base="results/$1/socp/${PROB_STR}"
    ls -dt ${base}/$2 2>/dev/null | head -1
}

echo "=== [ens-eval] $(date '+%Y-%m-%d %H:%M:%S') Job=$SLURM_JOB_ID ==="

# ── Exp 3: agg mode sweep (reuse vanilla penalty training) ──
VANILLA_DIR=$(latest_run nonsmooth_nonconvex "*_penalty_*_ens5_vanilla_*")
echo "Vanilla dir: $VANILLA_DIR"

if [ -n "$VANILLA_DIR" ]; then
    for agg in median best_obj best_merit; do
        echo "--- eval agg=$agg ---"
        python eval.py --run_dir "$VANILLA_DIR" --ensemble_agg $agg
    done
else
    echo "WARNING: vanilla penalty run dir not found, skipping agg evals"
fi

# ── Exp 2: pre vs post (reuse FSNet training, which used pre) ──
FSNET_DIR=$(latest_run nonconvex "*_FSNet_*_ens5_vanilla_*")
echo "FSNet dir: $FSNET_DIR"

if [ -n "$FSNET_DIR" ]; then
    echo "--- eval ensemble_post=post ---"
    python eval.py --run_dir "$FSNET_DIR" --ensemble_post post --wandb
else
    echo "WARNING: FSNet run dir not found, skipping post eval"
fi

echo "=== [ens-eval] done $(date '+%Y-%m-%d %H:%M:%S') ==="
