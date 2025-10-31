#!/bin/bash
#SBATCH --gres=gpu:l40s:1

# optional: load environment
# source ~/.bashrc && conda activate ml4opt
# cd FSNet/

for ckpt in 100 200; do
    for seed in 0 1 2; do
        python main.py \
            --method DC3 \
            --prob_type nonsmooth_nonconvex \
            --prob_name socp \
            --seed $seed \
            --num_epochs 300 \
            --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251031-162815_MLP_sup_partial_seed1_dropout0.1/model_${ckpt}.pt" # subopt 0.0

        python main.py \
            --method DC3 \
            --prob_type nonsmooth_nonconvex \
            --prob_name socp \
            --seed $seed \
            --num_epochs 300 \
            --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251031-163030_MLP_sup_partial_seed1_dropout0.1/model_${ckpt}.pt" # subopt 0.5

        python main.py \
            --method DC3 \
            --prob_type nonsmooth_nonconvex \
            --prob_name socp \
            --seed $seed \
            --num_epochs 300 \
            --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251031-163247_MLP_sup_partial_seed1_dropout0.1/model_${ckpt}.pt" # subopt 1.0

        python main.py \
            --method DC3 \
            --prob_type nonsmooth_nonconvex \
            --prob_name socp \
            --seed $seed \
            --num_epochs 300 \
            --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251031-163616_MLP_sup_partial_seed1_dropout0.1/model_${ckpt}.pt" # subopt 2.0

        # Draw something funny
        echo "============================================================"
        echo "==========         =====================         ==========="
        echo "==========         =====================         ==========="
        echo "==========         =====================         ==========="
        echo "============================================================"
    done
done