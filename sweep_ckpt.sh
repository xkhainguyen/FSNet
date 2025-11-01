#!/bin/bash
#SBATCH --gres=gpu:l40s:1

# optional: load environment
# source ~/.bashrc && conda activate ml4opt
# cd FSNet/

for ckpt in 20 60; do
    for seed in 0 1 2 3; do
        # python main.py \
        #     --method FSNet \
        #     --prob_type nonsmooth_nonconvex \
        #     --prob_name socp \
        #     --seed $seed \
        #     --num_epochs 300 \
        #     --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251030-232939_MLP_sup_seed1_dropout0.1/model_${ckpt}.pt" # subopt 0.0

        # python main.py \
        #     --method FSNet \
        #     --prob_type nonsmooth_nonconvex \
        #     --prob_name socp \
        #     --seed $seed \
        #     --num_epochs 300 \
        #     --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251030-233223_MLP_sup_seed1_dropout0.1/model_${ckpt}.pt" # subopt 0.5

        # python main.py \
        #     --method FSNet \
        #     --prob_type nonsmooth_nonconvex \
        #     --prob_name socp \
        #     --seed $seed \
        #     --num_epochs 300 \
        #     --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251030-233258_MLP_sup_seed1_dropout0.1/model_${ckpt}.pt" # subopt 1.0

        # python main.py \
        #     --method FSNet \
        #     --prob_type nonsmooth_nonconvex \
        #     --prob_name socp \
        #     --seed $seed \
        #     --num_epochs 300 \
        #     --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251031-073026_MLP_sup_seed1_dropout0.1/model_${ckpt}.pt" # subopt 2.0

        python main.py \
            --method FSNet \
            --prob_type nonsmooth_nonconvex \
            --prob_name socp \
            --seed $seed \
            --num_epochs 300 \
            --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251101-002602_MLP_sup_seed1_dropout0.1/model_${ckpt}.pt" # subopt 4.0

        python main.py \
            --method FSNet \
            --prob_type nonsmooth_nonconvex \
            --prob_name socp \
            --seed $seed \
            --num_epochs 300 \
            --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251101-003437_MLP_sup_seed1_dropout0.1/model_${ckpt}.pt" # subopt 10.0

        # Draw something funny
        echo "============================================================"
        echo "==========         =====================         ==========="
        echo "==========         =====================         ==========="
        echo "==========         =====================         ==========="
        echo "============================================================"
    done
done