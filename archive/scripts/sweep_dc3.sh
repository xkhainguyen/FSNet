#!/bin/bash
#SBATCH --gres=gpu:l40s:1

# optional: load environment
# source ~/.bashrc && conda activate ml4opt
# cd penalty/

# SWEEP SUBOPT RATIO WITH DC3 METHOD
for ckpt in 20 60 100 200; do
    for seed in 0 1 2; do
        python main.py \
            --method DC3 \
            --prob_type nonsmooth_nonconvex \
            --prob_name socp \
            --seed $seed \
            --num_epochs 500 \
            --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251031-162815_MLP_sup_partial_seed1_dropout0.1/model_${ckpt}.pt" # subopt 0.0

        # Draw something funny
        echo "============================================================"
        echo "==========   0.0      =====================         ==========="
        echo "==========         =====================         ==========="
        echo "==========         =====================         ==========="
        echo "============================================================"

        python main.py \
            --method DC3 \
            --prob_type nonsmooth_nonconvex \
            --prob_name socp \
            --seed $seed \
            --num_epochs 500 \
            --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251031-163030_MLP_sup_partial_seed1_dropout0.1/model_${ckpt}.pt" # subopt 0.5

        # Draw something funny
        echo "============================================================"
        echo "==========  0.5       =====================         ==========="
        echo "==========         =====================         ==========="
        echo "==========         =====================         ==========="
        echo "============================================================"

        python main.py \
            --method DC3 \
            --prob_type nonsmooth_nonconvex \
            --prob_name socp \
            --seed $seed \
            --num_epochs 500 \
            --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251031-163247_MLP_sup_partial_seed1_dropout0.1/model_${ckpt}.pt" # subopt 1.0

        # Draw something funny
        echo "============================================================"
        echo "==========         =====================         ==========="
        echo "==========   1.0      =====================         ==========="
        echo "==========         =====================         ==========="
        echo "============================================================"

        python main.py \
            --method DC3 \
            --prob_type nonsmooth_nonconvex \
            --prob_name socp \
            --seed $seed \
            --num_epochs 500 \
            --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251031-163616_MLP_sup_partial_seed1_dropout0.1/model_${ckpt}.pt" # subopt 2.0

        # Draw something funny
        echo "============================================================"
        echo "==========    2.0     =====================         ==========="
        echo "==========         =====================         ==========="
        echo "==========         =====================         ==========="
        echo "============================================================"

        python main.py \
            --method DC3 \
            --prob_type nonsmooth_nonconvex \
            --prob_name socp \
            --seed $seed \
            --num_epochs 500 \
            --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251105-173353_MLP_sup_partial_seed1_dropout0.1/model_${ckpt}.pt" # subopt 4.0

        # Draw something funny
        echo "============================================================"
        echo "==========    4.0     =====================         ==========="
        echo "==========         =====================         ==========="
        echo "==========         =====================         ==========="
        echo "============================================================"

        python main.py \
            --method DC3 \
            --prob_type nonsmooth_nonconvex \
            --prob_name socp \
            --seed $seed \
            --num_epochs 500 \
            --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251105-173555_MLP_sup_partial_seed1_dropout0.1/model_${ckpt}.pt" # subopt 6.0

        # Draw something funny
        echo "============================================================"
        echo "==========    6.0     =====================         ==========="
        echo "==========         =====================         ==========="
        echo "==========         =====================         ==========="
        echo "============================================================"

        # python main.py \
        #     --method DC3 \
        #     --prob_type nonsmooth_nonconvex \
        #     --prob_name socp \
        #     --seed $seed \
        #     --num_epochs 500 \
        #     --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251101-003437_MLP_sup_seed1_dropout0.1/model_${ckpt}.pt" # subopt 10.0

        # # Draw something funny
        # echo "============================================================"
        # echo "==========    10.0     =====================         ==========="
        # echo "==========         =====================         ==========="
        # echo "==========         =====================         ==========="
        # echo "============================================================"
    done
done


# # SWEEP TRAIN SIZE (TODO)
# for seed in 3; do
#     python main.py \
#         --method DC3 \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --seed $seed \
#         --num_epochs 500 \
#         --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251103-103706_MLP_sup_seed1_dropout0.1/model_20.pt" # subopt 2.0
#     # Draw something funny
#     echo "============================================================"
#     echo "==========  10       =====================         ==========="
#     echo "==========         =====================         ==========="
#     echo "==========         =====================         ==========="
#     echo "============================================================"

#     python main.py \
#         --method DC3 \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --seed $seed \
#         --num_epochs 500 \
#         --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251106-114016_MLP_sup_seed1_dropout0.1/model_20.pt" # subopt 2.0
#     # Draw something funny
#     echo "============================================================"
#     echo "==========  20       =====================         ==========="
#     echo "==========         =====================         ==========="
#     echo "==========         =====================         ==========="
#     echo "============================================================"

#     python main.py \
#         --method DC3 \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --seed $seed \
#         --num_epochs 500 \
#         --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251106-114103_MLP_sup_seed1_dropout0.1/model_20.pt" # subopt 2.0
#     # Draw something funny
#     echo "============================================================"
#     echo "==========  30       =====================         ==========="
#     echo "==========         =====================         ==========="
#     echo "==========         =====================         ==========="
#     echo "============================================================"

#     python main.py \
#         --method DC3 \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --seed $seed \
#         --num_epochs 500 \
#         --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251106-114138_MLP_sup_seed1_dropout0.1/model_20.pt" # subopt 2.0
#     # Draw something funny
#     echo "============================================================"
#     echo "==========  40       =====================         ==========="
#     echo "==========         =====================         ==========="
#     echo "==========         =====================         ==========="
#     echo "============================================================"
    
#     python main.py \
#         --method DC3 \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --seed $seed \
#         --num_epochs 500 \
#         --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251103-103719_MLP_sup_seed1_dropout0.1/model_20.pt" # subopt 2.0
#     # Draw something funny
#     echo "============================================================"
#     echo "==========  50       =====================         ==========="
#     echo "==========         =====================         ==========="
#     echo "==========         =====================         ==========="
#     echo "============================================================"

#     python main.py \
#         --method DC3 \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --seed $seed \
#         --num_epochs 500 \
#         --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251101-233217_MLP_sup_seed1_dropout0.1/model_20.pt" # subopt 2.0
#     # Draw something funny
#     echo "============================================================"
#     echo "==========  200       =====================         ==========="
#     echo "==========         =====================         ==========="
#     echo "==========         =====================         ==========="
#     echo "============================================================"
#     python main.py \
#         --method DC3 \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --seed $seed \
#         --num_epochs 500 \
#         --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251101-233236_MLP_sup_seed1_dropout0.1/model_20.pt" # subopt 2.0
#     # Draw something funny
#     echo "============================================================"
#     echo "==========  500       =====================         ==========="
#     echo "==========         =====================         ==========="
#     echo "==========         =====================         ==========="
#     echo "============================================================"
#     python main.py \
#         --method DC3 \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --seed $seed \
#         --num_epochs 500 \
#         --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251101-233303_MLP_sup_seed1_dropout0.1/model_20.pt" # subopt 2.0
#     # Draw something funny
#     echo "============================================================"
#     echo "==========  1000       =====================         ==========="
#     echo "==========         =====================         ==========="
#     echo "==========         =====================         ==========="
#     echo "============================================================"
#     python main.py \
#         --method DC3 \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --seed $seed \
#         --num_epochs 500 \
#         --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251101-233338_MLP_sup_seed1_dropout0.1/model_20.pt" # subopt 2.0
#     # Draw something funny
#     echo "============================================================"
#     echo "==========  4000       =====================         ==========="
#     echo "==========         =====================         ==========="
#     echo "==========         =====================         ==========="
#     echo "============================================================"

# done