#!/bin/bash
#SBATCH --gres=gpu:l40s:1

# optional: load environment
# source ~/.bashrc && conda activate ml4opt
# cd FSNet/

for seed in 1 2 3; do
#     python main.py \
#         --method FSNet \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --seed $seed \
#         --num_epochs 300 \
#         --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251103-103706_MLP_sup_seed1_dropout0.1/model_20.pt" # subopt 2.0
#     # Draw something funny
#     echo "============================================================"
#     echo "==========  10       =====================         ==========="
#     echo "==========         =====================         ==========="
#     echo "==========         =====================         ==========="
#     echo "============================================================"

    python main.py \
        --method FSNet \
        --prob_type nonsmooth_nonconvex \
        --prob_name socp \
        --seed $seed \
        --num_epochs 300 \
        --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251106-114016_MLP_sup_seed1_dropout0.1/model_20.pt" # subopt 2.0
    # Draw something funny
    echo "============================================================"
    echo "==========  20       =====================         ==========="
    echo "==========         =====================         ==========="
    echo "==========         =====================         ==========="
    echo "============================================================"

    python main.py \
        --method FSNet \
        --prob_type nonsmooth_nonconvex \
        --prob_name socp \
        --seed $seed \
        --num_epochs 300 \
        --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251106-114103_MLP_sup_seed1_dropout0.1/model_20.pt" # subopt 2.0
    # Draw something funny
    echo "============================================================"
    echo "==========  30       =====================         ==========="
    echo "==========         =====================         ==========="
    echo "==========         =====================         ==========="
    echo "============================================================"

    python main.py \
        --method FSNet \
        --prob_type nonsmooth_nonconvex \
        --prob_name socp \
        --seed $seed \
        --num_epochs 300 \
        --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251106-114138_MLP_sup_seed1_dropout0.1/model_20.pt" # subopt 2.0
    # Draw something funny
    echo "============================================================"
    echo "==========  40       =====================         ==========="
    echo "==========         =====================         ==========="
    echo "==========         =====================         ==========="
    echo "============================================================"

    # python main.py \
    #     --method FSNet \
    #     --prob_type nonsmooth_nonconvex \
    #     --prob_name socp \
    #     --seed $seed \
    #     --num_epochs 300 \
    #     --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251103-103719_MLP_sup_seed1_dropout0.1/model_20.pt" # subopt 2.0
    # # Draw something funny
    # echo "============================================================"
    # echo "==========  50       =====================         ==========="
    # echo "==========         =====================         ==========="
    # echo "==========         =====================         ==========="
    # echo "============================================================"

    # python main.py \
    #     --method FSNet \
    #     --prob_type nonsmooth_nonconvex \
    #     --prob_name socp \
    #     --seed $seed \
    #     --num_epochs 300 \
    #     --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251101-233217_MLP_sup_seed1_dropout0.1/model_20.pt" # subopt 2.0
    # # Draw something funny
    # echo "============================================================"
    # echo "==========  200       =====================         ==========="
    # echo "==========         =====================         ==========="
    # echo "==========         =====================         ==========="
    # echo "============================================================"
    # python main.py \
    #     --method FSNet \
    #     --prob_type nonsmooth_nonconvex \
    #     --prob_name socp \
    #     --seed $seed \
    #     --num_epochs 300 \
    #     --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251101-233236_MLP_sup_seed1_dropout0.1/model_20.pt" # subopt 2.0
    # # Draw something funny
    # echo "============================================================"
    # echo "==========  500       =====================         ==========="
    # echo "==========         =====================         ==========="
    # echo "==========         =====================         ==========="
    # echo "============================================================"
    # python main.py \
    #     --method FSNet \
    #     --prob_type nonsmooth_nonconvex \
    #     --prob_name socp \
    #     --seed $seed \
    #     --num_epochs 300 \
    #     --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251101-233303_MLP_sup_seed1_dropout0.1/model_20.pt" # subopt 2.0
    # # Draw something funny
    # echo "============================================================"
    # echo "==========  1000       =====================         ==========="
    # echo "==========         =====================         ==========="
    # echo "==========         =====================         ==========="
    # echo "============================================================"
    # python main.py \
    #     --method FSNet \
    #     --prob_type nonsmooth_nonconvex \
    #     --prob_name socp \
    #     --seed $seed \
    #     --num_epochs 300 \
    #     --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251101-233338_MLP_sup_seed1_dropout0.1/model_20.pt" # subopt 2.0
    # # Draw something funny
    # echo "============================================================"
    # echo "==========  4000       =====================         ==========="
    # echo "==========         =====================         ==========="
    # echo "==========         =====================         ==========="
    # echo "============================================================"

done

for size in 7000; do
    python main.py \
        --method sup \
        --prob_type nonsmooth_nonconvex \
        --prob_name socp \
        --dropout 0.1 \
        --lr 0.00005  \
        --train_size $size \
        --seed 1 \
        --en_subopt True \
        --subopt_ratio  2.0

    # Draw something funny
    echo "============================================================"
    echo "==========         =====================         ==========="
    echo "==========         =====================         ==========="
    echo "==========         =====================         ==========="
    echo "============================================================"
done
