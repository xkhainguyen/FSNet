#!/bin/bash

# for dropout in 0.05 0.1 0.2
# do
#   for seed in 0 1 2
#   do
#     python main.py --method sup_pen --prob_type nonsmooth_nonconvex --prob_name socp --dropout $dropout --seed $seed
#   done
# done

# # print
# 


# change results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251004-214029_MLP_sup_seed0_dropout0.1/model_20.pt to model_{$ckpt}


# for ckpt in 20 100 200 600; do
#     python main.py \
#         --method FSNet \
#         --prob_type nonsmooth_nonconvex \
#         --prob_name socp \
#         --dropout 0.1 \
#         --seed 0 \
#         --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251004-214029_MLP_sup_seed0_dropout0.1/model_${ckpt}.pt"
# done

python main.py \
    --method sup \
    --prob_type nonsmooth_nonconvex \
    --prob_name socp \
    --train_size 7000 \
    --seed 1 \
    --en_subopt 2 \
    --subopt_ratio -10.0 \
    --save_intermediate True 

python main.py \
    --method sup \
    --prob_type nonsmooth_nonconvex \
    --prob_name socp \
    --train_size 7000 \
    --seed 1 \
    --en_subopt 2 \
    --subopt_ratio 1.0 \
    --save_intermediate True 

python main.py \
    --method FSNet \
    --prob_type nonsmooth_nonconvex \
    --prob_name socp \
    --lr 0.0001  \
    --seed 1 \
    --num_epochs 300

python main.py \
    --method FSNet \
    --prob_type nonsmooth_nonconvex \
    --prob_name socp \
    --lr 0.0002  \
    --seed 1 \
    --num_epochs 300 \
    --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260105-234302_MLP_sup_seed1_dropout0.1_subopt_2_-10.0/model_60.pt"



python main.py \
    --method sup \
    --prob_type nonsmooth_nonconvex \
    --prob_name socp \
    --dropout 0.1 \
    --lr 0.0005  \
    --train_size 7000 \
    --seed 1 \
    --en_subopt True \
    --save_intermediate True \
    --subopt_ratio 6.0

python main.py \
    --method FSNet \
    --prob_type nonsmooth_nonconvex \
    --prob_name socp \
    --lr 0.0001  \
    --seed 1\

python main.py \
    --method FSNet \
    --prob_type nonsmooth_nonconvex \
    --prob_name socp \
    --dropout 0.1 \
    --lr 0.00005 \
    --seed 0 \
    --num_epochs 300 \
    --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251023-170626_MLP_sup_seed0_dropout0.1/model_40.pt"

python main.py \
    --method FSNet \
    --prob_type nonsmooth_nonconvex \
    --prob_name socp \
    --dropout 0.1 \
    --lr 0.0001 \
    --seed 0 \
    --num_epochs 200 \
    --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251010-090553_MLP_sup_pen_seed1_dropout0.1/model_40.pt"


# python main.py     --method penalty     --prob_type nonsmooth_nonconvex     --prob_name socp    --lr 0.00005  --seed 0     --num_epochs 500

# python main.py \
#     --method FSNet \
#     --prob_type nonsmooth_nonconvex \
#     --prob_name socp \
#     --dropout 0.1 \
#     --seed 1 \
#     --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251006-074528_MLP_sup_pen_seed0_dropout0.1/model_200.pt"

# python main.py \
#     --method FSNet \
#     --prob_type nonsmooth_nonconvex \
#     --prob_name socp \
#     --dropout 0.1 \
#     --seed 2 \
#     --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251006-074528_MLP_sup_pen_seed0_dropout0.1/model_200.pt"

# python main.py \
#     --method FSNet \
#     --prob_type nonsmooth_nonconvex \
#     --prob_name socp \
#     --dropout 0.1 \
#     --seed 0 \
#     --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251006-074528_MLP_sup_pen_seed0_dropout0.1/model_900.pt"

# python main.py \
#     --method FSNet \
#     --prob_type nonsmooth_nonconvex \
#     --prob_name socp \
#     --dropout 0.1 \
#     --seed 1 \
#     --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251006-074528_MLP_sup_pen_seed0_dropout0.1/model_900.pt"

# python main.py \
#     --method FSNet \
#     --prob_type nonsmooth_nonconvex \
#     --prob_name socp \
#     --dropout 0.1 \
#     --seed 2 \
#     --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251006-074528_MLP_sup_pen_seed0_dropout0.1/model_900.pt"

echo "All experiments completed."