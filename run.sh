#!/bin/bash


for seed in 1 2 3
do
    # python main.py     --method FSNet     --prob_type nonsmooth_nonconvex     --prob_name socp    --seed $seed     --num_epochs 300     --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251023-170626_MLP_sup_seed0_dropout0.1/model_20.pt"

    # echo "ASASASASASASASSSSSSSSSSSSSSSSSS"
    # echo "BACBASASSSSASDFPOEKPOKLMMOPPOKM"

    # python main.py     --method FSNet     --prob_type nonsmooth_nonconvex     --prob_name socp    --seed $seed     --num_epochs 300     --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251024-125513_MLP_sup_seed1_dropout0.1/model_20.pt"

    # python main.py     --method DC3     --prob_type nonsmooth_nonconvex     --prob_name socp    --lr 0.0005  --seed $seed     --num_epochs 500 

    python main.py     --method DC3     --prob_type nonsmooth_nonconvex     --prob_name socp    --seed $seed     --num_epochs 500 

    # python main.py     --method FSNet     --prob_type nonconvex     --prob_name socp    --seed $seed     --num_epochs 300     --checkpoint "results/nonconvex/socp/SOCPProblem-100-50-50-10000/20251024-113746_MLP_sup_seed1_dropout0.1/model_20.pt"

    # python main.py     --method DC3     --prob_type nonsmooth_nonconvex     --prob_name socp    --seed $seed     --num_epochs 300     --checkpoint "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251024-132810_MLP_sup_partial_seed1_dropout0.1/model_100.pt"
done



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

echo "All experiments completed."