#!/bin/bash

# for dropout in 0.05 0.1 0.2
# do
#   for seed in 0 1 2
#   do
#     python main.py --method sup_pen --prob_type nonsmooth_nonconvex --prob_name socp --dropout $dropout --seed $seed
#   done
# done

# # print
# echo "All experiments completed."

python main.py --method FSNet --prob_type nonsmooth_nonconvex --prob_name socp --dropout 0.1 --seed 0