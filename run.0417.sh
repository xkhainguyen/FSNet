# for seed in 0 1 2 3; do
#     echo "=============================================="
#     echo " Job started at: $(date '+%Y-%m-%d %H:%M:%S')"
#     echo " Job ID: $SLURM_JOB_ID"
#     echo " Node: $SLURM_NODELIST"
#     echo "=============================================="

#     python main.py \
#     --seed $seed \
#     --method penalty \
#     --prob_type nonsmooth_nonconvex \
#     --prob_name socp \
#     --network LocalContextMLPv2 \
#     --hidden_dim 1024

# done

for seed in 0 1 2 3; do
    echo "=============================================="
    echo " Job started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo " Job ID: $SLURM_JOB_ID"
    echo " Node: $SLURM_NODELIST"
    echo "=============================================="

    python main.py \
    --seed $seed \
    --method penalty \
    --prob_type nonsmooth_nonconvex \
    --prob_name socp \
    --network MLP \
    --hidden_dim 2048

done

