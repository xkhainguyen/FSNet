import numpy as np

######################################
# Train size 7000

fsnet_ft_sup_trainsize7000_over_sublevel = np.array([
    [
        "20260115-012027_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000",
        "20260115-013345_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000",
        "20260115-012025_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000",
        "20260115-013254_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000",
    ],
    # [
    #     "20260114-224213_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051731_penalty_seedseed1_model_990",
    #     "20260114-224213_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051731_penalty_seedseed1_model_990",
    #     "20260114-225544_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051731_penalty_seedseed1_model_990",
    #     "20260114-225547_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051731_penalty_seedseed1_model_990",
    # ],  # penalty
    # [
    #     "20260115-022127_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260115-021622_penalty_seedseed0_model_990",
    #     "20260115-022527_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260115-021622_penalty_seedseed0_model_990",
    #     "20260115-023352_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260115-021622_penalty_seedseed0_model_990",
    #     "20260115-023737_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260115-021622_penalty_seedseed0_model_990",
    # ],  # 0.0 penalty
    [
        "20260114-090924_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260114-053054_sup_seedseed0_model_210",
        "20260114-095319_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260114-053054_sup_seedseed0_model_210",
        "20260114-065003_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260114-053054_sup_seedseed0_model_210",
        "20260114-073431_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260114-053054_sup_seedseed0_model_210",
    ],  # 0.5
    # [
    #     "20260114-082836_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260114-061120_sup_seedseed0_model_120",
    #     "20260114-091103_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260114-061120_sup_seedseed0_model_120",
    #     "20260114-071827_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260114-061120_sup_seedseed0_model_120",
    #     "20260114-075903_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260114-061120_sup_seedseed0_model_120",
    # ],  # 1.5
    # [
    #     "20260114-174659_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260114-083124_sup_seedseed0_model_80",
    #     "20260114-182846_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260114-083124_sup_seedseed0_model_80",
    #     "20260114-183015_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260114-083124_sup_seedseed0_model_80",
    #     "20260114-191029_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260114-083124_sup_seedseed0_model_80",
    # ],  # 2.0
    [
        "20260114-170818_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260114-152404_sup_seedseed0_model_50",
        "20260114-172146_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260114-152404_sup_seedseed0_model_50",
        "20260114-173516_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260114-152404_sup_seedseed0_model_50",
        "20260114-174845_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260114-152404_sup_seedseed0_model_50",
    ],  # 3.0
    [
        "20260114-190933_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260114-170518_sup_seedseed0_model_70",
        "20260114-195044_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260114-170518_sup_seedseed0_model_70",
        "20260114-195528_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260114-170518_sup_seedseed0_model_70",
        "20260114-203653_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260114-170518_sup_seedseed0_model_70",
    ]  # 4.0
])  # matrix of shape (num_ckpt, num_seeds)

fsnet_ft_sup_pen_trainsize7000_over_sublevel = np.array([
    [
        "20260115-012027_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000",
        "20260115-013345_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000",
        "20260115-012025_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000",
        "20260115-013254_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000",
    ],
    # [
    #     "20260115-022127_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260115-021622_penalty_seedseed0_model_990",
    #     "20260115-022527_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260115-021622_penalty_seedseed0_model_990",
    #     "20260115-023352_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260115-021622_penalty_seedseed0_model_990",
    #     "20260115-023737_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260115-021622_penalty_seedseed0_model_990",
    # ],  # 0.0 penalty
    # [
    #     "20260114-224213_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051731_penalty_seedseed1_model_990",
    #     "20260114-224213_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051731_penalty_seedseed1_model_990",
    #     "20260114-225544_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051731_penalty_seedseed1_model_990",
    #     "20260114-225547_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051731_penalty_seedseed1_model_990",
    # ],  # penalty
    [
        "20260114-062152_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051920_sup_seedpen_model_990",
        "20260114-070402_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051920_sup_seedpen_model_990",
        "20260114-084153_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051920_sup_seedpen_model_990",
        "20260114-092335_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051920_sup_seedpen_model_990",
    ],  # 0.5
    [
        "20260114-065152_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260114-060752_sup_seedpen_model_990",
        "20260114-073226_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260114-060752_sup_seedpen_model_990",
        "20260114-080154_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260114-060752_sup_seedpen_model_990",
        "20260114-084429_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260114-060752_sup_seedpen_model_990",
    ],  # 1.5
    [
        "20260114-172035_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260114-082755_sup_seedpen_model_990",
        "20260114-180211_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260114-082755_sup_seedpen_model_990",
        "20260114-180320_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260114-082755_sup_seedpen_model_990",
        "20260114-184350_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260114-082755_sup_seedpen_model_990",
    ],  # 2.0
    [
        "20260114-153617_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260114-145943_sup_seedpen_model_990",
        "20260114-154225_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260114-145943_sup_seedpen_model_990",
        "20260114-161638_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260114-145943_sup_seedpen_model_990",
        "20260114-162226_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260114-145943_sup_seedpen_model_990",
    ],  # 3.0
    [
        "20260114-184305_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260114-145943_sup_seedpen_model_990",
        "20260114-192410_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260114-145943_sup_seedpen_model_990",
        "20260114-192900_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260114-145943_sup_seedpen_model_990",
        "20260114-201002_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260114-145943_sup_seedpen_model_990",
    ]  # 4.0
])  # matrix of shape (num_ckpt, num_seeds)

# how much do we need data (data weight)
fsnet_ft_sup_pen_trainsize7000_over_sublevel2 = np.array([
    [
        "20260115-012027_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000",
        "20260115-013345_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000",
        "20260115-012025_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000",
        "20260115-013254_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000",
    ],
    [
        "20260114-224213_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051731_penalty_seedseed1_model_990",
        "20260114-224213_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051731_penalty_seedseed1_model_990",
        "20260114-225544_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051731_penalty_seedseed1_model_990",
        "20260114-225547_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051731_penalty_seedseed1_model_990",
    ],  # penalty
    [
        "20260115-022127_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260115-021622_penalty_seedseed0_model_990",
        "20260115-022527_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260115-021622_penalty_seedseed0_model_990",
        "20260115-023352_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260115-021622_penalty_seedseed0_model_990",
        "20260115-023737_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260115-021622_penalty_seedseed0_model_990",
    ],  # 0.0 penalty
    [
        "20260114-065152_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260114-060752_sup_seedpen_model_990",
        "20260114-073226_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260114-060752_sup_seedpen_model_990",
        "20260114-080154_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260114-060752_sup_seedpen_model_990",
        "20260114-084429_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260114-060752_sup_seedpen_model_990",
    ],  # 1.5
    [
        "20260115-151210_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260115-145229_sup_seedpen_model_380",
        "20260115-151210_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260115-145229_sup_seedpen_model_380",
        "20260115-152425_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260115-145229_sup_seedpen_model_380",
        "20260115-152450_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260115-145229_sup_seedpen_model_380",
    ],  # 1.5 10.0
    [
        "20260115-153755_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260115-145229_sup_seedpen_model_630",
        "20260115-153756_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260115-145229_sup_seedpen_model_630",
        "20260115-154949_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260115-145229_sup_seedpen_model_630",
        "20260115-154950_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260115-145229_sup_seedpen_model_630",
    ],  # 1.5 100.0
])  # matrix of shape (num_ckpt, num_seeds)

# how much do we need data (data weight)
fsnet_ft_sup_pen_over_sublevel3 = np.array([
    [
        "20260115-012027_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000",
        "20260115-013345_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000",
        "20260115-012025_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000",
        "20260115-013254_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000",
    ],
    [
        "20260114-224213_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051731_penalty_seedseed1_model_990",
        "20260114-224213_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051731_penalty_seedseed1_model_990",
        "20260114-225544_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051731_penalty_seedseed1_model_990",
        "20260114-225547_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051731_penalty_seedseed1_model_990",
    ],  # penalty
    # [
    #     "20260115-022127_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260115-021622_penalty_seedseed0_model_990",
    #     "20260115-022527_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260115-021622_penalty_seedseed0_model_990",
    #     "20260115-023352_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260115-021622_penalty_seedseed0_model_990",
    #     "20260115-023737_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260115-021622_penalty_seedseed0_model_990",
    # ],  # 0.0 penalty
    # [
    #     "20260114-062152_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051920_sup_seedpen_model_990",
    #     "20260114-070402_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051920_sup_seedpen_model_990",
    #     "20260114-084153_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051920_sup_seedpen_model_990",
    #     "20260114-092335_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051920_sup_seedpen_model_990",
    # ],  # 0.5
    # [
    #     "20260115-174555_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260115-173954_sup_seedpen_model_450",
    #     "20260115-175810_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260115-173954_sup_seedpen_model_450",
    #     "20260115-181026_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260115-173954_sup_seedpen_model_450",
    #     "20260115-182240_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260115-173954_sup_seedpen_model_450",
    # ],  # 0.5 10.0
    # [
    #     "20260115-174614_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260115-174154_sup_seedpen_model_900",
    #     "20260115-180115_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260115-174154_sup_seedpen_model_900",
    #     "20260115-181603_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260115-174154_sup_seedpen_model_900",
    #     "20260115-183054_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260115-174154_sup_seedpen_model_900",
    # ],  # 0.5 100.0
    [
        "20260116-031254_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260116-022657_sup_seedpen_model_940",
        "20260116-034156_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260116-022657_sup_seedpen_model_940",
        "20260116-040759_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260116-022657_sup_seedpen_model_940",
        "20260116-043934_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260116-022657_sup_seedpen_model_940",
    ],  # 0.5 800
    # [
    #     "20260116-052255_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260116-051557_sup_seedpen_model_390",
    #     "20260116-053625_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260116-051557_sup_seedpen_model_390",
    #     "20260116-054957_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260116-051557_sup_seedpen_model_390",
    #     "20260116-060327_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260116-051557_sup_seedpen_model_390",
    # ], # 5 1000
])  # matrix of shape (num_ckpt, num_seeds)


fsnet_ft_sup_990_trainsize7000_over_sublevel = np.array([
    [
        "20260115-012027_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000",
        "20260115-013345_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000",
        "20260115-012025_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000",
        "20260115-013254_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000",
    ],
    # [
    #     "20260114-063411_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260114-053054_sup_seedseed0_model_990",
    #     "20260114-071630_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260114-053054_sup_seedseed0_model_990",
    #     "20260114-085456_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260114-053054_sup_seedseed0_model_990",
    #     "20260114-093636_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260114-053054_sup_seedseed0_model_990",
    # ],  # 0.5
    [
        "20260114-070411_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260114-061120_sup_seedseed0_model_990",
        "20260114-074445_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260114-061120_sup_seedseed0_model_990",
        "20260114-081421_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260114-061120_sup_seedseed0_model_990",
        "20260114-085658_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260114-061120_sup_seedseed0_model_990",
    ],  # 1.5
    [
        "20260114-173255_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260114-083124_sup_seedseed0_model_990",
        "20260114-181430_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260114-083124_sup_seedseed0_model_990",
        "20260114-181547_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260114-083124_sup_seedseed0_model_990",
        "20260114-185610_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260114-083124_sup_seedseed0_model_990",
    ],  # 2.0
    [
        "20260114-154840_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260114-152404_sup_seedseed0_model_990",
        "20260114-155446_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260114-152404_sup_seedseed0_model_990",
        "20260114-162903_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260114-152404_sup_seedseed0_model_990",
        "20260114-163445_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260114-152404_sup_seedseed0_model_990",
    ],  # 3.0
    [
        "20260114-185516_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260114-170518_sup_seedseed0_model_990",
        "20260114-193622_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260114-170518_sup_seedseed0_model_990",
        "20260114-194117_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260114-170518_sup_seedseed0_model_990",
        "20260114-202219_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260114-170518_sup_seedseed0_model_990",
    ],  # 4.0
])  # matrix of shape (num_ckpt, num_seeds)

fsnet_ft_sup_pen_w100_trainsize7000_over_sublevel = np.array([
    [
        "20260115-012027_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000",
        "20260115-013345_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000",
        "20260115-012025_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000",
        "20260115-013254_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000",
    ],
    # [
    #     "20260114-224213_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051731_penalty_seedseed1_model_990",
    #     "20260114-224213_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051731_penalty_seedseed1_model_990",
    #     "20260114-225544_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051731_penalty_seedseed1_model_990",
    #     "20260114-225547_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051731_penalty_seedseed1_model_990",
    # ],  # penalty
    # [
    #     "20260115-022127_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260115-021622_penalty_seedseed0_model_990",
    #     "20260115-022527_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260115-021622_penalty_seedseed0_model_990",
    #     "20260115-023352_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260115-021622_penalty_seedseed0_model_990",
    #     "20260115-023737_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260115-021622_penalty_seedseed0_model_990",
    # ],  # 0.0 penalty
    # [
    #     "20260115-174614_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260115-174154_sup_seedpen_model_900",
    #     "20260115-180115_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260115-174154_sup_seedpen_model_900",
    #     "20260115-181603_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260115-174154_sup_seedpen_model_900",
    #     "20260115-183054_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260115-174154_sup_seedpen_model_900",
    # ],  # 0.5 100.0
    [
        "20260117-203130_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000_finetune_20260115-174154_sup_seedpen_model_900",
        "20260117-203130_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000_finetune_20260115-174154_sup_seedpen_model_900",
        "20260117-204610_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000_finetune_20260115-174154_sup_seedpen_model_900",
        "20260117-204622_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000_finetune_20260115-174154_sup_seedpen_model_900",
    ],  # 0.5 100.0
    #     [
    #         "20260118-051412_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000_finetune_20260115-174154_sup_seedpen_model_990",
    # "20260118-051957_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000_finetune_20260115-174154_sup_seedpen_model_990",
    # "20260118-060930_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000_finetune_20260115-174154_sup_seedpen_model_990",
    # "20260118-061932_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000_finetune_20260115-174154_sup_seedpen_model_990",
    #     ], #0.5
    # [
    #     "20260115-153755_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260115-145229_sup_seedpen_model_630",
    #     "20260115-153756_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260115-145229_sup_seedpen_model_630",
    #     "20260115-154949_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260115-145229_sup_seedpen_model_630",
    #     "20260115-154950_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260115-145229_sup_seedpen_model_630",
    # ],  # 1.5 100.0
    # [
    #     "20260118-143534_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000_finetune_20260115-145229_sup_seedpen_model_990",
    # "20260118-143534_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000_finetune_20260115-145229_sup_seedpen_model_990",
    # "20260118-144908_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000_finetune_20260115-145229_sup_seedpen_model_990",
    # "20260118-144911_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000_finetune_20260115-145229_sup_seedpen_model_990",
    # ],
    [
        "20260117-211630_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000_finetune_20260115-145229_sup_seedpen_model_630",
        "20260117-212827_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000_finetune_20260115-145229_sup_seedpen_model_630",
        "20260117-214027_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000_finetune_20260115-145229_sup_seedpen_model_630",
        "20260117-215225_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000_finetune_20260115-145229_sup_seedpen_model_630",
    ],  # 1.5 100.0
    # [
    #     "20260115-190256_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260115-184954_sup_seedpen_model_430",
    #     "20260115-192056_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260115-184954_sup_seedpen_model_430",
    #     "20260115-193354_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260115-184954_sup_seedpen_model_430",
    #     "20260115-201054_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260115-184954_sup_seedpen_model_430",
    # ],  # 2.0
    [
        "20260117-211630_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184954_sup_seedpen_model_430",
        "20260117-213054_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184954_sup_seedpen_model_430",
        "20260117-214509_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184954_sup_seedpen_model_430",
        "20260117-215918_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184954_sup_seedpen_model_430",
    ],  # 2.0
    #     [
    #         "20260118-050045_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184954_sup_seedpen_model_990",
    # "20260118-050530_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184954_sup_seedpen_model_990",
    # "20260118-055607_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184954_sup_seedpen_model_990",
    # "20260118-060443_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184954_sup_seedpen_model_990",
    #     ], # 2.0
    # [
    #     "20260115-190316_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260115-184556_sup_seedpen_model_150",
    #     "20260115-193658_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260115-184556_sup_seedpen_model_150",
    #     "20260115-194926_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260115-184556_sup_seedpen_model_150",
    #     "20260115-202821_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260115-184556_sup_seedpen_model_150",
    # ],  # 3.0 100.0
    # [
    #     "20260118-023637_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_990",
    #     "20260118-025007_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_990",
    #     "20260118-030333_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_990",
    #     "20260118-031656_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_990",
    # ],
    #     [
    # "20260118-044720_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_990",
    # "20260118-045108_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_990",
    # "20260118-054241_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_990",
    # "20260118-055022_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_990",
    #     ], # 3.0
    [
        "20260117-185928_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_150",
        "20260117-185930_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_150",
        "20260117-191324_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_150",
        "20260117-191337_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_150",
    ],  # 3.0
    # [
    #     "20260115-191756_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260115-184556_sup_seedpen_model_400",
    #     "20260115-195323_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260115-184556_sup_seedpen_model_400",
    #     "20260115-200427_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260115-184556_sup_seedpen_model_400",
    #     "20260115-204303_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260115-184556_sup_seedpen_model_400",
    # ],  # 4.0 100.0
    [
        "20260117-200230_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_400",
        "20260117-200230_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_400",
        "20260117-201630_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_400",
        "20260117-201644_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_400",
    ],  # 4.0
    # [
    #     "20260118-043331_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_990",
    #     "20260118-043649_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_990",
    #     "20260118-052857_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_990",
    #     "20260118-053539_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_990",
    # ], # 4.0
    [
        "20260117-174228_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000_finetune_20260116-173042_sup_seedpen_model_430",
        "20260117-174228_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000_finetune_20260116-173042_sup_seedpen_model_430",
        "20260117-175720_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000_finetune_20260116-173042_sup_seedpen_model_430",
        "20260117-175809_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000_finetune_20260116-173042_sup_seedpen_model_430",
    ],  # 10.0
    # [
    #     "20260118-150337_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000_finetune_20260116-173042_sup_seedpen_model_990",
    # "20260118-150337_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000_finetune_20260116-173042_sup_seedpen_model_990",
    # "20260118-151724_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000_finetune_20260116-173042_sup_seedpen_model_990",
    # "20260118-151812_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000_finetune_20260116-173042_sup_seedpen_model_990",
    # ],
])  # matrix of shape (num_ckpt, num_seeds)

fsnet_ft_sup_pen_w100_ratio05_trainsize7000_over_nlabel = np.array([
    [
        "20260115-012027_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000",
        "20260115-013345_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000",
        "20260115-012025_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000",
        "20260115-013254_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000",
    ],
    [
        "20260116-031253_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260116-022657_sup_seedpen_model_970",
        "20260116-034002_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260116-022657_sup_seedpen_model_970",
        "20260116-041155_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260116-022657_sup_seedpen_model_970",
        "20260116-043620_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260116-022657_sup_seedpen_model_970",
    ],  # 50
    [
        "20260116-032612_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260116-022757_sup_seedpen_model_780",
        "20260116-035349_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260116-022757_sup_seedpen_model_780",
        "20260116-042406_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260116-022757_sup_seedpen_model_780",
        "20260116-044833_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260116-022757_sup_seedpen_model_780",
    ],  # 200
    [
        "20260116-031254_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260116-022657_sup_seedpen_model_940",
        "20260116-034156_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260116-022657_sup_seedpen_model_940",
        "20260116-040759_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260116-022657_sup_seedpen_model_940",
        "20260116-043934_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260116-022657_sup_seedpen_model_940",
    ],  # 800
    [
        "20260116-032722_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260116-022847_sup_seedpen_model_920",
        "20260116-035634_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260116-022847_sup_seedpen_model_920",
        "20260116-042343_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260116-022847_sup_seedpen_model_920",
        "20260116-045512_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260116-022847_sup_seedpen_model_920",
    ],  # 3000
    [
        "20260117-203130_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000_finetune_20260115-174154_sup_seedpen_model_900",
        "20260117-203130_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000_finetune_20260115-174154_sup_seedpen_model_900",
        "20260117-204610_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000_finetune_20260115-174154_sup_seedpen_model_900",
        "20260117-204622_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000_finetune_20260115-174154_sup_seedpen_model_900",
    ],
])  # matrix of shape (num_ckpt, num_seeds)

fsnet_ft_sup_pen_w100_ratio05_trainsize7000_over_ckpt = np.array([
    [
        "20260115-012027_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000",
        "20260115-013345_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000",
        "20260115-012025_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000",
        "20260115-013254_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000",
    ],
    # [
    #     "20260116-141744_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260116-122715_sup_seedpen_model_4990",
    #     "20260116-142658_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260116-122715_sup_seedpen_model_4990",
    #     "20260116-143218_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260116-122715_sup_seedpen_model_4990",
    #     "20260116-144136_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260116-122715_sup_seedpen_model_4990",
    # ],
    # [
    #     "20260116-065548_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260116-061857_sup_seedpen_model_4990",
    #     "20260116-065632_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260116-061857_sup_seedpen_model_4990",
    #     "20260116-072437_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260116-061857_sup_seedpen_model_4990",
    #     "20260116-072507_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260116-061857_sup_seedpen_model_4990",
    # ],
    # [
    #     "20260116-153100_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260116-151545_sup_seedpen_model_4990",
    #     "20260116-154643_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260116-151545_sup_seedpen_model_4990",
    #     "20260116-160153_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260116-151545_sup_seedpen_model_4990",
    #     "20260116-161648_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260116-151545_sup_seedpen_model_4990",
    # ],
    # [
    #     "20260116-123509_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260116-122715_sup_seedpen_model_2580",
    #     "20260116-124950_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260116-122715_sup_seedpen_model_2580",
    #     "20260116-130432_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260116-122715_sup_seedpen_model_2580",
    #     "20260116-131913_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260116-122715_sup_seedpen_model_2580",
    # ],
    # [
    #     "20260116-064002_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260116-061857_sup_seedpen_model_2170",
    #     "20260116-064055_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260116-061857_sup_seedpen_model_2170",
    #     "20260116-070856_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260116-061857_sup_seedpen_model_2170",
    #     "20260116-070932_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260116-061857_sup_seedpen_model_2170",
    # ]
    # [
    #     "20260116-153100_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260116-151545_sup_seedpen_model_980",
    #     "20260116-154546_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260116-151545_sup_seedpen_model_980",
    #     "20260116-160033_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260116-151545_sup_seedpen_model_980",
    #     "20260116-161522_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260116-151545_sup_seedpen_model_980",
    # ],
    [
        "20260118-023637_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_990",
        "20260118-025007_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_990",
        "20260118-030333_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_990",
        "20260118-031656_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_990",
    ],
    # [
    #     "20260117-185928_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_150",
    #     "20260117-185930_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_150",
    #     "20260117-191324_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_150",
    #     "20260117-191337_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_150",
    # ],  # 3.0 100.0

])  # matrix of shape (num_ckpt, num_seeds)

fsnet_ft_sup_pen_w100_ratio05_trainsize7000_over_ckpt2 = np.array([
    [
        "20260115-012027_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000",
        "20260115-013345_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000",
        "20260115-012025_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000",
        "20260115-013254_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000",
    ],
    [
        "20260118-033729_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000_finetune_20260118-032632_sup_seedpen_model_1990",
        "20260118-033731_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000_finetune_20260118-032632_sup_seedpen_model_1990",
        "20260118-040558_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000_finetune_20260118-032632_sup_seedpen_model_1990",
        "20260118-040703_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000_finetune_20260118-032632_sup_seedpen_model_1990",
    ],
    [
        "20260118-035147_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000_finetune_20260118-032632_sup_seedpen_model_120",
        "20260118-035151_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000_finetune_20260118-032632_sup_seedpen_model_120",
        "20260118-041928_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000_finetune_20260118-032632_sup_seedpen_model_120",
        "20260118-042119_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000_finetune_20260118-032632_sup_seedpen_model_120",
    ],

])  # matrix of shape (num_ckpt, num_seeds)

mse_pen_w100_ratio05_trainsize800 = np.array([
    [
        "20260116-022657_MLP_sup_pen_seed0_nepochs1000_lr0.0001_trainsize800_subopt_3_0.5",
        "20260116-150032_MLP_sup_pen_seed1_nepochs1000_lr0.0001_trainsize800_subopt_3_0.5",
        "20260116-150125_MLP_sup_pen_seed2_nepochs1000_lr0.0001_trainsize800_subopt_3_0.5",
        "20260116-150217_MLP_sup_pen_seed3_nepochs1000_lr0.0001_trainsize800_subopt_3_0.5",
    ],
    [
        "20260116-170915_MLP_sup_seed0_nepochs1000_lr0.0001_trainsize7000_subopt_3_10.0",
        "20260116-171222_MLP_sup_seed1_nepochs1000_lr0.0001_trainsize7000_subopt_3_10.0",
        "20260116-171528_MLP_sup_seed2_nepochs1000_lr0.0001_trainsize7000_subopt_3_10.0",
        "20260116-171834_MLP_sup_seed3_nepochs1000_lr0.0001_trainsize7000_subopt_3_10.0",
    ]
])

# how much do we need data (data weight)
main_ft_sup_pen_over_method = np.array([
    # [
    #     "20260118-172439_MLP_penalty_seed2_nepochs1000_lr0.0001_trainsize7000",
    #     "20260118-172757_MLP_penalty_seed3_nepochs1000_lr0.0001_trainsize7000",
    #     "20260118-174600_MLP_penalty_seed0_nepochs1000_lr0.0001_trainsize7000",
    #     "20260118-174916_MLP_penalty_seed1_nepochs1000_lr0.0001_trainsize7000",
    # ],
    # [
    #     "20260118-172633_MLP_penalty_seed2_nepochs1000_lr0.0001_trainsize7000_finetune_20260116-022657_sup_seedpen_model_940",
    #     "20260118-172946_MLP_penalty_seed3_nepochs1000_lr0.0001_trainsize7000_finetune_20260116-022657_sup_seedpen_model_940",
    #     "20260118-174602_MLP_penalty_seed0_nepochs1000_lr0.0001_trainsize7000_finetune_20260116-022657_sup_seedpen_model_940",
    #     "20260118-174918_MLP_penalty_seed1_nepochs1000_lr0.0001_trainsize7000_finetune_20260116-022657_sup_seedpen_model_940",
    # ],
    # [
    #     "20260118-175812_MLP_adaptive_penalty_seed0_nepochs1000_lr0.0001_trainsize7000",
    #     "20260118-180127_MLP_adaptive_penalty_seed1_nepochs1000_lr0.0001_trainsize7000",
    #     "20260118-180441_MLP_adaptive_penalty_seed2_nepochs1000_lr0.0001_trainsize7000",
    #     "20260118-180755_MLP_adaptive_penalty_seed3_nepochs1000_lr0.0001_trainsize7000",
    # ],
    # [
    #     "20260118-175817_MLP_adaptive_penalty_seed0_nepochs1000_lr0.0001_trainsize7000_finetune_20260116-022657_sup_seedpen_model_940",
    #     "20260118-180136_MLP_adaptive_penalty_seed1_nepochs1000_lr0.0001_trainsize7000_finetune_20260116-022657_sup_seedpen_model_940",
    #     "20260118-180454_MLP_adaptive_penalty_seed2_nepochs1000_lr0.0001_trainsize7000_finetune_20260116-022657_sup_seedpen_model_940",
    #     "20260118-180813_MLP_adaptive_penalty_seed3_nepochs1000_lr0.0001_trainsize7000_finetune_20260116-022657_sup_seedpen_model_940",
    # ],
    # [
    #     "20260119-151521_MLP_adaptive_penalty_seed3_nepochs1000_lr0.0001_trainsize7000_finetune_20260116-022657_sup_seedpen_model_200",
    #     "20260119-151925_MLP_adaptive_penalty_seed0_nepochs1000_lr0.0001_trainsize7000_finetune_20260116-022657_sup_seedpen_model_200",
    #     "20260119-151935_MLP_adaptive_penalty_seed2_nepochs1000_lr0.0001_trainsize7000_finetune_20260116-022657_sup_seedpen_model_200",
    #     "20260119-152239_MLP_adaptive_penalty_seed1_nepochs1000_lr0.0001_trainsize7000_finetune_20260116-022657_sup_seedpen_model_200",
    # ],
    # [
    #     "20260119-150847_MLP_adaptive_penalty_seed1_nepochs1000_lr0.00015_trainsize7000_finetune_20260116-022657_sup_seedpen_model_200",
    #     "20260119-150631_MLP_adaptive_penalty_seed2_nepochs1000_lr0.00015_trainsize7000_finetune_20260116-022657_sup_seedpen_model_200",
    #     "20260119-150035_MLP_adaptive_penalty_seed3_nepochs1000_lr0.00015_trainsize7000_finetune_20260116-022657_sup_seedpen_model_200",
    #     "20260119-150532_MLP_adaptive_penalty_seed0_nepochs1000_lr0.00015_trainsize7000_finetune_20260116-022657_sup_seedpen_model_200"
    # ],
    # [
    #     "20260119-155724_MLP_adaptive_penalty_seed3_nepochs1000_lr0.00013_trainsize7000_finetune_20260116-022657_sup_seedpen_model_200",
    #     "20260119-170031_MLP_adaptive_penalty_seed0_nepochs1000_lr0.00013_trainsize7000_finetune_20260116-022657_sup_seedpen_model_200",
    #     "20260119-170045_MLP_adaptive_penalty_seed2_nepochs1000_lr0.00013_trainsize7000_finetune_20260116-022657_sup_seedpen_model_200",
    #     "20260119-170348_MLP_adaptive_penalty_seed1_nepochs1000_lr0.00013_trainsize7000_finetune_20260116-022657_sup_seedpen_model_200",
    # ],
    #     [
    #         "20260118-225115_MLP_DC3_seed3_nepochs1000_lr0.0001_trainsize7000",
    #         "20260118-230024_MLP_DC3_seed2_nepochs1000_lr0.0001_trainsize7000",
    #         "20260118-222424_MLP_DC3_seed0_nepochs1000_lr0.0001_trainsize7000",
    #         "20260118-223637_MLP_DC3_seed1_nepochs1000_lr0.0001_trainsize7000",
    #     ],
    #     [
    #         "20260118-225653_MLP_DC3_seed2_nepochs1000_lr0.0001_trainsize7000_finetune_20260118-184700_sup_seedpartial_model_230",
    #         "20260118-230534_MLP_DC3_seed3_nepochs1000_lr0.0001_trainsize7000_finetune_20260118-184700_sup_seedpartial_model_230",
    #         "20260118-223717_MLP_DC3_seed0_nepochs1000_lr0.0001_trainsize7000_finetune_20260118-184700_sup_seedpartial_model_230",
    #         "20260118-224813_MLP_DC3_seed1_nepochs1000_lr0.0001_trainsize7000_finetune_20260118-184700_sup_seedpartial_model_230",
    #     ],
    #         [
    #         "20260119-005511_MLP_DC3_seed3_nepochs1000_lr0.0001_trainsize7000",
    #         "20260119-023841_MLP_DC3_seed1_nepochs1000_lr0.0001_trainsize7000",
    #         "20260119-005511_MLP_DC3_seed3_nepochs1000_lr0.0001_trainsize7000",
    #         "20260119-022157_MLP_DC3_seed0_nepochs1000_lr0.0001_trainsize7000",
    #     ],
    #     [
    # "20260119-014701_MLP_DC3_seed2_nepochs1000_lr5e-05_trainsize7000_finetune_20260118-184700_sup_seedpartial_model_230",
    # "20260119-021214_MLP_DC3_seed1_nepochs1000_lr5e-05_trainsize7000_finetune_20260118-184700_sup_seedpartial_model_230",
    # "20260119-020422_MLP_DC3_seed0_nepochs1000_lr5e-05_trainsize7000_finetune_20260118-184700_sup_seedpartial_model_230",
    # "20260119-023107_MLP_DC3_seed3_nepochs1000_lr5e-05_trainsize7000_finetune_20260118-184700_sup_seedpartial_model_230",
    # ]
])  # matrix of shape (num_ckpt, num_seeds)


train_with_merit = np.array([
    [
        "20260119-205612_MLP_FSNet_seed0_nepochs300_lr5e-05_trainsize7000",
        "20260119-204636_MLP_FSNet_seed3_nepochs300_lr5e-05_trainsize7000",
        "20260119-211051_MLP_FSNet_seed1_nepochs300_lr5e-05_trainsize7000",
        "20260119-211230_MLP_FSNet_seed2_nepochs300_lr5e-05_trainsize7000",
    ]
])

fsnet_pen = np.array([
    #     [
    #     "20260115-012027_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000",
    #     "20260115-013345_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000",
    #     "20260115-012025_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000",
    #     "20260115-013254_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000",
    # ],
    # [
    #     "20260115-022127_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260115-021622_penalty_seedseed0_model_990",
    #     "20260115-022527_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260115-021622_penalty_seedseed0_model_990",
    #     "20260115-023352_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260115-021622_penalty_seedseed0_model_990",
    #     "20260115-023737_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260115-021622_penalty_seedseed0_model_990",
    # ],  # 0.0 penalty
    # [
    #     "20260114-224213_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051731_penalty_seedseed1_model_990",
    #     "20260114-224213_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051731_penalty_seedseed1_model_990",
    #     "20260114-225544_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051731_penalty_seedseed1_model_990",
    #     "20260114-225547_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051731_penalty_seedseed1_model_990",
    # ],  # penalty
    # [
    #     "20260117-203130_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000_finetune_20260115-174154_sup_seedpen_model_900",
    #     "20260117-203130_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000_finetune_20260115-174154_sup_seedpen_model_900",
    #     "20260117-204610_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000_finetune_20260115-174154_sup_seedpen_model_900",
    #     "20260117-204622_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000_finetune_20260115-174154_sup_seedpen_model_900",
    # ],  # 0.5 100.0
    [
        "20260126-231016_MLP_semi_seed0_nepochs300_lr0.0001_trainsize7000_subopt_3_0.5",
        "20260126-231016_MLP_semi_seed2_nepochs300_lr0.0001_trainsize7000_subopt_3_0.5",
        "20260126-232302_MLP_semi_seed3_nepochs300_lr0.0001_trainsize7000_subopt_3_0.5",
        "20260126-232607_MLP_semi_seed1_nepochs300_lr0.0001_trainsize7000_subopt_3_0.5",
    ],  # semi-supervised
])