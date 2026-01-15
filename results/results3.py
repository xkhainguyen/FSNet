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
    ], # 0.0 penalty
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
    [
        "20260115-022127_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260115-021622_penalty_seedseed0_model_990",
        "20260115-022527_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260115-021622_penalty_seedseed0_model_990",
        "20260115-023352_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260115-021622_penalty_seedseed0_model_990",
        "20260115-023737_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260115-021622_penalty_seedseed0_model_990",
    ], # 0.0 penalty
    [
        "20260114-224213_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051731_penalty_seedseed1_model_990",
        "20260114-224213_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051731_penalty_seedseed1_model_990",
        "20260114-225544_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051731_penalty_seedseed1_model_990",
        "20260114-225547_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260114-051731_penalty_seedseed1_model_990",
    ],  # penalty
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
    ], # 0.0 penalty
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
    ], # 1.5 10.0
    [
        "20260115-153755_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260115-145229_sup_seedpen_model_630",
        "20260115-153756_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260115-145229_sup_seedpen_model_630",
        "20260115-154949_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260115-145229_sup_seedpen_model_630",
        "20260115-154950_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260115-145229_sup_seedpen_model_630",
    ] # 1.5 100.0
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
    ]  # 4.0
])  # matrix of shape (num_ckpt, num_seeds)


#######################################
#######################################

fsnet_ft_sup_subopt2_m10_trainsize7000_over_ckpt = np.array([
    [
        "20260107-024219_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000",
        "20260107-034140_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000",
        "20260107-032119_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000",
        "20260108-155633_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000",
    ],
    [
        "20260107-044919_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260106-202959_model_60",
        "20260107-050307_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260106-202959_model_60",
        "20260107-051636_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260106-202959_model_60",
        "20260108-003033_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260106-202959_model_60",
    ],
    [
        "20260107-053006_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260106-202959_model_400",
        "20260107-054450_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260106-202959_model_400",
        "20260107-055920_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260106-202959_model_400",
        "20260108-004411_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260106-202959_model_400",
    ],
])  # matrix of shape (num_ckpt, num_seeds)

fsnet_ft_sup_trainsize7000_over_ckpt = np.array([
    [
        "20260107-024219_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000",
        "20260107-034140_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000",
        "20260107-032119_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000",
        "20260108-155633_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000",
    ],
    [
        "20260107-044819_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260107-043823_model_70",
        "20260107-050629_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260107-043823_model_70",
        "20260107-052110_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260107-043823_model_70",
        "20260108-005832_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260107-043823_model_70",
    ],
    [
        "20260107-053746_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260107-043823_model_400",
        "20260107-055248_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260107-043823_model_400",
        "20260107-060833_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260107-043823_model_400",
        "20260108-011202_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260107-043823_model_400",
    ],
])  # matrix of shape (num_ckpt, num_seeds)

fsnet_ft_sup_pen_subopt2_1_trainsize7000_over_ckpt = np.array([
    [
        "20260107-024219_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000",
        "20260107-034140_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000",
        "20260107-032119_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000",
        "20260108-155633_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000",
    ],
    [
        "20260108-014627_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260108-014039_model_230",
        "20260108-020001_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260108-014039_model_230",
        "20260108-021338_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260108-014039_model_230",
        "20260108-022723_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260108-014039_model_230",
    ],
])  # matrix of shape (num_ckpt, num_seeds)

######################################
# Train size 1000

fsnet_ft_sup_subopt2_1_trainsize1000_over_ckpt = np.array([
    [
        "20260107-024219_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000",
        "20260107-034140_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000",
        "20260107-032119_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000",
        "20260108-155633_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000",
    ],
    [
        "20260108-041148_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260108-031328_model_280",
        "20260108-042644_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260108-031328_model_280",
        "20260108-044204_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260108-031328_model_280",
        "20260108-045711_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260108-031328_model_280",
    ],
])  # matrix of shape (num_ckpt, num_seeds)

fsnet_ft_sup_subopt2_1_trainsize1000_over_ckpt = np.array([
    [
        "20260107-024219_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000",
        "20260107-034140_MLP_FSNet_seed1_nepochs300_lr0.0001_trainsize7000",
        "20260107-032119_MLP_FSNet_seed2_nepochs300_lr0.0001_trainsize7000",
        "20260108-155633_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000",
    ],
    [
        "20260108-041148_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260108-031328_model_280",
        "20260108-042644_MLP_FSNet_seed1_nepochs300_lr0.0002_trainsize7000_finetune_20260108-031328_model_280",
        "20260108-044204_MLP_FSNet_seed2_nepochs300_lr0.0002_trainsize7000_finetune_20260108-031328_model_280",
        "20260108-045711_MLP_FSNet_seed3_nepochs300_lr0.0002_trainsize7000_finetune_20260108-031328_model_280",
    ],
])  # matrix of shape (num_ckpt, num_seeds)
