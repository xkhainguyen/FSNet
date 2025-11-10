import numpy as np

fsnet_ft_ckpt20_subopt20_over_train_size = np.array([
    [
        "20251103-103923_MLP_FSNet_seed0_dropout0.1_finetune_20251103-103706_model_20",
        "20251103-110520_MLP_FSNet_seed1_dropout0.1_finetune_20251103-103706_model_20",
        "20251103-113323_MLP_FSNet_seed2_dropout0.1_finetune_20251103-103706_model_20",
        "20251103-115837_MLP_FSNet_seed3_dropout0.1_finetune_20251103-103706_model_20",
    ], # 10
    [
        "20251106-114900_MLP_FSNet_seed0_dropout0.1_finetune_20251106-114016_model_20",
        "20251106-145546_MLP_FSNet_seed1_dropout0.1_finetune_20251106-114016_model_20",
        "20251106-153706_MLP_FSNet_seed2_dropout0.1_finetune_20251106-114016_model_20",
        "20251106-161755_MLP_FSNet_seed3_dropout0.1_finetune_20251106-114016_model_20",
    ], # 20
    [
        "20251106-120534_MLP_FSNet_seed0_dropout0.1_finetune_20251106-114103_model_20",
        "20251106-151130_MLP_FSNet_seed1_dropout0.1_finetune_20251106-114103_model_20",
        "20251106-155154_MLP_FSNet_seed2_dropout0.1_finetune_20251106-114103_model_20",
        "20251106-163305_MLP_FSNet_seed3_dropout0.1_finetune_20251106-114103_model_20",
    ], # 30
    [
        "20251106-121915_MLP_FSNet_seed0_dropout0.1_finetune_20251106-114138_model_20",
        "20251106-152422_MLP_FSNet_seed1_dropout0.1_finetune_20251106-114138_model_20",
        "20251106-160452_MLP_FSNet_seed2_dropout0.1_finetune_20251106-114138_model_20",
        "20251106-164540_MLP_FSNet_seed3_dropout0.1_finetune_20251106-114138_model_20",
    ], # 40
    [
        "20251103-105147_MLP_FSNet_seed0_dropout0.1_finetune_20251103-103719_model_20",
        "20251103-112026_MLP_FSNet_seed1_dropout0.1_finetune_20251103-103719_model_20",
        "20251103-114546_MLP_FSNet_seed2_dropout0.1_finetune_20251103-103719_model_20",
        "20251103-115906_MLP_FSNet_seed3_dropout0.1_finetune_20251103-103719_model_20",
    ], # 50
    [
        "20251101-234017_MLP_FSNet_seed0_dropout0.1_finetune_20251101-233217_model_20",
        "20251102-003328_MLP_FSNet_seed1_dropout0.1_finetune_20251101-233217_model_20",
        "20251102-012534_MLP_FSNet_seed2_dropout0.1_finetune_20251101-233217_model_20",
        "20251102-012156_MLP_FSNet_seed3_dropout0.1_finetune_20251101-233217_model_20",
    ], # 200
    [
        "20251101-235251_MLP_FSNet_seed0_dropout0.1_finetune_20251101-233236_model_20",
        "20251102-004558_MLP_FSNet_seed1_dropout0.1_finetune_20251101-233236_model_20",
        "20251102-013943_MLP_FSNet_seed2_dropout0.1_finetune_20251101-233236_model_20",
        "20251102-013453_MLP_FSNet_seed3_dropout0.1_finetune_20251101-233236_model_20",
    ], # 500
    [
        "20251102-000639_MLP_FSNet_seed0_dropout0.1_finetune_20251101-233303_model_20",
        "20251102-005943_MLP_FSNet_seed1_dropout0.1_finetune_20251101-233303_model_20",
        "20251102-015453_MLP_FSNet_seed2_dropout0.1_finetune_20251101-233303_model_20",
        "20251102-014842_MLP_FSNet_seed3_dropout0.1_finetune_20251101-233303_model_20",
    ], # 1000
    [
        "20251102-001926_MLP_FSNet_seed0_dropout0.1_finetune_20251101-233338_model_20",
        "20251102-011248_MLP_FSNet_seed1_dropout0.1_finetune_20251101-233338_model_20",
        "20251102-010834_MLP_FSNet_seed2_dropout0.1_finetune_20251101-233338_model_20",
        "20251102-020113_MLP_FSNet_seed3_dropout0.1_finetune_20251101-233338_model_20",
    ], # 4000
    [
        "20251031-130001_MLP_FSNet_seed0_dropout0.1_finetune_20251031-073026_model_20",
        "20251031-133322_MLP_FSNet_seed1_dropout0.1_finetune_20251031-073026_model_20",
        "20251031-140451_MLP_FSNet_seed2_dropout0.1_finetune_20251031-073026_model_20",
        "20251031-143652_MLP_FSNet_seed3_dropout0.1_finetune_20251031-073026_model_20",
    ], # ckpt20_subopt20_train_size7000
    [
        "20251030-234500_MLP_FSNet_seed0_dropout0.1_finetune_20251030-232939_model_20",
        "20251031-005543_MLP_FSNet_seed1_dropout0.1_finetune_20251030-232939_model_20",
        "20251031-011200_MLP_FSNet_seed2_dropout0.1_finetune_20251030-232939_model_20",
        "20251031-012540_MLP_FSNet_seed3_dropout0.1_finetune_20251030-232939_model_20",
    ], # ckpt20_subopt00_train_size7000
    [
        "20251031-203828_MLP_FSNet_seed0_dropout0.1",
        "20251031-205035_MLP_FSNet_seed1_dropout0.1",
        "20251031-210247_MLP_FSNet_seed2_dropout0.1",
        "20251031-211532_MLP_FSNet_seed3_dropout0.1"
    ],
]) # matrix of shape (num_ckpt, num_seeds)

# Finetune checkpoint 20 with suboptimality 0.0 over different dataset sizes 
fsnet_ft_ckpt20_subopt00_over_train_size = np.array([
    [
        "20251031-124440_MLP_FSNet_seed0_dropout0.1_finetune_20251031-075451_model_20",
        "20251031-133706_MLP_FSNet_seed1_dropout0.1_finetune_20251031-075451_model_20",
        "20251031-143142_MLP_FSNet_seed2_dropout0.1_finetune_20251031-075451_model_20",
        "20251031-153128_MLP_FSNet_seed3_dropout0.1_finetune_20251031-075451_model_20",
    ],
    [
        "20251031-125708_MLP_FSNet_seed0_dropout0.1_finetune_20251031-074829_model_20",
        "20251031-134943_MLP_FSNet_seed1_dropout0.1_finetune_20251031-074829_model_20",
        "20251031-144555_MLP_FSNet_seed2_dropout0.1_finetune_20251031-074829_model_20",
        "20251031-154355_MLP_FSNet_seed3_dropout0.1_finetune_20251031-074829_model_20",
    ], #500
    [
        "20251031-130941_MLP_FSNet_seed0_dropout0.1_finetune_20251031-074856_model_20",
        "20251031-140212_MLP_FSNet_seed1_dropout0.1_finetune_20251031-074856_model_20",
        "20251031-150350_MLP_FSNet_seed2_dropout0.1_finetune_20251031-074856_model_20",
        "20251031-155732_MLP_FSNet_seed3_dropout0.1_finetune_20251031-074856_model_20",
    ],
    [
        "20251031-132213_MLP_FSNet_seed0_dropout0.1_finetune_20251031-074930_model_20",
        "20251031-141608_MLP_FSNet_seed1_dropout0.1_finetune_20251031-074930_model_20",
        "20251031-151626_MLP_FSNet_seed2_dropout0.1_finetune_20251031-074930_model_20",
        "20251031-161024_MLP_FSNet_seed3_dropout0.1_finetune_20251031-074930_model_20"
    ],
    [
        "20251030-234500_MLP_FSNet_seed0_dropout0.1_finetune_20251030-232939_model_20",
        "20251031-005543_MLP_FSNet_seed1_dropout0.1_finetune_20251030-232939_model_20",
        "20251031-011200_MLP_FSNet_seed2_dropout0.1_finetune_20251030-232939_model_20",
        "20251031-012540_MLP_FSNet_seed3_dropout0.1_finetune_20251030-232939_model_20",
    ],
    [
        "20251031-203828_MLP_FSNet_seed0_dropout0.1",
        "20251031-205035_MLP_FSNet_seed1_dropout0.1",
        "20251031-210247_MLP_FSNet_seed2_dropout0.1",
        "20251031-211532_MLP_FSNet_seed3_dropout0.1"
    ],
]) # matrix of shape (num_ckpt, num_seeds)

# Finetune checkpoint 20 of different suboptimalities
fsnet_ft_ckpt20_train_size7000_over_subopt = np.array([
    [
        "20251031-203828_MLP_FSNet_seed0_dropout0.1",
        "20251031-205035_MLP_FSNet_seed1_dropout0.1",
        "20251031-210247_MLP_FSNet_seed2_dropout0.1",
        "20251031-211532_MLP_FSNet_seed3_dropout0.1"
    ], # baseline
    [
        "20251030-234500_MLP_FSNet_seed0_dropout0.1_finetune_20251030-232939_model_20",
        "20251031-005543_MLP_FSNet_seed1_dropout0.1_finetune_20251030-232939_model_20",
        "20251031-011200_MLP_FSNet_seed2_dropout0.1_finetune_20251030-232939_model_20",
        "20251031-012540_MLP_FSNet_seed3_dropout0.1_finetune_20251030-232939_model_20",
    ], # 0.0
    [
        "20251030-234745_MLP_FSNet_seed0_dropout0.1_finetune_20251030-233223_model_20",
        "20251031-005527_MLP_FSNet_seed1_dropout0.1_finetune_20251030-233223_model_20",
        "20251031-011053_MLP_FSNet_seed2_dropout0.1_finetune_20251030-233223_model_20",
        "20251031-012437_MLP_FSNet_seed3_dropout0.1_finetune_20251030-233223_model_20",
    ], # 0.5
    [
        "20251031-124515_MLP_FSNet_seed0_dropout0.1_finetune_20251030-233258_model_20",
        "20251031-131637_MLP_FSNet_seed1_dropout0.1_finetune_20251030-233258_model_20",
        "20251031-134843_MLP_FSNet_seed2_dropout0.1_finetune_20251030-233258_model_20",
        "20251031-142108_MLP_FSNet_seed3_dropout0.1_finetune_20251030-233258_model_20",
    ], # 1.0
    [
        "20251031-130001_MLP_FSNet_seed0_dropout0.1_finetune_20251031-073026_model_20",
        "20251031-133322_MLP_FSNet_seed1_dropout0.1_finetune_20251031-073026_model_20",
        "20251031-140451_MLP_FSNet_seed2_dropout0.1_finetune_20251031-073026_model_20",
        "20251031-143652_MLP_FSNet_seed3_dropout0.1_finetune_20251031-073026_model_20",
    ], # 2.0
    [
        "20251101-004609_MLP_FSNet_seed0_dropout0.1_finetune_20251101-002602_model_20",
        "20251101-012021_MLP_FSNet_seed1_dropout0.1_finetune_20251101-002602_model_20",
        "20251101-015622_MLP_FSNet_seed2_dropout0.1_finetune_20251101-002602_model_20",
        "20251101-023133_MLP_FSNet_seed3_dropout0.1_finetune_20251101-002602_model_20",
    ], # 4.0
    [
        "20251101-232533_MLP_FSNet_seed0_dropout0.1_finetune_20251101-232010_model_20",
        "20251101-234224_MLP_FSNet_seed1_dropout0.1_finetune_20251101-232010_model_20",
        "20251102-000057_MLP_FSNet_seed2_dropout0.1_finetune_20251101-232010_model_20",
        "20251102-001948_MLP_FSNet_seed3_dropout0.1_finetune_20251101-232010_model_20",
    ],
    [
        "20251101-010011_MLP_FSNet_seed0_dropout0.1_finetune_20251101-003437_model_20",
        "20251101-013354_MLP_FSNet_seed1_dropout0.1_finetune_20251101-003437_model_20",
        "20251101-021048_MLP_FSNet_seed2_dropout0.1_finetune_20251101-003437_model_20",
        "20251101-024556_MLP_FSNet_seed3_dropout0.1_finetune_20251101-003437_model_20",
    ], # 10.0       
]) # matrix of shape (num_ckpt, num_seeds)

# Suboptimality 10.0 at different checkpoints
fsnet_ft_subopt100_train_size7000_over_ckpt = np.array([
    [
        "20251031-203828_MLP_FSNet_seed0_dropout0.1",
        "20251031-205035_MLP_FSNet_seed1_dropout0.1",
        "20251031-210247_MLP_FSNet_seed2_dropout0.1",
        "20251031-211532_MLP_FSNet_seed3_dropout0.1",
    ],
    [
        "20251101-010011_MLP_FSNet_seed0_dropout0.1_finetune_20251101-003437_model_20",
        "20251101-013354_MLP_FSNet_seed1_dropout0.1_finetune_20251101-003437_model_20",
        "20251101-021048_MLP_FSNet_seed2_dropout0.1_finetune_20251101-003437_model_20",
        "20251101-024556_MLP_FSNet_seed3_dropout0.1_finetune_20251101-003437_model_20",
    ],
    [
        "20251101-032818_MLP_FSNet_seed0_dropout0.1_finetune_20251101-003437_model_60",
        "20251101-040830_MLP_FSNet_seed1_dropout0.1_finetune_20251101-003437_model_60",
        "20251101-044648_MLP_FSNet_seed2_dropout0.1_finetune_20251101-003437_model_60",
        "20251101-052807_MLP_FSNet_seed3_dropout0.1_finetune_20251101-003437_model_60",
    ],
]) # matrix of shape (num_ckpt, num_seeds)

# Suboptimality 6.0 at different checkpoints
fsnet_ft_subopt60_train_size7000_over_ckpt = np.array([
    [
        "20251031-203828_MLP_FSNet_seed0_dropout0.1",
        "20251031-205035_MLP_FSNet_seed1_dropout0.1",
        "20251031-210247_MLP_FSNet_seed2_dropout0.1",
        "20251031-211532_MLP_FSNet_seed3_dropout0.1",
    ],
    [
        "20251101-232533_MLP_FSNet_seed0_dropout0.1_finetune_20251101-232010_model_20",
        "20251101-234224_MLP_FSNet_seed1_dropout0.1_finetune_20251101-232010_model_20",
        "20251102-000057_MLP_FSNet_seed2_dropout0.1_finetune_20251101-232010_model_20",
        "20251102-001948_MLP_FSNet_seed3_dropout0.1_finetune_20251101-232010_model_20",
    ],
    [
        "20251102-003834_MLP_FSNet_seed0_dropout0.1_finetune_20251101-232010_model_60",
        "20251102-005853_MLP_FSNet_seed1_dropout0.1_finetune_20251101-232010_model_60",
        "20251102-011918_MLP_FSNet_seed2_dropout0.1_finetune_20251101-232010_model_60",
        "20251102-013941_MLP_FSNet_seed3_dropout0.1_finetune_20251101-232010_model_60",
    ],
]) # matrix of shape (num_ckpt, num_seeds)

# Suboptimality 4.0 at different checkpoints
fsnet_ft_subopt40_train_size7000_over_ckpt = np.array([
    [
        "20251031-203828_MLP_FSNet_seed0_dropout0.1",
        "20251031-205035_MLP_FSNet_seed1_dropout0.1",
        "20251031-210247_MLP_FSNet_seed2_dropout0.1",
        "20251031-211532_MLP_FSNet_seed3_dropout0.1"
    ],
    [
        "20251101-004609_MLP_FSNet_seed0_dropout0.1_finetune_20251101-002602_model_20",
        "20251101-012021_MLP_FSNet_seed1_dropout0.1_finetune_20251101-002602_model_20",
        "20251101-015622_MLP_FSNet_seed2_dropout0.1_finetune_20251101-002602_model_20",
        "20251101-023133_MLP_FSNet_seed3_dropout0.1_finetune_20251101-002602_model_20",
    ],
    [
        "20251101-030629_MLP_FSNet_seed0_dropout0.1_finetune_20251101-002602_model_60",
        "20251101-034907_MLP_FSNet_seed1_dropout0.1_finetune_20251101-002602_model_60",
        "20251101-042726_MLP_FSNet_seed2_dropout0.1_finetune_20251101-002602_model_60",
        "20251101-050812_MLP_FSNet_seed3_dropout0.1_finetune_20251101-002602_model_60",
    ],
]) # matrix of shape (num_ckpt, num_seeds)

# Suboptimality 2.0 at different checkpoints
fsnet_ft_subopt20_train_size7000_over_ckpt = np.array([
    [
        "20251031-203828_MLP_FSNet_seed0_dropout0.1",
        "20251031-205035_MLP_FSNet_seed1_dropout0.1",
        "20251031-210247_MLP_FSNet_seed2_dropout0.1",
        "20251031-211532_MLP_FSNet_seed3_dropout0.1"
    ],
    [
        "20251031-130001_MLP_FSNet_seed0_dropout0.1_finetune_20251031-073026_model_20",
        "20251031-133322_MLP_FSNet_seed1_dropout0.1_finetune_20251031-073026_model_20",
        "20251031-140451_MLP_FSNet_seed2_dropout0.1_finetune_20251031-073026_model_20",
        "20251031-143652_MLP_FSNet_seed3_dropout0.1_finetune_20251031-073026_model_20",
    ],
    [
        "20251031-150659_MLP_FSNet_seed0_dropout0.1_finetune_20251031-073026_model_60",
        "20251031-154150_MLP_FSNet_seed1_dropout0.1_finetune_20251031-073026_model_60",
        "20251031-161550_MLP_FSNet_seed2_dropout0.1_finetune_20251031-073026_model_60",
        "20251031-164945_MLP_FSNet_seed3_dropout0.1_finetune_20251031-073026_model_60",
    ],
    [
        "20251031-172350_MLP_FSNet_seed0_dropout0.1_finetune_20251031-073026_model_100",
        "20251031-183433_MLP_FSNet_seed1_dropout0.1_finetune_20251031-073026_model_100",
        "20251031-190906_MLP_FSNet_seed2_dropout0.1_finetune_20251031-073026_model_100",
        "20251031-194329_MLP_FSNet_seed3_dropout0.1_finetune_20251031-073026_model_100",
    ],
    [
        "20251031-205728_MLP_FSNet_seed0_dropout0.1_finetune_20251031-073026_model_200",
        "20251031-213622_MLP_FSNet_seed1_dropout0.1_finetune_20251031-073026_model_200",
        "20251031-221450_MLP_FSNet_seed2_dropout0.1_finetune_20251031-073026_model_200",
        "20251031-225329_MLP_FSNet_seed3_dropout0.1_finetune_20251031-073026_model_200",
    ]
]) # matrix of shape (num_ckpt, num_seeds)

# Suboptimality 1.0 at different checkpoints
fsnet_ft_subopt10_train_size7000_over_ckpt = np.array([
    [
        "20251031-203828_MLP_FSNet_seed0_dropout0.1",
        "20251031-205035_MLP_FSNet_seed1_dropout0.1",
        "20251031-210247_MLP_FSNet_seed2_dropout0.1",
        "20251031-211532_MLP_FSNet_seed3_dropout0.1"
    ],
    [
        "20251031-124515_MLP_FSNet_seed0_dropout0.1_finetune_20251030-233258_model_20",
        "20251031-131637_MLP_FSNet_seed1_dropout0.1_finetune_20251030-233258_model_20",
        "20251031-134843_MLP_FSNet_seed2_dropout0.1_finetune_20251030-233258_model_20",
        "20251031-142108_MLP_FSNet_seed3_dropout0.1_finetune_20251030-233258_model_20",
    ],
    [
        "20251031-145207_MLP_FSNet_seed0_dropout0.1_finetune_20251030-233258_model_60",
        "20251031-152614_MLP_FSNet_seed1_dropout0.1_finetune_20251030-233258_model_60",
        "20251031-160107_MLP_FSNet_seed2_dropout0.1_finetune_20251030-233258_model_60",
        "20251031-163506_MLP_FSNet_seed3_dropout0.1_finetune_20251030-233258_model_60",
    ],
    [
        "20251031-170902_MLP_FSNet_seed0_dropout0.1_finetune_20251030-233258_model_100",
        "20251031-181937_MLP_FSNet_seed1_dropout0.1_finetune_20251030-233258_model_100",
        "20251031-185405_MLP_FSNet_seed2_dropout0.1_finetune_20251030-233258_model_100",
        "20251031-192839_MLP_FSNet_seed3_dropout0.1_finetune_20251030-233258_model_100",
    ],
    [
        "20251031-203756_MLP_FSNet_seed0_dropout0.1_finetune_20251030-233258_model_200",
        "20251031-211657_MLP_FSNet_seed1_dropout0.1_finetune_20251030-233258_model_200",
        "20251031-215556_MLP_FSNet_seed2_dropout0.1_finetune_20251030-233258_model_200",
        "20251031-223414_MLP_FSNet_seed3_dropout0.1_finetune_20251030-233258_model_200",
    ]
]) # matrix of shape (num_ckpt, num_seeds)

# Suboptimality 0.5 at different checkpoints
fsnet_ft_subopt05_train_size7000_over_ckpt = np.array([
    [
        "20251031-203828_MLP_FSNet_seed0_dropout0.1",
        "20251031-205035_MLP_FSNet_seed1_dropout0.1",
        "20251031-210247_MLP_FSNet_seed2_dropout0.1",
        "20251031-211532_MLP_FSNet_seed3_dropout0.1"
    ],
    [
        "20251030-234745_MLP_FSNet_seed0_dropout0.1_finetune_20251030-233223_model_20",
        "20251031-005527_MLP_FSNet_seed1_dropout0.1_finetune_20251030-233223_model_20",
        "20251031-011053_MLP_FSNet_seed2_dropout0.1_finetune_20251030-233223_model_20",
        "20251031-012437_MLP_FSNet_seed3_dropout0.1_finetune_20251030-233223_model_20",
    ],
    [
        "20251031-000347_MLP_FSNet_seed0_dropout0.1_finetune_20251030-233223_model_60",
        "20251031-014015_MLP_FSNet_seed1_dropout0.1_finetune_20251030-233223_model_60",
        "20251031-015555_MLP_FSNet_seed2_dropout0.1_finetune_20251030-233223_model_60",
        "20251031-021142_MLP_FSNet_seed3_dropout0.1_finetune_20251030-233223_model_60",
    ],
    [
        "20251031-001737_MLP_FSNet_seed0_dropout0.1_finetune_20251030-233223_model_100",
        "20251031-022558_MLP_FSNet_seed1_dropout0.1_finetune_20251030-233223_model_100",
        "20251031-024211_MLP_FSNet_seed2_dropout0.1_finetune_20251030-233223_model_100",
        "20251031-025820_MLP_FSNet_seed3_dropout0.1_finetune_20251030-233223_model_100",
    ],
    [
        "20251031-003411_MLP_FSNet_seed0_dropout0.1_finetune_20251030-233223_model_200",
        "20251031-031359_MLP_FSNet_seed1_dropout0.1_finetune_20251030-233223_model_200",
        "20251031-033136_MLP_FSNet_seed2_dropout0.1_finetune_20251030-233223_model_200",
        "20251031-034722_MLP_FSNet_seed3_dropout0.1_finetune_20251030-233223_model_200",
    ]
]) # matrix of shape (num_ckpt, num_seeds)

# Suboptimality 0.0 at different checkpoints
fsnet_ft_subopt00_train_size7000_over_ckpt = np.array([
    [
        "20251031-203828_MLP_FSNet_seed0_dropout0.1",
        "20251031-205035_MLP_FSNet_seed1_dropout0.1",
        "20251031-210247_MLP_FSNet_seed2_dropout0.1",
        "20251031-211532_MLP_FSNet_seed3_dropout0.1"
    ],
    [
        "20251030-234500_MLP_FSNet_seed0_dropout0.1_finetune_20251030-232939_model_20",
        "20251031-005543_MLP_FSNet_seed1_dropout0.1_finetune_20251030-232939_model_20",
        "20251031-011200_MLP_FSNet_seed2_dropout0.1_finetune_20251030-232939_model_20",
        "20251031-012540_MLP_FSNet_seed3_dropout0.1_finetune_20251030-232939_model_20",
    ],
    [
        "20251030-235837_MLP_FSNet_seed0_dropout0.1_finetune_20251030-232939_model_60",
        "20251031-014302_MLP_FSNet_seed1_dropout0.1_finetune_20251030-232939_model_60",
        "20251031-015852_MLP_FSNet_seed2_dropout0.1_finetune_20251030-232939_model_60",
        "20251031-021430_MLP_FSNet_seed3_dropout0.1_finetune_20251030-232939_model_60",
    ],
    [
        "20251031-001414_MLP_FSNet_seed0_dropout0.1_finetune_20251030-232939_model_100",
        "20251031-022921_MLP_FSNet_seed1_dropout0.1_finetune_20251030-232939_model_100",
        "20251031-024552_MLP_FSNet_seed2_dropout0.1_finetune_20251030-232939_model_100",
        "20251031-030129_MLP_FSNet_seed3_dropout0.1_finetune_20251030-232939_model_100",
    ],
    [
        "20251031-003458_MLP_FSNet_seed0_dropout0.1_finetune_20251030-232939_model_200",
        "20251031-031716_MLP_FSNet_seed1_dropout0.1_finetune_20251030-232939_model_200",
        "20251031-033222_MLP_FSNet_seed2_dropout0.1_finetune_20251030-232939_model_200",
        "20251031-034823_MLP_FSNet_seed3_dropout0.1_finetune_20251030-232939_model_200",
    ]
]) # matrix of shape (num_ckpt, num_seeds)

##################################
###### DC3 #######################
##################################

# dc3_ft_subopt100_train_size7000_over_ckpt = np.array([
    # [
    #     "20251107-110210_MLP_DC3_seed0_dropout0.1"
    #     "20251106-172023_MLP_DC3_seed1_dropout0.1",
    #     "20251106-172451_MLP_DC3_seed2_dropout0.1",
    #     # "20251106-172918_MLP_DC3_seed3_dropout0.1",
    # ],
#     [
#         "20251031-170438_MLP_DC3_seed0_dropout0.1_finetune_20251031-163616_model_20",
#         "20251031-171212_MLP_DC3_seed1_dropout0.1_finetune_20251031-163616_model_20",
#         "20251031-171947_MLP_DC3_seed2_dropout0.1_finetune_20251031-163616_model_20",
#     ], # 0.0
#     [
#         "20251031-172723_MLP_DC3_seed0_dropout0.1_finetune_20251031-163616_model_60",
#         "20251031-173457_MLP_DC3_seed1_dropout0.1_finetune_20251031-163616_model_60",
#         "20251031-174232_MLP_DC3_seed2_dropout0.1_finetune_20251031-163616_model_60",
#     ],
#     [
#         "20251031-182238_MLP_DC3_seed0_dropout0.1_finetune_20251031-163616_model_100",
#         "20251031-183014_MLP_DC3_seed1_dropout0.1_finetune_20251031-163616_model_100",
#         "20251031-183751_MLP_DC3_seed2_dropout0.1_finetune_20251031-163616_model_100",
#     ],
#     [
#         "20251031-184531_MLP_DC3_seed0_dropout0.1_finetune_20251031-163616_model_200",
#         "20251031-185311_MLP_DC3_seed1_dropout0.1_finetune_20251031-163616_model_200",
#         "20251031-190051_MLP_DC3_seed2_dropout0.1_finetune_20251031-163616_model_200",
#     ],
# ])

dc3_ft_subopt60_train_size7000_over_ckpt = np.array([
    [
        "20251107-110210_MLP_DC3_seed0_dropout0.1",
        "20251106-172023_MLP_DC3_seed1_dropout0.1",
        "20251106-172451_MLP_DC3_seed2_dropout0.1",
        # "20251106-172918_MLP_DC3_seed3_dropout0.1",
    ],
    [
        "20251106-174104_MLP_DC3_seed0_dropout0.1_finetune_20251105-173555_model_20",
        "20251106-180757_MLP_DC3_seed1_dropout0.1_finetune_20251105-173555_model_20",
        "20251106-183455_MLP_DC3_seed2_dropout0.1_finetune_20251105-173555_model_20",
    ], # 0.0
    [
        "20251106-190149_MLP_DC3_seed0_dropout0.1_finetune_20251105-173555_model_60",
        "20251106-192847_MLP_DC3_seed1_dropout0.1_finetune_20251105-173555_model_60",
        "20251106-195542_MLP_DC3_seed2_dropout0.1_finetune_20251105-173555_model_60",
    ],
    [
        "20251106-202236_MLP_DC3_seed0_dropout0.1_finetune_20251105-173555_model_100",
        "20251106-204933_MLP_DC3_seed1_dropout0.1_finetune_20251105-173555_model_100",
        "20251106-211637_MLP_DC3_seed2_dropout0.1_finetune_20251105-173555_model_100",
    ],
    [
        "20251106-214350_MLP_DC3_seed0_dropout0.1_finetune_20251105-173555_model_200",
        "20251106-221100_MLP_DC3_seed1_dropout0.1_finetune_20251105-173555_model_200",
        "20251106-223804_MLP_DC3_seed2_dropout0.1_finetune_20251105-173555_model_200",
    ],
])

dc3_ft_subopt40_train_size7000_over_ckpt = np.array([
    [
        "20251107-110210_MLP_DC3_seed0_dropout0.1",
        "20251106-172023_MLP_DC3_seed1_dropout0.1",
        "20251106-172451_MLP_DC3_seed2_dropout0.1",
        # "20251106-172918_MLP_DC3_seed3_dropout0.1",
    ],
    [
        "20251106-173635_MLP_DC3_seed0_dropout0.1_finetune_20251105-173353_model_20",
        "20251106-180328_MLP_DC3_seed1_dropout0.1_finetune_20251105-173353_model_20",
        "20251106-183026_MLP_DC3_seed2_dropout0.1_finetune_20251105-173353_model_20",
    ], # 0.0
    [
        "20251106-185721_MLP_DC3_seed0_dropout0.1_finetune_20251105-173353_model_60",
        "20251106-192417_MLP_DC3_seed1_dropout0.1_finetune_20251105-173353_model_60",
        "20251106-195113_MLP_DC3_seed2_dropout0.1_finetune_20251105-173353_model_60",
    ],
    [
        "20251106-201807_MLP_DC3_seed0_dropout0.1_finetune_20251105-173353_model_100",
        "20251106-204503_MLP_DC3_seed1_dropout0.1_finetune_20251105-173353_model_100",
        "20251106-211205_MLP_DC3_seed2_dropout0.1_finetune_20251105-173353_model_100",
    ],
    [
        "20251106-213918_MLP_DC3_seed0_dropout0.1_finetune_20251105-173353_model_200",
        "20251106-220630_MLP_DC3_seed1_dropout0.1_finetune_20251105-173353_model_200",
        "20251106-223333_MLP_DC3_seed2_dropout0.1_finetune_20251105-173353_model_200",
    ],
])

dc3_ft_subopt20_train_size7000_over_ckpt = np.array([
    [
        "20251107-110210_MLP_DC3_seed0_dropout0.1",
        "20251106-172023_MLP_DC3_seed1_dropout0.1",
        "20251106-172451_MLP_DC3_seed2_dropout0.1",
        # "20251106-172918_MLP_DC3_seed3_dropout0.1",
    ],
    [
        "20251106-173206_MLP_DC3_seed0_dropout0.1_finetune_20251031-163616_model_20",
        "20251106-175858_MLP_DC3_seed1_dropout0.1_finetune_20251031-163616_model_20",
        "20251106-182556_MLP_DC3_seed2_dropout0.1_finetune_20251031-163616_model_20",
    ], # 0.0
    [
        "20251106-185252_MLP_DC3_seed0_dropout0.1_finetune_20251031-163616_model_60",
        "20251106-191948_MLP_DC3_seed1_dropout0.1_finetune_20251031-163616_model_60",
        "20251106-194644_MLP_DC3_seed2_dropout0.1_finetune_20251031-163616_model_60",
    ],
    [
        "20251106-201337_MLP_DC3_seed0_dropout0.1_finetune_20251031-163616_model_100",
        "20251106-204034_MLP_DC3_seed1_dropout0.1_finetune_20251031-163616_model_100",
        "20251106-210734_MLP_DC3_seed2_dropout0.1_finetune_20251031-163616_model_100",
    ],
    [
        "20251106-213445_MLP_DC3_seed0_dropout0.1_finetune_20251031-163616_model_200",
        "20251106-220159_MLP_DC3_seed1_dropout0.1_finetune_20251031-163616_model_200",
        "20251106-222903_MLP_DC3_seed2_dropout0.1_finetune_20251031-163616_model_200",
    ],
])


dc3_ft_subopt10_train_size7000_over_ckpt = np.array([
    [
        "20251107-110210_MLP_DC3_seed0_dropout0.1",
        "20251106-172023_MLP_DC3_seed1_dropout0.1",
        "20251106-172451_MLP_DC3_seed2_dropout0.1",
        # "20251106-172918_MLP_DC3_seed3_dropout0.1",
    ],
    [
        "20251106-172736_MLP_DC3_seed0_dropout0.1_finetune_20251031-163247_model_20",
        "20251106-175430_MLP_DC3_seed1_dropout0.1_finetune_20251031-163247_model_20",
        "20251106-182127_MLP_DC3_seed2_dropout0.1_finetune_20251031-163247_model_20",
    ], # 0.0
    [
        "20251106-184823_MLP_DC3_seed0_dropout0.1_finetune_20251031-163247_model_60",
        "20251106-191519_MLP_DC3_seed1_dropout0.1_finetune_20251031-163247_model_60",
        "20251106-194216_MLP_DC3_seed2_dropout0.1_finetune_20251031-163247_model_60",
    ],
    [
        "20251106-200908_MLP_DC3_seed0_dropout0.1_finetune_20251031-163247_model_100",
        "20251106-203604_MLP_DC3_seed1_dropout0.1_finetune_20251031-163247_model_100",
        "20251106-210302_MLP_DC3_seed2_dropout0.1_finetune_20251031-163247_model_100",
    ],
    [
        "20251106-213012_MLP_DC3_seed0_dropout0.1_finetune_20251031-163247_model_200",
        "20251106-215728_MLP_DC3_seed1_dropout0.1_finetune_20251031-163247_model_200",
        "20251106-222432_MLP_DC3_seed2_dropout0.1_finetune_20251031-163247_model_200",
    ],
])


dc3_ft_subopt05_train_size7000_over_ckpt = np.array([
    [
        "20251107-110210_MLP_DC3_seed0_dropout0.1",
        "20251106-172023_MLP_DC3_seed1_dropout0.1",
        "20251106-172451_MLP_DC3_seed2_dropout0.1",
        # "20251106-172918_MLP_DC3_seed3_dropout0.1",
    ],
    [
        "20251106-172305_MLP_DC3_seed0_dropout0.1_finetune_20251031-163030_model_20",
        "20251106-175002_MLP_DC3_seed1_dropout0.1_finetune_20251031-163030_model_20",
        "20251106-181657_MLP_DC3_seed2_dropout0.1_finetune_20251031-163030_model_20",
    ], # 0.0
    [
        "20251106-184354_MLP_DC3_seed0_dropout0.1_finetune_20251031-163030_model_60",
        "20251106-191048_MLP_DC3_seed1_dropout0.1_finetune_20251031-163030_model_60",
        "20251106-193746_MLP_DC3_seed2_dropout0.1_finetune_20251031-163030_model_60",
    ],
    [
        "20251106-200439_MLP_DC3_seed0_dropout0.1_finetune_20251031-163030_model_100",
        "20251106-203135_MLP_DC3_seed1_dropout0.1_finetune_20251031-163030_model_100",
        "20251106-205831_MLP_DC3_seed2_dropout0.1_finetune_20251031-163030_model_100",
    ],
    [
        "20251106-212540_MLP_DC3_seed0_dropout0.1_finetune_20251031-163030_model_200",
        "20251106-215255_MLP_DC3_seed1_dropout0.1_finetune_20251031-163030_model_200",
        "20251106-222002_MLP_DC3_seed2_dropout0.1_finetune_20251031-163030_model_200",
    ],
])

dc3_ft_subopt00_train_size7000_over_ckpt = np.array([
    [
        "20251107-110210_MLP_DC3_seed0_dropout0.1",
        "20251106-172023_MLP_DC3_seed1_dropout0.1",
        "20251106-172451_MLP_DC3_seed2_dropout0.1",
        # "20251106-172918_MLP_DC3_seed3_dropout0.1",
    ],
    [
        "20251106-171835_MLP_DC3_seed0_dropout0.1_finetune_20251031-162815_model_20",
        "20251106-174533_MLP_DC3_seed1_dropout0.1_finetune_20251031-162815_model_20",
        "20251106-181227_MLP_DC3_seed2_dropout0.1_finetune_20251031-162815_model_20",
    ], # 0.0
    [
        "20251106-183925_MLP_DC3_seed0_dropout0.1_finetune_20251031-162815_model_60",
        "20251106-190618_MLP_DC3_seed1_dropout0.1_finetune_20251031-162815_model_60",
        "20251106-193317_MLP_DC3_seed2_dropout0.1_finetune_20251031-162815_model_60",
    ],
    [
        "20251106-200011_MLP_DC3_seed0_dropout0.1_finetune_20251031-162815_model_100",
        "20251106-202705_MLP_DC3_seed1_dropout0.1_finetune_20251031-162815_model_100",
        "20251106-205403_MLP_DC3_seed2_dropout0.1_finetune_20251031-162815_model_100",
    ],
    [
        "20251106-212108_MLP_DC3_seed0_dropout0.1_finetune_20251031-162815_model_200",
        "20251106-214822_MLP_DC3_seed1_dropout0.1_finetune_20251031-162815_model_200",
        "20251106-221531_MLP_DC3_seed2_dropout0.1_finetune_20251031-162815_model_200",
    ],
])

dc3_ft_ckpt20_train_size7000_over_subopt = np.array([
    [
        "20251107-110210_MLP_DC3_seed0_dropout0.1",
        "20251106-172023_MLP_DC3_seed1_dropout0.1",
        "20251106-172451_MLP_DC3_seed2_dropout0.1",
        # "20251106-172918_MLP_DC3_seed3_dropout0.1",
    ],
    [
        "20251106-171835_MLP_DC3_seed0_dropout0.1_finetune_20251031-162815_model_20",
        "20251106-174533_MLP_DC3_seed1_dropout0.1_finetune_20251031-162815_model_20",
        "20251106-181227_MLP_DC3_seed2_dropout0.1_finetune_20251031-162815_model_20",
    ], # 0.0
    [
        "20251106-172305_MLP_DC3_seed0_dropout0.1_finetune_20251031-163030_model_20",
        "20251106-175002_MLP_DC3_seed1_dropout0.1_finetune_20251031-163030_model_20",
        "20251106-181657_MLP_DC3_seed2_dropout0.1_finetune_20251031-163030_model_20",
    ], # 0.5
    [
        "20251106-172736_MLP_DC3_seed0_dropout0.1_finetune_20251031-163247_model_20",
        "20251106-175430_MLP_DC3_seed1_dropout0.1_finetune_20251031-163247_model_20",
        "20251106-182127_MLP_DC3_seed2_dropout0.1_finetune_20251031-163247_model_20",
    ], # 1.0
    [
        "20251106-173206_MLP_DC3_seed0_dropout0.1_finetune_20251031-163616_model_20",
        "20251106-175858_MLP_DC3_seed1_dropout0.1_finetune_20251031-163616_model_20",
        "20251106-182556_MLP_DC3_seed2_dropout0.1_finetune_20251031-163616_model_20",
    ], # 2.0
    [
        "20251106-173635_MLP_DC3_seed0_dropout0.1_finetune_20251105-173353_model_20",
        "20251106-180328_MLP_DC3_seed1_dropout0.1_finetune_20251105-173353_model_20",
        "20251106-183026_MLP_DC3_seed2_dropout0.1_finetune_20251105-173353_model_20",
    ], # 4.0
    [
        "20251106-174104_MLP_DC3_seed0_dropout0.1_finetune_20251105-173555_model_20",
        "20251106-180757_MLP_DC3_seed1_dropout0.1_finetune_20251105-173555_model_20",
        "20251106-183455_MLP_DC3_seed2_dropout0.1_finetune_20251105-173555_model_20",
    ], # 6.0
])

##########################################
######## PENALTY
##########################################

penalty_ft_subopt00_train_size7000_over_ckpt = np.array([
    [
        "20251106-170432_MLP_penalty_seed0_dropout0.1",
        "20251106-170740_MLP_penalty_seed1_dropout0.1",
        "20251106-170924_MLP_penalty_seed2_dropout0.1",
        "20251106-171108_MLP_penalty_seed3_dropout0.1",
    ],
    [
        "20251105-163157_MLP_penalty_seed0_dropout0.1_finetune_20251030-232939_model_20",
        "20251105-164410_MLP_penalty_seed1_dropout0.1_finetune_20251030-232939_model_20",
        "20251105-165621_MLP_penalty_seed2_dropout0.1_finetune_20251030-232939_model_20",
        "20251106-163157_MLP_penalty_seed3_dropout0.1_finetune_20251030-232939_model_20",
    ], # 0.0
    [
        "20251107-103631_MLP_penalty_seed0_dropout0.1_finetune_20251030-232939_model_60",
        "20251105-171231_MLP_penalty_seed1_dropout0.1_finetune_20251030-232939_model_60",
        "20251105-172435_MLP_penalty_seed2_dropout0.1_finetune_20251030-232939_model_60",
        "20251105-173646_MLP_penalty_seed3_dropout0.1_finetune_20251030-232939_model_60",
    ],
    [
        "20251105-174852_MLP_penalty_seed0_dropout0.1_finetune_20251030-232939_model_100",
        "20251105-180103_MLP_penalty_seed1_dropout0.1_finetune_20251030-232939_model_100",
        "20251105-181313_MLP_penalty_seed2_dropout0.1_finetune_20251030-232939_model_100",
        "20251105-182523_MLP_penalty_seed3_dropout0.1_finetune_20251030-232939_model_100",
    ],
    [
        "20251105-183732_MLP_penalty_seed0_dropout0.1_finetune_20251030-232939_model_200",
        "20251105-184942_MLP_penalty_seed1_dropout0.1_finetune_20251030-232939_model_200",
        "20251105-190154_MLP_penalty_seed2_dropout0.1_finetune_20251030-232939_model_200",
        "20251105-191410_MLP_penalty_seed3_dropout0.1_finetune_20251030-232939_model_200",
    ],
])

penalty_ft_subopt05_train_size7000_over_ckpt = np.array([
    [
        "20251106-170432_MLP_penalty_seed0_dropout0.1",
        "20251106-170740_MLP_penalty_seed1_dropout0.1",
        "20251106-170924_MLP_penalty_seed2_dropout0.1",
        "20251106-171108_MLP_penalty_seed3_dropout0.1",
    ],
    [
        "20251105-163342_MLP_penalty_seed0_dropout0.1_finetune_20251030-233223_model_20",
        "20251105-164555_MLP_penalty_seed1_dropout0.1_finetune_20251030-233223_model_20",
        "20251105-165805_MLP_penalty_seed2_dropout0.1_finetune_20251030-233223_model_20",
        "20251106-163340_MLP_penalty_seed3_dropout0.1_finetune_20251030-233223_model_20",
    ], # 0.0
    [
        "20251107-103823_MLP_penalty_seed0_dropout0.1_finetune_20251030-233223_model_60",
        "20251105-171415_MLP_penalty_seed1_dropout0.1_finetune_20251030-233223_model_60",
        "20251105-172620_MLP_penalty_seed2_dropout0.1_finetune_20251030-233223_model_60",
        "20251105-173829_MLP_penalty_seed3_dropout0.1_finetune_20251030-233223_model_60",
    ],
    [
        "20251105-175037_MLP_penalty_seed0_dropout0.1_finetune_20251030-233223_model_100",
        "20251105-180248_MLP_penalty_seed1_dropout0.1_finetune_20251030-233223_model_100",
        "20251105-181458_MLP_penalty_seed2_dropout0.1_finetune_20251030-233223_model_100",
        "20251105-182706_MLP_penalty_seed3_dropout0.1_finetune_20251030-233223_model_100",
    ],
    [
        "20251105-183916_MLP_penalty_seed0_dropout0.1_finetune_20251030-233223_model_200",
        "20251105-185126_MLP_penalty_seed1_dropout0.1_finetune_20251030-233223_model_200",
        "20251105-190338_MLP_penalty_seed2_dropout0.1_finetune_20251030-233223_model_200",
        "20251105-191555_MLP_penalty_seed3_dropout0.1_finetune_20251030-233223_model_200",
    ],
])

penalty_ft_subopt10_train_size7000_over_ckpt = np.array([
    [
        "20251106-170432_MLP_penalty_seed0_dropout0.1",
        "20251106-170740_MLP_penalty_seed1_dropout0.1",
        "20251106-170924_MLP_penalty_seed2_dropout0.1",
        "20251106-171108_MLP_penalty_seed3_dropout0.1",
    ],
    [
        "20251105-163527_MLP_penalty_seed0_dropout0.1_finetune_20251030-233258_model_20",
        "20251105-164740_MLP_penalty_seed1_dropout0.1_finetune_20251030-233258_model_20",
        "20251105-165949_MLP_penalty_seed2_dropout0.1_finetune_20251030-233258_model_20",
        "20251106-163523_MLP_penalty_seed3_dropout0.1_finetune_20251030-233258_model_20",
    ], # 0.0
    [
        "20251105-170351_MLP_penalty_seed0_dropout0.1_finetune_20251030-233258_model_60",
        "20251105-171559_MLP_penalty_seed1_dropout0.1_finetune_20251030-233258_model_60",
        "20251105-172804_MLP_penalty_seed2_dropout0.1_finetune_20251030-233258_model_60",
        "20251105-174012_MLP_penalty_seed3_dropout0.1_finetune_20251030-233258_model_60",
    ],
    [
        "20251105-175221_MLP_penalty_seed0_dropout0.1_finetune_20251030-233258_model_100",
        "20251105-180432_MLP_penalty_seed1_dropout0.1_finetune_20251030-233258_model_100",
        "20251105-181642_MLP_penalty_seed2_dropout0.1_finetune_20251030-233258_model_100",
        "20251105-182850_MLP_penalty_seed3_dropout0.1_finetune_20251030-233258_model_100",
    ],
    [
        "20251105-184100_MLP_penalty_seed0_dropout0.1_finetune_20251030-233258_model_200",
        "20251105-185311_MLP_penalty_seed1_dropout0.1_finetune_20251030-233258_model_200",
        "20251105-190523_MLP_penalty_seed2_dropout0.1_finetune_20251030-233258_model_200",
        "20251105-191739_MLP_penalty_seed3_dropout0.1_finetune_20251030-233258_model_200",
    ],
])

penalty_ft_subopt20_train_size7000_over_ckpt = np.array([
    [
        "20251106-170432_MLP_penalty_seed0_dropout0.1",
        "20251106-170740_MLP_penalty_seed1_dropout0.1",
        "20251106-170924_MLP_penalty_seed2_dropout0.1",
        "20251106-171108_MLP_penalty_seed3_dropout0.1",
    ],
    [
        "20251105-163711_MLP_penalty_seed0_dropout0.1_finetune_20251031-073026_model_20",
        "20251105-164925_MLP_penalty_seed1_dropout0.1_finetune_20251031-073026_model_20",
        "20251107-130520_MLP_penalty_seed2_dropout0.1_finetune_20251031-073026_model_20",
        "20251106-163705_MLP_penalty_seed3_dropout0.1_finetune_20251031-073026_model_20",
    ], # 0.0
    [
        "20251105-170536_MLP_penalty_seed0_dropout0.1_finetune_20251031-073026_model_60",
        "20251105-171743_MLP_penalty_seed1_dropout0.1_finetune_20251031-073026_model_60",
        "20251105-172948_MLP_penalty_seed2_dropout0.1_finetune_20251031-073026_model_60",
        "20251105-174155_MLP_penalty_seed3_dropout0.1_finetune_20251031-073026_model_60",
    ],
    [
        "20251105-175406_MLP_penalty_seed0_dropout0.1_finetune_20251031-073026_model_100",
        "20251105-180616_MLP_penalty_seed1_dropout0.1_finetune_20251031-073026_model_100",
        "20251105-181826_MLP_penalty_seed2_dropout0.1_finetune_20251031-073026_model_100",
        "20251105-183035_MLP_penalty_seed3_dropout0.1_finetune_20251031-073026_model_100",
    ],
    [
        "20251105-184245_MLP_penalty_seed0_dropout0.1_finetune_20251031-073026_model_200",
        "20251105-185456_MLP_penalty_seed1_dropout0.1_finetune_20251031-073026_model_200",
        "20251105-190708_MLP_penalty_seed2_dropout0.1_finetune_20251031-073026_model_200",
        "20251105-191924_MLP_penalty_seed3_dropout0.1_finetune_20251031-073026_model_200",
    ],
])

penalty_ft_subopt40_train_size7000_over_ckpt = np.array([
    [
        "20251106-170432_MLP_penalty_seed0_dropout0.1",
        "20251106-170740_MLP_penalty_seed1_dropout0.1",
        "20251106-170924_MLP_penalty_seed2_dropout0.1",
        "20251106-171108_MLP_penalty_seed3_dropout0.1",
    ],
    [
        "20251105-163856_MLP_penalty_seed0_dropout0.1_finetune_20251101-002602_model_20",
        "20251105-165110_MLP_penalty_seed1_dropout0.1_finetune_20251101-002602_model_20",
        "20251107-104028_MLP_penalty_seed2_dropout0.1_finetune_20251101-002602_model_20",
        "20251106-163848_MLP_penalty_seed3_dropout0.1_finetune_20251101-002602_model_20",
    ], # 0.0
    [
        "20251105-170720_MLP_penalty_seed0_dropout0.1_finetune_20251101-002602_model_60",
        "20251105-171926_MLP_penalty_seed1_dropout0.1_finetune_20251101-002602_model_60",
        "20251105-173132_MLP_penalty_seed2_dropout0.1_finetune_20251101-002602_model_60",
        "20251105-174339_MLP_penalty_seed3_dropout0.1_finetune_20251101-002602_model_60",
    ],
    [
        "20251105-175550_MLP_penalty_seed0_dropout0.1_finetune_20251101-002602_model_100",
        "20251105-180800_MLP_penalty_seed1_dropout0.1_finetune_20251101-002602_model_100",
        "20251105-182010_MLP_penalty_seed2_dropout0.1_finetune_20251101-002602_model_100",
        "20251105-183220_MLP_penalty_seed3_dropout0.1_finetune_20251101-002602_model_100",
    ],
    [
        "20251105-184429_MLP_penalty_seed0_dropout0.1_finetune_20251101-002602_model_200",
        "20251105-185640_MLP_penalty_seed1_dropout0.1_finetune_20251101-002602_model_200",
        "20251105-190852_MLP_penalty_seed2_dropout0.1_finetune_20251101-002602_model_200",
        "20251105-192109_MLP_penalty_seed3_dropout0.1_finetune_20251101-002602_model_200",
    ],
])

penalty_ft_subopt60_train_size7000_over_ckpt = np.array([
    [
        "20251106-170432_MLP_penalty_seed0_dropout0.1",
        "20251106-170740_MLP_penalty_seed1_dropout0.1",
        "20251106-170924_MLP_penalty_seed2_dropout0.1",
        "20251106-171108_MLP_penalty_seed3_dropout0.1",
    ],
    [
        "20251105-164040_MLP_penalty_seed0_dropout0.1_finetune_20251101-232010_model_20",
        "20251105-165253_MLP_penalty_seed1_dropout0.1_finetune_20251101-232010_model_20",
        "20251107-104215_MLP_penalty_seed2_dropout0.1_finetune_20251101-232010_model_20",
        "20251106-164031_MLP_penalty_seed3_dropout0.1_finetune_20251101-232010_model_20",
    ], # 0.0
    [
        "20251105-170904_MLP_penalty_seed0_dropout0.1_finetune_20251101-232010_model_60",
        "20251105-172109_MLP_penalty_seed1_dropout0.1_finetune_20251101-232010_model_60",
        "20251105-173317_MLP_penalty_seed2_dropout0.1_finetune_20251101-232010_model_60",
        "20251105-174523_MLP_penalty_seed3_dropout0.1_finetune_20251101-232010_model_60",
    ],
    [
        "20251105-175735_MLP_penalty_seed0_dropout0.1_finetune_20251101-232010_model_100",
        "20251105-180945_MLP_penalty_seed1_dropout0.1_finetune_20251101-232010_model_100",
        "20251105-182154_MLP_penalty_seed2_dropout0.1_finetune_20251101-232010_model_100",
        "20251105-183404_MLP_penalty_seed3_dropout0.1_finetune_20251101-232010_model_100",
    ],
    [
        "20251105-184613_MLP_penalty_seed0_dropout0.1_finetune_20251101-232010_model_200",
        "20251105-185825_MLP_penalty_seed1_dropout0.1_finetune_20251101-232010_model_200",
        "20251105-191041_MLP_penalty_seed2_dropout0.1_finetune_20251101-232010_model_200",
        "20251105-192253_MLP_penalty_seed3_dropout0.1_finetune_20251101-232010_model_200",
    ],
])


penalty_ft_subopt100_train_size7000_over_ckpt = np.array([
    [
        "20251106-170432_MLP_penalty_seed0_dropout0.1",
        "20251106-170740_MLP_penalty_seed1_dropout0.1",
        "20251106-170924_MLP_penalty_seed2_dropout0.1",
        "20251106-171108_MLP_penalty_seed3_dropout0.1",
    ],
    [
        "20251105-164225_MLP_penalty_seed0_dropout0.1_finetune_20251101-003437_model_20",
        "20251105-165437_MLP_penalty_seed1_dropout0.1_finetune_20251101-003437_model_20",
        "20251107-104403_MLP_penalty_seed2_dropout0.1_finetune_20251101-003437_model_20",
        "20251106-164214_MLP_penalty_seed3_dropout0.1_finetune_20251101-003437_model_20",
    ], # 0.0
    [
        "20251105-171048_MLP_penalty_seed0_dropout0.1_finetune_20251101-003437_model_60",
        "20251105-172252_MLP_penalty_seed1_dropout0.1_finetune_20251101-003437_model_60",
        "20251105-173501_MLP_penalty_seed2_dropout0.1_finetune_20251101-003437_model_60",
        "20251105-174708_MLP_penalty_seed3_dropout0.1_finetune_20251101-003437_model_60",
    ],
    [
        "20251105-175919_MLP_penalty_seed0_dropout0.1_finetune_20251101-003437_model_100",
        "20251105-181129_MLP_penalty_seed1_dropout0.1_finetune_20251101-003437_model_100",
        "20251105-182339_MLP_penalty_seed2_dropout0.1_finetune_20251101-003437_model_100",
        "20251105-183548_MLP_penalty_seed3_dropout0.1_finetune_20251101-003437_model_100",
    ],
    [
        "20251105-184758_MLP_penalty_seed0_dropout0.1_finetune_20251101-003437_model_200",
        "20251105-190009_MLP_penalty_seed1_dropout0.1_finetune_20251101-003437_model_200",
        "20251105-191225_MLP_penalty_seed2_dropout0.1_finetune_20251101-003437_model_200",
        "20251105-192437_MLP_penalty_seed3_dropout0.1_finetune_20251101-003437_model_200",
    ],
])

# Finetune checkpoint 20 of different suboptimalities
penalty_ft_ckpt20_train_size7000_over_subopt = np.array([
    [
        "20251106-170432_MLP_penalty_seed0_dropout0.1",
        "20251106-170740_MLP_penalty_seed1_dropout0.1",
        "20251106-170924_MLP_penalty_seed2_dropout0.1",
        "20251106-171108_MLP_penalty_seed3_dropout0.1",
    ], # baseline
    [
        "20251105-163157_MLP_penalty_seed0_dropout0.1_finetune_20251030-232939_model_20",
        "20251105-164410_MLP_penalty_seed1_dropout0.1_finetune_20251030-232939_model_20",
        "20251105-165621_MLP_penalty_seed2_dropout0.1_finetune_20251030-232939_model_20",
        "20251106-163157_MLP_penalty_seed3_dropout0.1_finetune_20251030-232939_model_20",
    ], # 0.0
    [
        "20251105-163342_MLP_penalty_seed0_dropout0.1_finetune_20251030-233223_model_20",
        "20251105-164555_MLP_penalty_seed1_dropout0.1_finetune_20251030-233223_model_20",
        "20251105-165805_MLP_penalty_seed2_dropout0.1_finetune_20251030-233223_model_20",
        "20251106-163340_MLP_penalty_seed3_dropout0.1_finetune_20251030-233223_model_20",
    ], # 0.5
    [
        "20251105-163527_MLP_penalty_seed0_dropout0.1_finetune_20251030-233258_model_20",
        "20251105-164740_MLP_penalty_seed1_dropout0.1_finetune_20251030-233258_model_20",
        "20251105-165949_MLP_penalty_seed2_dropout0.1_finetune_20251030-233258_model_20",
        "20251106-163523_MLP_penalty_seed3_dropout0.1_finetune_20251030-233258_model_20",
    ], # 1.0
    [
        "20251105-163711_MLP_penalty_seed0_dropout0.1_finetune_20251031-073026_model_20",
        "20251105-164925_MLP_penalty_seed1_dropout0.1_finetune_20251031-073026_model_20",
        "20251107-130520_MLP_penalty_seed2_dropout0.1_finetune_20251031-073026_model_20",
        "20251106-163705_MLP_penalty_seed3_dropout0.1_finetune_20251031-073026_model_20",
    ], # 2.0
    [
        "20251105-163856_MLP_penalty_seed0_dropout0.1_finetune_20251101-002602_model_20",
        "20251105-165110_MLP_penalty_seed1_dropout0.1_finetune_20251101-002602_model_20",
        "20251107-104028_MLP_penalty_seed2_dropout0.1_finetune_20251101-002602_model_20",
        "20251106-163848_MLP_penalty_seed3_dropout0.1_finetune_20251101-002602_model_20",
    ], # 4.0
    [
        "20251105-164040_MLP_penalty_seed0_dropout0.1_finetune_20251101-232010_model_20",
        "20251105-165253_MLP_penalty_seed1_dropout0.1_finetune_20251101-232010_model_20",
        "20251107-104215_MLP_penalty_seed2_dropout0.1_finetune_20251101-232010_model_20",
        "20251106-164031_MLP_penalty_seed3_dropout0.1_finetune_20251101-232010_model_20",
    ],
    [
        "20251105-164225_MLP_penalty_seed0_dropout0.1_finetune_20251101-003437_model_20",
        "20251105-165437_MLP_penalty_seed1_dropout0.1_finetune_20251101-003437_model_20",
        "20251107-104403_MLP_penalty_seed2_dropout0.1_finetune_20251101-003437_model_20",
        "20251106-164214_MLP_penalty_seed3_dropout0.1_finetune_20251101-003437_model_20",
    ], # 10.0       
]) # matrix of shape (num_ckpt, num_seeds)

penalty_ft_ckpt20_subopt20_over_train_size = np.array([
    [
        "20251106-115116_MLP_penalty_seed0_dropout0.1_finetune_20251103-103706_model_20",
        "20251106-120742_MLP_penalty_seed1_dropout0.1_finetune_20251103-103706_model_20",
        "20251106-122404_MLP_penalty_seed2_dropout0.1_finetune_20251103-103706_model_20",
        "20251106-150824_MLP_penalty_seed3_dropout0.1_finetune_20251103-103706_model_20",
    ], # 10
    [
        "20251106-115305_MLP_penalty_seed0_dropout0.1_finetune_20251106-114016_model_20",
        "20251106-120930_MLP_penalty_seed1_dropout0.1_finetune_20251106-114016_model_20",
        "20251106-122554_MLP_penalty_seed2_dropout0.1_finetune_20251106-114016_model_20",
        "20251106-151010_MLP_penalty_seed3_dropout0.1_finetune_20251106-114016_model_20",
    ], # 20
    [
        "20251106-115454_MLP_penalty_seed0_dropout0.1_finetune_20251106-114103_model_20",
        "20251106-121120_MLP_penalty_seed1_dropout0.1_finetune_20251106-114103_model_20",
        "20251106-122743_MLP_penalty_seed2_dropout0.1_finetune_20251106-114103_model_20",
        "20251106-151155_MLP_penalty_seed3_dropout0.1_finetune_20251106-114103_model_20",
    ], # 30
    [
        "20251106-115643_MLP_penalty_seed0_dropout0.1_finetune_20251106-114138_model_20",
        "20251106-121309_MLP_penalty_seed1_dropout0.1_finetune_20251106-114138_model_20",
        "20251106-122933_MLP_penalty_seed2_dropout0.1_finetune_20251106-114138_model_20",
        "20251106-151341_MLP_penalty_seed3_dropout0.1_finetune_20251106-114138_model_20",
    ], # 40
    [
        "20251106-115833_MLP_penalty_seed0_dropout0.1_finetune_20251103-103719_model_20",
        "20251106-121458_MLP_penalty_seed1_dropout0.1_finetune_20251103-103719_model_20",
        "20251106-123121_MLP_penalty_seed2_dropout0.1_finetune_20251103-103719_model_20",
        "20251106-151526_MLP_penalty_seed3_dropout0.1_finetune_20251103-103719_model_20",
    ], # 50
    [
        "20251107-132155_MLP_penalty_seed0_dropout0.1_finetune_20251101-233217_model_20",
        "20251107-132900_MLP_penalty_seed1_dropout0.1_finetune_20251101-233217_model_20",
        "20251107-133605_MLP_penalty_seed2_dropout0.1_finetune_20251101-233217_model_20",
        "20251107-134302_MLP_penalty_seed3_dropout0.1_finetune_20251101-233217_model_20",
    ], # 200
    [
        "20251107-132341_MLP_penalty_seed0_dropout0.1_finetune_20251101-233236_model_20",
        "20251107-133046_MLP_penalty_seed1_dropout0.1_finetune_20251101-233236_model_20",
        "20251107-133749_MLP_penalty_seed2_dropout0.1_finetune_20251101-233236_model_20",
        "20251107-134446_MLP_penalty_seed3_dropout0.1_finetune_20251101-233236_model_20",
    ], # 500
    [
        "20251107-132527_MLP_penalty_seed0_dropout0.1_finetune_20251101-233303_model_20",
        "20251107-133233_MLP_penalty_seed1_dropout0.1_finetune_20251101-233303_model_20",
        "20251107-133933_MLP_penalty_seed2_dropout0.1_finetune_20251101-233303_model_20",
        "20251107-134630_MLP_penalty_seed3_dropout0.1_finetune_20251101-233303_model_20",
    ], # 1000
    [
        "20251107-132714_MLP_penalty_seed0_dropout0.1_finetune_20251101-233338_model_20",
        "20251107-133419_MLP_penalty_seed1_dropout0.1_finetune_20251101-233338_model_20",
        "20251107-134117_MLP_penalty_seed2_dropout0.1_finetune_20251101-233338_model_20",
        "20251107-134814_MLP_penalty_seed3_dropout0.1_finetune_20251101-233338_model_20",
    ], # 4000
    [
        "20251105-163711_MLP_penalty_seed0_dropout0.1_finetune_20251031-073026_model_20",
        "20251105-164925_MLP_penalty_seed1_dropout0.1_finetune_20251031-073026_model_20",
        "20251107-130520_MLP_penalty_seed2_dropout0.1_finetune_20251031-073026_model_20",
        "20251106-163705_MLP_penalty_seed3_dropout0.1_finetune_20251031-073026_model_20",
    ], # ckpt20_subopt20_train_size7000
    [
        "20251105-163157_MLP_penalty_seed0_dropout0.1_finetune_20251030-232939_model_20",
        "20251105-164410_MLP_penalty_seed1_dropout0.1_finetune_20251030-232939_model_20",
        "20251105-165621_MLP_penalty_seed2_dropout0.1_finetune_20251030-232939_model_20",
        "20251106-163157_MLP_penalty_seed3_dropout0.1_finetune_20251030-232939_model_20",
    ], # ckpt20_subopt00_train_size7000
    [
        "20251106-170432_MLP_penalty_seed0_dropout0.1",
        "20251106-170740_MLP_penalty_seed1_dropout0.1",
        "20251106-170924_MLP_penalty_seed2_dropout0.1",
        "20251106-171108_MLP_penalty_seed3_dropout0.1",
    ],
]) # matrix of shape (num_ckpt, num_seeds)

