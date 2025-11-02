import numpy as np

fsnet_ft_ckpt20_subopt20_over_train_size = np.array([
    [
        "20251101-234017_MLP_FSNet_seed0_dropout0.1_finetune_20251101-233217_model_20",
        "20251102-003328_MLP_FSNet_seed1_dropout0.1_finetune_20251101-233217_model_20",
        "20251102-012534_MLP_FSNet_seed2_dropout0.1_finetune_20251101-233217_model_20",
        "20251102-012156_MLP_FSNet_seed3_dropout0.1_finetune_20251101-233217_model_20",
    ],
    [
        "20251101-235251_MLP_FSNet_seed0_dropout0.1_finetune_20251101-233236_model_20",
        "20251102-004558_MLP_FSNet_seed1_dropout0.1_finetune_20251101-233236_model_20",
        "20251102-013943_MLP_FSNet_seed2_dropout0.1_finetune_20251101-233236_model_20",
        "20251102-013453_MLP_FSNet_seed3_dropout0.1_finetune_20251101-233236_model_20",
    ],
    [
        "20251102-000639_MLP_FSNet_seed0_dropout0.1_finetune_20251101-233303_model_20",
        "20251102-005943_MLP_FSNet_seed1_dropout0.1_finetune_20251101-233303_model_20",
        "20251102-015453_MLP_FSNet_seed2_dropout0.1_finetune_20251101-233303_model_20",
        "20251102-014842_MLP_FSNet_seed3_dropout0.1_finetune_20251101-233303_model_20",
    ],
    [
        "20251102-001926_MLP_FSNet_seed0_dropout0.1_finetune_20251101-233338_model_20",
        "20251102-011248_MLP_FSNet_seed1_dropout0.1_finetune_20251101-233338_model_20",
        "20251102-010834_MLP_FSNet_seed2_dropout0.1_finetune_20251101-233338_model_20",
        "20251102-020113_MLP_FSNet_seed3_dropout0.1_finetune_20251101-233338_model_20",
    ],
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
    ],
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
        "20251031-211532_MLP_FSNet_seed3_dropout0.1"
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
        "20251031-211532_MLP_FSNet_seed3_dropout0.1"
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


dc3_ft_subopt20_train_size7000_over_ckpt = np.array([
    [
        "20251102-112406_MLP_DC3_seed0_dropout0.1",
        "20251102-112606_MLP_DC3_seed1_dropout0.1",
        "20251102-112806_MLP_DC3_seed2_dropout0.1",
        "20251102-113006_MLP_DC3_seed3_dropout0.1",
    ],
    [
        "20251031-170438_MLP_DC3_seed0_dropout0.1_finetune_20251031-163616_model_20",
        "20251031-171212_MLP_DC3_seed1_dropout0.1_finetune_20251031-163616_model_20",
        "20251031-171947_MLP_DC3_seed2_dropout0.1_finetune_20251031-163616_model_20",
    ], # 0.0
    [
        "20251031-172723_MLP_DC3_seed0_dropout0.1_finetune_20251031-163616_model_60",
        "20251031-173457_MLP_DC3_seed1_dropout0.1_finetune_20251031-163616_model_60",
        "20251031-174232_MLP_DC3_seed2_dropout0.1_finetune_20251031-163616_model_60",
    ],
    [
        "20251031-182238_MLP_DC3_seed0_dropout0.1_finetune_20251031-163616_model_100",
        "20251031-183014_MLP_DC3_seed1_dropout0.1_finetune_20251031-163616_model_100",
        "20251031-183751_MLP_DC3_seed2_dropout0.1_finetune_20251031-163616_model_100",
    ],
    [
        "20251031-184531_MLP_DC3_seed0_dropout0.1_finetune_20251031-163616_model_200",
        "20251031-185311_MLP_DC3_seed1_dropout0.1_finetune_20251031-163616_model_200",
        "20251031-190051_MLP_DC3_seed2_dropout0.1_finetune_20251031-163616_model_200",
    ],
])


dc3_ft_subopt10_train_size7000_over_ckpt = np.array([
    [
        "20251102-112406_MLP_DC3_seed0_dropout0.1",
        "20251102-112606_MLP_DC3_seed1_dropout0.1",
        "20251102-112806_MLP_DC3_seed2_dropout0.1",
        "20251102-113006_MLP_DC3_seed3_dropout0.1",
    ],
    [
        "20251031-170244_MLP_DC3_seed0_dropout0.1_finetune_20251031-163247_model_20",
        "20251031-171019_MLP_DC3_seed1_dropout0.1_finetune_20251031-163247_model_20",
        "20251031-171754_MLP_DC3_seed2_dropout0.1_finetune_20251031-163247_model_20",
    ], # 0.0
    [
        "20251031-172529_MLP_DC3_seed0_dropout0.1_finetune_20251031-163247_model_60",
        "20251031-173304_MLP_DC3_seed1_dropout0.1_finetune_20251031-163247_model_60",
        "20251031-174038_MLP_DC3_seed2_dropout0.1_finetune_20251031-163247_model_60",
    ],
    [
        "20251031-182044_MLP_DC3_seed0_dropout0.1_finetune_20251031-163247_model_100",
        "20251031-182820_MLP_DC3_seed1_dropout0.1_finetune_20251031-163247_model_100",
        "20251031-183556_MLP_DC3_seed2_dropout0.1_finetune_20251031-163247_model_100",
    ],
    [
        "20251031-184336_MLP_DC3_seed0_dropout0.1_finetune_20251031-163247_model_200",
        "20251031-185116_MLP_DC3_seed1_dropout0.1_finetune_20251031-163247_model_200",
        "20251031-185856_MLP_DC3_seed2_dropout0.1_finetune_20251031-163247_model_200",
    ],
])


dc3_ft_subopt05_train_size7000_over_ckpt = np.array([
    [
        "20251102-112406_MLP_DC3_seed0_dropout0.1",
        "20251102-112606_MLP_DC3_seed1_dropout0.1",
        "20251102-112806_MLP_DC3_seed2_dropout0.1",
        "20251102-113006_MLP_DC3_seed3_dropout0.1",
    ],
    [
        "20251031-170049_MLP_DC3_seed0_dropout0.1_finetune_20251031-163030_model_20",
        "20251031-170825_MLP_DC3_seed1_dropout0.1_finetune_20251031-163030_model_20",
        "20251031-171601_MLP_DC3_seed2_dropout0.1_finetune_20251031-163030_model_20",
    ], # 0.0
    [
        "20251031-172335_MLP_DC3_seed0_dropout0.1_finetune_20251031-163030_model_60",
        "20251031-173110_MLP_DC3_seed1_dropout0.1_finetune_20251031-163030_model_60",
        "20251031-173845_MLP_DC3_seed2_dropout0.1_finetune_20251031-163030_model_60",
    ],
    [
        "20251031-181851_MLP_DC3_seed0_dropout0.1_finetune_20251031-163030_model_100",
        "20251031-182626_MLP_DC3_seed1_dropout0.1_finetune_20251031-163030_model_100",
        "20251031-183401_MLP_DC3_seed2_dropout0.1_finetune_20251031-163030_model_100",
    ],
    [
        "20251031-184140_MLP_DC3_seed0_dropout0.1_finetune_20251031-163030_model_200",
        "20251031-184921_MLP_DC3_seed1_dropout0.1_finetune_20251031-163030_model_200",
        "20251031-185701_MLP_DC3_seed2_dropout0.1_finetune_20251031-163030_model_200",
    ],
])

dc3_ft_subopt00_train_size7000_over_ckpt = np.array([
    [
        "20251102-112406_MLP_DC3_seed0_dropout0.1",
        "20251102-112606_MLP_DC3_seed1_dropout0.1",
        "20251102-112806_MLP_DC3_seed2_dropout0.1",
        "20251102-113006_MLP_DC3_seed3_dropout0.1",
    ],
    [
        "20251031-165855_MLP_DC3_seed0_dropout0.1_finetune_20251031-162815_model_20",
        "20251031-170631_MLP_DC3_seed1_dropout0.1_finetune_20251031-162815_model_20",
        "20251031-171407_MLP_DC3_seed2_dropout0.1_finetune_20251031-162815_model_20",
    ], # 0.0
    [
        "20251031-172141_MLP_DC3_seed0_dropout0.1_finetune_20251031-162815_model_60",
        "20251031-172917_MLP_DC3_seed1_dropout0.1_finetune_20251031-162815_model_60",
        "20251031-173651_MLP_DC3_seed2_dropout0.1_finetune_20251031-162815_model_60",
    ],
    [
        "20251031-181657_MLP_DC3_seed0_dropout0.1_finetune_20251031-162815_model_100",
        "20251031-182432_MLP_DC3_seed1_dropout0.1_finetune_20251031-162815_model_100",
        "20251031-183207_MLP_DC3_seed2_dropout0.1_finetune_20251031-162815_model_100",
    ],
    [
        "20251031-183946_MLP_DC3_seed0_dropout0.1_finetune_20251031-162815_model_200",
        "20251031-184726_MLP_DC3_seed1_dropout0.1_finetune_20251031-162815_model_200",
        "20251031-185506_MLP_DC3_seed2_dropout0.1_finetune_20251031-162815_model_200",
    ],
])

dc3_ft_ckpt20_train_size7000_over_subopt = np.array([
    [
        "20251102-112406_MLP_DC3_seed0_dropout0.1",
        "20251102-112606_MLP_DC3_seed1_dropout0.1",
        "20251102-112806_MLP_DC3_seed2_dropout0.1",
        "20251102-113006_MLP_DC3_seed3_dropout0.1",
    ],
    [
        "20251031-165855_MLP_DC3_seed0_dropout0.1_finetune_20251031-162815_model_20",
        "20251031-170631_MLP_DC3_seed1_dropout0.1_finetune_20251031-162815_model_20",
        "20251031-171407_MLP_DC3_seed2_dropout0.1_finetune_20251031-162815_model_20",
    ], # 0.0
    [
        "20251031-170049_MLP_DC3_seed0_dropout0.1_finetune_20251031-163030_model_20",
        "20251031-170825_MLP_DC3_seed1_dropout0.1_finetune_20251031-163030_model_20",
        "20251031-171601_MLP_DC3_seed2_dropout0.1_finetune_20251031-163030_model_20",
    ],
    [
        "20251031-170244_MLP_DC3_seed0_dropout0.1_finetune_20251031-163247_model_20",
        "20251031-171019_MLP_DC3_seed1_dropout0.1_finetune_20251031-163247_model_20",
        "20251031-171754_MLP_DC3_seed2_dropout0.1_finetune_20251031-163247_model_20",
    ],
    [
        "20251031-170438_MLP_DC3_seed0_dropout0.1_finetune_20251031-163616_model_20",
        "20251031-171212_MLP_DC3_seed1_dropout0.1_finetune_20251031-163616_model_20",
        "20251031-171947_MLP_DC3_seed2_dropout0.1_finetune_20251031-163616_model_20",
    ],
])