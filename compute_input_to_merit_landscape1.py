import os
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import numpy as np

from utils.trainer_2 import load_instance, Trainer

# -----------------------------------------------------------------------------
# Global setup
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

PROBLEM_TYPES = ["convex", "nonconvex", "nonsmooth_nonconvex"]
PROBLEM_NAMES = ["qp", "qcqp", "socp"]

eq_weight = 1e5
ineq_weight = 1e5


# -----------------------------------------------------------------------------
# Config loader
# -----------------------------------------------------------------------------
def create_parser(arg_list=None):
    parser = argparse.ArgumentParser(description="Neural Network Optimization")

    # General parameters
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--method", type=str)
    parser.add_argument("--prob_type", type=str, choices=PROBLEM_TYPES)
    parser.add_argument("--prob_name", type=str, choices=PROBLEM_NAMES)
    parser.add_argument("--prob_size", type=int, nargs="+", default=[100, 50, 50, 10000])
    parser.add_argument("--network", type=str, default="MLP")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--ablation", type=bool, default=False)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--en_subopt", type=bool, default=False)
    parser.add_argument("--subopt_ratio", type=float, default=0.0)
    parser.add_argument("--save_intermediate", type=bool, default=False)

    # dataset parameters
    parser.add_argument("--train_size", type=int, default=-1)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--val_size", type=int)
    parser.add_argument("--test_size", type=int)
    parser.add_argument("--dropout", type=float)

    # Neural network parameters
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lr_decay", type=float)
    parser.add_argument("--lr_decay_step", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--num_layers", type=int)

    # Feasibility seeking parameters
    parser.add_argument("--scale", type=float)
    parser.add_argument("--dist_weight", type=float)
    parser.add_argument("--max_diff_iter", type=int)

    args = parser.parse_args(arg_list)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # overrides
    if args.method:
        config["seed"] = args.seed
        config["method"] = args.method
    if args.prob_type:
        config["prob_type"] = args.prob_type
    if args.prob_name:
        config["prob_name"] = args.prob_name
    if args.prob_size:
        config["prob_size"] = args.prob_size
    if args.network:
        config["network"] = args.network

    config["checkpoint"] = args.checkpoint
    config["en_subopt"] = args.en_subopt
    config["subopt_ratio"] = args.subopt_ratio
    config["save_intermediate"] = args.save_intermediate

    # dataset overrides
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.train_size:
        config["train_size"] = args.train_size
    if args.val_size:
        config["val_size"] = args.val_size
    if args.test_size:
        config["test_size"] = args.test_size

    # NN overrides
    if args.lr:
        config["lr"] = args.lr
    if args.lr_decay:
        config["lr_decay"] = args.lr_decay
    if args.lr_decay_step:
        config["lr_decay_step"] = args.lr_decay_step
    if args.num_epochs:
        config["num_epochs"] = args.num_epochs
    if args.hidden_dim:
        config["hidden_dim"] = args.hidden_dim
    if args.num_layers:
        config["num_layers"] = args.num_layers
    if args.dropout:
        config["dropout"] = args.dropout

    # FS params
    if args.scale:
        config["FSNet"]["scale"] = args.scale
        config["S3Net"]["scale"] = args.scale
        config["semi"]["scale"] = args.scale
    if args.dist_weight is not None:
        config["FSNet"]["dist_weight"] = args.dist_weight
        config["S3Net"]["dist_weight"] = args.dist_weight
        config["semi"]["dist_weight"] = args.dist_weight
    if args.max_diff_iter is not None:
        config["FSNet"]["max_diff_iter"] = args.max_diff_iter
        config["S3Net"]["max_diff_iter"] = args.max_diff_iter
        config["semi"]["max_diff_iter"] = args.max_diff_iter

    config["ablation"] = args.ablation
    return args, config


# -----------------------------------------------------------------------------
# Sample structure helpers
# Assumes dataset returns: x | (x, ...) | dict with "x" or first tensor as x
# -----------------------------------------------------------------------------
def _extract_x_from_sample(sample):
    if torch.is_tensor(sample):
        return sample
    if isinstance(sample, (tuple, list)):
        return sample[0]
    if isinstance(sample, dict):
        if "x" in sample and torch.is_tensor(sample["x"]):
            return sample["x"]
        for v in sample.values():
            if torch.is_tensor(v):
                return v
        raise ValueError("Dict sample has no tensor entry to treat as x.")
    raise TypeError(f"Unsupported sample type: {type(sample)}")


def _set_x_in_sample(sample, new_x):
    if torch.is_tensor(sample):
        return new_x
    if isinstance(sample, tuple):
        return (new_x,) + sample[1:]
    if isinstance(sample, list):
        s = list(sample)
        s[0] = new_x
        return s
    if isinstance(sample, dict):
        s = dict(sample)
        if "x" in s and torch.is_tensor(s["x"]):
            s["x"] = new_x
            return s
        for k, v in s.items():
            if torch.is_tensor(v):
                s[k] = new_x
                return s
        raise ValueError("Dict sample has no tensor entry to treat as x.")
    raise TypeError(f"Unsupported sample type: {type(sample)}")


# -----------------------------------------------------------------------------
# Infer bounds from a loader of the base dataset
# -----------------------------------------------------------------------------
@torch.no_grad()
def infer_x_bounds_from_loader(test_loader):
    mins, maxs = None, None
    for batch in test_loader:
        if torch.is_tensor(batch):
            x = batch
        elif isinstance(batch, (tuple, list)):
            x = batch[0]
        elif isinstance(batch, dict):
            x = batch["x"] if "x" in batch else next(v for v in batch.values() if torch.is_tensor(v))
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = x.to(DEVICE, non_blocking=True)

        bmin = x.min(dim=0).values
        bmax = x.max(dim=0).values

        mins = bmin if mins is None else torch.minimum(mins, bmin)
        maxs = bmax if maxs is None else torch.maximum(maxs, bmax)

    return mins, maxs


# -----------------------------------------------------------------------------
# Big dataset: Cartesian product of (grid point) x (test sample)
# Each item returns:
#   (X_grid_sample, grid_id)
# where X_grid_sample has dims dim_i/dim_j clamped to grid coordinates.
# -----------------------------------------------------------------------------
class GridPointCartesianTestDataset(Dataset):
    def __init__(self, base_dataset, alphas, betas, *, dim_i, dim_j):
        assert len(alphas) == len(betas)
        self.base = base_dataset
        self.alphas = alphas
        self.betas = betas
        self.dim_i = dim_i
        self.dim_j = dim_j
        self.G = len(alphas)
        self.N = len(base_dataset)

    def __len__(self):
        return self.G * self.N

    def __getitem__(self, idx):
        g = idx // self.N
        n = idx - g * self.N

        sample = self.base[n]
        x = _extract_x_from_sample(sample).clone()

        x[self.dim_i] = float(self.alphas[g])
        x[self.dim_j] = float(self.betas[g])

        # For your evaluate_batch: we only want the input tensor, not (x,y) etc.
        # If your base dataset returns (X, Y_true), we drop Y_true here.
        return x, g


def _collate_x_and_grid_id(batch):
    xs, gids = zip(*batch)
    X = torch.stack(xs, dim=0)  # [B, D]
    gids = torch.tensor(gids, dtype=torch.long)
    return X, gids


# -----------------------------------------------------------------------------
# Batched grid computation using evaluator.evaluate_batch(model, input_batch)
# -----------------------------------------------------------------------------
@torch.no_grad()
def compute_grid_merit_batched(
    model,
    base_trainer,
    base_dataset,
    *,
    dim_i=0,
    dim_j=1,
    xnum=51,
    ynum=51,
    margin=0.0,
    batch_size=256,
    num_workers=0,
    verbose=True,
):
    # 1) Infer bounds on the ORIGINAL test data
    tmp_loader = DataLoader(
        base_dataset,
        batch_size=4096,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    mins, maxs = infer_x_bounds_from_loader(tmp_loader)
    mins = mins.detach().cpu().numpy()
    maxs = maxs.detach().cpu().numpy()

    lo_i, hi_i = float(mins[dim_i]), float(maxs[dim_i])
    lo_j, hi_j = float(mins[dim_j]), float(maxs[dim_j])

    if margin > 0:
        lo_i = lo_i + margin * (hi_i - lo_i)
        hi_i = hi_i - margin * (hi_i - lo_i)
        lo_j = lo_j + margin * (hi_j - lo_j)
        hi_j = hi_j - margin * (hi_j - lo_j)

    xs = np.linspace(lo_i, hi_i, xnum, dtype=np.float32)  # alpha axis
    ys = np.linspace(lo_j, hi_j, ynum, dtype=np.float32)  # beta axis
    Xg, Yg = np.meshgrid(xs, ys)

    # 2) Flatten grid (row-major: beta outer, alpha inner)
    alphas = np.tile(xs[None, :], (ynum, 1)).reshape(-1)  # [G]
    betas = np.tile(ys[:, None], (1, xnum)).reshape(-1)   # [G]
    G = alphas.shape[0]

    # 3) Build big dataset + loader
    big_ds = GridPointCartesianTestDataset(
        base_dataset, alphas, betas, dim_i=dim_i, dim_j=dim_j
    )

    loader = DataLoader(
        big_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_x_and_grid_id,
    )

    # 4) Accumulate per-grid sums on CPU (scatter_add)
    sum_obj = torch.zeros(G, device="cpu")
    sum_eq = torch.zeros(G, device="cpu")
    sum_ineq = torch.zeros(G, device="cpu")
    count = torch.zeros(G, device="cpu")

    model = model.to(DEVICE).eval()

    if not hasattr(base_trainer.evaluator, "evaluate_batch"):
        raise AttributeError("base_trainer.evaluator must implement evaluate_batch(model, input_batch).")

    total = len(loader)
    for it, (X_batch, gids) in enumerate(loader, start=1):
        # evaluate_batch expects input_batch tensor [B, D]
        metrics = base_trainer.evaluator.evaluate_batch(model, X_batch)

        # Ensure dtype matches accumulator dtype (float32)
        obj   = metrics["objective"].to(dtype=sum_obj.dtype, device="cpu")
        eqv   = metrics["eq_violation_l1"].to(dtype=sum_eq.dtype, device="cpu")
        ineqv = metrics["ineq_violation_l1"].to(dtype=sum_ineq.dtype, device="cpu")

        gids = gids.to("cpu")

        sum_obj.scatter_add_(0, gids, obj)
        sum_eq.scatter_add_(0, gids, eqv)
        sum_ineq.scatter_add_(0, gids, ineqv)
        count.scatter_add_(0, gids, torch.ones_like(obj))

        if verbose and (it % max(1, total // 20) == 0):
            print(f"[{it:>6}/{total}] processed", flush=True)

    mean_obj = sum_obj / count.clamp_min(1)
    mean_eq = sum_eq / count.clamp_min(1)
    mean_ineq = sum_ineq / count.clamp_min(1)

    Z_flat = mean_obj + eq_weight * mean_eq + ineq_weight * mean_ineq
    Z = Z_flat.numpy().reshape(ynum, xnum)

    return Xg, Yg, Z


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Base trainer for dataset + evaluator
    args, config = create_parser([
        "--method", "FSNet",
        "--prob_type", "nonsmooth_nonconvex",
        "--prob_name", "socp",
        "--batch_size", "256",
    ])
    opt_problem, result_save_dir = load_instance(config)
    base_trainer = Trainer(opt_problem=opt_problem, config=config, save_dir=result_save_dir)

    # Load model checkpoint (EDIT PATH)
    args, config = create_parser([
        "--method", "FSNet",
        "--prob_type", "nonsmooth_nonconvex",
        "--prob_name", "socp",
        "--checkpoint",
        "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260115-013254_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000/model.pt",
        "--batch_size", "256",
    ])
    opt_problem, result_save_dir = load_instance(config)
    trainer_dummy = Trainer(opt_problem=opt_problem, config=config, save_dir=result_save_dir)
    model = trainer_dummy.train().to(DEVICE)

    # Compute batched grid surface
    X, Y, Z = compute_grid_merit_batched(
        model,
        base_trainer,
        base_trainer.opt_problem.test_dataset,
        dim_i=0,        # EDIT: which 2 input dims to vary
        dim_j=1,
        xnum=51,
        ynum=51,
        margin=0.0,
        batch_size=256,
        num_workers=0,
        verbose=True,
    )

    # Save
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "input_grid_merit_surface_batched.npz")
    np.savez(out_path, X=X, Y=Y, Z=Z)
    print(f"Saved X,Y,Z to: {out_path}", flush=True)
