import os
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import numpy as np

from utils.trainer_2 import load_instance, Trainer

# ---------------------------------------------------------------------
# Global setup
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Config loader (kept compatible with your current workflow)
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Merit evaluation (scalar over a loader)
# ---------------------------------------------------------------------
def make_eval_merit(base_trainer):
    def eval_merit(net, loader):
        net.eval()
        metrics = base_trainer.evaluator.evaluate(net, loader)
        merit = (
            metrics["objective"]
            + eq_weight * metrics["eq_violation_l1_mean"]
            + ineq_weight * metrics["ineq_violation_l1_mean"]
        )
        # should already be float; just in case:
        if torch.is_tensor(merit):
            merit = float(merit.detach().cpu().item())
        else:
            merit = float(merit)
        return merit

    return eval_merit


# ---------------------------------------------------------------------
# Utilities to handle dataset sample structure
# Assumes dataset returns either:
#   - x
#   - (x, ...)
#   - {"x": x, ...} or dict with first tensor as x
# ---------------------------------------------------------------------
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
        if "x" in sample and torch.is_tensor(sample["x"]):
            s = dict(sample)
            s["x"] = new_x
            return s
        # replace first tensor key
        s = dict(sample)
        for k, v in s.items():
            if torch.is_tensor(v):
                s[k] = new_x
                return s
        raise ValueError("Dict sample has no tensor entry to treat as x.")

    raise TypeError(f"Unsupported sample type: {type(sample)}")


# ---------------------------------------------------------------------
# Infer data bounds for x from the test_loader
# (min/max per dimension)
# ---------------------------------------------------------------------
@torch.no_grad()
def infer_x_bounds_from_loader(test_loader):
    mins = None
    maxs = None

    for batch in test_loader:
        # batch could be tensor / tuple / dict
        if torch.is_tensor(batch):
            x = batch
        elif isinstance(batch, (tuple, list)):
            x = batch[0]
        elif isinstance(batch, dict):
            x = batch["x"] if "x" in batch else next(v for v in batch.values() if torch.is_tensor(v))
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        # ensure [B, D] shape
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = x.to(DEVICE)

        bmin = x.min(dim=0).values
        bmax = x.max(dim=0).values

        mins = bmin if mins is None else torch.minimum(mins, bmin)
        maxs = bmax if maxs is None else torch.maximum(maxs, bmax)

    return mins, maxs


# ---------------------------------------------------------------------
# Dataset wrapper: clamp two coordinates (dim_i, dim_j) of x
# for every sample, to the provided (alpha, beta).
# ---------------------------------------------------------------------
class Clamp2DimsDataset(Dataset):
    def __init__(self, base_dataset, *, dim_i, dim_j, alpha, beta):
        self.base = base_dataset
        self.dim_i = dim_i
        self.dim_j = dim_j
        self.alpha = alpha
        self.beta = beta

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        x = _extract_x_from_sample(sample)

        # x could be shape [D] or more; we assume last dim is feature dim
        x = x.clone()
        x[self.dim_i] = self.alpha
        x[self.dim_j] = self.beta

        return _set_x_in_sample(sample, x)


# ---------------------------------------------------------------------
# Compute input-space grid (within test data bounds) -> X, Y, Z
# Z is dataset-average merit for each (alpha, beta) clamp
# ---------------------------------------------------------------------
@torch.no_grad()
def compute_input_grid_surface_from_data(
    model,
    base_trainer,
    test_loader,
    *,
    dim_i=0,
    dim_j=1,
    xnum=51,
    ynum=51,
    margin=0.0,          # optionally shrink bounds inward (e.g., 0.02)
    levels_seed=0,       # only for print formatting cadence; not random
    verbose=True,
):
    model = model.to(DEVICE)
    model.eval()

    eval_merit = make_eval_merit(base_trainer)

    mins, maxs = infer_x_bounds_from_loader(test_loader)
    mins = mins.detach().cpu().numpy()
    maxs = maxs.detach().cpu().numpy()

    lo_i, hi_i = float(mins[dim_i]), float(maxs[dim_i])
    lo_j, hi_j = float(mins[dim_j]), float(maxs[dim_j])

    # optional margin shrink
    if margin > 0:
        span_i = hi_i - lo_i
        span_j = hi_j - lo_j
        lo_i += margin * span_i
        hi_i -= margin * span_i
        lo_j += margin * span_j
        hi_j -= margin * span_j

    xs = np.linspace(lo_i, hi_i, xnum, dtype=np.float32)  # alpha axis
    ys = np.linspace(lo_j, hi_j, ynum, dtype=np.float32)  # beta axis

    Z = np.empty((ynum, xnum), dtype=np.float32)

    base_dataset = base_trainer.opt_problem.test_dataset

    total = xnum * ynum
    k = 0

    for j, beta in enumerate(ys):
        for i, alpha in enumerate(xs):
            # Clamp every test sample to this (alpha,beta) for dims (i,j)
            ds = Clamp2DimsDataset(
                base_dataset,
                dim_i=dim_i,
                dim_j=dim_j,
                alpha=torch.tensor(alpha, device="cpu").item(),
                beta=torch.tensor(beta, device="cpu").item(),
            )
            loader = DataLoader(
                ds,
                batch_size=test_loader.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )

            Z[j, i] = float(eval_merit(model, loader))

            k += 1
            if verbose and (k % max(1, total // 20) == 0):
                print(
                    f"[{k:>6}/{total}] alpha={alpha:+.4g} beta={beta:+.4g} merit={Z[j,i]:.6g}",
                    flush=True,
                )

    X, Y = np.meshgrid(xs, ys)
    return X, Y, Z


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Build base_trainer (for evaluator + datasets)
    args, config = create_parser([
        "--method", "FSNet",
        "--prob_type", "nonsmooth_nonconvex",
        "--prob_name", "socp",
        "--batch_size", "256",
    ])
    opt_problem, result_save_dir = load_instance(config)
    base_trainer = Trainer(opt_problem=opt_problem, config=config, save_dir=result_save_dir)

    test_loader = DataLoader(
        base_trainer.opt_problem.test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )

    # 2) Load model checkpoint (EDIT THIS PATH)
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

    # 3) Compute input-grid landscape within test data bounds
    # EDIT: choose which two input dims to visualize
    dim_i = 0
    dim_j = 1

    X, Y, Z = compute_input_grid_surface_from_data(
        model,
        base_trainer,
        test_loader,
        dim_i=dim_i,
        dim_j=dim_j,
        xnum=51,
        ynum=51,
        margin=0.0,   # set e.g. 0.02 to avoid extreme edges if needed
        verbose=True,
    )

    # 4) Save X,Y,Z
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"input_merit_surface_dim{dim_i}_dim{dim_j}21312321.npz")
    np.savez(out_path, X=X, Y=Y, Z=Z)
    print(f"Saved X,Y,Z to: {out_path}", flush=True)
