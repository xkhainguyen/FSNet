import os
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
from utils.trainer_2 import load_instance, Trainer
import copy
import numpy as np
import matplotlib.pyplot as plt

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
# Merit evaluation (assumes evaluator already handles device placement)
# -----------------------------------------------------------------------------
def make_eval_merit(base_trainer):
    def eval_merit(net, loader):
        net.eval()
        val_metrics = base_trainer.evaluator.evaluate(net, loader)
        merit = (
            val_metrics["objective"]
            + eq_weight * val_metrics["eq_violation_l1_mean"]
            + ineq_weight * val_metrics["ineq_violation_l1_mean"]
        )
        return merit
    return eval_merit

# --------------------------
# Helpers
# --------------------------
def _to_scalar(x):
    if isinstance(x, (float, int)):
        return float(x)
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    raise TypeError(f"eval_merit must return float/int/tensor scalar, got {type(x)}")


def _move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return type(obj)(_move_to_device(v, device) for v in obj)
    return obj


def _detach_clone(obj):
    if torch.is_tensor(obj):
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {k: _detach_clone(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return type(obj)(_detach_clone(v) for v in obj)
    return obj


def _get_single_sample_from_loader(test_loader, device):
    """
    Returns:
      sample0: a single dataset sample (same structure as dataset __getitem__)
      x_get: function(sample)->x_tensor
      x_set: function(sample, new_x)->sample_with_new_x
    Supports:
      - sample is Tensor
      - sample is (x, ...)
      - sample is dict with key 'x' OR first tensor entry treated as x
    """
    batch0 = next(iter(test_loader))

    # Convert batch -> single sample with same structure
    if torch.is_tensor(batch0):
        sample0 = batch0[0]
        sample0 = _move_to_device(_detach_clone(sample0), device)

        def x_get(s): return s
        def x_set(s, new_x): return new_x

        return sample0, x_get, x_set

    if isinstance(batch0, (tuple, list)):
        # assume first element is x, and all tensor-like have batch dim
        elems = []
        for v in batch0:
            if torch.is_tensor(v):
                elems.append(v[0])
            else:
                elems.append(v)
        sample0 = tuple(elems) if isinstance(batch0, tuple) else list(elems)
        sample0 = _move_to_device(_detach_clone(sample0), device)

        def x_get(s): return s[0]

        def x_set(s, new_x):
            if isinstance(s, tuple):
                return (new_x,) + s[1:]
            s2 = list(s)
            s2[0] = new_x
            return s2

        return sample0, x_get, x_set

    if isinstance(batch0, dict):
        sample0 = {}
        for k, v in batch0.items():
            if torch.is_tensor(v):
                sample0[k] = v[0]
            else:
                sample0[k] = v
        sample0 = _move_to_device(_detach_clone(sample0), device)

        if "x" in sample0 and torch.is_tensor(sample0["x"]):
            x_key = "x"
        else:
            # pick first tensor key as x
            x_key = None
            for k, v in sample0.items():
                if torch.is_tensor(v):
                    x_key = k
                    break
            if x_key is None:
                raise ValueError("Could not find a tensor input in dict sample.")

        def x_get(s): return s[x_key]

        def x_set(s, new_x):
            s2 = dict(s)
            s2[x_key] = new_x
            return s2

        return sample0, x_get, x_set

    raise TypeError(f"Unsupported batch type from test_loader: {type(batch0)}")


class _SingleSampleDataset(Dataset):
    def __init__(self, sample):
        self.sample = sample

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.sample


def _single_loader(sample):
    # num_workers=0 avoids fork issues on clusters; pin_memory optional
    return DataLoader(_SingleSampleDataset(sample), batch_size=1, shuffle=False, num_workers=0)


def _random_directions_like(x0, seed):
    """
    Generate two reproducible random directions in input space.
    Compatible with older PyTorch versions.
    """
    g = torch.Generator(device=x0.device)
    g.manual_seed(seed)

    d1 = torch.randn(
        x0.shape,
        device=x0.device,
        dtype=x0.dtype,
        generator=g,
    )
    d2 = torch.randn(
        x0.shape,
        device=x0.device,
        dtype=x0.dtype,
        generator=g,
    )

    d1 = d1 / (d1.norm() + 1e-12)
    d2 = d2 / (d2.norm() + 1e-12)
    return d1, d2

# --------------------------
# Main: compute X, Y, Z
# --------------------------
@torch.no_grad()
def compute_input_merit_surface_2d(
    model,
    test_loader,
    eval_merit,
    *,
    device=None,
    xmin=-1.0, xmax=1.0, xnum=51,
    ymin=-1.0, ymax=1.0, ynum=51,
    eps=0.1,
    verbose=True,
):
    """
    Perturb a single base test input x0 as:
      x(alpha,beta) = x0 + eps * (alpha*d1 + beta*d2)
    Evaluate merit at each point using eval_merit(model, single_loader(sample)).

    Returns: X, Y, Z (numpy arrays)
    """
    if device is None:
        device = next(model.parameters()).device

    model = model.to(device)
    model.eval()

    sample0, x_get, x_set = _get_single_sample_from_loader(test_loader, device)
    x0 = x_get(sample0)

    if not torch.is_tensor(x0):
        raise ValueError("Extracted x0 is not a tensor; cannot build input-space surface.")

    d1, d2 = _random_directions_like(x0, 1)

    xs = np.linspace(xmin, xmax, xnum, dtype=np.float32)
    ys = np.linspace(ymin, ymax, ynum, dtype=np.float32)
    Z = np.empty((ynum, xnum), dtype=np.float32)

    total = xnum * ynum
    k = 0

    for j, beta in enumerate(ys):
        for i, alpha in enumerate(xs):
            x = x0 + eps * (float(alpha) * d1 + float(beta) * d2)
            sample = x_set(sample0, x)

            m = eval_merit(model, _single_loader(sample))
            Z[j, i] = _to_scalar(m)

            k += 1
            if verbose and (k % max(1, total // 20) == 0):
                print(f"[{k:>6}/{total}] alpha={alpha:+.3f} beta={beta:+.3f} merit={Z[j,i]:.6g}", flush=True)

    X, Y = np.meshgrid(xs, ys)
    return X, Y, Z


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Build base_trainer (for evaluator + datasets)
    args, config = create_parser([
        "--method", "FSNet",
        "--prob_type", "nonsmooth_nonconvex",
        "--prob_name", "socp",
    ])
    opt_problem, result_save_dir = load_instance(config)
    base_trainer = Trainer(opt_problem=opt_problem, config=config, save_dir=result_save_dir)

    test_loader = DataLoader(
        base_trainer.opt_problem.test_dataset,
        batch_size=256,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )

    eval_merit = make_eval_merit(base_trainer)

    # Load model_A0
    args, config = create_parser([
        "--method", "FSNet",
        "--prob_type", "nonsmooth_nonconvex",
        "--prob_name", "socp",
        "--checkpoint",
        "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260117-191337_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_150/model.pt",
    ])
    opt_problem, result_save_dir = load_instance(config)
    trainer_dummy = Trainer(opt_problem=opt_problem, config=config, save_dir=result_save_dir)
    model_A0 = trainer_dummy.train()
    model_A0 = model_A0.to(DEVICE)

    # Load model_B0
    args, config = create_parser([
        "--method", "FSNet",
        "--prob_type", "nonsmooth_nonconvex",
        "--prob_name", "socp",
        "--checkpoint",
        "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260115-013254_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000/model.pt",
    ])
    opt_problem, result_save_dir = load_instance(config)
    trainer_dummy = Trainer(opt_problem=opt_problem, config=config, save_dir=result_save_dir)
    model_B0 = trainer_dummy.train()
    model_B0 = model_B0.to(DEVICE)

    # Pick device
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- EDIT: choose your grid + step ----
    xmin, xmax, xnum = -1.0, 1.0, 51
    ymin, ymax, ynum = -1.0, 1.0, 51
    eps = 0.1  # scale of input perturbation (increase/decrease as needed)

    # ---- Compute surface ----
    X, Y, Z = compute_input_merit_surface_2d(
        model_B0,
        test_loader,
        eval_merit,
        device=DEVICE,
        xmin=xmin, xmax=xmax, xnum=xnum,
        ymin=ymin, ymax=ymax, ynum=ynum,
        eps=eps,
        verbose=True,
    )

    # ---- Save ----
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "vanilla_input_merit_surface1.npz")
    np.savez(out_path, X=X, Y=Y, Z=Z)
    print(f"Saved X,Y,Z to: {out_path}", flush=True)