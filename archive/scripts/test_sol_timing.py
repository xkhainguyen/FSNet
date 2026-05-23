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
def make_eval_parallel_time(base_trainer):
    def eval_parallel_time(net):
        net.eval()
        input = base_trainer.opt_problem.test_dataset[:][0]
        val_metrics = base_trainer.evaluator.evaluate_batch(net, input)
        total_time = val_metrics["sol_time"]
        return total_time
    return eval_parallel_time

def make_eval_serial_time(base_trainer):
    def eval_serial_time(net):
        # for each instance in loader, evaluate one by one
        net.eval()
        input = base_trainer.opt_problem.test_dataset[:][0]
        total_time = 0.0
        for i in range(input.shape[0]):
            input_i = input[i:i+1]
            val_metrics = base_trainer.evaluator.evaluate_batch(net, input_i)
            total_time += val_metrics["sol_time"]
        return total_time
    return eval_serial_time



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

    eval_parallel_time = make_eval_parallel_time(base_trainer)
    eval_serial_time = make_eval_serial_time(base_trainer)

    # Load model_A0
    args, config = create_parser([
        "--method", "FSNet",
        "--prob_type", "nonsmooth_nonconvex",
        "--prob_name", "socp",
        "--checkpoint",
        "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260115-012027_MLP_FSNet_seed0_nepochs300_lr0.0001_trainsize7000/model.pt",
    ])
    # args, config = create_parser([
    #     "--method", "FSNet",
    #     "--prob_type", "nonsmooth_nonconvex",
    #     "--prob_name", "socp",
    #     "--checkpoint",
    #     "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260116-031254_MLP_FSNet_seed0_nepochs300_lr0.0002_trainsize7000_finetune_20260116-022657_sup_seedpen_model_940/model.pt",
    # ])
    opt_problem, result_save_dir = load_instance(config)
    trainer_dummy = Trainer(opt_problem=opt_problem, config=config, save_dir=result_save_dir)
    model = trainer_dummy.train()
    model = model.to(DEVICE)

    # Pick device
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # print time per sample, run 4 times and take average of the last 3
    n_runs = 4
    parallel_times = []
    serial_times = []
    for run in range(n_runs):
        p_time = eval_parallel_time(model)
        s_time = eval_serial_time(model)
        print(f"Run {run+1}: Parallel time = {p_time:.4f}s, Serial time = {s_time:.4f}s")
        parallel_times.append(p_time)
        serial_times.append(s_time)
    # print mean and std
    parallel_times = np.array(parallel_times[1:])
    serial_times = np.array(serial_times[1:])
    print(f"Average Parallel time: {parallel_times.mean():.4f}s ± {parallel_times.std():.4f}s")
    print(f"Average Serial time: {serial_times.mean():.4f}s ± {serial_times.std():.4f}s")