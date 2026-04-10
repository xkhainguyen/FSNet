import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset

from eval import load_single_model, resolve_checkpoints
from utils.evaluator import Evaluator
from utils.trainer import DEVICE, load_instance


RUN = "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260315-172824_FSNet_seed0_e300_lr1e-04_n7000_ens5_vanilla_pre"
OUT = "results/fsnet_seed0_winner_regions_fast.png"
N = 50
BATCH_SIZE = 64
MERIT_EQ_WEIGHT = 1e5
MERIT_INEQ_WEIGHT = 1e5


def eval_member(model, evaluator, dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    out = {k: [] for k in ["final_merit", "move_l2"]}

    for x_batch, _ in loader:
        x_batch = x_batch.to(DEVICE)
        with torch.no_grad():
            y_raw = evaluator.opt_problem.scale(model(x_batch))
            y_final = evaluator._post_process_predictions(x_batch, y_raw)
            final_merit = (
                evaluator.opt_problem.obj_fn(y_final)
                + MERIT_EQ_WEIGHT * evaluator.opt_problem.eq_resid(x_batch, y_final).abs().sum(dim=1)
                + MERIT_INEQ_WEIGHT * evaluator.opt_problem.ineq_resid(x_batch, y_final).abs().sum(dim=1)
            )
            move_l2 = torch.linalg.norm(y_final - y_raw, dim=1)

        out["final_merit"].append(final_merit.detach().cpu().numpy())
        out["move_l2"].append(move_l2.detach().cpu().numpy())

    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    print("loading checkpoints...", flush=True)
    ckpts = resolve_checkpoints(RUN)
    first = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    cfg = dict(first["config"])
    cfg["ensemble_post"] = "post"
    cfg["ensemble_agg"] = "best_merit"
    cfg["_eval_only"] = True

    problem, _ = load_instance(cfg)
    evaluator = Evaluator(problem, "FSNet", cfg)
    x_test, y_test = problem.test_dataset.tensors
    subset = TensorDataset(x_test[:N], y_test[:N])

    member_out = []
    for i, ckpt in enumerate(ckpts):
        print(f"evaluating member {i}...", flush=True)
        model, _ = load_single_model(ckpt, problem)
        member_out.append(eval_member(model, evaluator, subset))

    final_merit = np.stack([m["final_merit"] for m in member_out], axis=0)
    move_l2 = np.stack([m["move_l2"] for m in member_out], axis=0)

    winner = final_merit.argmin(axis=0)
    disagreement_std = final_merit.std(axis=0)
    winner_move_l2 = move_l2[winner, np.arange(len(winner))]

    active_members = np.where(np.bincount(winner, minlength=final_merit.shape[0]) >= 5)[0]
    mask = np.isin(winner, active_members)

    x = disagreement_std[mask]
    y = winner_move_l2[mask]
    labels = winner[mask]

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    for m in active_members:
        mm = labels == m
        ax.scatter(x[mm], y[mm], s=28, alpha=0.85, color=cmap(m), label=f"Member {m}")

    ax.set_xlabel("disagreement_std")
    ax.set_ylabel("winner_move_l2")
    ax.set_title(f"FSNet seed0 winner regions (first {N} test samples)")
    ax.legend(fontsize=8)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"saved_plot: {OUT}", flush=True)


if __name__ == "__main__":
    main()
