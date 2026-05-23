import itertools
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
import warnings

from eval import load_single_model, resolve_checkpoints
from utils.evaluator import Evaluator
from utils.trainer import DEVICE, load_instance


RUN_GROUPS = {
    0: "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260315-172824_FSNet_seed0_e300_lr1e-04_n7000_ens5_vanilla_pre",
    100: "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260315-174003_FSNet_seed100_e300_lr1e-04_n7000_ens5_vanilla_pre",
    2025: "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260316-003057_FSNet_seed2025_e300_lr1e-04_n7000_ens5_vanilla_pre",
}

DATASET_PATH = "datasets/nonsmooth_nonconvex/socp/random2025_socp_dataset_var100_ineq50_eq50_ex10000"
BATCH_SIZE = 256
MERIT_EQ_WEIGHT = 1e5
MERIT_INEQ_WEIGHT = 1e5
OUT_PATH = "results/fsnet_winner_metric_regions.png"


def evaluate_member(model, evaluator, dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    out = {k: [] for k in ["raw_merit", "final_merit", "repair_gain", "move_l2", "Y_final"]}

    for x_batch, _ in loader:
        x_batch = x_batch.to(DEVICE)
        with torch.no_grad():
            y_raw = evaluator.opt_problem.scale(model(x_batch))
            y_final = evaluator._post_process_predictions(x_batch, y_raw)
            raw_merit = (
                evaluator.opt_problem.obj_fn(y_raw)
                + MERIT_EQ_WEIGHT * evaluator.opt_problem.eq_resid(x_batch, y_raw).abs().sum(dim=1)
                + MERIT_INEQ_WEIGHT * evaluator.opt_problem.ineq_resid(x_batch, y_raw).abs().sum(dim=1)
            )
            final_merit = (
                evaluator.opt_problem.obj_fn(y_final)
                + MERIT_EQ_WEIGHT * evaluator.opt_problem.eq_resid(x_batch, y_final).abs().sum(dim=1)
                + MERIT_INEQ_WEIGHT * evaluator.opt_problem.ineq_resid(x_batch, y_final).abs().sum(dim=1)
            )
            move_l2 = torch.linalg.norm(y_final - y_raw, dim=1)
            repair_gain = raw_merit - final_merit

        for k, v in {
            "raw_merit": raw_merit,
            "final_merit": final_merit,
            "repair_gain": repair_gain,
            "move_l2": move_l2,
            "Y_final": y_final,
        }.items():
            out[k].append(v.detach().cpu().numpy())

    return {k: np.concatenate(v) for k, v in out.items()}


def mean_pairwise_l2(preds):
    vals = []
    n_members = preds.shape[0]
    for i in range(n_members):
        for j in range(i + 1, n_members):
            vals.append(np.linalg.norm(preds[i] - preds[j], axis=1))
    return np.mean(np.stack(vals, axis=0), axis=0)


def load_run(seed, run_dir):
    ckpts = resolve_checkpoints(run_dir)
    first = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    cfg = dict(first["config"])
    cfg["ensemble_post"] = "post"
    cfg["ensemble_agg"] = "best_merit"
    cfg["_eval_only"] = True
    problem, _ = load_instance(cfg)
    evaluator = Evaluator(problem, "FSNet", cfg)

    member_out = []
    for ckpt in ckpts:
        model, _ = load_single_model(ckpt, problem)
        member_out.append(evaluate_member(model, evaluator, problem.test_dataset))

    arr = {k: np.stack([m[k] for m in member_out], axis=0) for k in member_out[0].keys()}
    winner = arr["final_merit"].argmin(axis=0)
    sorted_raw = np.sort(arr["raw_merit"], axis=0)
    sorted_final = np.sort(arr["final_merit"], axis=0)
    features = {
        "raw_disagree_std": arr["raw_merit"].std(axis=0),
        "repair_gain_std": arr["repair_gain"].std(axis=0),
        "move_std": arr["move_l2"].std(axis=0),
        "final_pw_l2": mean_pairwise_l2(arr["Y_final"]),
        "raw_margin": sorted_raw[1] - sorted_raw[0],
        "final_margin": sorted_final[1] - sorted_final[0],
        "winner_move_l2": arr["move_l2"][winner, np.arange(len(winner))],
        "winner_repair_gain": arr["repair_gain"][winner, np.arange(len(winner))],
    }
    return {"seed": seed, "winner": winner, "features": features, "win_counts": np.bincount(winner, minlength=arr["final_merit"].shape[0])}


def score_pair(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=0, stratify=y
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        clf = QuadraticDiscriminantAnalysis(reg_param=1e-3)
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)
    return balanced_accuracy_score(y_test, pred)


def main():
    runs = [load_run(seed, run_dir) for seed, run_dir in RUN_GROUPS.items()]
    feature_names = list(runs[0]["features"].keys())

    fig, axes = plt.subplots(1, len(runs), figsize=(7 * len(runs), 6), constrained_layout=True)
    if len(runs) == 1:
        axes = [axes]

    summary_rows = []
    cmap = plt.get_cmap("tab10")

    for ax, rr in zip(axes, runs):
        y = rr["winner"]
        active_members = np.where(rr["win_counts"] >= 25)[0]
        mask = np.isin(y, active_members)
        y_use = y[mask]

        best = None
        for f1, f2 in itertools.combinations(feature_names, 2):
            x = np.column_stack([rr["features"][f1][mask], rr["features"][f2][mask]])
            try:
                score = score_pair(x, y_use)
            except Exception:
                continue
            if best is None or score > best[0]:
                best = (score, f1, f2, x)

        score, f1, f2, x = best
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            clf = QuadraticDiscriminantAnalysis(reg_param=1e-3)
            clf.fit(x, y_use)

        x1_min, x1_max = np.quantile(x[:, 0], [0.01, 0.99])
        x2_min, x2_max = np.quantile(x[:, 1], [0.01, 0.99])
        pad1 = 0.08 * max(x1_max - x1_min, 1e-9)
        pad2 = 0.08 * max(x2_max - x2_min, 1e-9)
        gx, gy = np.meshgrid(
            np.linspace(x1_min - pad1, x1_max + pad1, 300),
            np.linspace(x2_min - pad2, x2_max + pad2, 300),
        )
        grid = np.column_stack([gx.ravel(), gy.ravel()])
        gz = clf.predict(grid).reshape(gx.shape)

        levels = np.arange(gz.max() + 2) - 0.5
        ax.contourf(gx, gy, gz, levels=levels, cmap="Pastel1", alpha=0.45)

        for m in active_members:
            m_mask = y_use == m
            ax.scatter(
                x[m_mask, 0],
                x[m_mask, 1],
                s=14,
                alpha=0.8,
                color=cmap(m),
                label=f"Member {m}",
            )

        ax.set_title(f"FSNet seed {rr['seed']}\n{f1} vs {f2}\nbalanced acc={score:.3f}")
        ax.set_xlabel(f1)
        ax.set_ylabel(f2)
        ax.legend(loc="best", fontsize=8)
        summary_rows.append((rr["seed"], f1, f2, score, list(active_members)))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=180)
    print(f"saved_plot: {OUT_PATH}")
    print("best_pairs:")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
