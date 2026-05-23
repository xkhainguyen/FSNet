import pickle
from collections import Counter

import numpy as np
import torch

from eval import resolve_checkpoints, load_single_model
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
TEST_SLICE = slice(8000, 10000)


def evaluate_member(model, evaluator, dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    out = {k: [] for k in ["raw_merit", "final_merit", "move_l2", "Y_final"]}

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

        for k, v in {
            "raw_merit": raw_merit,
            "final_merit": final_merit,
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


def main():
    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)

    y_oracle = np.asarray(dataset["Y"])[TEST_SLICE]
    q = np.asarray(dataset["Q"])
    p = np.asarray(dataset["p"])
    oracle_obj = (0.5 * (y_oracle @ q) * y_oracle + p * np.sin(y_oracle)).sum(axis=1) + 0.1 * np.linalg.norm(y_oracle, axis=1) + 20.0
    intrinsic = (oracle_obj - oracle_obj.min()) / (oracle_obj.max() - oracle_obj.min())

    for seed, run_dir in RUN_GROUPS.items():
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
        raw_best = arr["raw_merit"].argmin(axis=0)
        final_best = arr["final_merit"].argmin(axis=0)
        best_single = arr["final_merit"].mean(axis=1).argmin()
        gain = arr["final_merit"][best_single] - arr["final_merit"].min(axis=0)
        final_pw = mean_pairwise_l2(arr["Y_final"])
        signatures = [tuple(np.argsort(arr["final_merit"][:, i]).tolist()) for i in range(arr["final_merit"].shape[1])]

        print(f"\nSEED {seed} TOP SIGNATURES")
        for sig, count in Counter(signatures).most_common(6):
            mask = np.array([s == sig for s in signatures])
            winner_members = final_best[mask]
            winner_move = arr["move_l2"][winner_members, np.arange(mask.sum())]
            flip = raw_best[mask] != final_best[mask]
            print(
                {
                    "sig": sig,
                    "count": count,
                    "share": round(count / len(signatures), 3),
                    "winner": sig[0],
                    "runner_up": sig[1],
                    "flip": round(float(flip.mean()), 3),
                    "gain": round(float(gain[mask].mean()), 3),
                    "move": round(float(winner_move.mean()), 3),
                    "intr": round(float(intrinsic[mask].mean()), 3),
                    "obj": round(float(oracle_obj[mask].mean()), 3),
                    "final_pw": round(float(final_pw[mask].mean()), 3),
                }
            )

        print("WINNER FAMILIES")
        for winner in sorted(np.unique(final_best)):
            mask = final_best == winner
            top2_family = Counter(tuple(list(s)[:2]) for s in np.array(signatures, dtype=object)[mask])
            dom_sig, dom_count = top2_family.most_common(1)[0]
            flip = raw_best[mask] != final_best[mask]
            print(
                {
                    "winner": int(winner),
                    "wins": int(mask.sum()),
                    "dom_top2": dom_sig,
                    "dom_share": round(dom_count / mask.sum(), 3),
                    "flip": round(float(flip.mean()), 3),
                    "gain": round(float(gain[mask].mean()), 3),
                    "move": round(float(arr["move_l2"][winner, mask].mean()), 3),
                    "intr": round(float(intrinsic[mask].mean()), 3),
                    "obj": round(float(oracle_obj[mask].mean()), 3),
                    "final_pw": round(float(final_pw[mask].mean()), 3),
                }
            )


if __name__ == "__main__":
    main()
