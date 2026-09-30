"""Are the SL labels global optima? Multistart IPOPT on held-out instances.

The labels (datasets/nonsmooth_nonconvex/socp/make_data.py) are single IPOPT solves from CasADi's
default start (zero) of a nonconvex problem: min 0.5 y'Qy + p'sin(y) + 0.1 ||y|| s.t. A y = x,
||G_i cos(y) + h_i|| <= C_i'y + d_i, -5 <= y <= 5. For each of the first N held-out (test) instances
this re-solves the same NLP from K random starts (uniform in [-5, 5]; start 0 = zero, the label's
own start) and compares the best feasible objective with the label's objective.
Writes figures/landscape/fig1/label_multistart.json and prints a summary.

  python label_multistart.py --n 100 --starts 20 --workers 16
"""
import argparse
import json
import pickle
from multiprocessing import Pool

import numpy as np

PATH = "datasets/nonsmooth_nonconvex/socp/random2025_socp_dataset_var100_ineq50_eq50_ex10000"
D = pickle.load(open(PATH, "rb"))
TEST0 = 7000 + 1000  # load_problem(train 7000, val 1000, test 2000): the test split follows train and val


def objective(y):
    return 0.5 * y @ (D["Q"] @ y) + D["p"] @ np.sin(y) + 0.1 * np.linalg.norm(y)


def violation(y, x):
    eq = np.abs(D["A"] @ y - x).max()
    ineq = max(0.0, max(np.linalg.norm(D["G"][i] @ np.cos(y) + D["h"][i]) - (D["C"][i] @ y + D["d"][i])
                        for i in range(len(D["C"]))))
    box = max(0.0, np.abs(y).max() - 5)
    return max(eq, ineq, box)


def solve_instance(args):
    import casadi as ca
    k, starts, seed = args
    x = D["X"][TEST0 + k]
    n, m, q = D["Q"].shape[0], D["A"].shape[0], len(D["C"])
    y = ca.MX.sym("y", n)
    t = ca.MX.sym("t")
    f = 0.5 * ca.mtimes(y.T, ca.mtimes(D["Q"], y)) + ca.dot(D["p"], ca.sin(y)) + 0.1 * t
    g = [D["A"] @ y - x] + [ca.norm_2(D["G"][i] @ ca.cos(y) + D["h"][i]) - (ca.dot(D["C"][i], y) + D["d"][i])
                            for i in range(q)] + [ca.dot(y, y) - t ** 2]
    solver = ca.nlpsol("s", "ipopt", {"x": ca.vertcat(y, t), "f": f, "g": ca.vertcat(*g)},
                       {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": 3000})
    lbg = np.r_[np.zeros(m), -np.inf * np.ones(q + 1)]
    ubg = np.r_[np.zeros(m), np.zeros(q + 1)]
    lbx, ubx = np.r_[-5 * np.ones(n), 0.0], np.r_[5 * np.ones(n), np.inf]
    rng = np.random.default_rng(seed + k)
    label = np.asarray(D["Y"][TEST0 + k])
    out = {"k": k, "label_obj": float(objective(label)), "label_viol": float(violation(label, x)), "starts": []}
    for s in range(starts):
        y0 = np.zeros(n) if s == 0 else rng.uniform(-5, 5, n)
        res = solver(x0=np.r_[y0, np.linalg.norm(y0)], lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
        ys = res["x"].full().ravel()[:n]
        out["starts"].append({"ok": bool(solver.stats()["success"]), "obj": float(objective(ys)),
                              "viol": float(violation(ys, x)), "dist_to_label": float(np.linalg.norm(ys - label))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--starts", type=int, default=20)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tol", type=float, default=1e-6, help="max violation to count a solution as feasible")
    args = ap.parse_args()
    with Pool(args.workers) as pool:
        res = pool.map(solve_instance, [(k, args.starts, 0) for k in range(args.n)])
    json.dump(res, open("figures/landscape/fig1/label_multistart.json", "w"))
    gaps, better, distinct = [], 0, []
    for r in res:
        feas = [s for s in r["starts"] if s["viol"] < args.tol]
        best = min(s["obj"] for s in feas) if feas else np.inf
        gaps.append(r["label_obj"] - best)
        better += best < r["label_obj"] - 1e-6
        distinct.append(len({round(s["obj"], 4) for s in feas}))
        r0 = r["starts"][0]
    gaps = np.array(gaps)
    print(f"instances {args.n}, starts {args.starts}")
    print(f"label objective: mean {np.mean([r['label_obj'] for r in res]):.4f}, max violation "
          f"{max(r['label_viol'] for r in res):.2e}")
    print(f"start 0 (zero, the label's start) reproduces the label: "
          f"{np.mean([abs(r['starts'][0]['obj'] - r['label_obj']) < 1e-4 for r in res]):.0%}")
    print(f"instances where some start finds a better feasible point than the label: {better}/{args.n}")
    print(f"improvement label - best: median {np.median(gaps):.4f}, mean {gaps.mean():.4f}, max {gaps.max():.4f}")
    print(f"distinct feasible local optima per instance (rounded 1e-4): median {np.median(distinct):.0f}, "
          f"max {max(distinct)}")


if __name__ == "__main__":
    main()
