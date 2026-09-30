"""Exact held-out own training loss and merit at every trained network (aligned checkpoints, seeds 0-9),
evaluated like the landscape grids (compute_landscape_compare.eval_points_grouped: 1000 test
instances, batches of 500, FS L-BFGS 50 iterations, tol 1e-9). Used to normalize landscapes by each
method's best trained network. Writes figures/landscape/fig1/seed_losses.json.
"""
import json

import torch

import compute_landscape_compare as C
from fig1_common import own_loss

A = "figures/landscape/fig1/aligned"
METHODS = ["M1", "M2", "M3f", "M4", "M3r1", "M3r08", "M1r1", "M4r1"]


def main():
    prob = C.load_problem()
    X, Y = [t[:1000].to(C.DEVICE) for t in prob.test_dataset.tensors]
    fs = dict(val_tol=1e-9, memory=30, max_iter=50, scale=1000)
    out = {}
    for m in METHODS:
        net, _ = C.load_model(prob, f"{A}/{m}_s0.pt")
        params = list(net.parameters())
        thetas = [torch.nn.utils.parameters_to_vector(C.load_model(prob, f"{A}/{m}_s{s}.pt")[0].parameters()).detach()
                  for s in range(10)]
        setters = [lambda th=th: torch.nn.utils.vector_to_parameters(th, params) for th in thetas]
        comps = C.eval_points_grouped(net, prob, X, Y, setters, fs if m.startswith("M3") else None, 500)
        rows = []
        for s, c in enumerate(comps):
            merit = c["obj_fs"] + 1e5 * c["viol_l1_fs"] if m.startswith("M3") else c["obj"] + 1e5 * c["viol_l1"]
            rows.append({"seed": s, "own": own_loss(m, c), "merit": merit})
        out[m] = rows
        print(m, " ".join(f"{r['own']:.4g}" for r in rows), flush=True)
    json.dump(out, open("figures/landscape/fig1/seed_losses.json", "w"), indent=1)


if __name__ == "__main__":
    main()
