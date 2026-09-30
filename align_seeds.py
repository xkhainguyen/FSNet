"""Permutation-align trained seeds to seed 0 (weight matching, Ainsworth et al. "Git Re-Basin").

MLP layout (dropout 0): mlp.0 Linear(in,h), SiLU, mlp.2/4/6 Linear(h,h), SiLU, mlp.8 Linear(h,out).
For each hidden layer l, B's units are reordered by pi_l to maximize <W_A, W_B'> summed over
the incoming weights (with pi_{l-1}), the bias, and the outgoing weights (with pi_{l+1});
coordinate ascent over layers with the Hungarian algorithm until no layer changes.
Permuting hidden units leaves the network function unchanged (checked on held-out inputs).

--method act: activation matching instead. For each hidden layer, B's units are reordered to
maximize the summed correlation of post-SiLU activations with A's units on training inputs
(Hungarian per layer; layers are independent because activations do not depend on permutations).

Writes aligned checkpoints to figures/landscape/fig1/aligned/{M}_s{k}.pt (--method act:
figures/landscape/fig1/aligned_act/) and prints the relative weight distance to seed 0 before and
after alignment.
"""
import glob
import os

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from compute_landscape_compare import load_problem, load_model

D = "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000"
OUT = "figures/landscape/fig1/aligned"
SPEC = {"M1": ("sup_pen", "obj0.1_eq10.0_ineq10.0", "0.0003"),
        "M2": ("sup_pen", "obj0.1_eq100000.0_ineq100000.0", "0.0001"),
        "M4": ("penalty", "obj1.0_eq10.0_ineq10.0", "0.001")}
LAYERS = ["mlp.0", "mlp.2", "mlp.4", "mlp.6", "mlp.8"]  # last one is the output layer


def ckpt(meth, w, lr, s):
    runs = [r for r in sorted(glob.glob(f"{D}/*_MLP_{meth}_seed{s}_nepochs3000_lr{lr}_trainsize7000_{w}_penl1_dropout0.0_lrschedcosine_etamin1e-06"))
            if os.path.exists(r + "/model.pt")]
    return runs[-1] + "/model.pt"


def weight_matching(A, B, iters=50):
    """A, B: state dicts. Returns permutations (index arrays) for the 4 hidden layers."""
    W = lambda sd, l: sd[f"{l}.weight"].double().cpu().numpy()
    bias = lambda sd, l: sd[f"{l}.bias"].double().cpu().numpy()
    nh = len(LAYERS) - 1
    h = W(A, LAYERS[0]).shape[0]
    perms = [np.arange(h) for _ in range(nh)]
    rng = np.random.default_rng(0)
    for _ in range(iters):
        changed = False
        for l in rng.permutation(nh):
            WA, WB = W(A, LAYERS[l]), W(B, LAYERS[l])
            if l > 0:
                WB = WB[:, perms[l - 1]]
            S = WA @ WB.T + np.outer(bias(A, LAYERS[l]), bias(B, LAYERS[l]))
            WA_out, WB_out = W(A, LAYERS[l + 1]), W(B, LAYERS[l + 1])
            if l + 1 < nh:
                WB_out = WB_out[perms[l + 1], :]
            S += WA_out.T @ WB_out
            _, col = linear_sum_assignment(S, maximize=True)
            if not np.array_equal(col, perms[l]):
                perms[l], changed = col, True
        if not changed:
            break
    return perms


def hidden_acts(net, X):
    """Post-activation outputs of the 4 hidden layers (mlp.1, 3, 5, 7), each (n, h)."""
    acts, hooks = [], [net.mlp[k].register_forward_hook(lambda _m, _i, o: acts.append(o.detach()))
                        for k in (1, 3, 5, 7)]
    with torch.no_grad():
        net(X)
    for h in hooks:
        h.remove()
    return acts


def activation_matching(netA, netB, X):
    perms = []
    for a, b in zip(hidden_acts(netA, X), hidden_acts(netB, X)):
        a = (a - a.mean(0)) / (a.std(0) + 1e-8)
        b = (b - b.mean(0)) / (b.std(0) + 1e-8)
        C = (a.T @ b / len(a)).double().cpu().numpy()
        perms.append(linear_sum_assignment(C, maximize=True)[1])
    return perms


def apply(B, perms):
    out = {k: v.clone() for k, v in B.items()}
    for l, name in enumerate(LAYERS):
        w = out[f"{name}.weight"]
        if l < len(perms):
            w = w[perms[l]]
            out[f"{name}.bias"] = out[f"{name}.bias"][perms[l]]
        if l > 0:
            w = w[:, perms[l - 1]]
        out[f"{name}.weight"] = w
    return out


def flat(sd):
    return torch.cat([sd[k].reshape(-1).double() for k in sorted(sd)])


def main():
    import argparse
    from fig1_common import ckpt as ck
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["M1", "M2", "M4"])
    ap.add_argument("--nseeds", type=int, default=10)
    ap.add_argument("--method", choices=["weight", "act"], default="weight")
    ap.add_argument("--n_act", type=int, default=5000, help="training inputs for activation matching")
    args = ap.parse_args()
    out = OUT if args.method == "weight" else OUT + "_act"
    os.makedirs(out, exist_ok=True)
    prob = load_problem()
    X = prob.test_dataset.tensors[0][:500].to(prob.device)
    Xtr = prob.train_dataset.tensors[0][:args.n_act].to(prob.device)
    print("| model | seed | weight distance to seed 0: before | after alignment | max output change (should be ~0) |")
    print("|---|---:|---:|---:|---:|")
    for m in args.models:
        c0 = torch.load(ck(m, 0), map_location="cpu", weights_only=False)
        A = c0["model_state_dict"]
        torch.save(c0, f"{out}/{m}_s0.pt")
        netA, _ = load_model(prob, ck(m, 0))
        for s in range(1, args.nseeds):
            ps = ck(m, s)
            cs = torch.load(ps, map_location="cpu", weights_only=False)
            B = cs["model_state_dict"]
            net, _ = load_model(prob, ps)
            perms = weight_matching(A, B) if args.method == "weight" else activation_matching(netA, net, Xtr)
            Bp = apply(B, perms)
            with torch.no_grad():
                y_before = net(X)
                net.load_state_dict(Bp)
                y_after = net(X)
            before = ((flat(A) - flat(B)).norm() / flat(A).norm()).item()
            after = ((flat(A) - flat(Bp)).norm() / flat(A).norm()).item()
            cs["model_state_dict"] = Bp
            torch.save(cs, f"{out}/{m}_s{s}.pt")
            print(f"| {m} | {s} | {before:.3f} | {after:.3f} | {(y_after - y_before).abs().max().item():.1e} |", flush=True)

if __name__ == "__main__":
    main()
