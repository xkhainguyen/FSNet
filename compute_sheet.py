"""Piecewise-planar sheet through many trained seeds (permutation-aligned, align_seeds.py).

Seeds sit on the sites of a triangular lattice. Every point of the lattice region lies in one
small (unit, equilateral) triangle; its weights are the barycentric combination of that
triangle's three seeds, i.e. exactly the plane through them. Neighbouring triangles share an
edge, so the sheet is continuous and passes through every seed. Nothing is extrapolated.

Layouts:  tri10  rows of 4,3,2,1 (10 seeds, 1 interior)
          hex19  hexagon of radius 2 (19 seeds: centre + ring of 6 interior, ring of 12 on the edge)
--order k assigns seeds to sites by a random permutation with seed k (0 = identity), to check
that the picture does not depend on which seeds are neighbours.

  python compute_sheet.py --model M2 --layout hex19 --order 0 --split test --out ...
"""
import argparse
import itertools

import numpy as np
import torch

from compute_landscape_compare import load_problem, load_model, eval_components, COMPONENTS

A_DIR = "figures/landscape/fig1/aligned"


def lattice(kind):
    if kind == "tri3":
        pts = [(0.0, 0.0), (1.0, 0.0), (0.5, np.sqrt(3) / 2)]
    elif kind == "tri10":
        pts = [(c + 0.5 * r, r * np.sqrt(3) / 2) for r in range(4) for c in range(4 - r)]
    elif kind == "hex19":
        pts = [(q + 0.5 * r, r * np.sqrt(3) / 2) for q in range(-2, 3) for r in range(-2, 3) if abs(q + r) <= 2]
    else:
        raise ValueError(kind)
    pos = np.array(pts)
    tris = [t for t in itertools.combinations(range(len(pos)), 3)
            if all(abs(np.linalg.norm(pos[a] - pos[b]) - 1) < 1e-6 for a, b in itertools.combinations(t, 2))]
    return pos, tris


def barycentric(p, a, b, c):
    v0, v1, v2 = b - a, c - a, p - a
    d = v0[0] * v1[1] - v1[0] * v0[1]
    l1 = (v2[0] * v1[1] - v1[0] * v2[1]) / d
    l2 = (v0[0] * v2[1] - v2[0] * v0[1]) / d
    return np.array([1 - l1 - l2, l1, l2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layout", choices=["tri3", "tri10", "hex19"], default="hex19")
    ap.add_argument("--order", type=int, default=0, help="random seed-to-site permutation (0 = identity)")
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--n", type=int, default=161)
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--margin", type=float, default=0.15, help="empty plot padding around the lattice")
    ap.add_argument("--fs", action="store_true", help="also compute the FS-layer components (needed for M3)")
    ap.add_argument("--aligned_dir", default=A_DIR, help="aligned checkpoints (align_seeds.py)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prob = load_problem()
    ds = prob.test_dataset if args.split == "test" else prob.train_dataset
    X, Y = [t[:args.n_eval].to(prob.device) for t in ds.tensors]
    pos, tris = lattice(args.layout)
    seeds = np.arange(len(pos))
    if args.order:
        seeds = np.random.default_rng(args.order).permutation(len(pos))
    net, _ = load_model(prob, f"{args.aligned_dir}/{args.model}_s0.pt")
    thetas = []
    for s in seeds:
        m, _ = load_model(prob, f"{args.aligned_dir}/{args.model}_s{s}.pt")
        thetas.append(torch.nn.utils.parameters_to_vector(m.parameters()).detach())
    mg = args.margin
    xs = np.linspace(pos[:, 0].min() - mg, pos[:, 0].max() + mg, args.n)
    ys = np.linspace(pos[:, 1].min() - mg, pos[:, 1].max() + mg, args.n)
    Z = {k: np.full((len(ys), len(xs)), np.nan) for k in COMPONENTS}
    params = list(net.parameters())
    done = 0
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            p = np.array([x, y])
            for t in tris:
                lam = barycentric(p, *pos[list(t)])
                if (lam >= -1e-9).all():
                    theta = sum(float(l) * thetas[k] for l, k in zip(lam, t))
                    torch.nn.utils.vector_to_parameters(theta, params)
                    fs_kw = dict(val_tol=1e-9, memory=30, max_iter=50, scale=1000) if args.fs else None
                    for key, v in eval_components(net, prob, X, Y, fs_kw, 1000).items():
                        Z[key][j, i] = v
                    done += 1
                    break
        if j % 20 == 0:
            print(f"row {j + 1}/{len(ys)}, {done} points", flush=True)
    np.savez(args.out, xs=xs, ys=ys, points=pos, names=np.array([f"s{s}" for s in seeds]),
             tris=np.array(tris), components=np.array(COMPONENTS), layout=args.layout, order=args.order,
             split=args.split, **{f"plane/{k}": v for k, v in Z.items()})
    print(f"saved {args.out} ({done} points)")


if __name__ == "__main__":
    main()
