"""Curved seed sheet: learn one Bezier control point per lattice edge (mode connectivity on a sheet).

Same lattice and seed-to-site assignment as compute_sheet.py (ordering 0). Each lattice triangle
with seeds a, b, c becomes the quadratic Bezier triangle

    theta(l) = sum_i l_i^2 theta_i + 2 sum_{i<j} l_i l_j m_ij,     l = barycentric coordinates,

whose edges are the quadratic Bezier curves of Garipov et al. (connect_seeds.py), so neighbouring
triangles share their edge curve and the sheet is continuous. All edge points m_ij start at the
straight-line midpoints (i.e. the flat sheet) and are trained jointly with Adam (the model's own training lr, linear warm-up then cosine decay) on the model's own
training loss (Trainer.compute_batch_loss with the checkpoint's config), at a uniformly random
triangle and barycentric point per step, on training minibatches. Only m_ij are trained; the
seeds stay fixed. Evaluate with compute_sheet.py --mids.

  python connect_sheet.py --model M1 --layout tri10 --out figures/landscape/fig1/mids_tri10_M1.pt
"""
import argparse
import itertools

import numpy as np
import torch

from compute_landscape_compare import load_problem, load_model
from compute_sheet import A_DIR, lattice
from utils.trainer import Trainer


def edges_of(tris):
    return sorted({tuple(sorted(e)) for t in tris for e in itertools.combinations(t, 2)})


def bezier_point(lam, t, thetas, mids, eid):
    """lam: 3 barycentric weights for the triangle t (site indices)."""
    theta = sum(float(lam[i]) ** 2 * thetas[t[i]] for i in range(3))
    for i, j in itertools.combinations(range(3), 2):
        theta = theta + 2 * float(lam[i] * lam[j]) * mids[eid[tuple(sorted((t[i], t[j])))]]
    return theta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layout", choices=["tri3", "tri10", "hex19"], default="tri10")
    ap.add_argument("--aligned_dir", default=A_DIR)
    ap.add_argument("--steps", type=int, default=9000, help="~1500 updates per edge on tri10")
    ap.add_argument("--lr", type=float, default=None, help="default: the model's own training lr")
    ap.add_argument("--warmup", type=int, default=500, help="linear lr warm-up steps, then cosine decay")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    prob = load_problem()
    pos, tris = lattice(args.layout)
    net, cfg = load_model(prob, f"{args.aligned_dir}/{args.model}_s0.pt")
    cfg = dict(cfg)
    cfg.setdefault("checkpoint", None)
    lr = args.lr if args.lr is not None else float(cfg[cfg["method"]]["lr"])
    trainer = Trainer(prob, cfg)
    names = [n for n, _ in net.named_parameters()]
    shapes = [p.shape for _, p in net.named_parameters()]
    sizes = [p.numel() for _, p in net.named_parameters()]
    thetas = [torch.nn.utils.parameters_to_vector(load_model(prob, f"{args.aligned_dir}/{args.model}_s{s}.pt")[0]
                                                  .parameters()).detach() for s in range(len(pos))]
    edges = edges_of(tris)
    eid = {e: k for k, e in enumerate(edges)}
    mids = torch.nn.Parameter(torch.stack([(thetas[a] + thetas[b]) / 2 for a, b in edges]))

    def loss(theta, X, Y):
        ps = {n: c.view(s) for n, c, s in zip(names, torch.split(theta, sizes), shapes)}
        out = torch.func.functional_call(net, ps, (X,))
        return trainer.compute_batch_loss(X, out, Y, {"epoch": 0})[0].mean()

    Xtr, Ytr = [t.to(prob.device) for t in prob.train_dataset.tensors]
    opt = torch.optim.Adam([mids], lr=lr)
    w = args.warmup
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: (k + 1) / w if k < w else
                                              0.5 * (1 + np.cos(np.pi * (k - w) / max(1, args.steps - w))))
    print(f"{args.model}: {len(edges)} edges, lr {lr:g}, warm-up {w}, {args.steps} steps", flush=True)
    run = []
    for step in range(args.steps):
        t = tris[rng.integers(len(tris))]
        lam = rng.dirichlet([1.0, 1.0, 1.0])
        idx = torch.randint(0, len(Xtr), (args.batch,), device=Xtr.device)
        opt.zero_grad()
        l = loss(bezier_point(lam, t, thetas, mids, eid), Xtr[idx], Ytr[idx])
        l.backward()
        opt.step()
        sched.step()
        run.append(l.item())
        if step % 500 == 0 or step == args.steps - 1:
            print(f"step {step}: mean loss over last 500 {np.mean(run[-500:]):.4g}", flush=True)
    torch.save({"edges": edges, "mids": mids.detach().cpu(), "layout": args.layout, "model": args.model,
                "aligned_dir": args.aligned_dir, "steps": args.steps, "lr": lr, "warmup": w}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
