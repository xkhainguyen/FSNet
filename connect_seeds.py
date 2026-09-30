"""Mode connectivity between two trained seeds (Garipov et al., 2018), own training loss.

Quadratic Bezier path theta(t) = (1-t)^2 theta_a + 2t(1-t) theta_m + t^2 theta_b.
theta_m starts at the straight-line midpoint and is trained with Adam on the model's own
training loss (Trainer.compute_batch_loss with the checkpoint's config) at random t,
on training minibatches. The path is then evaluated on held-out test samples.

barrier = max_t L(theta(t)) / max(L(theta_a), L(theta_b)):
~1 means the two minima are connected by a low path; >> 1 means separate basins.

  python connect_seeds.py --a path/a.pt --b path/b.pt --out pair.json
"""
import argparse
import json

import numpy as np
import torch

from compute_landscape_compare import load_problem, load_model
from utils.trainer import Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--n_t", type=int, default=41)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(0)
    prob = load_problem()
    ma, cfg = load_model(prob, args.a)
    mb, _ = load_model(prob, args.b)
    cfg = dict(cfg)
    cfg.setdefault("checkpoint", None)
    trainer = Trainer(prob, cfg)

    names = [n for n, _ in ma.named_parameters()]
    shapes = [p.shape for _, p in ma.named_parameters()]
    sizes = [p.numel() for _, p in ma.named_parameters()]
    ta = torch.nn.utils.parameters_to_vector(ma.parameters()).detach()
    tb = torch.nn.utils.parameters_to_vector(mb.parameters()).detach()
    tm = ((ta + tb) / 2).clone().requires_grad_(True)

    def loss(theta, X, Y):
        ps = {n: c.view(s) for n, c, s in zip(names, torch.split(theta, sizes), shapes)}
        out = torch.func.functional_call(ma, ps, (X,))
        return trainer.compute_batch_loss(X, out, Y, {"epoch": 0})[0].mean()

    def point(t, m):
        return (1 - t) ** 2 * ta + 2 * t * (1 - t) * m + t ** 2 * tb

    Xtr, Ytr = [t.to(prob.device) for t in prob.train_dataset.tensors]
    Xte, Yte = [t[:args.n_eval].to(prob.device) for t in prob.test_dataset.tensors]
    opt = torch.optim.Adam([tm], lr=args.lr)
    for step in range(args.steps):
        idx = torch.randint(0, len(Xtr), (args.batch,), device=Xtr.device)
        t = torch.rand(()).item()
        opt.zero_grad()
        l = loss(point(t, tm), Xtr[idx], Ytr[idx])
        l.backward()
        opt.step()
        if step % 300 == 0:
            print(f"step {step}: t={t:.2f} loss {l.item():.4g}", flush=True)

    ts = np.linspace(0, 1, args.n_t)
    with torch.no_grad():
        curve = [loss(point(float(t), tm.detach()), Xte, Yte).item() for t in ts]
        line = [loss((1 - float(t)) * ta + float(t) * tb, Xte, Yte).item() for t in ts]
    ends = max(curve[0], curve[-1])
    res = dict(a=args.a, b=args.b, ts=ts.tolist(), curve=curve, line=line,
               barrier_curve=max(curve) / ends, barrier_line=max(line) / ends,
               end_losses=[curve[0], curve[-1]], dist=(ta - tb).norm().item() / ta.norm().item())
    print(f"barrier: curve {res['barrier_curve']:.3g}x, straight line {res['barrier_line']:.3g}x, "
          f"relative distance {res['dist']:.3f}")
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
