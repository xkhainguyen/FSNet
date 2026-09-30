"""How far did training move the weights from their own random initialization?"""
import torch

from compute_landscape_compare import load_problem, load_model
from utils.trainer import create_model
from align_seeds import ckpt, SPEC

prob = load_problem()
print("| model | seed | ||theta - theta_init|| / ||theta_init|| | per layer (mlp.0, 2, 4, 6, 8) |")
print("|---|---:|---:|---|")
for m, (meth, w, lr) in SPEC.items():
    for s in (0, 1):
        net, cfg = load_model(prob, ckpt(meth, w, lr, s))
        torch.manual_seed(s)  # main.py seeds before the Trainer creates the model
        init = create_model(prob, cfg["method"], cfg)
        a, b = dict(net.named_parameters()), dict(init.named_parameters())
        tot = (torch.cat([(a[k] - b[k]).reshape(-1) for k in a]).norm()
               / torch.cat([b[k].reshape(-1) for k in a]).norm()).item()
        per = [((a[f"mlp.{i}.weight"] - b[f"mlp.{i}.weight"]).norm() / b[f"mlp.{i}.weight"].norm()).item()
               for i in (0, 2, 4, 6, 8)]
        print(f"| {m} | {s} | {tot:.3f} | " + ", ".join(f"{v:.3f}" for v in per) + " |")
