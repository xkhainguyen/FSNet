"""Collapsed hard-FS seeds vs sigmoid saturation of the output layer (held-out inputs, CPU is fine).

The MLP ends in a sigmoid scaled to the variable bounds (models/neural_networks.py,
optimization_utils.scale). Prints, per trained network, the share of outputs within 1e-3 of a
bound and the mean sigmoid slope p(1 - p) (0.25 at most); collapsed = merit after FS > 10
(m3_candidates.py). Result 2026-09-30: all 8 collapsed seeds 100% saturated (slope 3e-5 to 1e-4),
all good seeds 0% (slope 0.23).
"""
import torch
import compute_landscape_compare as C
prob = C.load_problem()
X = prob.test_dataset.tensors[0][:1000].to(C.DEVICE)
A = "figures/landscape/fig1/aligned"
rows = [("M1", 0, "good")] + [("M3f", 0, "good")] + [("M3r1", s, "collapsed" if s in (1, 3, 8) else "good") for s in range(10)] \
     + [("M3r08", s, "collapsed" if s in (2, 5, 6, 7, 8) else "good") for s in range(10)]
print("| method | seed | status | sigmoid outputs saturated (<1e-3 or >1-1e-3) | mean sigmoid'(z) = p(1-p) |")
print("|---|---:|---|---:|---:|")
for m, s, st in rows:
    net, _ = C.load_model(prob, f"{A}/{m}_s{s}.pt")
    with torch.no_grad():
        p = net(X)
    sat = ((p < 1e-3) | (p > 1 - 1e-3)).double().mean().item()
    print(f"| {m} | {s} | {st} | {sat:.0%} | {(p * (1 - p)).mean().item():.3g} |")
