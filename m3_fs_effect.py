"""Does FS shape the training loss of the rho = 0 M3 variants? At the trained weights (held-out):
label error with / without FS, FS correction size, violation raw / after FS."""
import glob, os
import torch
from compute_landscape_compare import load_problem, load_model, eval_components
D = "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000"
prob = load_problem()
X, Y = [t[:500].to(prob.device) for t in prob.test_dataset.tensors]
fs = dict(val_tol=1e-9, memory=30, max_iter=50, scale=1000)
print("| variant | seed | 100 huber(y - y*) | 100 huber(FS(y) - y*) | FS effect on label term | ||FS(y) - y||^2 | violation raw | after FS |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for name, tag in [("M3 (rho 10)", "eq10.0_ineq10.0_penl1"), ("M3a (rho 0)", "eq0.0_ineq0.0_penl1"), ("M3c (rho 0 + dist)", "eq0.0_ineq0.0_dist5.0_penl1"), ("M3d (rho 10 warm-up 300 + dist)", "eq10.0_ineq10.0_warm300_dist5.0_penl1")]:
    for s in range(3):
        runs = [r for r in sorted(glob.glob(f"{D}/*_MLP_sup_pen_fs_seed{s}_nepochs3000_lr0.0003_trainsize7000_obj0.1_{tag}_dropout0.0_*")) if os.path.exists(r + "/model.pt")]
        if not runs:
            continue
        net, _ = load_model(prob, runs[-1] + "/model.pt")
        c = eval_components(net, prob, X, Y, fs, 500)
        eff = abs(c["huber_fs"] - c["huber"]) / c["huber"]
        print(f"| {name} | {s} | {100*c['huber']:.2f} | {100*c['huber_fs']:.2f} | {eff:.1%} | {c['dist_fs']:.3f} | {c['viol_l1']:.3g} | {c['viol_l1_fs']:.3g} |", flush=True)
