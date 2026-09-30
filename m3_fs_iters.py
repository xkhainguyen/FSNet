"""Step 1: are the rho = 0 M3 variants only short of FS iterations? Re-score their checkpoints with
more FS iterations at evaluation (tolerance 1e-9), held-out; no retraining."""
import glob, os
from compute_landscape_compare import load_problem, load_model, eval_components
D = "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000"
prob = load_problem()
X, Y = [t[:500].to(prob.device) for t in prob.test_dataset.tensors]
VARIANTS = [("M3 (rho 10)", "eq10.0_ineq10.0_penl1"), ("M3a (rho 0)", "eq0.0_ineq0.0_penl1"),
            ("M3c (rho 0 + dist)", "eq0.0_ineq0.0_dist5.0_penl1"),
            ("M3d (warm-up + dist)", "eq10.0_ineq10.0_warm300_dist5.0_penl1")]
print("| variant | seed | FS iters | violation after FS | 100 huber(FS(y) - y*) | FS effect on label term |")
print("|---|---:|---:|---:|---:|---:|")
for name, tag in VARIANTS:
    for s in range(3):
        runs = [r for r in sorted(glob.glob(f"{D}/*_MLP_sup_pen_fs_seed{s}_nepochs3000_lr0.0003_trainsize7000_obj0.1_{tag}_dropout0.0_*"))
                if os.path.exists(r + "/model.pt")]
        if not runs:
            continue
        net, _ = load_model(prob, runs[-1] + "/model.pt")
        for it in (50, 200, 500):
            c = eval_components(net, prob, X, Y, dict(val_tol=1e-9, memory=30, max_iter=it, scale=1000), 500)
            print(f"| {name} | {s} | {it} | {c['viol_l1_fs']:.3g} | {100*c['huber_fs']:.2f} | "
                  f"{abs(c['huber_fs'] - c['huber']) / c['huber']:.1%} |", flush=True)
