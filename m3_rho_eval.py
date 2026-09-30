"""M3 rho sweep: deployment quality (after FS, 500 iters) and FS effect on the training loss (held-out)."""
import glob, os
from compute_landscape_compare import load_problem, load_model, eval_components
D = "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000"
prob = load_problem()
X, Y = [t[:500].to(prob.device) for t in prob.test_dataset.tensors]
print("| rho | objective after FS | violation after FS | merit after FS | FS effect on label term | FS effect on own loss |")
print("|---:|---:|---:|---:|---:|---:|")
for rho in ["10.0", "3.0", "1.0", "0.3", "0.0"]:
    runs = [r for r in sorted(glob.glob(f"{D}/*_MLP_sup_pen_fs_seed0_nepochs3000_lr0.0003_trainsize7000_obj0.1_eq{rho}_ineq{rho}_penl1_dropout0.0_*")) if os.path.exists(r + "/model.pt")]
    if not runs:
        print(f"| {rho} | (not finished) | | | | |"); continue
    net, _ = load_model(prob, runs[-1] + "/model.pt")
    c = eval_components(net, prob, X, Y, dict(val_tol=1e-9, memory=30, max_iter=500, scale=1000), 500)
    own_fs = 100 * c["huber_fs"] + float(rho) * c["viol_l1"]
    own_raw = 100 * c["huber"] + float(rho) * c["viol_l1"]
    print(f"| {rho} | {c['obj_fs']:.3f} | {c['viol_l1_fs']:.2g} | {c['obj_fs'] + 1e5 * c['viol_l1_fs']:.4g} | "
          f"{abs(c['huber_fs'] - c['huber']) / c['huber']:.1%} | {abs(own_fs - own_raw) / own_raw:.1%} |", flush=True)
