"""Head-to-head perturbation mode comparison on 256 test instances.

Four parallelism modes (B=batch over instances, K=20 perturbations per instance):
  (a) B=1   SEQ over both: 256 instances × 20 sequential L-BFGS calls (batch=1)
  (b) B=1   VEC over pert: 256 instances ×  1 vectorised L-BFGS call (batch=20)
  (c) B=256 SEQ over pert:  1 call ×       20 sequential L-BFGS calls (batch=256)
  (d) B=256 VEC over both:  1 call ×        1 vectorised L-BFGS call (batch=256*20=5120)

Reports Merit, Obj, EqVio, IneqVio, wall_time for each.
"""
import os, sys, time
import torch
from torch.utils.data import Subset, DataLoader

ROOT = '/orcd/scratch/orcd/008/khain/FSNet'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from utils.trainer import load_instance
from utils.evaluator import Evaluator
import yaml

CKPT = 'results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260315-181820_FSNet_seed0_e300_lr1e-04_n7000/model.pt'
N_EVAL = 256
K = 20
EPS = 0.1

# Load checkpoint
ckpt = torch.load(CKPT, map_location='cuda', weights_only=False)
config = ckpt['config']
config['inference_perturb_k'] = K
config['inference_perturb_eps'] = EPS
config['inference_perturb_dist'] = 'gauss'
config['inference_perturb_keep_original'] = True
config['ensemble_agg'] = 'best_merit'
config['ensemble_post'] = 'post'
config['skip_repair'] = False
config['test_batch_sizes'] = []

# Build problem + model
opt_problem, _ = load_instance(config)
from utils.trainer import create_model
model = create_model(opt_problem, config['method'], config).cuda()
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# First-256 subset of test set
test_set = Subset(opt_problem.test_dataset, range(N_EVAL))
evaluator = Evaluator(opt_problem, config['method'], config)


def run_mode(B, vectorize, label):
    config['vectorize_repair'] = vectorize
    loader = DataLoader(test_set, batch_size=B, shuffle=False)

    # Warmup
    torch.cuda.synchronize()
    t0 = time.time()
    all_obj = []
    all_eq = []
    all_ineq = []
    for X, Y_true in loader:
        X = X.cuda()
        # Use _get_final_prediction directly to bypass per-batch eval overhead
        with torch.no_grad():
            Y_final = evaluator._get_final_prediction(model, X)
        obj = opt_problem.obj_fn(Y_final)
        eq = opt_problem.eq_resid(X, Y_final).abs().sum(dim=1)
        ineq = opt_problem.ineq_resid(X, Y_final).abs().sum(dim=1)
        all_obj.append(obj.cpu()); all_eq.append(eq.cpu()); all_ineq.append(ineq.cpu())
    torch.cuda.synchronize()
    t1 = time.time()
    obj = torch.cat(all_obj); eq = torch.cat(all_eq); ineq = torch.cat(all_ineq)
    merit = obj + 1e6 * (eq + ineq)
    print(f"{label:<35} | Obj={obj.mean().item():7.4f} | "
          f"EqL1={eq.mean().item():.4e} | IneqL1={ineq.mean().item():.4e} | "
          f"Merit={merit.mean().item():9.4g} | wall={t1-t0:6.2f} s")

print(f"\n=== K={K} ε={EPS} on {N_EVAL} instances (FSNet hdim=2048, ρ=1e6) ===\n")
print(f"{'mode':<35} | {'Obj':>10} | {'EqL1':>11} | {'IneqL1':>11} | {'Merit':>10} | {'wall':>8}")
print('-' * 100)

# Warm up GPU
_warm = torch.randn(256, 100).cuda(); _ = (_warm @ _warm.T).sum(); torch.cuda.synchronize()

run_mode(B=256, vectorize=False, label='(c) B=256, SEQ over K')
run_mode(B=256, vectorize=True,  label='(d) B=256, VEC over K')
run_mode(B=1,   vectorize=True,  label='(b) B=1,   VEC over K')
run_mode(B=1,   vectorize=False, label='(a) B=1,   SEQ over K')

print()
