"""Test what makes B=1 L-BFGS terminate worse than B=256."""
import os, sys, time
import torch

ROOT = '/orcd/scratch/orcd/008/khain/FSNet'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from torch.utils.data import Subset, DataLoader
from utils.trainer import load_instance, create_model
from utils.evaluator import Evaluator
import utils.lbfgs as lbfgs

CKPT = 'results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260315-181820_FSNet_seed0_e300_lr1e-04_n7000/model.pt'
N_EVAL = 256

ckpt = torch.load(CKPT, map_location='cuda', weights_only=False)
config = ckpt['config']
config['ensemble_post'] = 'post'
config['ensemble_agg'] = 'best_merit'
config['inference_perturb_k'] = 0
config['vectorize_repair'] = False

opt_problem, _ = load_instance(config)
model = create_model(opt_problem, config['method'], config).cuda()
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

test_set = Subset(opt_problem.test_dataset, range(N_EVAL))

def run(B, label):
    evaluator = Evaluator(opt_problem, config['method'], config)
    loader = DataLoader(test_set, batch_size=B, shuffle=False)
    all_obj, all_eq, all_ineq = [], [], []
    torch.cuda.synchronize() ; t0 = time.time()
    for X, _ in loader:
        X = X.cuda()
        with torch.no_grad():
            Y_final = evaluator._get_final_prediction(model, X)
        all_obj.append(opt_problem.obj_fn(Y_final).cpu())
        all_eq.append(opt_problem.eq_resid(X, Y_final).abs().sum(dim=1).cpu())
        all_ineq.append(opt_problem.ineq_resid(X, Y_final).abs().sum(dim=1).cpu())
    torch.cuda.synchronize() ; t1 = time.time()
    obj = torch.cat(all_obj); eq = torch.cat(all_eq); ineq = torch.cat(all_ineq)
    merit = obj + 1e6 * (eq + ineq)
    print(f"{label:<60} | EqL1={eq.mean().item():.3e} | Merit={merit.mean().item():9.4f} | wall={t1-t0:6.2f} s")

_ = (torch.randn(256, 100).cuda() @ torch.randn(100, 100).cuda()).sum(); torch.cuda.synchronize()

# Monkey-patch the convergence check to use only val_converged (disable grad early-exit)
orig_check = lbfgs._check_convergence
def val_only_check(f_val_per_sample, g, config):
    return f_val_per_sample / config.scale < config.val_tol
def both_check(f_val_per_sample, g, config):
    return (f_val_per_sample / config.scale < config.val_tol) & (g.norm(dim=1) < config.grad_tol)

print("\n=== Baseline (orig: val OR grad) ===")
run(B=256, label='B=256, val|grad')
run(B=1,   label='B=1,   val|grad')

print("\n=== val-only convergence (no grad early-exit) ===")
lbfgs._check_convergence = val_only_check
run(B=256, label='B=256, val-only')
run(B=1,   label='B=1,   val-only')

print("\n=== val AND grad (both required) ===")
lbfgs._check_convergence = both_check
run(B=256, label='B=256, val&grad')
run(B=1,   label='B=1,   val&grad')

lbfgs._check_convergence = orig_check
