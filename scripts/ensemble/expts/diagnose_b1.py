"""Isolate where mode (a) loses quality at batch=1.

Hypothesis A: L-BFGS itself behaves differently at batch=1 vs batch=256.
  Test: K=1 (no perturbation) at B=1 vs B=256 — same NN, same starting points.
  If Merit differs, L-BFGS is batch-dependent.

Hypothesis B: The K=20 perturbation generation differs between B=1 sequential
  (each B=1 batch gets fresh noise) and B=256 sequential (one (K,256,D) noise
  tensor shared across instances).
  Test: K=20 at B=1 vs B=256 with seeded perturbations.
"""
import os, sys, time
import torch

ROOT = '/orcd/scratch/orcd/008/khain/FSNet'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from torch.utils.data import Subset, DataLoader
from utils.trainer import load_instance, create_model
from utils.evaluator import Evaluator

CKPT = 'results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260315-181820_FSNet_seed0_e300_lr1e-04_n7000/model.pt'
N_EVAL = 256

ckpt = torch.load(CKPT, map_location='cuda', weights_only=False)
config = ckpt['config']
config['ensemble_agg'] = 'best_merit'
config['ensemble_post'] = 'post'
config['skip_repair'] = False
config['test_batch_sizes'] = []

opt_problem, _ = load_instance(config)
model = create_model(opt_problem, config['method'], config).cuda()
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
evaluator = Evaluator(opt_problem, config['method'], config)

test_set = Subset(opt_problem.test_dataset, range(N_EVAL))

def run(B, K, eps, label):
    config['inference_perturb_k'] = K
    config['inference_perturb_eps'] = eps
    config['vectorize_repair'] = False  # SEQ over K
    loader = DataLoader(test_set, batch_size=B, shuffle=False)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    all_obj, all_eq, all_ineq = [], [], []
    torch.cuda.synchronize() ; t0 = time.time()
    for X, Y_true in loader:
        X = X.cuda()
        with torch.no_grad():
            Y_final = evaluator._get_final_prediction(model, X)
        all_obj.append(opt_problem.obj_fn(Y_final).cpu())
        all_eq.append(opt_problem.eq_resid(X, Y_final).abs().sum(dim=1).cpu())
        all_ineq.append(opt_problem.ineq_resid(X, Y_final).abs().sum(dim=1).cpu())
    torch.cuda.synchronize() ; t1 = time.time()
    obj = torch.cat(all_obj); eq = torch.cat(all_eq); ineq = torch.cat(all_ineq)
    merit = obj + 1e6 * (eq + ineq)
    print(f"{label:<40} | Obj={obj.mean().item():7.4f} | EqL1={eq.mean().item():.3e} | IneqL1={ineq.mean().item():.3e} | Merit={merit.mean().item():9.4f} | wall={t1-t0:6.2f} s")

# Warm GPU
_ = (torch.randn(256, 100).cuda() @ torch.randn(100, 100).cuda()).sum(); torch.cuda.synchronize()

print("\n=== H_A: L-BFGS at K=1 (no perturbation) ===")
# K=0 disables perturbation entirely
run(B=256, K=0, eps=0, label='B=256, no perturb')
run(B=1,   K=0, eps=0, label='B=1,   no perturb')

print("\n=== H_B: L-BFGS at K=20 perturbation (same seed) ===")
run(B=256, K=20, eps=0.1, label='B=256, K=20 SEQ')
run(B=1,   K=20, eps=0.1, label='B=1,   K=20 SEQ')
