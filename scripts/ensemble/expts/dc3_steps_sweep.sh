#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 02:00:00
#SBATCH -J dc3-steps-sweep
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

source /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/_common.sh

export PYTHONUNBUFFERED=1

DC3_SOCP="results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260528-171119_DC3_e1000_lr5e-05_n7000_hdim1024_seed0"

cat > /tmp/dc3_steps_eval.py <<'PY'
import os, sys, time, copy
import torch
ROOT = '/orcd/scratch/orcd/008/khain/FSNet'
sys.path.insert(0, ROOT); os.chdir(ROOT)
from torch.utils.data import Subset, DataLoader
from utils.trainer import load_instance, create_model
from utils.evaluator import Evaluator

CKPT = sys.argv[1] + '/model.pt'
ckpt = torch.load(CKPT, map_location='cuda', weights_only=False)

def run(max_corr_steps, corr_lr, K, eps):
    config = copy.deepcopy(ckpt['config'])
    config['DC3']['max_corr_steps'] = max_corr_steps
    config['DC3']['corr_lr'] = corr_lr
    config['inference_perturb_k'] = K
    config['inference_perturb_eps'] = eps
    config['ensemble_agg'] = 'best_merit'
    config['ensemble_post'] = 'post'
    config['vectorize_repair'] = True   # K perturbations broadcast into batch dim
    config['test_batch_sizes'] = []
    opt_problem, _ = load_instance(config)
    model = create_model(opt_problem, config['method'], config).cuda()
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    evaluator = Evaluator(opt_problem, config['method'], config)
    loader = DataLoader(opt_problem.test_dataset, batch_size=256, shuffle=False)
    all_obj=[];all_eq=[];all_ineq=[]
    torch.cuda.synchronize() ; t0=time.time()
    for X,_ in loader:
        X = X.cuda()
        with torch.no_grad():
            Y_final = evaluator._get_final_prediction(model, X)
        all_obj.append(opt_problem.obj_fn(Y_final).cpu())
        all_eq.append(opt_problem.eq_resid(X, Y_final).abs().sum(dim=1).cpu())
        all_ineq.append(opt_problem.ineq_resid(X, Y_final).abs().sum(dim=1).cpu())
    torch.cuda.synchronize() ; t1=time.time()
    obj = torch.cat(all_obj); eq=torch.cat(all_eq); ineq=torch.cat(all_ineq)
    merit = obj + 1e6*(eq+ineq)
    print(f"  steps={max_corr_steps:>5} lr={corr_lr:>8.0e} K={K:>3} eps={eps:>5} | Obj={obj.mean().item():7.3f} EqL1={eq.mean().item():.3e} IneqL1={ineq.mean().item():.3e} | Merit={merit.mean().item():10.4f} | wall={t1-t0:5.1f}s", flush=True)

print(f"\n=== DC3 + nonsmooth SOCP: grad_steps budget sweep ===\n", flush=True)
print("--- No perturb (K=0) ---", flush=True)
for steps in [30, 100, 300, 1000, 3000]:
    run(steps, 1e-6, 0, 0)
print(flush=True)
print("--- Larger corr_lr at default max_corr_steps=30 ---", flush=True)
for lr in [1e-6, 1e-5, 1e-4, 1e-3]:
    run(30, lr, 0, 0)
print(flush=True)
print("--- With K=100 eps=0.01 perturb at increasing budget ---", flush=True)
for steps in [30, 300, 1000]:
    run(steps, 1e-6, 100, 0.01)
PY
python /tmp/dc3_steps_eval.py "$DC3_SOCP"

echo "=== done $(date) ==="
