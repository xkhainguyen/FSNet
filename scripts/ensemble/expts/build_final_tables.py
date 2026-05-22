"""Build cleaned-up tables for the presentation from results.csv."""
import csv, os, sys

DEF = '/orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/results.csv'
rows = list(csv.DictReader(open(DEF)))

def f(s, d=None):
    if s in (None,'','None'): return d
    try: return float(s)
    except: return d

def fmt(x, p=4):
    if x is None: return '-'
    if x == 0: return '0'
    a = abs(x)
    if a >= 1e4 or a < 1e-2: return f"{x:.{p-1}g}"
    return f"{x:.{p}g}"

def find_one(method, ensemble_size, post=None, agg=None, skip=False, repIt=None, pK=0, eps=0, dist='gauss'):
    """Find the row with matching params (256 batch)."""
    for r in rows:
        if r['method'] != method: continue
        if int(r.get('batch_size','0')) != 256: continue
        if int(r.get('ensemble_size','1')) != ensemble_size: continue
        if post and (r.get('ensemble_post') != post): continue
        if agg and (r.get('ensemble_agg') != agg): continue
        if (r.get('skip_repair') in ('True','true', True)) != skip: continue
        rmio = r.get('repair_max_iter_override')
        rmio = int(rmio) if rmio not in (None,'','None') else None
        if rmio != repIt: continue
        if int(r.get('inference_perturb_k','0')) != pK: continue
        peps = f(r.get('inference_perturb_eps'), 0)
        if pK > 1 and abs(peps - eps) > 1e-9: continue
        if pK > 1 and r.get('inference_perturb_dist','gauss') != dist: continue
        return r
    return None

def get(method, **kw):
    r = find_one(method, **kw)
    if r is None: return ('-','-','-','-','-','-')
    return (
        fmt(f(r.get('objective'))),
        fmt(f(r.get('merit_mean'))),
        fmt(f(r.get('opt_gap_mean'))),
        fmt(f(r.get('eq_violation_l1_mean'))),
        fmt(f(r.get('ineq_violation_l1_mean'))),
        fmt(f(r.get('eval_time_seconds'))),
    )

print("=" * 100)
print("TABLE 1 — Baselines (FSNet)")
print("=" * 100)
print(f"{'Config':<40} | {'Obj':>7} | {'Merit':>10} | {'OptGap%':>8} | {'EqVio':>10} | {'IneqVio':>10}")
print("-"*100)
# Single FSNet (hdim=2048)
r = find_one('FSNet', ensemble_size=1, pK=0, skip=False, repIt=None)
if r:
    print(f"{'FSNet single (hdim=2048)':<40} | " + ' | '.join(g.rjust(w) for g,w in zip(get('FSNet', ensemble_size=1)[:5], [7,10,8,10,10])))

# FSNet ens5 vanilla post+best_merit (these are duplicates between two runs - take first ens5_vanilla one)
for label, cfg in [
    ('FSNet ens5 vanilla post+mean',     dict(ensemble_size=5, post='post', agg='mean')),
    ('FSNet ens5 vanilla post+median',   dict(ensemble_size=5, post='post', agg='median')),
    ('FSNet ens5 vanilla post+best_obj', dict(ensemble_size=5, post='post', agg='best_obj')),
    ('FSNet ens5 vanilla post+best_merit', dict(ensemble_size=5, post='post', agg='best_merit')),
    ('FSNet ens5 vanilla pre+best_merit', dict(ensemble_size=5, post='pre', agg='best_merit')),
]:
    vals = get('FSNet', **cfg)
    print(f"{label:<40} | " + ' | '.join(g.rjust(w) for g,w in zip(vals[:5], [7,10,8,10,10])))

print()
print("=" * 100)
print("TABLE 2 — FSNet single + perturbation (hdim=2048)")
print("=" * 100)
print(f"{'Config':<30} | {'Obj':>7} | {'Merit':>10} | {'OptGap%':>8} | {'EqVio':>10} | {'IneqVio':>10}")
print("-"*100)
for K in [1,5,10,20]:
    if K == 1:
        vals = get('FSNet', ensemble_size=1, pK=0)
        print(f"{'(no perturb)':<30} | " + ' | '.join(g.rjust(w) for g,w in zip(vals[:5], [7,10,8,10,10])))
    else:
        for eps in [0.01, 0.05, 0.1, 0.2]:
            vals = get('FSNet', ensemble_size=1, pK=K, eps=eps)
            print(f"{'K=%d eps=%s'%(K,eps):<30} | " + ' | '.join(g.rjust(w) for g,w in zip(vals[:5], [7,10,8,10,10])))

print()
print("=" * 100)
print("TABLE 3 — FSNet repair_max_iter sweep (single, hdim=2048)")
print("=" * 100)
print(f"{'max_iter':<12} | {'Obj':>7} | {'Merit':>10} | {'EqVio':>10} | {'IneqVio':>10}")
print("-"*100)
# skip_repair
vals = get('FSNet', ensemble_size=1, skip=True, repIt=None)
print(f"{'0 (skip)':<12} | " + ' | '.join(g.rjust(w) for g,w in [(vals[0],7), (vals[1],10), (vals[3],10), (vals[4],10)]))
for it in [1, 5, 10, 20, 50]:
    vals = get('FSNet', ensemble_size=1, repIt=it)
    print(f"{str(it):<12} | " + ' | '.join(g.rjust(w) for g,w in [(vals[0],7), (vals[1],10), (vals[3],10), (vals[4],10)]))

print()
print("=" * 100)
print("TABLE 4 — Penalty baselines (eq=10 ineq=10, hdim=1024)")
print("=" * 100)
print(f"{'Config':<40} | {'Obj':>7} | {'Merit':>10} | {'OptGap%':>8} | {'EqVio':>10} | {'IneqVio':>10}")
print("-"*100)
for label, cfg in [
    ('penalty ens5 vanilla pre+mean',     dict(ensemble_size=5, post='pre', agg='mean')),
    ('penalty ens5 vanilla pre+median',   dict(ensemble_size=5, post='pre', agg='median')),
    ('penalty ens5 vanilla pre+best_obj', dict(ensemble_size=5, post='pre', agg='best_obj')),
    ('penalty ens5 vanilla pre+best_merit', dict(ensemble_size=5, post='pre', agg='best_merit')),
    ('penalty ens20 vanilla pre+best_merit',dict(ensemble_size=20, post='pre', agg='best_merit')),
]:
    vals = get('penalty', **cfg)
    print(f"{label:<40} | " + ' | '.join(g.rjust(w) for g,w in zip(vals[:5], [7,10,8,10,10])))

