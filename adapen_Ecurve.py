"""Does ANY SL checkpoint E beat cold-start vanilla at equal total budget?

AdaPen at its tuned config (lr=1e-5, eqW=10), maxt0.5, 800 labels, B=952.3 s.
Every E is charged gen + 0.16*E + 0.18*epochs = B, so later checkpoints buy
fewer SSL epochs. Vanilla spends the whole budget on SSL.

Reports the full E curve rather than its best point, and marks where the merit
rule's k* lands per seed, so "is there a good E" and "does the rule find it" stay
separate questions.
"""
import glob
import os
import pickle
from collections import defaultdict

import numpy as np

RES = "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000"
KSTAR = {0: 750, 1: 800, 2: 500, 3: 700}
merit = lambda m: m["objective"] + 1e5 * (m["eq_violation_l1_mean"]
                                          + m["ineq_violation_l1_mean"])


def scan(manifests, eqw=10):
    eps = set()
    for mf in manifests:
        if os.path.exists(mf):
            eps |= {f"_nepochs{l.split(chr(9))[6]}_" for l in open(mf)}
    idx = {}
    for d in glob.glob(f"{RES}/*_MLP_adaptive_penalty_seed*_eq{float(eqw)}_*"):
        if not any(e in os.path.basename(d) for e in eps):
            continue
        p = os.path.join(d, "results.pkl")
        if not os.path.exists(p):
            continue
        try:
            r = pickle.load(open(p, "rb"))
        except Exception:
            continue
        c = r.get("config", {})
        m = c.get("method", "adaptive_penalty")
        idx.setdefault((int(r.get("seed", -1)), c.get(m, {}).get("num_epochs"),
                        c.get("checkpoint") or "-",
                        float(c.get(m, {}).get("lr", 0))), []).append(r)
    return idx


def collect(mf, idx, out, want_lr):
    """Only rows at want_lr. The lrlow manifest carries vanilla at TWO learning
    rates; keying by seed alone let the 3e-6 vanilla overwrite the 1e-5 one and
    silently compared the E-sweep against the wrong (worse) baseline.
    """
    if not os.path.exists(mf):
        return
    for line in open(mf):
        _, B, s, q, arm, k, ep, ck, lr = line.rstrip("\n").split("\t")
        if abs(float(lr) - want_lr) > 1e-12:
            continue
        h = idx.get((int(s), int(ep), ck, float(lr)), [])
        if h:
            mm = h[-1]["test_results"]["batch_size_comparison"][256]["metrics"]
            out[arm][int(s)] = (merit(mm), mm["eq_violation_l1_mean"], int(ep))


def main():
    mfs = ["budget_manifest_adapen_Esweep.tsv", "budget_manifest_adapen_lrlow.tsv"]
    idx = scan(mfs)
    d = defaultdict(dict)
    for mf in mfs:
        collect(mf, idx, d, 1e-5)

    van = {s: v for s, v in d.get("vanilla", {}).items()}
    if not van:
        print("no vanilla runs at lr=1e-5 yet")
        return
    vm = np.array([v[0] for v in van.values()])
    print("adaptive_penalty, maxt0.5, 800 labels, B=952.3s, lr=1e-5, eqW=10")
    print(f"Merit = obj + 1e5*vio.  vanilla = {vm.mean():.4e} +- {vm.std():.1e} "
          f"(n={len(vm)}, {list(van.values())[0][2]} ep)\n")
    hdr = (f"{'E':>5} {'SSL ep':>7} {'n':>2} {'Merit':>22} {'eq_l1':>8} | "
           f"{'van/E':>7} {'wins':>5}  k*seeds")
    print(hdr); print("-" * len(hdr))
    best = None
    for arm in sorted((a for a in d if a.startswith("E")),
                      key=lambda a: int(a[1:])):
        E = int(arm[1:])
        g = d[arm]
        v = np.array([x[0] for x in g.values()])
        e = np.array([x[1] for x in g.values()])
        sh = sorted(set(g) & set(van))
        rr, w = "", ""
        if sh:
            r = np.array([van[s][0] / g[s][0] for s in sh])
            gm = float(np.exp(np.log(r).mean()))
            rr, w = f"{gm:.4f}", f"{int((r > 1).sum())}/{len(sh)}"
            if best is None or gm > best[1]:
                best = (E, gm, int((r > 1).sum()), len(sh))
        ks = [s for s in range(4) if KSTAR[s] == E]
        print(f"{E:>5} {list(g.values())[0][2]:>7} {len(v):>2} "
              f"{v.mean():>13.4e}+-{v.std():7.1e} {e.mean():>8.4f} | "
              f"{rr:>7} {w:>5}  {ks if ks else ''}")
    if best:
        E, gm, w, n = best
        print(f"\nbest E = {E}: van/E = {gm:.4f}x, beats vanilla on {w}/{n} seeds")
        print("  (>1.0 means that checkpoint beats cold-start vanilla)")
        print(f"  merit rule picks E in {sorted(set(KSTAR.values()))}")


if __name__ == "__main__":
    main()
