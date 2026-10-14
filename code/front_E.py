"""
Front of the error constant against the coefficient bound, E_min(B), on the families
CRK7 (7,7), CRK8 (9,8) and CCK7(6) (8,7/7,6), and the behaviour of the minimizers for
large B, as in the order 5-6 paper.

Chains of constrained minimizations (optimize_E.Family.minimize, soft bound, rho = 10):
  * descending chains: start at distinct local minima of an optimize_E multistart
    (data/Emin_*.json) and lower the bound step by step from 1e3 to 0.45, each
    minimization warm-started from the previous one;
  * ascending chains: start at the stored tableaux and raise the bound from 0.45 to 1e3.
E_min(B) is the smallest E over all chains that satisfy the bound.  For every point
the smallest distance between two abscissae (including c_1 = 0) and the largest
moduli of A and of b are recorded: a minimizer whose coefficients grow without
bound while the abscissae coalesce is a degenerate (confluent) limit, not a tableau.
Results are merged into ../data/front_E.json (one record per chain, keyed by family).
Usage:  python front_E.py FAMILY LEVELS EMIN_JSON[,EMIN_JSON] K TABLEAU[,TABLEAU] [NPROC]
        e.g. python front_E.py CRK7 7,7 Emin_77_5.json 6 CRK7-opt,CRK7s7b 4
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
BOUNDS = [1e3, 300, 100, 50, 30, 20, 15, 10, 7, 5, 4, 3, 2.5, 2, 1.5, 1.25, 1.0,
          0.8, 0.7, 0.6, 0.55, 0.5, 0.45]

_FAM = {}
def family(levels):
    if levels not in _FAM:
        import optimize_E as O
        _FAM[levels] = O.Family(levels)
    return _FAM[levels]

def info(Fm, x):
    A, W = Fm.P.unpack(x[None])
    A = A[0]
    c = A.sum(1)
    S = len(c)
    gap = min(abs(c[i] - c[j]) for i in range(S) for j in range(i))
    return dict(gap=float(gap), maxA=float(np.abs(A).max()), maxb=float(max(np.abs(w[0]).max() for w in W)),
                c=[[float(v.real), float(v.imag)] for v in c])

def norm_maxcoef(Fm, x):
    """max|coef| after the normalization Im b_1 = Im b_2 = Im b_3 = 0 of the inert weights (all levels),
    applied only if the first three stages are real; the minimization penalizes the unnormalized moduli."""
    A, W = Fm.P.unpack(np.array(x)[None])
    A = A[0]
    real3 = np.abs(A[:3, :3].imag).max() < 1e-12
    m = float(np.abs(A).max())
    for w in W:
        b = w[0].copy()
        if real3:
            b[:3] = b[:3].real
        m = max(m, float(np.abs(b).max()))
    return m

def add_normalized(fn):
    """Add maxcoef_n to every point of a front file and to the B=5 minima files used as starts."""
    d = json.load(open(fn))
    for ch in d:
        Fm = family(ch["levels"])
        for p in ch["points"]:
            if p["x"] is not None:
                p["maxcoef_n"] = norm_maxcoef(Fm, p["x"])
    json.dump(d, open(fn, "w"))

def chain(args):
    fam, levels, label, x0, bounds = args
    Fm = family(levels)
    x = np.array(x0)
    out = []
    t0 = time.time()
    for B in bounds:
        x1, E, it = Fm.minimize(x, B, rho=10.0, iters=800)
        r = float(np.abs(Fm.residual(x1)).max())
        ok = r < 1e-10 and np.all(np.isfinite(x1))
        if ok:
            x = x1
        rec = dict(B=B, E=E if ok else None, maxcoef=Fm.max_coef(x1) if ok else None,
                   resid=r, iters=it, x=x1.tolist() if ok else None)
        if ok:
            rec.update(info(Fm, x1))
        out.append(rec)
    return fam, label, out, time.time() - t0

def distinct(results, k):
    sel = []
    for r in sorted(results, key=lambda d: d["E"]):
        if r["resid"] > 1e-11:
            continue
        if all(abs(r["E"] - q["E"]) > 2e-5 * r["E"] or abs(r["maxcoef"] - q["maxcoef"]) > 0.05 * r["maxcoef"]
               for q in sel):
            sel.append(r)
        if len(sel) == k:
            break
    return sel

if __name__ == "__main__" and sys.argv[1] == "--norm":       # python front_E.py --norm  (after the runs)
    for fn in ("front_E.json", "front_E_CCK7.json"):
        if os.path.exists(os.path.join(DATA, fn)):
            add_normalized(os.path.join(DATA, fn))
    for fn, levels in (("Emin_77_5.json", "7,7"), ("Emin_77_5_polish.json", "7,7"), ("Emin_98_5.json", "9,8"),
                       ("Emin_98_5_polish.json", "9,8"), ("Emin_87_76_5.json", "8,7/7,6"),
                       ("Emin_87_76_5_front.json", "8,7/7,6"), ("Emin_87_76_5_front2.json", "8,7/7,6"),
                       ("Emin_87_76_5_pol_r1.json", "8,7/7,6"), ("Emin_87_76_5_final.json", "8,7/7,6")):
        if not os.path.exists(os.path.join(DATA, fn)):
            continue
        d = json.load(open(os.path.join(DATA, fn)))
        for r in d["results"]:
            r["maxcoef_n"] = norm_maxcoef(family(levels), r["x"])
        json.dump(d, open(os.path.join(DATA, fn), "w"), indent=1)
    print("normalized max|coef| added")
elif __name__ == "__main__":
    fam, levels, files, k, names = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    nproc = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    import optimize_E as O
    Fm = family(levels)
    tasks = []
    for fn in files.split(","):
        d = json.load(open(os.path.join(DATA, fn)))
        for r in distinct(d["results"], k):
            tasks.append((fam, levels, f"down:{fn}:{r['start']}", r["x"], BOUNDS))
    for lab, x0 in O.collect_starts(Fm, names):
        x0 = Fm.newton_project(x0, iters=12)
        if np.abs(Fm.residual(x0)).max() < 1e-11:
            tasks.append((fam, levels, "up:" + lab, x0.tolist(), BOUNDS[::-1]))
        else:
            print("  skip", lab)
    print(len(tasks), "chains", flush=True)
    fn = os.path.join(DATA, os.environ.get("FRONT_OUT", "front_E.json"))   # separate file for a parallel run
    res = [c for c in (json.load(open(fn)) if os.path.exists(fn) else []) if c["family"] != fam]
    t0 = time.time()
    with Pool(nproc) as pool:
        for fam_, label, out, dt in pool.imap_unordered(chain, tasks):
            res.append(dict(family=fam_, levels=levels, chain=label, points=out))
            pts = {p["B"]: p for p in out}
            show = " ".join(f"B{B:g}:" + ("-" if pts[B]["E"] is None else f"{pts[B]['E']:.4e}/{pts[B]['maxcoef']:.3g}")
                            for B in (1e3, 100, 5, 2, 1, 0.5))
            print(f"[{time.time() - t0:5.0f}s] {fam_:5s} {label:40s} ({dt:4.0f}s) {show}", flush=True)
            json.dump(res, open(fn, "w"))
    print("done", time.time() - t0)
