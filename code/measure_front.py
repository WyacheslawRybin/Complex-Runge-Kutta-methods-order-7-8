"""
Measured constants of tableaux from the fronts E_min(B) of front_E.py (data/front_E*.json), as for the
CCK6(5) front points of the order 5-6 paper: the interior local minimum of CRK7 with small coefficients,
and for the pair CCK7(6) the constrained minima at B = 2 and B = 5 and the interior local minimum.
The points are refined at 60 digits (inert Im b of the real stages set to zero) and measured with the
procedure of measure.py.  Writes ../data/front_candidates.json (refined coefficients and E of all levels)
and ../data/measured_front.json.
Usage:  python measure_front.py [NPROC]
"""
import json, os, sys
from multiprocessing import Pool
import numpy as np
import mpmath as mp
import measure as M
import front_E as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
CAND = os.path.join(DATA, "front_candidates.json")

def front_points(fam):
    out = []
    for fn in ("front_E.json", "front_E_CCK7.json"):
        if os.path.exists(os.path.join(DATA, fn)):
            for ch in json.load(open(os.path.join(DATA, fn))):
                if ch["family"] == fam:
                    Fm = F.family(ch["levels"])
                    out += [dict(p, levels=ch["levels"], chain=ch["chain"], Mn=F.norm_maxcoef(Fm, p["x"]))
                            for p in ch["points"] if p["E"] is not None]
    return out

def pick():
    """name -> (levels, x, chain, B): the selection rules, applied to the current front files."""
    best = lambda pts: min(pts, key=lambda p: p["E"])
    c7, k7 = front_points("CRK7"), front_points("CCK7")
    sel = {"CRK7-L": best([p for p in c7 if p["Mn"] <= 1.5]),          # interior minimum, max|coef| = 1.31
           "CCK7-B2": best([p for p in k7 if p["Mn"] <= 2.002]),
           "CCK7-B5": best([p for p in k7 if p["Mn"] <= 5.005]),
           "CCK7-L": best([p for p in k7 if p["Mn"] <= 100])}             # interior minimum
    return {n: (p["levels"], p["x"], p["chain"], p["B"]) for n, p in sel.items()}

def build(name, levels_str, x):
    import tableaux as TB
    from crk import Tableau
    Fm = F.family(levels_str)
    A, W = Fm.P.unpack(np.array(x)[None])
    S = A.shape[1]
    Am = [[mp.mpc(complex(A[0, i, j])) for j in range(S)] for i in range(S)]
    lev = [(m, p, [mp.mpc(complex(v)) for v in W[L][0]]) for L, (m, p) in enumerate(Fm.levels)]
    T, _ = TB.clean_inert(Tableau(name, Am, lev, True), tol=1e-10)
    R, shift, r = T.normalized().refine(iters=40, rcond=1e-14)
    R, real = TB.clean_inert(R, tol=mp.mpf(10) ** -40)
    R.name = name
    return R, shift, r, real

def save_candidates():
    out = {}
    with mp.workdps(60):
        for name, (levels, x, chain, B) in pick().items():
            R, shift, r, real = build(name, levels, x)
            E = [float(R.E(L)) for L in range(len(R.levels))]
            out[name] = dict(levels=[[m, p] for m, p, _ in R.levels], chain=chain, B=B,
                             shift=float(shift), resid=float(r), real_stages=real, E=E,
                             maxcoef=float(R.maxcoef()), c=[[str(mp.re(v)), str(mp.im(v))] for v in R.c],
                             A=[[[str(mp.re(v)), str(mp.im(v))] for v in row] for row in R.A],
                             b=[[[str(mp.re(v)), str(mp.im(v))] for v in b] for _, _, b in R.levels])
            print(f"{name:8s} from {chain} @B{B:g}: refine shift {float(shift):.1e} resid {float(r):.1e}  "
                  f"E {['%.4e' % e for e in E]}  max|coef| {float(R.maxcoef()):.4f}  real stages {real}", flush=True)
    json.dump(out, open(CAND, "w"), indent=1)
    return out

def load_candidates():
    from crk import Tableau
    d = json.load(open(CAND))
    cv = lambda u: mp.mpc(mp.mpf(u[0]), mp.mpf(u[1]))
    return {n: Tableau(n, [[cv(u) for u in row] for row in v["A"]],
                       [(m, p, [cv(u) for u in b]) for (m, p), b in zip(v["levels"], v["b"])], True)
            for n, v in d.items()}

def _init():
    M._init()
    with mp.workdps(60):
        M._TAB.update(load_candidates())

if __name__ == "__main__":
    from problems import ORDER
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    names = list(save_candidates())
    tasks = [(n, n, 0, q) for n in names for q in ORDER]
    res = {}
    fn = os.path.join(DATA, "measured_front.json")
    with Pool(nproc, initializer=_init) as pool:
        for label, q, rows, dt in pool.imap_unordered(M.task, tasks):
            res.setdefault(label, {})[q] = dict(rows=rows)
            print(f"{label:8s} {q:14s} ({dt:5.0f}s) " + " ".join("-" if r["Ch"] is None else f"{r['Ch']:.4e}"
                                                             for r in rows[-4:]), flush=True)
            json.dump(res, open(fn, "w"), indent=1)
        todo = []
        for lab in res:
            for q, cell in res[lab].items():
                Ch = [r["Ch"] for r in cell["rows"] if r["err"]]
                R = [2 * Ch[k + 1] - Ch[k] for k in range(len(Ch) - 1)]
                if len(R) > 1 and abs(R[-1] - R[-2]) > 0.01 * abs(R[-1]):
                    todo.append((lab, lab, 0, q, [7, 8]))
        print(len(todo), "cells to extend", flush=True)
        for label, q, rows in pool.imap_unordered(M.task_ext, todo):
            res[label][q]["rows"] += rows
            json.dump(res, open(fn, "w"), indent=1)
    print("done")
