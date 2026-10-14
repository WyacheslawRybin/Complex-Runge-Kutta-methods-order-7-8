"""
Turn a double-precision result into a named tableau of the paper: inert directions
removed (Im of real stages), minimum-norm Newton refinement of all levels jointly at
60 digits, full verification by check_tableaux.check, and storage in
../data/tableaux.json (refined coefficients) and ../data/check_added.json.

    python add_tableau.py NAME SOURCE
SOURCE:
    emin:FILE.json#k        k-th result of optimize_E.py (levels from the file)
    casc:FILE.json#k        k-th tableau of check_casc.py (already refined)
    pair:NAME_A+NAME_B      not used here (pairs are built by embedded.py)
"""
import json, os, sys
import numpy as np
import mpmath as mp
from crk import Tableau
import tableaux as TB
import check_tableaux as CT
from cascade_search import Problem

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

def from_emin(fn, k, name):
    d = json.load(open(fn))
    levels = [tuple(l) for l in d["levels"]]
    P = Problem(levels)
    x = np.array(d["results"][k]["x"])
    A, W = P.unpack(x[None])
    S = P.S
    Am = [[mp.mpc(complex(A[0, i, j])) for j in range(S)] for i in range(S)]
    lev = [(m, p, [mp.mpc(complex(v)) for v in W[L][0]]) for L, (m, p) in enumerate(levels)]
    return Tableau(name, Am, lev, True)

def from_casc(fn, k, name):
    d = json.load(open(fn))[k]
    S = d["S"]
    cv = lambda u: mp.mpc(mp.mpf(u[0]), mp.mpf(u[1]))
    A = [[cv(d["A"][i][j]) if j < i else mp.mpc(0) for j in range(S)] for i in range(S)]
    lev = [(d["S"], d["p"], [cv(u) for u in d["b"][0]]), (d["MH"], d["q"], [cv(u) for u in d["b"][1]])]
    return Tableau(name, A, lev, True)

def add(name, T):
    mp.mp.dps = CT.DPS
    T, real = TB.clean_inert(T, tol=1e-10)
    # E-minima can lie on a locus where the Jacobian loses rank (CRK7-opt: b_2 -> 0); the
    # minimum-norm Newton step then needs a small pseudo-inverse cutoff and more iterations
    T = T.normalized()
    R, shift, r = T.refine(iters=40, rcond=1e-14)
    print(f"  refine: shift {mp.nstr(shift, 3)}, residual {mp.nstr(r, 3)}")
    R, real = TB.clean_inert(R, tol=mp.mpf(10) ** -40)
    R.name = name
    _, info = CT.check((name, R))
    info["real_stages_list"] = real
    print(CT.line(name, info), flush=True)
    fn = os.path.join(DATA, "tableaux.json")
    tab = json.load(open(fn))
    tab[name] = dict(s=info["s"], cost=info["cost"], projected=True, levels=info["levels"],
                     A=info["A"], b=info["b"])
    json.dump(tab, open(fn, "w"), indent=1)
    fc = os.path.join(DATA, "check_added.json")
    chk = json.load(open(fc)) if os.path.exists(fc) else {}
    chk[name] = info
    json.dump(chk, open(fc, "w"), indent=1)
    return info

if __name__ == "__main__":
    name, src = sys.argv[1], sys.argv[2]
    mp.mp.dps = CT.DPS
    kind, rest = src.split(":", 1)
    fn, _, k = rest.partition("#")
    T = {"emin": from_emin, "casc": from_casc}[kind](fn, int(k or 0), name)
    add(name, T)
