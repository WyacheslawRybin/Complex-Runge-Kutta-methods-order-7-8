"""
Named tableaux taken from the fronts E_min(B) of front_E.py (data/front_E.json):
    CRK7-s    interior local minimum of E_8 with small coefficients (max|coef| = 1.31)
    CRK8-s    smallest E_9 found with all coefficients <= 1 (point of the front at B = 1)
    CRK8-opt  smallest E_9 found under max|coef| <= 5 (interior local minimum, max|coef| = 3.13)
max|coef| is the normalized one (inert Im b of the real stages set to zero).  Each point is refined
and checked by add_tableau.add and stored in ../data/tableaux.json and ../data/check_added.json.
Usage:  python add_front.py [NAME ...]
"""
import sys
import numpy as np
import mpmath as mp
from crk import Tableau
import check_tableaux as CT
import front_E as F
from add_tableau import add
from measure_front import front_points

RULES = {"CRK7-s": ("CRK7", 1.5), "CRK8-s": ("CRK8", 1.0), "CRK8-opt": ("CRK8", 5.0)}

def point(fam, bmax):
    pts = [p for p in front_points(fam) if p["Mn"] <= bmax * 1.001]
    return min(pts, key=lambda p: p["E"])

def tableau(name, p):
    Fm = F.family(p["levels"])
    A, W = Fm.P.unpack(np.array(p["x"])[None])
    S = A.shape[1]
    Am = [[mp.mpc(complex(A[0, i, j])) for j in range(S)] for i in range(S)]
    lev = [(m, q, [mp.mpc(complex(v)) for v in W[L][0]]) for L, (m, q) in enumerate(Fm.levels)]
    return Tableau(name, Am, lev, True)

if __name__ == "__main__":
    mp.mp.dps = CT.DPS
    for name in sys.argv[1:] or list(RULES):
        fam, bmax = RULES[name]
        p = point(fam, bmax)
        print(f"{name}: chain {p['chain']} B={p['B']:g} E={p['E']:.5e} max|coef|_n={p['Mn']:.4f}", flush=True)
        add(name, tableau(name, p))
