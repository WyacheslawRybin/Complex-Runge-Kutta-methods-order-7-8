# Jacobian singular values in multiple precision (the float64 SVD has a floor near 1e-16 * sv[0])
# Usage (from code/):  python rank_mp.py CRK7-opt CRK8-opt "CCK7(6)"   prints the singular values from index 40 on
import sys, mpmath as mp
sys.path.insert(0, '.')
import check_tableaux as CT
from tableaux import load_refined
mp.mp.dps = 60
TT = load_refined()
for name in sys.argv[1:]:
    T = TT[name]
    x = T.get_x()
    F = lambda xx: [v for lev in T.set_x(xx).residuals() for v in lev]
    r0 = F(x); h = mp.mpf(10) ** -28
    cols = []
    for k in range(len(x)):
        xp = x[:]; xp[k] += h
        cols.append([(a - b) / h for a, b in zip(F(xp), r0)])
    T.set_x(x)
    J = mp.matrix(len(r0), len(x))
    for k, col in enumerate(cols):
        for i, v in enumerate(col):
            J[i, k] = v
    sv = mp.svd_r(J, compute_uv=False)
    sv = sorted([sv[i] for i in range(len(sv))], reverse=True)
    print(name, len(r0), len(x), " ".join(mp.nstr(v, 3) for v in sv[40:]), flush=True)
