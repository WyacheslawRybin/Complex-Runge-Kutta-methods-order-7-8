"""
Cash-Karp cascade on the coefficient matrix of CCK7(6): for every level (m, q) on the
leading m stages, the minimum-norm projected weights of order q (multiple precision),
their residual, and the empirical order on two problems with a fixed step.
    python cascade_levels.py
Writes ../data/cascade_levels.json.
"""
import json, os, sys
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crk
import measure
from tableaux import load_refined

LEVELS = [(7, 6), (6, 5), (5, 4), (4, 3), (3, 2), (2, 2), (1, 1)]
PROBS = {"quadratic": (40, 80), "nonautonomous": (160, 320)}
NLOW = (1280, 2560)          # levels of order <= 3: further into the asymptotic range

def level_weights(T, m, q):
    ph = T.phi(q)
    ts = crk.trees_upto(q)
    cplx = [i for i in range(m) if any(mp.im(ph[t][i]) != 0 for t in ts)]
    M = mp.matrix(len(ts), m + len(cplx))
    rhs = mp.matrix(len(ts), 1)
    for r, t in enumerate(ts):
        for i in range(m):
            M[r, i] = mp.re(ph[t][i])
        for k, i in enumerate(cplx):
            M[r, m + k] = -mp.im(ph[t][i])
        rhs[r] = mp.mpf(1) / crk.gamma(t)
    U, S, V = mp.svd_r(M)
    tol = mp.mpf(10) ** (-mp.mp.dps // 2) * S[0]
    x = mp.matrix(M.cols, 1)
    for k in range(len(S)):
        if S[k] > tol:
            coef = sum(U[r, k] * rhs[r] for r in range(M.rows)) / S[k]
            for j in range(M.cols):
                x[j] += coef * V[k, j]
    b = [mp.mpc(x[i], 0) for i in range(m)]
    for k, i in enumerate(cplx):
        b[i] += mp.mpc(0, x[m + k])
    res = max(abs(mp.re(mp.fsum(b[i] * ph[t][i] for i in range(m))) - mp.mpf(1) / crk.gamma(t)) for t in ts)
    return b + [mp.mpc(0)] * (T.s - m), res

if __name__ == "__main__":
    mp.mp.dps = 60
    T = load_refined()["CCK7(6)"]
    measure._init()
    mp.mp.dps = measure.DPS
    out = {}
    base = len(T.levels)
    for m, q in LEVELS:
        with mp.workdps(60):
            b, res = level_weights(T, m, q)
        T.levels.append((m, q, b))
        L = len(T.levels) - 1
        row = dict(m=m, q=q, resid=float(res), maxb=float(max(abs(v) for v in b)))
        for prob, (n1, n2) in PROBS.items():
            if q <= 3:
                n1, n2 = NLOW
            e1, e2 = measure._run(T, L, prob, n1), measure._run(T, L, prob, n2)
            row[prob] = float(mp.log(e1 / e2, 2))
        out[f"({m},{q})"] = row
        print(f"level ({m},{q}): resid {row['resid']:.1e}  max|b| {row['maxb']:.3f}  "
              + "  ".join(f"{p}: {row[p]:.3f}" for p in PROBS), flush=True)
    json.dump(out, open(os.path.join(HERE, "..", "data", "cascade_levels.json"), "w"), indent=1)
