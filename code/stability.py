"""
Linear stability of every method.

For a projected method applied to a real linear system y' = J y the step is
y_{n+1} = P(hJ) y_n with the REAL polynomial P(z) = 1 + sum_k Re(b^T A^{k-1} 1) z^k,
so the eigenvalues of the step matrix are P(h lambda) for the (possibly complex)
eigenvalues lambda of J.  The stability region is {z : |P(z)| <= 1}.
We check this directly on 2x2 real matrices with eigenvalues sigma +- i omega, and
compute the real and imaginary stability intervals, the area of the region (left
half-plane part and total), and the same quantities per function evaluation.
Writes ../data/stability.json.
"""
import json, os, sys
from math import factorial
import numpy as np
import mpmath as mp
from tableaux import load_refined, real_methods

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

def poly(T, L=0):
    return np.array([float(v) for v in T.stab_coeffs(L)])

def Tm(m):
    return np.array([1.0 / factorial(k) for k in range(m + 1)])

def real_interval(co):
    """Left end of the stability interval on the negative real axis (connected to 0)."""
    P = np.polynomial.Polynomial(co)
    x = np.linspace(0, -12, 1_200_001)
    bad = np.abs(P(x)) > 1 + 1e-13
    bad[0] = False
    i = np.argmax(bad[1:]) + 1 if bad[1:].any() else len(x) - 1
    # refine the crossing by bisection on |P| - 1
    a, b = x[i - 1], x[i]
    for _ in range(80):
        m = 0.5 * (a + b)
        if abs(P(m)) > 1:
            b = m
        else:
            a = m
    return a

def imag_interval(co):
    """Largest y with |P(i s)| <= 1 for 0 <= s <= y.  g(y) = |P(iy)|^2 - 1 is an even
    polynomial; if its lowest non-vanishing coefficient is positive the axis is left
    at once (interval 0), otherwise the interval ends at the first positive root of g."""
    P = np.polynomial.Polynomial(co)
    # P(iy) = Re + i Im as polynomials in y
    n = len(co)
    re = np.zeros(n); im = np.zeros(n)
    for k, a in enumerate(co):
        ik = (1j) ** k
        re[k] += a * ik.real; im[k] += a * ik.imag
    g = (np.polynomial.Polynomial(re) ** 2 + np.polynomial.Polynomial(im) ** 2 - 1).coef
    g[np.abs(g) < 1e-14 * max(1.0, np.abs(g).max())] = 0.0
    nz = np.nonzero(g)[0]
    if len(nz) == 0 or g[nz[0]] > 0:
        return 0.0
    # first crossing of |P(iy)| = 1 by a grid scan and bisection (the roots of g are
    # ill-conditioned when the top coefficients of P are tiny, e.g. Verner8(7))
    f = lambda y: abs(P(1j * y)) - 1.0
    y = np.linspace(0.0, 20.0, 20001)[1:]
    out = np.nonzero(np.abs(P(1j * y)) > 1.0 + 1e-12)[0]
    if len(out) == 0:
        return np.inf
    a, b = (y[out[0] - 1] if out[0] > 0 else 0.0), y[out[0]]
    for _ in range(60):
        m = 0.5 * (a + b)
        if f(m) > 0: b = m
        else: a = m
    return a

def area(co, box=(-8.0, 2.0, -7.0, 7.0), n=2001):
    P = np.polynomial.Polynomial(co)
    X, Y = np.meshgrid(np.linspace(box[0], box[1], n), np.linspace(box[2], box[3], n))
    inside = np.abs(P(X + 1j * Y)) <= 1
    dA = (box[1] - box[0]) / (n - 1) * (box[3] - box[2]) / (n - 1)
    return float(inside.sum() * dA), float((inside & (X <= 0)).sum() * dA)

def check_2x2(T, L=0, samples=((-1.0, 0.7), (-2.5, 1.3), (-0.3, 2.9), (-3.1, 0.2)), h=1.0):
    """Spectral radius of one projected step on real 2x2 systems vs |P(h lambda)|."""
    m, p, b = T.levels[L]
    co = poly(T, L)
    worst = 0.0
    for sg, om in samples:
        J = [[mp.mpf(sg), mp.mpf(om)], [mp.mpf(-om), mp.mpf(sg)]]
        f = lambda t, y: [J[0][0] * y[0] + J[0][1] * y[1], J[1][0] * y[0] + J[1][1] * y[1]]
        S = []
        for e in ([1, 0], [0, 1]):
            y = [mp.mpf(e[0]), mp.mpf(e[1])]
            K = []
            for i in range(m):
                yi = [y[q] + h * mp.fsum(T.A[i][j] * K[j][q] for j in range(i)) for q in range(2)]
                K.append(f(0, yi))
            S.append([mp.re(y[q] + h * mp.fsum(b[i] * K[i][q] for i in range(m))) for q in range(2)])
        M = np.array([[float(S[0][0]), float(S[1][0])], [float(S[0][1]), float(S[1][1])]])
        rho = max(abs(np.linalg.eigvals(M)))
        val = abs(np.polynomial.Polynomial(co)(complex(sg, om) * h))
        worst = max(worst, abs(rho - val))
    return worst

ROWS = [
    # label, tableau, level, evaluations per step (stages of the propagated formula)
    ("CRK7 ($T_7$)", "CRK7-opt", 0, 7),
    ("CCK7(6)", "CCK7(6)", 0, 8),
    ("CCK7(6) embedded", "CCK7(6)", 1, 7),
    ("CRK8-opt", "CRK8-opt", 0, 9),
    ("CRK8-s", "CRK8-s", 0, 9),
    ("Butcher7", "Butcher7", 0, 9),
    ("Verner7(6)", "Verner7(6)", 0, 10),
    ("Vern7", "Vern7", 0, 9),
    ("Fehlberg7", "Fehlberg7", 0, 11),
    ("CooperVerner8", "CooperVerner8", 0, 11),
    ("DP8(7)", "DP8(7)", 0, 13),
    ("DOP853", "DOP853", 0, 12),
    ("Fehlberg8", "Fehlberg8", 0, 13),
    ("Verner8(7)", "Verner8(7)", 0, 13),
]
COMPLEX = ["CRK7-opt", "CRK7-s", "CCK7(6)", "CCK7-casc", "CRK8-opt", "CRK8-s"]

def main():
    mp.mp.dps = 40
    TT = load_refined()
    TT.update(real_methods())
    out = {}
    for lab, key, L, ev in ROWS:
        if key not in TT:
            print(f"{lab:24s} (missing)")
            continue
        T = TT[key]
        co = poly(T, L)
        r, i = real_interval(co), imag_interval(co)
        A_tot, A_left = area(co)
        out[lab] = dict(key=key, level=L, evals=ev, coeffs=co.tolist(), real=r, imag=i,
                        real_per_eval=-r / ev, imag_per_eval=i / ev, area=A_tot, area_left=A_left,
                        area_left_per_eval2=A_left / ev ** 2, check2x2=check_2x2(T, L))
        print(f"{lab:24s} real [{r:.4f}, 0]  imag {i:.4f}  per eval {-r / ev:.3f}  area(left) {A_left:.3f}"
              f"  2x2 check {out[lab]['check2x2']:.1e}")
    # every level of every complex tableau: deviation of P from T_m
    dev = {}
    # P - T_p for every level: zero below degree p+1; the free tail (s > p) is reported
    for key in [k for k in COMPLEX if k in TT]:
        T = TT[key]
        dev[key] = []
        for L in range(len(T.levels)):
            m, p, _ = T.levels[L]
            co = poly(T, L)
            dev[key].append(dict(upto_p=float(np.max(np.abs(co[:p + 1] - Tm(p)))),
                                 tail=[float(c * factorial(k)) for k, c in enumerate(co) if k > p]))
    out["_Tm_deviation"] = dev
    for m in range(1, 9):
        co = Tm(m)
        out[f"_T{m}"] = dict(real=real_interval(co), imag=imag_interval(co), area_left=area(co)[1])
    print("max deviation of P from T_p up to z^p over all levels:", max(d["upto_p"] for v in dev.values() for d in v))
    for k, v in dev.items():
        print(f"  {k:12s} tail k!*c_k (k > p):", [[round(x, 5) for x in d["tail"]] for d in v])
    print({k: (round(v["real"], 4), round(v["imag"], 4)) for k, v in out.items() if k.startswith("_T") and "real" in v})
    json.dump(out, open(os.path.join(DATA, "stability.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
