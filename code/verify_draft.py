"""
Check, one by one, the numerical claims of the CRK7 draft ("Seventh order at seven
stages", appendix tableaux B0, Bbal, Blo) and of COMPLEX_RK_FINAL.md (CRK7s7b,
CRK8s9) against the refined tableaux of data/tableaux.json.
Writes ../data/verify_draft.json and prints a claim/measured list.
"""
import json, os, sys, math
import numpy as np
import mpmath as mp
import crk
from tableaux import load_refined
from problems import P, references

mp.mp.dps = 60
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

def jac(T, h):
    x = T.get_x()
    F = lambda xx: [v for lev in T.set_x(xx).residuals() for v in lev]
    r0 = F(x)
    J = np.zeros((len(r0), len(x)))
    for k in range(len(x)):
        xp = x[:]; xp[k] += h
        J[:, k] = [float((a - b) / h) for a, b in zip(F(xp), r0)]
    return J

def M_sv(T):
    m, p, b = T.levels[0]
    ph = T.phi(p)
    ts = crk.trees_upto(p)
    Pm = np.array([[float(mp.re(ph[t][i])) for i in range(m)] for t in ts])
    Qm = np.array([[float(mp.im(ph[t][i])) for i in range(m)] for t in ts])
    return np.linalg.svd(np.hstack([Pm, -Qm]), compute_uv=False)

def order_next(T):
    m, p, b = T.levels[0]
    ph = T.phi(p + 1)
    r = [abs(mp.re(mp.fsum(b[i] * ph[t][i] for i in range(m))) - mp.mpf(1) / crk.gamma(t)) for t in crk.TREES[p + 1]]
    return float(min(r)), float(max(r))

def real_interval(co):
    """Left end of the stability interval on the negative real axis: first x<0 with |P(x)| = 1."""
    Pz = lambda x: mp.polyval(list(reversed(co)), x)
    x, step = mp.mpf(0), mp.mpf("0.001")
    while abs(Pz(x - step)) <= 1:
        x -= step
    return mp.findroot(lambda z: abs(Pz(z)) - 1, (x - step, x), solver="bisect")

def integrate(T, f, y0, Tend, N):
    m, p, b = T.levels[0]
    A, c = T.A, T.c
    h = mp.mpf(Tend) / N
    y = [mp.mpf(v) for v in y0]
    t = mp.mpf(0)
    for _ in range(N):
        K = []
        for i in range(m):
            yi = [y[q] + h * mp.fsum(A[i][j] * K[j][q] for j in range(i)) for q in range(len(y))] if i else y
            K.append(f(t + c[i] * h, yi))
        y = [mp.re(y[q] + h * mp.fsum(b[i] * K[i][q] for i in range(m))) for q in range(len(y))]
        t += h
    return y

def generic(t, y):
    return [mp.sin(y[1]) + y[2] * y[0], mp.cos(y[0]) - mp.exp(-y[2]), y[0] * y[1] - mp.sin(y[2]) + mp.mpf(1) / 4]

def richardson(T, f, y0, Tend, ns):
    Y = [integrate(T, f, y0, Tend, n) for n in ns]
    d = [mp.sqrt(mp.fsum((a - b) ** 2 for a, b in zip(Y[k], Y[k + 1]))) for k in range(len(ns) - 1)]
    return [float(mp.log(d[k] / d[k + 1], 2)) for k in range(len(d) - 1)], [float(v) for v in d]

def main():
    TT = load_refined()
    ref = references()
    out = {}
    print("=== draft claims ===")
    for name in ["CRK7-B0", "CRK7-Bbal", "CRK7-Blo", "CRK7s7b"]:
        T = TT[name]
        d = {}
        sv = M_sv(T)
        d["M_sv"] = sv.tolist()
        J = jac(T, mp.mpf(10) ** -30)
        jsv = np.linalg.svd(J, compute_uv=False)
        d["J_sv_first_last_nonzero_next"] = [float(jsv[0]), float(jsv[47]), float(jsv[48])]
        d["order8_resid_min_max"] = order_next(T)
        co = T.stab_coeffs(0)
        d["stab_interval"] = float(real_interval(co))
        d["E"] = float(T.E())
        d["maxcoef"] = float(T.maxcoef())
        # Lorenz T = 0.5, N = 20..320, errors against the reference
        lor = P["Lorenz05"]
        errs = []
        with mp.workdps(40):
            for N in (20, 40, 80, 160, 320):
                y = integrate(T, lor["f_mp"], lor["y0"], lor["T"], N)
                errs.append(float(mp.sqrt(mp.fsum((a - b) ** 2 for a, b in zip(y, ref["Lorenz05"])))))
        d["lorenz_err"] = errs
        d["lorenz_orders"] = [math.log2(errs[k] / errs[k + 1]) for k in range(4)]
        out[name] = d
        print(f"{name}: E {d['E']:.7e} max|coef| {d['maxcoef']:.7f}")
        print(f"   M sv: " + ", ".join(f"{v:.3g}" for v in sv))
        print(f"   J sv: first {jsv[0]:.3g}, 48th {jsv[47]:.3g}, 49th {jsv[48]:.3g} (gap {jsv[47] / jsv[48]:.2g})")
        print(f"   order-8 residuals: min {d['order8_resid_min_max'][0]:.2e}, max {d['order8_resid_min_max'][1]:.2e}")
        print(f"   real stability interval [{d['stab_interval']:.6f}, 0]")
        print(f"   Lorenz T=0.5 errors " + " ".join(f"{e:.4g}" for e in errs) + "  orders "
              + " ".join(f"{o:.3f}" for o in d["lorenz_orders"]), flush=True)
    # rank at a random point
    rng = np.random.default_rng(0)
    T = TT["CRK7-B0"]
    x = [mp.mpf(float(v)) for v in rng.normal(0, 0.5, len(T.get_x()))]
    Tr = T.set_x(x)
    sv = np.linalg.svd(jac(Tr, mp.mpf(10) ** -30), compute_uv=False)
    rr = int(np.sum(sv > 1e-12 * sv[0]))
    out["random_point_rank"] = [rr, len(sv)]
    print(f"rank of the Jacobian at a random point: {rr} of {len(sv)} free variables (Im b1 excluded)")
    # Richardson orders (FINAL.md 2.2: Lorenz y0 = (1,1,1), T = 0.5; generic y0 = (0.3,-0.2,0.1), T = 1)
    print("=== FINAL.md 2.2: Richardson orders ===")
    lor = P["Lorenz05"]
    for name in ["CRK8s9", "CRK7s7b"]:
        T = TT[name]
        with mp.workdps(50):
            o1, d1 = richardson(T, lor["f_mp"], ["1", "1", "1"], "0.5", [20, 40, 80, 160, 320])
            o2, d2 = richardson(T, generic, ["0.3", "-0.2", "0.1"], 1, [8, 16, 32, 64, 128])
        out[name + "_richardson"] = dict(lorenz=o1, generic=o2, lorenz_d=d1, generic_d=d2)
        print(f"{name}: Lorenz " + " ".join(f"{v:.3f}" for v in o1) + " | generic " + " ".join(f"{v:.3f}" for v in o2), flush=True)
    json.dump(out, open(os.path.join(OUT, "verify_draft.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
