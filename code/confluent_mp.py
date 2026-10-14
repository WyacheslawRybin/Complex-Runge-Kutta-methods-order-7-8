"""
The confluent limit scheme L (Section 3.4, eq. (Lscheme)) in multiple precision:
    k_1 = f(y_n),  G = h f'(y_n) f(y_n),
    k_i = f(y_n + h(gamma_i k_1 + alpha_i G + sum_{j=3}^{i-1} a_ij k_j)),  i = 3..8,
    y_{n+1} = Re(y_n + h(beta_1 k_1 + beta_G G + sum_{i=3}^{8} b_i k_i)).
Elementary weights: Phi_1(t) = [t = tau], Phi_G(t) = [t = [tau]], and for i >= 3
Phi_i(t) = prod over the subtrees u of t of (gamma_i Phi_1(u) + alpha_i Phi_G(u) + sum_j a_ij Phi_j(u)).
Reads the 25-digit coefficients of the note (../data/confluent_L_note.txt), refines them by
minimum-norm Gauss-Newton iterations on the 200 projected conditions of order <= 8 (68 real
unknowns; Im beta_1 = Im beta_G = 0 are inert), and reports the residual, the rank of the
Jacobian, the norm of the ninth-order error coefficients and max|coef|.
Writes ../data/confluent_L.json (45 digits).
    python confluent_mp.py
"""
import json, os, sys
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crk

DATA = os.path.join(HERE, "..", "data")
STAGES = range(3, 9)
NAMES = ([f"γ_{i}" for i in STAGES] + [f"α_{i}" for i in STAGES]
         + [f"a_{i}{j}" for i in STAGES for j in range(3, i)] + [f"b_{i}" for i in STAGES])

def read_note():
    vals = {}
    for line in open(os.path.join(DATA, "confluent_L_note.txt"), encoding="utf-8"):
        k, re_s, im_s = [x.strip() for x in line.split("|")]
        vals[k] = mp.mpc(mp.mpf(re_s), mp.mpf(im_s))
    return vals

def pack(v):
    x = []
    for k in NAMES:
        x += [mp.re(v[k]), mp.im(v[k])]
    return x + [mp.re(v["β_1"]), mp.re(v["β_G"])]

def unpack(x):
    v = {k: mp.mpc(x[2 * n], x[2 * n + 1]) for n, k in enumerate(NAMES)}
    v["β_1"], v["β_G"] = mp.mpc(x[-2], 0), mp.mpc(x[-1], 0)
    return v

def weights(v, pmax):
    """Phi_t for stages 1, G, 3..8 (dict over trees of dict over stage keys)."""
    tau, ttau = (), ((),)
    cache = {}
    def Phi(t):
        if t in cache:
            return cache[t]
        out = {1: mp.mpf(1) if t == tau else mp.mpf(0), "G": mp.mpf(1) if t == ttau else mp.mpf(0)}
        for i in STAGES:
            prod = mp.mpc(1)
            for u in t:
                pu = Phi(u)
                prod *= (v[f"γ_{i}"] * pu[1] + v[f"α_{i}"] * pu["G"]
                         + mp.fsum(v[f"a_{i}{j}"] * pu[j] for j in range(3, i)))
            out[i] = prod
        cache[t] = out
        return out
    for t in crk.trees_upto(pmax):
        Phi(t)
    return cache

def bphi(v, ph):
    return v["β_1"] * ph[1] + v["β_G"] * ph["G"] + mp.fsum(v[f"b_{i}"] * ph[i] for i in STAGES)

def residual(x, p=8):
    v = unpack(x)
    ph = weights(v, p)
    return [mp.re(bphi(v, ph[t])) - mp.mpf(1) / crk.gamma(t) for t in crk.trees_upto(p)]

def jacobian(x, d):
    cols = []
    for k in range(len(x)):
        xp, xm = list(x), list(x)
        xp[k] += d; xm[k] -= d
        rp, rm = residual(xp), residual(xm)
        cols.append([(a - b) / (2 * d) for a, b in zip(rp, rm)])
    return mp.matrix([[cols[k][r] for k in range(len(x))] for r in range(len(cols[0]))])

def refine(x, iters=8):
    for it in range(iters):
        r = residual(x)
        rmax = max(abs(v) for v in r)
        print(f"  iter {it}: max|r| = {mp.nstr(rmax, 3)}", flush=True)
        if rmax < mp.mpf(10) ** (-(mp.mp.dps - 25)):
            break
        J = jacobian(x, mp.mpf(10) ** (-(mp.mp.dps // 2 - 5)))
        U, S, V = mp.svd_r(J)
        tol = mp.mpf(10) ** -20 * S[0]
        dx = [mp.mpf(0)] * len(x)
        for k in range(len(S)):
            if S[k] > tol:
                c = mp.fsum(U[i, k] * r[i] for i in range(J.rows)) / S[k]
                for j in range(len(x)):
                    dx[j] -= c * V[k, j]
        x = [a + b for a, b in zip(x, dx)]
    return x, rmax, S

if __name__ == "__main__":
    mp.mp.dps = 90
    x = pack(read_note())
    x, rmax, S = refine(x)
    r = max(abs(v) for v in residual(x))
    rank = sum(1 for s in S if s > mp.mpf(10) ** -20 * S[0])
    v = unpack(x)
    ph = weights(v, 9)
    e9 = [(mp.re(bphi(v, ph[t])) - mp.mpf(1) / crk.gamma(t)) / crk.sigma(t) for t in crk.TREES[9]]
    E9 = mp.sqrt(mp.fsum(e ** 2 for e in e9))
    maxcoef = max(abs(v[k]) for k in v)
    gap = [mp.nstr(S[k], 3) for k in range(max(0, rank - 2), min(len(S), rank + 2))]
    print(f"L: max|r| = {mp.nstr(r, 3)}, rank {rank} of {len(x)} (singular values around the gap {gap}), "
          f"E_9 = {mp.nstr(E9, 4)}, max|coef| = {mp.nstr(maxcoef, 5)}")
    out = dict(resid=float(r), rank=rank, n=len(x), E9=float(E9), maxcoef=float(maxcoef),
               coef={k: [mp.nstr(mp.re(v[k]), 45), mp.nstr(mp.im(v[k]), 45)] for k in NAMES + ["β_1", "β_G"]})
    json.dump(out, open(os.path.join(DATA, "confluent_L.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
