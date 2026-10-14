"""
Batched Levenberg-Marquardt search for complex Cash-Karp tableaux with real
projection: one strictly lower triangular A (S x S) shared by several levels
(m, p); level L uses the first m stages and its own weights and must satisfy
Re(w^T Phi_t[:m]) = 1/gamma(t) for all trees with rho(t) <= p.

All starts of a batch are iterated together (numpy broadcasting over the batch
axis), each with its own damping.  Hits (max residual < HIT) are written to
../data/search/<tag>_<seed>.npz together with the best residual of every start,
so that a null result can be read against the calibration runs.

Usage:
  python cascade_search.py LEVELS NBATCH [seed] [warm]
     LEVELS  e.g. "7,7/6,6"  or "7,7/6,6/5,5/4,4/3,3/2,2/1,1"  or "6,6/5,5"
     warm    none | crk7 | crk6 | cck6 | crk8 | mix   (where the A matrices start)
"""
import os, sys, time
import numpy as np
from itertools import combinations_with_replacement

# ----------------------------------------------------------------- trees
def _order(t):
    return 1 + sum(_order(u) for u in t)

def trees_by_order(pmax):
    by = {1: [()]}
    for n in range(2, pmax + 1):
        pool = [t for k in range(1, n) for t in by[k]]
        out, seen = [], set()
        def ms(total, start):
            if total == 0:
                yield []
                return
            for i in range(start, len(pool)):
                w = _order(pool[i])
                if w <= total:
                    for rest in ms(total - w, i):
                        yield [pool[i]] + rest
        for m in ms(n - 1, 0):
            key = tuple(sorted(m, key=repr))
            if key not in seen:
                seen.add(key); out.append(key)
        by[n] = out
    return by

TREES = trees_by_order(10)

def gamma(t):
    g = _order(t)
    for u in t:
        g *= gamma(u)
    return g

def sigma(t):
    from math import factorial
    out = 1
    for u in set(t):
        k = t.count(u)
        out *= sigma(u) ** k * factorial(k)
    return out

# ----------------------------------------------------------------- problem
def chain(k):
    """The tall tree of order k (bushy-free chain), as a nested tuple."""
    t = ()
    for _ in range(k - 1):
        t = (t,)
    return t

class Problem:
    def __init__(self, levels, S=None, extra=()):
        """extra: (L, k) pairs -- level L must MISS the chain condition of order k:
        Re(w^T A^{k-1} 1) = 0 instead of 1/k!.  Used to force an embedded formula on all
        stages to differ from the propagated one (it then has order exactly k-1)."""
        self.levels = [tuple(l) for l in levels]
        self.S = S or max(m for m, _ in self.levels)
        self.pmax = max(p for _, p in self.levels)
        S = self.S
        self.aidx = [(i, j) for i in range(S) for j in range(i)]
        self.NA = len(self.aidx)
        self.I = np.array([i for i, _ in self.aidx]); self.Jc = np.array([j for _, j in self.aidx])
        self.K = np.arange(self.NA)
        self.woff, off = [], self.NA
        for m, _ in self.levels:
            self.woff.append(off); off += m
        self.nz = off
        drop = {self.nz + o for o in self.woff}            # Im w_1 of every level
        self.keep = np.array([q for q in range(2 * self.nz) if q not in drop])
        self.n = len(self.keep)
        self.rows = [(L, t) for L, (m, p) in enumerate(self.levels)
                     for k in range(1, p + 1) for t in TREES[k]]
        self.rhs = np.array([1.0 / gamma(t) for _, t in self.rows])
        self.extra = [tuple(e) for e in extra]
        for L, k in self.extra:
            self.rows.append((L, chain(k)))
            self.rhs = np.append(self.rhs, 0.0)
            self.pmax = max(self.pmax, k)
        self.nr = len(self.rows)
        self.trees = [t for k in range(1, self.pmax + 1) for t in TREES[k]]

    def unpack(self, X):
        B = X.shape[0]
        XX = np.zeros((B, 2 * self.nz)); XX[:, self.keep] = X
        Z = XX[:, :self.nz] + 1j * XX[:, self.nz:]
        A = np.zeros((B, self.S, self.S), complex)
        A[:, self.I, self.Jc] = Z[:, :self.NA]
        W = [Z[:, o:o + m] for o, (m, _) in zip(self.woff, self.levels)]
        return A, W

    def pack(self, A, W):
        B = A.shape[0]
        Z = np.zeros((B, self.nz), complex)
        Z[:, :self.NA] = A[:, self.I, self.Jc]
        for o, w in zip(self.woff, W):
            Z[:, o:o + w.shape[1]] = w
        return np.concatenate([Z.real, Z.imag], 1)[:, self.keep]

    def phi(self, A, pmax=None, deriv=True):
        pmax = pmax or self.pmax
        B, S = A.shape[0], self.S
        P, dP, AP, dAP = {}, {}, {}, {}
        def ap(u):
            if u not in AP:
                pu = get(u)
                AP[u] = np.einsum('bij,bj->bi', A, pu)
                if deriv:
                    d = np.einsum('bij,bjk->bik', A, dP[u])
                    d[:, self.I, self.K] += pu[:, self.Jc]
                    dAP[u] = d
            return AP[u]
        def get(t):
            if t in P:
                return P[t]
            v = np.ones((B, S), complex)
            dv = np.zeros((B, S, self.NA), complex) if deriv else None
            for u in t:
                w = ap(u)
                if deriv:
                    dv = dv * w[:, :, None] + v[:, :, None] * dAP[u]
                v = v * w
            P[t] = v
            if deriv:
                dP[t] = dv
            return v
        for k in range(1, pmax + 1):
            for t in TREES[k]:
                get(t)
        return P, dP

    def FJ(self, X, jac=True):
        A, W = self.unpack(X)
        P, dP = self.phi(A, deriv=jac)
        B = X.shape[0]
        r = np.empty((B, self.nr))
        Jc = np.zeros((B, self.nr, self.nz), complex) if jac else None
        for q, (L, t) in enumerate(self.rows):
            m = self.levels[L][0]; w = W[L]
            v = P[t][:, :m]
            r[:, q] = np.einsum('bi,bi->b', w, v).real - self.rhs[q]
            if jac:
                Jc[:, q, :self.NA] = np.einsum('bi,bik->bk', w, dP[t][:, :m])
                Jc[:, q, self.woff[L]:self.woff[L] + m] = v
        if not jac:
            return r, None
        J = np.concatenate([Jc.real, -Jc.imag], 2)[:, :, self.keep]
        return r, J

    def init_weights(self, A):
        """Least-squares weights for given A (variable projection start)."""
        P, _ = self.phi(A, deriv=False)
        W = []
        for L, (m, p) in enumerate(self.levels):
            ts = [t for k in range(1, p + 1) for t in TREES[k]]
            rhs = np.array([1.0 / gamma(t) for t in ts])
            for LL, k in self.extra:
                if LL == L:
                    ts.append(chain(k)); rhs = np.append(rhs, 0.0)
            M = np.stack([P[t][:, :m] for t in ts], 1)            # (B, nt, m)
            Mr = np.concatenate([M.real, -M.imag], 2)
            uv = np.stack([np.linalg.lstsq(Mr[b], rhs, rcond=None)[0] for b in range(A.shape[0])])
            W.append(uv[:, :m] + 1j * uv[:, m:])
        return W

    def error_consts(self, A, W):
        P, _ = self.phi(A, self.pmax + 1, deriv=False)
        out = []
        for (m, p), w in zip(self.levels, W):
            e = np.stack([((np.einsum('bi,bi->b', w, P[t][:, :m])).real - 1.0 / gamma(t)) / sigma(t)
                          for t in TREES[p + 1]], 1)
            out.append(np.linalg.norm(e, axis=1))
        return np.stack(out, 1)

def parse_levels(levels_str):
    """ "8,7/8,6+1:7" -> levels [(8,7),(8,6)], extra [(1,7)] """
    main, *ex = levels_str.split("+")
    levels = [tuple(int(v) for v in l.split(",")) for l in main.split("/")]
    extra = [tuple(int(v) for v in e.split(":")) for e in ex]
    return levels, extra

# ----------------------------------------------------------------- batched LM
def lm(prob, X, iters=3000, tol=1e-13, verbose=False):
    B, n = X.shape
    mu = np.full(B, 1e-3)
    r, J = prob.FJ(X)
    f = np.linalg.norm(r, axis=1)
    active = np.ones(B, bool)
    eye = np.eye(n)
    for it in range(iters):
        idx = np.where(active)[0]
        if len(idx) == 0:
            break
        Ja, ra = J[idx], r[idx]
        JtJ = np.einsum('bri,brj->bij', Ja, Ja)
        g = np.einsum('bri,br->bi', Ja, ra)
        diag = np.einsum('bii->bi', JtJ)
        H = JtJ + mu[idx, None, None] * (eye[None] * (1.0 + diag[:, :, None]))
        try:
            dx = -np.linalg.solve(H, g[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            dx = -np.stack([np.linalg.lstsq(H[k], g[k], rcond=None)[0] for k in range(len(idx))])
        Xn = X[idx] + dx
        rn, Jn = prob.FJ(Xn)
        fn = np.linalg.norm(rn, axis=1)
        ok = np.isfinite(fn) & (fn < f[idx])
        acc = idx[ok]
        X[acc], r[acc], J[acc], f[acc] = Xn[ok], rn[ok], Jn[ok], fn[ok]
        mu[acc] = np.maximum(mu[acc] / 3, 1e-15)
        rej = idx[~ok]
        mu[rej] *= 4
        active[f < tol] = False
        active[mu > 1e12] = False
        active[np.abs(X).max(1) > 50] = False                 # runaway coefficients
        if verbose and it % 500 == 0:
            print(f"    it {it}: active {active.sum()} best {np.sort(f)[:3]}", flush=True)
    # minimum-norm Gauss-Newton polishing
    for _ in range(6):
        dx = np.stack([np.linalg.lstsq(J[b], -r[b], rcond=1e-12)[0] for b in range(B)])
        rn, Jn = prob.FJ(X + dx)
        fn = np.linalg.norm(rn, axis=1)
        ok = fn < f
        X[ok], r[ok], J[ok], f[ok] = (X + dx)[ok], rn[ok], Jn[ok], fn[ok]
    return X, f, np.abs(r).max(1)

def local_dim(prob, x):
    _, J = prob.FJ(x[None])
    sv = np.linalg.svd(J[0], compute_uv=False)
    rank = int(np.sum(sv > 1e-9 * sv[0]))
    return len(x) - rank, rank, sv

# ----------------------------------------------------------------- warm starts
def warm_pool(kind, S):
    """List of complex A matrices (S x S, padded/truncated) from known tableaux."""
    import mpmath as mp
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    out = []
    def addT(T):
        A = np.zeros((S, S), complex)
        k = min(S, T.s)
        for i in range(k):
            for j in range(i):
                A[i, j] = complex(T.A[i][j])
        out.append(A)
    if kind in ("crk7", "mix"):
        import tableaux as TB
        for T in list(TB.complex_named().values())[:4] + TB.pool("77"):
            if T.s == 7:
                addT(T)
    if kind in ("crk8", "mix"):
        import tableaux as TB
        for T in [TB.archive()["CRK8s9"]] + TB.pool("98"):
            addT(T)
    if kind in ("crk6", "cck6", "mix"):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Paper order 5-6", "code", "base", "code"))
        import importlib
        old = sys.modules.pop("tableaux", None)
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Paper order 5-6", "code", "base", "code"))
        T56 = importlib.import_module("tableaux")
        d = T56.load_refined()
        for nm, T in d.items():
            if not T.projected:
                continue
            if kind == "crk6" and not nm.startswith("CRK6"):
                continue
            if kind == "cck6" and not nm.startswith("CCK6"):
                continue
            if kind == "mix" or T.s == 6:
                addT(T)
        sys.modules.pop("tableaux", None)
        if old is not None:
            sys.modules["tableaux"] = old
    return out

def starts(prob, B, rng, warm):
    S = prob.S
    A = np.zeros((B, S, S), complex)
    tril = np.tril(np.ones((S, S)), -1)
    for b in range(B):
        if warm and rng.random() < 0.6:
            A0 = warm[rng.integers(len(warm))].copy()
            # random new rows where the warm matrix is smaller than S
            for i in range(S):
                if np.all(A0[i, :i] == 0) and i > 0:
                    sc = rng.choice([0.3, 0.5])
                    A0[i, :i] = rng.normal(0, sc, i) + 1j * rng.normal(0, sc, i)
            sc = rng.choice([0.0, 0.01, 0.05, 0.15, 0.3])
            A[b] = A0 + tril * (rng.normal(0, sc, (S, S)) + 1j * rng.normal(0, sc, (S, S)))
        else:
            sc = rng.choice([0.3, 0.5, 0.8])
            A[b] = tril * (rng.normal(0, sc, (S, S)) + 1j * rng.normal(0, sc, (S, S)))
    return prob.pack(A, prob.init_weights(A))

def run(levels_str, nbatch, seed=0, warm_kind="none", B=48, iters=3000, hit=1e-11, S=None):
    levels, extra = parse_levels(levels_str)
    prob = Problem(levels, S, extra)
    tag = levels_str.replace(",", "").replace("/", "_").replace("+", "x").replace(":", "") + (f"_S{S}" if S else "") + f"_{warm_kind}"
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "search")
    os.makedirs(outdir, exist_ok=True)
    fn = os.path.join(outdir, f"{tag}_{seed}.npz")
    rng = np.random.default_rng(seed)
    warm = warm_pool(warm_kind, prob.S) if warm_kind != "none" else []
    print(f"[{tag} seed {seed}] S={prob.S} levels={prob.levels} unknowns={prob.n} conditions={prob.nr} "
          f"warm={len(warm)}", flush=True)
    hits, best, t0 = [], [], time.time()
    for nb in range(nbatch):
        X0 = starts(prob, B, rng, warm)
        X, f, rmax = lm(prob, X0.copy(), iters=iters)
        best += list(rmax)
        for b in np.where(rmax < hit)[0]:
            d, rank, _ = local_dim(prob, X[b])
            A, W = prob.unpack(X[b:b + 1])
            C = prob.error_consts(A, W)[0]
            hits.append(dict(x=X[b], dim=d, rank=rank, C=C, maxcoef=np.abs(np.concatenate(
                [A[0][prob.I, prob.Jc]] + [w[0] for w in W])).max()))
        a = np.sort(best)
        print(f"  [{time.time() - t0:7.0f}s] batch {nb + 1}/{nbatch} starts {len(best)} hits {len(hits)} "
              f"dims {sorted({h['dim'] for h in hits})} best max|r|: " + " ".join(f"{v:.1e}" for v in a[:5]),
              flush=True)
        np.savez(fn, levels=np.array(levels), S=prob.S, best=np.array(best),
                 X=np.array([h["x"] for h in hits]) if hits else np.zeros((0, prob.n)),
                 dims=np.array([h["dim"] for h in hits]), C=np.array([h["C"] for h in hits]) if hits else np.zeros((0, len(levels))),
                 maxcoef=np.array([h["maxcoef"] for h in hits]))
    return hits

if __name__ == "__main__":
    a = sys.argv
    run(a[1], int(a[2]), int(a[3]) if len(a) > 3 else 0, a[4] if len(a) > 4 else "none",
        S=int(a[5]) if len(a) > 5 else None)
