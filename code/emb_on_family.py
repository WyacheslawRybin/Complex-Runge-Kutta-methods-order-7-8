"""
Distance from a family of projected tableaux to the existence of an embedded level.

The propagated level (s, p) defines a solution set F in the real unknowns x.  For an
embedded level (m, q) on the leading m stages the conditions are linear in its weights,
so the level exists at x iff the variable-projection residual
    rho(x) = (I - M M^+) r,      M = M_q(A[:m,:m]) = [P | -Q],  r_t = 1/gamma(t),
vanishes.  For m = s the propagated weights already solve it, and a DIFFERENT embedded
formula exists iff M has an essential kernel; then the objective is the smallest
essential singular value of M (rank-drop test).

We minimize ||rho|| (or sigma_min) over F by Levenberg-Marquardt in the tangent space
of F (finite differences along an orthonormal tangent basis) with a minimum-norm Newton
return to F after every step, from every known root of the family.  A zero minimum
gives a Cash-Karp tableau (refined afterwards on the joint system); a positive minimum
over many components is the quantitative form of a null result.

Usage:  python emb_on_family.py MAIN EMB STARTS [NPROC] [NRANDOM]
   MAIN  "7,7" | "8,7" | "9,8"     EMB "6,6" | "7,6" | "7,7" (m,q)
   STARTS pool77 | pool98 | seeds87[:N] | named
Writes ../data/embfam_<MAIN>_<EMB>.json
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
from optimize_E import Family, x_from_tableau
from cascade_search import TREES, gamma

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

class EmbObjective:
    def __init__(self, Fm, m, q):
        self.Fm, self.m, self.q = Fm, m, q
        self.ts = [t for k in range(1, q + 1) for t in TREES[k]]
        self.rhs = np.array([1.0 / gamma(t) for t in self.ts])
        self.full = (m == Fm.P.S)

    def M(self, x):
        P = self.Fm.P
        A, W = P.unpack(x[None])
        ph, _ = P.phi(A, self.q, deriv=False)
        Mc = np.stack([ph[t][0, :self.m] for t in self.ts])
        M = np.hstack([Mc.real, -Mc.imag[:, 1:]])
        keep = np.abs(M).max(0) > 1e-12            # drop inert columns (Im b on real stages)
        return M[:, keep]

    def vec(self, x):
        M = self.M(x)
        if self.full:
            sv = np.linalg.svd(M, compute_uv=False)
            return np.array([sv[-1] / sv[0]])
        z, *_ = np.linalg.lstsq(M, self.rhs, rcond=1e-13)
        return M @ z - self.rhs

def minimize(Fm, obj, x, iters=300, fd=1e-7):
    x = Fm.newton_project(x, iters=10)
    g = obj.vec(x); f = float(g @ g)
    mu = 1e-4
    for it in range(iters):
        T = Fm.tangent_basis(x)
        if T.shape[1] == 0:
            break
        G = np.stack([(obj.vec(Fm.newton_project(x + fd * T[:, k], iters=3)) - g) / fd for k in range(T.shape[1])], 1)
        H, gr = G.T @ G, G.T @ g
        improved = False
        for _ in range(10):
            d = -np.linalg.solve(H + mu * np.eye(len(gr)) * max(1e-30, np.trace(H) / len(gr)), gr)
            st = np.linalg.norm(d)
            if st > 0.2:
                d *= 0.2 / st
            xc = Fm.newton_project(x + T @ d, iters=8)
            if np.abs(Fm.residual(xc)).max() < 1e-11:
                gc = obj.vec(xc); fc = float(gc @ gc)
                if fc < f:
                    rel = (f - fc) / f
                    x, g, f, improved = xc, gc, fc, True
                    mu = max(mu / 4, 1e-12)
                    break
            mu *= 8
        if not improved or rel < 1e-10 or f < 1e-26:
            break
    return x, np.sqrt(f), it

def _job(args):
    main, emb, label, x0, iters = args
    Fm = Family(main)
    m, q = (int(v) for v in emb.split(","))
    obj = EmbObjective(Fm, m, q)
    x0 = Fm.newton_project(np.array(x0), iters=12)
    if np.abs(Fm.residual(x0)).max() > 1e-11:
        return dict(start=label, failed=True)
    f0 = float(np.linalg.norm(obj.vec(x0)))
    x, f, it = minimize(Fm, obj, x0, iters=iters)
    return dict(start=label, f_start=f0, f=float(f), dim=int(Fm.tangent_basis(x).shape[1]),
                maxcoef=Fm.max_coef(x), E=Fm.error_constant(x), iters=it, x=x.tolist())

def collect(Fm, spec):
    import mpmath as mp
    import tableaux as TB
    out = []
    if spec in ("pool77", "pool98"):
        for T in TB.pool(spec[4:]):
            out.append((T.name, x_from_tableau(Fm, T)))
    elif spec.startswith("seeds87"):
        X = np.fromfile(os.path.join(TB.ARCH, "seeds87.bin")).reshape(-1, 72)
        n = int(spec.split(":")[1].rstrip("]")) if ":" in spec else len(X)
        for k, xv in enumerate(X[:n]):
            T = TB._from_flat([float(v) for v in xv], 8, f"s87[{k}]", 7)
            if T.maxcoef() < 20:
                out.append((T.name, x_from_tableau(Fm, T)))
    elif spec == "named":
        for nm, T in TB.complex_named().items():
            if T.s == Fm.P.S and T.p == Fm.p:
                out.append((nm, x_from_tableau(Fm, T)))
    return out

def main(mainlev, emb, spec, nproc=10, nrandom=2, iters=300, seed=0):
    Fm = Family(mainlev)
    rng = np.random.default_rng(seed)
    jobs = []
    for label, x0 in collect(Fm, spec):
        x0 = Fm.newton_project(x0, iters=12)
        if np.abs(Fm.residual(x0)).max() > 1e-11:
            continue
        jobs.append((mainlev, emb, label, x0.tolist(), iters))
        T = Fm.tangent_basis(x0)
        for k in range(nrandom):
            xr = Fm.newton_project(x0 + T @ (0.2 * rng.standard_normal(T.shape[1])), iters=12)
            if np.abs(Fm.residual(xr)).max() < 1e-11:
                jobs.append((mainlev, emb, f"{label}+r{k}", xr.tolist(), iters))
    print(f"main {mainlev}, embedded {emb}: {len(jobs)} starts", flush=True)
    res, t0 = [], time.time()
    with Pool(nproc) as pool:
        for d in pool.imap_unordered(_job, jobs):
            if d.get("failed"):
                continue
            res.append(d)
            print(f"  [{time.time() - t0:6.0f}s] {d['start']:16s} {d['f_start']:.3e} -> {d['f']:.3e}  dim {d['dim']:2d} "
                  f"max|coef| {d['maxcoef']:8.3f}  E {d['E']:.3e} ({d['iters']} it)", flush=True)
    res.sort(key=lambda d: d["f"])
    tag = f"{mainlev.replace(',', '')}_{emb.replace(',', '')}_{spec.replace(':', '').replace('[', '').replace(']', '')}"
    json.dump(dict(main=mainlev, emb=emb, results=res), open(os.path.join(DATA, f"embfam_{tag}.json"), "w"), indent=1)
    if res:
        print("best:", res[0]["start"], f"{res[0]['f']:.3e}", flush=True)

if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], a[3], int(a[4]) if len(a) > 4 else 10, int(a[5]) if len(a) > 5 else 2)
