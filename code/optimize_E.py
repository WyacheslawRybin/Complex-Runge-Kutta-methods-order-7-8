"""
Minimize the leading error constant E_{p+1} = ||e||_2 of the propagated level on the
solution set of a family of projected tableaux, under the bound max|coef| <= BOUND
(all a_ij and all weights of all levels), as in the 5-6 paper:
Levenberg-Marquardt on  [ e(x) ; rho * max(0, |z_k| - BOUND) ]  restricted to the
tangent space of the solution set, with a minimum-norm Newton return to the exact
order conditions after every step.  Double precision; results are refined and
re-evaluated in multiple precision by check_tableaux.py.

Families: any LEVELS string of cascade_search.py ("7,7", "9,8", "7,7/6,6", ...).
Starts: refined tableaux (data/tableaux.json), the pools of the search archive, hits
of cascade_search.py, and random tangent displacements of them.

Usage:  python optimize_E.py LEVELS BOUND START[,START...] [NRANDOM] [NPROC]
        START: a tableau name, "pool77", "pool98", or a path to a search .npz
Writes ../data/Emin_<LEVELS>_<BOUND>.json
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
from cascade_search import Problem, TREES, gamma, sigma, parse_levels

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

class Family:
    def __init__(self, levels_str):
        self.levels, self.extra = parse_levels(levels_str)
        self.P = Problem(self.levels, None, self.extra)
        self.m, self.p = self.levels[0]
        self.etrees = TREES[self.p + 1]
        self.sig = np.array([sigma(t) for t in self.etrees], float)
        self.g = np.array([1.0 / gamma(t) for t in self.etrees])

    def residual(self, x):
        return self.P.FJ(x[None], jac=False)[0][0]

    def rJ(self, x):
        r, J = self.P.FJ(x[None])
        return r[0], J[0]

    def newton_project(self, x, iters=10, tol=1e-14):
        for _ in range(iters):
            r, J = self.rJ(x)
            if np.abs(r).max() < tol:
                break
            x = x + np.linalg.lstsq(J, -r, rcond=1e-12)[0]
        return x

    def tangent_basis(self, x):
        _, J = self.rJ(x)
        U, s, Vt = np.linalg.svd(J)
        rank = int(np.sum(s > 1e-9 * s[0]))
        return Vt[rank:].T

    def err_and_jac(self, x):
        P = self.P
        A, W = P.unpack(x[None])
        ph, dph = P.phi(A, self.p + 1)
        m = self.m
        e = np.empty(len(self.etrees))
        Jc = np.zeros((len(self.etrees), P.nz), complex)
        for q, t in enumerate(self.etrees):
            v, dv = ph[t][0, :m], dph[t][0, :m]
            e[q] = ((W[0][0] @ v).real - self.g[q]) / self.sig[q]
            Jc[q, :P.NA] = (W[0][0] @ dv) / self.sig[q]
            Jc[q, P.woff[0]:P.woff[0] + m] = v / self.sig[q]
        return e, np.hstack([Jc.real, -Jc.imag])[:, P.keep]

    def error_constant(self, x):
        return float(np.linalg.norm(self.err_and_jac(x)[0]))

    def moduli(self, x):
        P = self.P
        xx = np.zeros(2 * P.nz); xx[P.keep] = x
        return xx[:P.nz], xx[P.nz:]

    def max_coef(self, x):
        re, im = self.moduli(x)
        return float(np.hypot(re, im).max())

    def pen_and_jac(self, x, bound, rho):
        P = self.P
        re, im = self.moduli(x)
        mod = np.hypot(re, im)
        act = np.where(mod > bound)[0]
        pen = rho * (mod[act] - bound)
        J = np.zeros((len(act), 2 * P.nz))
        for r, k in enumerate(act):
            J[r, k] = rho * re[k] / mod[k]
            J[r, P.nz + k] = rho * im[k] / mod[k]
        return pen, J[:, P.keep]

    def F(self, x, bound, rho):
        e, _ = self.err_and_jac(x)
        pen, _ = self.pen_and_jac(x, bound, rho)
        return float(e @ e + pen @ pen)

    def minimize(self, x, bound, rho=10.0, iters=400, verbose=False):
        x = self.newton_project(x, iters=10)
        f = self.F(x, bound, rho)
        mu, rel = 1e-6, 1.0
        for it in range(iters):
            T = self.tangent_basis(x)
            if T.shape[1] == 0:
                break
            e, Je = self.err_and_jac(x)
            pen, Jp = self.pen_and_jac(x, bound, rho)
            R = np.concatenate([e, pen])
            JT = np.vstack([Je, Jp]) @ T
            H, g = JT.T @ JT, JT.T @ R
            improved = False
            for _ in range(12):
                d = -np.linalg.solve(H + mu * np.eye(len(g)) * max(1e-30, np.trace(H) / len(g)), g)
                xc = self.newton_project(x + T @ d, iters=8)
                if np.abs(self.residual(xc)).max() < 1e-12:
                    fc = self.F(xc, bound, rho)
                    if fc < f:
                        rel = (f - fc) / f
                        x, f, improved = xc, fc, True
                        mu = max(mu / 4, 1e-12)
                        break
                mu *= 8
            if verbose and it % 20 == 0:
                print(f"    it {it:3d}  E = {self.error_constant(x):.6e}  max|coef| = {self.max_coef(x):.3f}", flush=True)
            if not improved or rel < 1e-12:
                break
        return x, self.error_constant(x), it

# ----------------------------------------------------------------- starts
def x_from_tableau(Fm, T):
    """Pack an mpmath Tableau (single level or matching levels) into the real unknowns."""
    P = Fm.P
    A = np.zeros((1, P.S, P.S), complex)
    for i in range(P.S):
        for j in range(i):
            A[0, i, j] = complex(T.A[i][j])
    W = []
    for L, (m, p) in enumerate(P.levels):
        if L < len(T.levels):
            W.append(np.array([[complex(v) for v in T.levels[L][2][:m]]]))
        else:
            W.append(None)
    if any(w is None for w in W):
        W0 = P.init_weights(A)                 # least squares, extra conditions included
        W = [w if w is not None else w0 for w, w0 in zip(W, W0)]
    W = [w.copy() for w in W]
    for w in W:
        w[:, 0] = w[:, 0].real
    return P.pack(A, W)[0]

def collect_starts(Fm, spec):
    import mpmath as mp
    import tableaux as TB
    mp.mp.dps = 30
    out = []
    for item in spec.split(","):
        if item in ("pool77", "pool98"):
            for T in TB.pool(item[4:]):
                out.append((T.name, x_from_tableau(Fm, T)))
        elif item.startswith("casc:"):                 # casc:FILE.bin  (cascsearch.c hits)
            fn = item[5:]
            (S, p), (MH, q) = Fm.levels[0], Fm.levels[1]
            for T in TB.casc_hits(fn, S, MH, p, q, os.path.basename(fn)):
                out.append((T.name, x_from_tableau(Fm, T)))
        elif item.startswith("json:"):                 # json:Emin_X.json[#k,k,...]  (earlier minima)
            fn, _, ks = item[5:].partition("#")
            res = json.load(open(fn))["results"]
            for k in ([int(v) for v in ks.split("+")] if ks else range(len(res))):
                out.append((f"{os.path.basename(fn)}#{k}", np.array(res[k]["x"])))
        elif item.endswith(".npz"):
            d = np.load(item)
            for k, x in enumerate(d["X"]):
                out.append((f"{os.path.basename(item)}#{k}", x))
        else:
            ref = TB.load_refined()
            out.append((item, x_from_tableau(Fm, ref[item])))
    return out

def _job(args):
    levels_str, bound, label, x0, iters = args
    Fm = Family(levels_str)
    x0 = Fm.newton_project(np.array(x0), iters=12)
    if np.abs(Fm.residual(x0)).max() > 1e-11:
        return dict(start=label, failed=True)
    E0 = Fm.error_constant(x0)
    x, E, it = Fm.minimize(x0, bound, iters=iters)
    return dict(start=label, E_start=E0, E=E, maxcoef=Fm.max_coef(x),
                resid=float(np.abs(Fm.residual(x)).max()), iters=it, x=x.tolist())

def main(levels_str, bound, spec, nrandom=8, nproc=8, seed=1, iters=600, out_tag=None):
    Fm = Family(levels_str)
    rng = np.random.default_rng(seed)
    base = collect_starts(Fm, spec)
    jobs = []
    for label, x0 in base:
        x0 = Fm.newton_project(x0, iters=12)
        if np.abs(Fm.residual(x0)).max() > 1e-11:
            print(f"  skip {label}: residual {np.abs(Fm.residual(x0)).max():.1e}")
            continue
        jobs.append((levels_str, bound, label, x0.tolist(), iters))
        T = Fm.tangent_basis(x0)
        for k in range(nrandom):
            sig = [0.05, 0.15, 0.4][k % 3]
            xr = Fm.newton_project(x0 + T @ (sig * rng.standard_normal(T.shape[1])), iters=12)
            if np.abs(Fm.residual(xr)).max() < 1e-11:
                jobs.append((levels_str, bound, f"{label}+rand{k}", xr.tolist(), iters))
    print(f"{levels_str}: {len(jobs)} starts, bound {bound}", flush=True)
    res, t0 = [], time.time()
    with Pool(nproc) as pool:
        for d in pool.imap_unordered(_job, jobs):
            if d.get("failed"):
                continue
            res.append(d)
            print(f"  [{time.time() - t0:6.0f}s] {d['start']:22s} E {d['E_start']:.4e} -> {d['E']:.6e}  "
                  f"max|coef| {d['maxcoef']:8.3f}  resid {d['resid']:.1e}  ({d['iters']} it)", flush=True)
    res.sort(key=lambda d: d["E"])
    tag = out_tag or levels_str.replace(",", "").replace("/", "_").replace("+", "x").replace(":", "") + f"_{bound:g}"
    json.dump(dict(levels=Fm.levels, bound=bound, results=res),
              open(os.path.join(DATA, f"Emin_{tag}.json"), "w"), indent=1)
    if res:
        print("best:", res[0]["start"], f"E = {res[0]['E']:.6e}, max|coef| = {res[0]['maxcoef']:.3f}")

if __name__ == "__main__":
    # python optimize_E.py LEVELS BOUND START[,START] [NRANDOM] [NPROC] [ITERS] [OUT_TAG]
    a = sys.argv
    main(a[1], float(a[2]), a[3], int(a[4]) if len(a) > 4 else 8, int(a[5]) if len(a) > 5 else 8,
         iters=int(a[6]) if len(a) > 6 else 600, out_tag=a[7] if len(a) > 7 else None)
