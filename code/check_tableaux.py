"""
Verification of every complex tableau of orders 7 and 8 in multiple precision
(and of the real reference methods where it makes sense):
  * Newton refinement of the printed coefficients (minimum-norm correction),
  * residual of the projected order conditions and of the full complex ones,
  * leading error constant E_{p+1} (sigma-weighted, as in the 5-6 paper) and the
    unweighted norm of the order p+1 residuals, max |coefficient|, min/max |b_i|,
  * structure: number of real leading stages, C(2) and pure C(3) per stage,
    D(1) per stage (real part and full), c_3 against 2/p, c_s, Re b_2,
  * stability polynomial against the truncated exponential (T_s when s = p),
  * rank of the Jacobian of the order conditions at two difference steps
    (true zeros scale with h) and the local dimension of the solution set,
  * rank of the weight matrix M_p(A) = [P | -Q].
Writes ../data/tableaux.json (refined coefficients, 45 digits) and
../data/check_tableaux.json (all diagnostics).

Usage:  python check_tableaux.py [named|pool77|pool98|all]
"""
import json, os, sys, time
from math import factorial
from multiprocessing import Pool
import numpy as np
import mpmath as mp
import crk
import tableaux as TB

DPS = 60
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(OUT, exist_ok=True)

def jac(T, h):
    x = T.get_x()
    F = lambda xx: [v for lev in T.set_x(xx).residuals() for v in lev]
    r0 = F(x)
    J = np.zeros((len(r0), len(x)))
    for k in range(len(x)):
        xp = x[:]; xp[k] += h
        J[:, k] = [float((a - b) / h) for a, b in zip(F(xp), r0)]
    return J

def jac_rank(T):
    """Rank by SVD at h = 1e-25 and 1e-35: an SV that drops with h is a zero."""
    sv1 = np.linalg.svd(jac(T, mp.mpf(10) ** -25), compute_uv=False)
    sv2 = np.linalg.svd(jac(T, mp.mpf(10) ** -35), compute_uv=False)
    rank = int(np.sum(sv1 > 1e-12 * sv1[0]))
    gap = sv1[rank - 1] / sv1[rank] if rank < len(sv1) else float("inf")
    stable = bool(np.all(np.abs(sv1[:rank] / sv2[:rank] - 1) < 1e-6))
    return rank, len(sv1), gap, stable, sv1

def weight_matrix(T, q, m=None):
    """M_q(A) restricted to the first m stages; columns Re b_i, -Im b_i."""
    m = m or T.levels[0][0]
    ph = T.phi(q)
    ts = crk.trees_upto(q)
    P = np.array([[float(mp.re(ph[t][i])) for i in range(m)] for t in ts])
    Q = np.array([[float(mp.im(ph[t][i])) for i in range(m)] for t in ts])
    rhs = np.array([1.0 / crk.gamma(t) for t in ts])
    return (np.hstack([P, -Q]) if T.projected else P), rhs

def rank(M, tol=1e-10):
    sv = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(sv > tol * sv[0])), sv

def structure(T):
    s, A, c = T.s, T.A, T.c
    m, p, b = T.levels[0]
    tiny = mp.mpf(10) ** (-(mp.mp.dps // 2))
    r = 0
    while r < s and all(abs(mp.im(A[r][j])) < tiny for j in range(r)):
        r += 1
    Ac = [mp.fsum(A[i][j] * c[j] for j in range(i)) for i in range(s)]
    Ac2 = [mp.fsum(A[i][j] * c[j] ** 2 for j in range(i)) for i in range(s)]
    C2 = [Ac[i] - c[i] ** 2 / 2 for i in range(s)]
    C3 = [Ac2[i] - c[i] ** 3 / 3 for i in range(s)]
    D1 = [mp.fsum(b[i] * A[i][j] for i in range(j + 1, m)) - b[j] * (1 - c[j]) for j in range(m)]
    mx = lambda v: float(max(abs(u) for u in v)) if v else 0.0
    return dict(
        real_stages=r,
        c=[[float(mp.re(v)), float(mp.im(v))] for v in c],
        c3_minus_2_over_p=float(abs(c[2] - mp.mpf(2) / p)) if s > 2 else None,
        cs_minus_1=float(abs(c[s - 1] - 1)),
        Re_b2=float(mp.re(b[1])),
        C2_per_stage=[float(abs(v)) for v in C2],
        C2_max_from3=mx(C2[2:]),
        C3_per_stage=[float(abs(v)) for v in C3],
        D1_re_max=mx([mp.re(v) for v in D1]),
        D1_full_per_stage=[float(abs(v)) for v in D1],
        D1_full_max_after_r=mx(D1[r:]),
        min_node_gap=float(min(abs(c[i] - c[j]) for i in range(s) for j in range(i))),
    )

def stability(T, L=0):
    co = T.stab_coeffs(L)
    p = T.levels[L][1]
    dev = [float(co[k] - mp.mpf(1) / factorial(k)) for k in range(len(co))]
    return dict(coeffs=[float(v) for v in co], max_dev_upto_p=max(abs(v) for v in dev[:p + 1]),
                tail=[float(v) for v in co[p + 1:]])

def cstr(z, d=45):
    return [mp.nstr(mp.re(z), d), mp.nstr(mp.im(z), d)]

def inert_normalized(T):
    """Im b_j = 0 on the leading real stages on every level (Proposition 2); changes neither order nor E."""
    m = 1
    while m < T.s and all(abs(mp.im(T.A[m][j])) < mp.mpf(10) ** -40 for j in range(m)):
        m += 1
    T.levels = [(mm, p, [mp.mpc(mp.re(v), 0) if i < m else v for i, v in enumerate(b)]) for mm, p, b in T.levels]
    return T

def check(item):
    name, T = item
    mp.mp.dps = DPS
    t0 = time.time()
    r_printed = max(abs(v) for lev in T.residuals() for v in lev)
    if T.projected:
        R, shift, r_ref = T.refine(iters=14)
        R = inert_normalized(R)
    else:
        R, shift, r_ref = T, mp.mpf(0), r_printed
    m, p, b = R.levels[0]
    eu = R.error_coefficients(0)
    unweighted = mp.sqrt(mp.fsum(((mp.re(mp.fsum(b[i] * R.phi(p + 1)[t][i] for i in range(m)))
                                   - mp.mpf(1) / crk.gamma(t))) ** 2 for t in crk.TREES[p + 1]))
    info = dict(s=R.s, cost=R.cost, projected=R.projected, levels=[(mm, pp) for mm, pp, _ in R.levels],
                resid_printed=float(r_printed), resid_refined=float(r_ref), refine_shift=float(shift),
                E=[float(R.E(L)) for L in range(len(R.levels))],
                E_unweighted=float(unweighted), E_inf=float(max(abs(v) for v in eu)),
                maxcoef=float(R.maxcoef()),
                maxA=float(max(abs(R.A[i][j]) for i in range(R.s) for j in range(i))),
                max_b=float(max(abs(v) for v in b)), min_b=float(min(abs(v) for v in b)),
                full_complex_resid=float(R.full_complex_residual(0)) if R.projected else 0.0,
                stability=[stability(R, L) for L in range(len(R.levels))])
    if R.projected:
        info["structure"] = structure(R)
        rk, n, gap, stable, sv = jac_rank(R)
        info.update(jac_rank=rk, n_free=n, local_dim=n - rk, jac_gap=float(gap), jac_stable=stable,
                    jac_sv=[float(v) for v in sv])
        M, _ = weight_matrix(R, p)
        mr, msv = rank(M, 1e-12)
        info.update(M_rank=mr, M_shape=list(M.shape), M_sv=[float(v) for v in msv])
        info["A"] = [[cstr(R.A[i][j]) for j in range(i)] for i in range(R.s)]
        info["b"] = [[cstr(v) for v in bb] for _, _, bb in R.levels]
    info["time"] = time.time() - t0
    return name, info

def line(name, info):
    s = f"{name:13s} s={info['s']:2d} p={info['levels'][0][1]} resid {info['resid_printed']:8.1e}"
    if info["projected"]:
        st = info["structure"]
        s += (f"->{info['resid_refined']:8.1e} (shift {info['refine_shift']:7.1e})  "
              f"E={info['E'][0]:.6e} (unw {info['E_unweighted']:.3e}) max|c|={info['maxcoef']:.5f} "
              f"|A|<={info['maxA']:.3f} |b| in [{info['min_b']:.3e},{info['max_b']:.3f}]  fullC {info['full_complex_resid']:.2g}  "
              f"J {info['jac_rank']}/{info['n_free']} dim {info['local_dim']} gap {info['jac_gap']:.1e}"
              f"{'' if info['jac_stable'] else ' UNSTABLE'}  M {info['M_rank']}{tuple(info['M_shape'])}  "
              f"real={st['real_stages']} C2(i>=3) {st['C2_max_from3']:.1e} D1re {st['D1_re_max']:.1e} "
              f"c3-2/p {st['c3_minus_2_over_p']:.1e} cs-1 {st['cs_minus_1']:.1e} Re b2 {st['Re_b2']:.1e} "
              f"gap(c) {st['min_node_gap']:.1e} stab-dev {info['stability'][0]['max_dev_upto_p']:.1e}")
    else:
        s += f"  E={info['E'][0]:.6e} max|c|={info['maxcoef']:.4f}"
    return s + f"  [{info['time']:.0f}s]"

def main(which):
    mp.mp.dps = DPS
    items = []
    if which in ("named", "all"):
        items += list(TB.complex_named().items())
    if which in ("pool77", "all"):
        items += [(T.name, T) for T in TB.pool("77")]
    if which in ("pool98", "all"):
        items += [(T.name, T) for T in TB.pool("98")]
    res = {}
    with Pool(min(10, len(items))) as P:
        for name, info in P.imap(check, items):
            res[name] = info
            print(line(name, info), flush=True)
    if which in ("named", "all"):
        for name, T in TB.real_methods().items():
            name, info = check((name, T))
            res[name] = info
            print(line(name, info), flush=True)
    fn = os.path.join(OUT, f"check_{which}.json")
    json.dump(res, open(fn, "w"), indent=1)
    if which in ("named", "all"):
        keep = {k: dict(s=v["s"], cost=v["cost"], projected=True, levels=v["levels"], A=v["A"], b=v["b"])
                for k, v in res.items() if v["projected"]}
        json.dump(keep, open(os.path.join(OUT, "tableaux.json"), "w"), indent=1)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "named")
