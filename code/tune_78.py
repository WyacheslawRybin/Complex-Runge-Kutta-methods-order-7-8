"""
tune_78.py -- tune a projected tableau of order 7 or 8 to ONE problem (Section "Tuning a tableau to one
problem"), keeping the order conditions of every level exact.  Port of tune_cck6.py of the 5-6 paper:
the descent moves in the tangent space of the solution set, returns to it by minimum-norm Newton steps
after every move, and minimizes the geometric mean of the global errors at N/2, N and 2N steps on the
problem under the bound max|coef| <= 5.

    python tune_78.py tune CRK7-opt|CRK8-opt|CCK7(6) [N]   -> tune_runs/tuned_<name>.npy, tune_<name>.log
    python tune_78.py eval [NAME ...]                      -> ../data/tuned.json (mp refinement, C on Rossler)
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RUNS = os.path.join(HERE, "tune_runs")
DATA = os.path.join(HERE, "..", "data")
os.makedirs(RUNS, exist_ok=True)

def f(t, y):
    """Rossler system (Appendix B); must accept complex t and y."""
    return np.array([-y[1] - y[2], y[0] + 0.2 * y[1], 0.2 + y[2] * (y[0] - 5.7)])

y0 = np.array([1.0, 1.0, 1.0])
T = 5.0
N = 100
MAX_COEF = 5.0
ROUNDS = 40

# ---------------------------------------------------------------- rooted trees
def order(t):
    return 1 + sum(order(u) for u in t)

def trees_by_order(pmax):
    by = {1: [()]}
    for n in range(2, pmax + 1):
        pool = [t for k in range(1, n) for t in by[k]]
        out, seen = [], set()
        def multisets(total, start):
            if total == 0:
                yield []
                return
            for i in range(start, len(pool)):
                w = order(pool[i])
                if w <= total:
                    for rest in multisets(total - w, i):
                        yield [pool[i]] + rest
        for ms in multisets(n - 1, 0):
            key = tuple(sorted(ms, key=repr))
            if key not in seen:
                seen.add(key)
                out.append(key)
        by[n] = out
    return by

def gamma(t):
    g = order(t)
    for u in t:
        g *= gamma(u)
    return g

def sigma(t):
    out = 1
    for u in set(t):
        k = t.count(u)
        out *= sigma(u) ** k
        for q in range(2, k + 1):
            out *= q
    return out

# ---------------------------------------------------------------- setup from the refined tableau
def setup(name):
    global S, LEV, TREES, ROWS, RHS, AIDX, NA, WOFF, NZ, KEEP, PMAX, A0, W0
    global FAMILY
    FAMILY = name[:4]
    d = json.load(open(os.path.join(DATA, "tableaux.json")))[name]
    S = d["s"]
    LEV = [tuple(l) for l in d["levels"]]
    A0 = np.zeros((S, S), complex)
    for i in range(S):
        for j in range(i):
            A0[i, j] = float(d["A"][i][j][0]) + 1j * float(d["A"][i][j][1])
    W0 = [np.array([float(v[0]) + 1j * float(v[1]) for v in b]) for b in d["b"]]
    PMAX = max(p for _, p in LEV)
    TREES = trees_by_order(PMAX + 1)
    ROWS = [(L, t) for L, (m, p) in enumerate(LEV) for k in range(1, p + 1) for t in TREES[k]]
    RHS = np.array([1.0 / gamma(t) for _, t in ROWS])
    AIDX = [(i, j) for i in range(S) for j in range(i)]
    NA = len(AIDX)
    WOFF, off = [], NA
    for m, _ in LEV:
        WOFF.append(off)
        off += m
    NZ = off
    drop = {NZ + o for o in WOFF}                 # Im w_1 of every level: k_1 is real
    KEEP = np.array([q for q in range(2 * NZ) if q not in drop])

def unpack(x):
    xx = np.zeros(2 * NZ)
    xx[KEEP] = x
    z = xx[:NZ] + 1j * xx[NZ:]
    A = np.zeros((S, S), complex)
    for k, (i, j) in enumerate(AIDX):
        A[i, j] = z[k]
    return A, [z[o:o + m] for o, (m, _) in zip(WOFF, LEV)]

def pack(A, W):
    z = np.zeros(NZ, complex)
    for k, (i, j) in enumerate(AIDX):
        z[k] = A[i, j]
    for o, w in zip(WOFF, W):
        z[o:o + len(w)] = w
    return np.concatenate([z.real, z.imag])[KEEP]

def phi_all(A, pmax, deriv=True):
    cache = {}
    def Phi(t):
        if t in cache:
            return cache[t]
        v = np.ones(S, complex)
        dv = np.zeros((S, NA), complex) if deriv else None
        for u in t:
            pu, dpu = Phi(u)
            w = A @ pu
            if deriv:
                dw = A @ dpu
                for k, (i, j) in enumerate(AIDX):
                    dw[i, k] += pu[j]
                dv = dv * w[:, None] + v[:, None] * dw
            v = v * w
        cache[t] = (v, dv)
        return v, dv
    for k in range(1, pmax + 1):
        for t in TREES[k]:
            Phi(t)
    return cache

def residual_and_jacobian(x):
    A, W = unpack(x)
    ph = phi_all(A, PMAX)
    r = np.empty(len(ROWS))
    Jc = np.zeros((len(ROWS), NZ), complex)
    for q, (L, t) in enumerate(ROWS):
        m = LEV[L][0]
        v, dv = ph[t]
        r[q] = (W[L] @ v[:m]).real - RHS[q]
        Jc[q, :NA] = W[L] @ dv[:m]
        Jc[q, WOFF[L]:WOFF[L] + m] = v[:m]
    return r, np.hstack([Jc.real, -Jc.imag])[:, KEEP]

def residual(x):
    return residual_and_jacobian(x)[0]

def error_constant(x, L=0):
    A, W = unpack(x)
    m, p = LEV[L]
    ph = phi_all(A, p + 1, deriv=False)
    e = [((W[L] @ ph[t][0][:m]).real - 1.0 / gamma(t)) / sigma(t) for t in TREES[p + 1]]
    return float(np.linalg.norm(e))

def max_coef(x):
    A, W = unpack(x)
    return max(np.abs(A).max(), max(np.abs(w).max() for w in W))

def newton_project(x, iters=6):
    for _ in range(iters):
        r, J = residual_and_jacobian(x)
        if np.linalg.norm(r) < 1e-14:
            break
        x = x - np.linalg.lstsq(J, r, rcond=1e-12)[0]
    return x

def active_grads(x, tol=1e-2):
    """Gradients (in x) of |z_q| for the coefficients within tol of the bound MAX_COEF."""
    xx = np.zeros(2 * NZ)
    xx[KEEP] = x
    z = xx[:NZ] + 1j * xx[NZ:]
    out = []
    for q in np.nonzero(np.abs(z) > MAX_COEF - tol)[0]:
        a = np.zeros(2 * NZ)
        a[q], a[NZ + q] = z[q].real / abs(z[q]), z[q].imag / abs(z[q])
        out.append(a[KEEP])
    return out

DIM = {"CRK7": 7, "CRK8": 10, "CCK7": 14}       # local dimensions from the multiple-precision ranks (Table 2)

def tangent_basis(x):
    """Null space of the Jacobian of the known dimension; at CRK7-opt the double-precision spectrum has one more
    singular value (2e-14 s_max) below any fixed threshold that would keep the null spaces of CRK8 and CCK7."""
    _, J = residual_and_jacobian(x)
    _, sv, Vt = np.linalg.svd(J)
    return Vt[len(x) - DIM[FAMILY]:].T

def integrate(x, n_steps, level=0):
    A, W = unpack(x)
    m = LEV[level][0]
    w = W[level]
    c = A.sum(axis=1)
    h = T / n_steps
    y = np.array(y0, float)
    for step in range(n_steps):
        t = step * h
        K = []
        for i in range(m):
            Y = y + h * sum(A[i, j] * K[j] for j in range(i)) if i else y.astype(complex)
            K.append(np.asarray(f(t + c[i] * h, Y), complex))
        y = np.real(y + h * sum(w[i] * K[i] for i in range(m)))
    return y

def errors(x, reference):
    return np.array([np.linalg.norm(integrate(x, n) - reference) for n in N_LIST])

def objective(x, reference, scale):
    return float(np.exp(np.mean(np.log(errors(x, reference) / scale))))

def tune(x, reference, scale, log, rounds=ROUNDS, step0=2e-3):
    best = objective(x, reference, scale)
    step = step0
    for it in range(rounds):
        Tn = tangent_basis(x)
        g = np.zeros(Tn.shape[1])
        eps = 1e-2                      # the round-off noise of the objective is about 1e-4 relative
        for k in range(Tn.shape[1]):
            xp = newton_project(x + eps * Tn[:, k], iters=12)
            xm = newton_project(x - eps * Tn[:, k], iters=12)
            g[k] = (objective(xp, reference, scale) - objective(xm, reference, scale)) / (2 * eps)
        ng = np.linalg.norm(g)
        if ng < 1e-14:
            break
        u = -g / ng
        # active bound: turn the direction slightly inward for every coefficient at |z| = MAX_COEF that it would increase
        C = [c for c in (Tn.T @ a for a in active_grads(x)) if c @ u > -1e-3 * np.linalg.norm(c)]
        if C:
            Cm = np.array(C).T
            tgt = -0.05 * np.linalg.norm(Cm, axis=0)
            u = u - Cm @ np.linalg.lstsq(Cm.T @ Cm, Cm.T @ u - tgt, rcond=None)[0]
            if np.linalg.norm(u) < 1e-12:
                break
            u /= np.linalg.norm(u)
        d = Tn @ u
        improved = False
        for _ in range(14):
            xc = newton_project(x + step * d, iters=12)
            if np.linalg.norm(residual(xc)) < 1e-12 and max_coef(xc) <= MAX_COEF:
                val = objective(xc, reference, scale)
                if val < best:
                    x, best, improved = xc, val, True
                    step *= 1.6
                    break
            step /= 2.0
        log("  round %2d  relative error = %.4f  dim tangent = %d  max|coef| = %.3f  step = %.1e"
            % (it + 1, best, Tn.shape[1], max_coef(x), step))
        if not improved:
            break
    return x, best

def reference_solution():
    import mpmath as mp
    from problems import references
    return np.array([float(v) for v in references()["Rossler"]])

def run_tune(name, n=N):
    global N_LIST
    N_LIST = (n // 2, n, 2 * n)
    setup(name)
    fn_log = os.path.join(RUNS, f"tune_{name}.log")
    out = open(fn_log, "w", encoding="utf-8")
    def log(s):
        print(s, flush=True)
        out.write(s + "\n"); out.flush()
    x0 = pack(A0, W0)
    log(f"{name}: levels {LEV}, real unknowns {len(x0)}, conditions {len(ROWS)}, N_LIST {N_LIST}")
    log("start: residual of the order conditions = %.2e" % np.linalg.norm(residual(x0)))
    x0 = newton_project(x0)
    log("after Newton                            = %.2e" % np.linalg.norm(residual(x0)))
    ref = reference_solution()
    scale = errors(x0, ref)
    log("errors of the starting tableau: " + " ".join("%.3e" % v for v in scale)
        + "  E_{p+1} = %.4e  max|coef| = %.3f" % (error_constant(x0), max_coef(x0)))
    x1, _ = tune(x0, ref, scale, log)
    x1 = newton_project(x1, iters=10)
    e1 = errors(x1, ref)
    log("tuned tableau, errors: " + " ".join("%.3e" % v for v in e1))
    log("improvement factors: " + " ".join("%.2f" % v for v in scale / e1))
    log("E_{p+1} = %.4e (start %.4e),  max|coef| = %.3f" % (error_constant(x1), error_constant(x0), max_coef(x1)))
    np.save(os.path.join(RUNS, f"tuned_{name}.npy"), x1)

def run_eval(names=("CRK7-opt", "CRK7-s", "CRK8-opt", "CRK8-s", "CCK7(6)")):
    """Refine the tuned tableaux in mp and measure the asymptotic constant C on the Rossler system."""
    import mpmath as mp
    import crk
    from problems import P, references
    mp.mp.dps = 34
    fo = os.path.join(DATA, "tuned.json")
    res = json.load(open(fo)) if os.path.exists(fo) else {}
    for name in names:
        fn = os.path.join(RUNS, f"tuned_{name}.npy")
        if not os.path.exists(fn):
            continue
        setup(name)
        A, W = unpack(np.load(fn))
        lev = [(m, p, [mp.mpc(complex(v)) for v in W[L]]) for L, (m, p) in enumerate(LEV)]
        Tt = crk.Tableau(f"tuned-{name}", [[mp.mpc(complex(A[i, j])) for j in range(S)] for i in range(S)], lev)
        with mp.workdps(60):
            Tt, shift, r = Tt.refine()
            if r > mp.mpf(10) ** -40:
                # the default Newton refinement stalls near a rank drop: as in add_tableau.py, remove the inert
                # directions, normalize and refine with a smaller pseudo-inverse cutoff
                import tableaux as TB
                T2, _ = TB.clean_inert(Tt, tol=1e-10)
                Tt, shift2, r = T2.normalized().refine(iters=40, rcond=1e-14)
                shift = max(shift, shift2)
        p = Tt.levels[0][1]
        pr, ref = P["Rossler"], references()["Rossler"]
        Ns = [50, 100, 200, 400, 800, 1600, 3200]
        e = []
        for n in Ns:
            y = crk.integrate_mp(Tt, pr["f_mp"], pr["y0"], 0, mp.mpf(pr["T"]), n, 0)
            e.append(float(mp.sqrt(mp.fsum((a - b) ** 2 for a, b in zip(y, ref)))))
        Ch = [v / (5.0 / n) ** p for v, n in zip(e, Ns)]
        R = [2 * Ch[k + 1] - Ch[k] for k in range(len(Ch) - 1)]
        res[name] = dict(Ns=Ns, err=e, Ch=Ch, C=R[-1], conv=abs(R[-1] - R[-2]) / abs(R[-1]),
                         E=float(Tt.E()), maxcoef=float(Tt.maxcoef()), shift=float(shift), resid=float(r))
        print(name, "shift %.1e" % float(shift), "errors", " ".join(f"{v:.3e}" for v in e),
              f"C = {R[-1]:.4e} ({res[name]['conv']:.1%})  E = {res[name]['E']:.4e}", flush=True)
    json.dump(res, open(fo, "w"), indent=1)

if __name__ == "__main__":
    if sys.argv[1] == "tune":
        run_tune(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else N)
    else:
        run_eval(*(tuple(sys.argv[2:]),) if len(sys.argv) > 2 else ())
