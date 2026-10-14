"""
Variable step size experiments with embedded pairs (double precision, numpy).

Every pair is run with the same standard controller (HNW I, Sect. II.4):
  err = || (y - yhat) / (Atol + Rtol max(|y_n|, |y_{n+1}|)) ||_rms,  Atol = Rtol = tol,
  h_new = h * min(facmax, max(0.2, 0.9 err^(-1/(q+1)))),  facmax = 5 (1 after a rejection),
with local extrapolation (the higher-order solution is propagated) and the
starting step of HNW I, Sect. II.4.  k_1 = f(t_n, y_n) is real and is reused
after a rejection; for FSAL pairs the last stage of an accepted step is reused.
The projected methods take the real part of both solutions after every step.

Writes ../data/adaptive.json: for every pair, problem and tolerance the global
error at T, the number of f evaluations, accepted and rejected steps.
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
TOLS = [10.0 ** (-k / 2) for k in range(6, 27)]          # 1e-3 ... 1e-13

# pairs of the paper: (name, kind, source)
#   dp:   embedded weights for a fixed matrix (embedded.json, built by embedded.py)
#   casc: complex Cash-Karp tableau with two levels (tableaux.json)
#   real: real pair (rk_tableaux.json via tableaux.real_methods)
PAIRS = [("CRK7(5)", "dp", "CRK7-opt"), ("CRK7-s(5)", "dp", "CRK7-s"), ("CCK7(6)", "casc", "CCK7(6)"),
         ("Verner7(6)", "real", "Verner7(6)"), ("Vern7", "real", "Vern7"),
         ("CRK8(6)", "dp", "CRK8-opt"), ("CRK8-s(6)", "dp", "CRK8-s"),
         ("DP8(7)", "real", "DP8(7)"), ("Verner8(7)", "real", "Verner8(7)"), ("DOP853", "dop853", "DOP853")]

def load_pairs(names=None):
    sys.path.insert(0, HERE)
    import mpmath as mp
    from tableaux import load_refined, real_methods, RK_JSON, _exact
    mp.mp.dps = 40
    TT = load_refined()
    TT.update(real_methods())
    fe = os.path.join(DATA, "embedded.json")
    emb = json.load(open(fe)) if os.path.exists(fe) else {}
    pairs = {}
    def arr(T):
        s = T.s
        return np.array([[complex(T.A[i][j]) for j in range(s)] for i in range(s)])
    def vec(v, s):
        out = np.zeros(s, complex); out[:len(v)] = [complex(x) for x in v]; return out
    for name, kind, base in PAIRS:
        if names and name not in names:
            continue
        if kind == "dp":
            if name not in emb or base not in TT:
                continue
            T = TT[base]
            bh = [complex(float(a), float(b)) for a, b in emb[name]["bh"]]
            pairs[name] = dict(A=arr(T), b=vec(T.levels[0][2], T.s), bh=vec(bh, T.s), p=T.levels[0][1],
                               q=emb[name]["q"], projected=True, fsal=False)
        elif kind in ("casc", "real"):
            if base not in TT:
                continue
            T = TT[base]
            A = arr(T); b = vec(T.levels[0][2], T.s); bh = vec(T.levels[1][2], T.s)
            fsal = (not T.projected) and np.allclose(A[-1, :-1], b[:-1], atol=1e-15) and abs(b[-1]) < 1e-15
            pairs[name] = dict(A=A if T.projected else A.real, b=b if T.projected else b.real,
                               bh=bh if T.projected else bh.real, p=T.levels[0][1], q=T.levels[1][1],
                               projected=T.projected, fsal=bool(fsal))
        elif kind == "dop853":
            d = json.load(open(RK_JSON, encoding="utf-8"))["DOP853"]
            s = d["stages"]
            A = np.array([[float(_exact(d["A_exact"][i][j])) if j < len(d["A_exact"][i]) else 0.0
                           for j in range(s)] for i in range(s)])
            pairs[name] = dict(A=A, b=np.array([float(_exact(v)) for v in d["b_exact"]]),
                               bh=np.array([float(_exact(v)) for v in d["bhat_exact"]]),
                               bh3=np.array([float(_exact(v)) for v in d["bhat3_exact"]]),
                               p=8, q=7, projected=False, fsal=False, dop853=True)
    return pairs

def solve(pair, f, y0, T, tol, max_steps=2_000_000):
    A, b, bh, q, proj, fsal = pair["A"], pair["b"], pair["bh"], pair["q"], pair["projected"], pair["fsal"]
    s = len(b); c = A.sum(1); p = pair["p"]
    y = np.array(y0, float); t = 0.0
    sc = lambda u, v: tol + tol * np.maximum(np.abs(u), np.abs(v))
    nf = 0
    f0 = np.real(f(t, y)) * 1.0; nf += 1
    # starting step (HNW I, II.4)
    d0 = np.sqrt(np.mean((y / sc(y, y)) ** 2)); d1 = np.sqrt(np.mean((f0 / sc(y, y)) ** 2))
    h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
    f1 = np.real(f(t + h0, y + h0 * f0)); nf += 1
    d2 = np.sqrt(np.mean(((f1 - f0) / sc(y, y)) ** 2)) / h0
    h1 = max(1e-6, h0 * 1e-3) if max(d1, d2) <= 1e-15 else (0.01 / max(d1, d2)) ** (1.0 / (p + 1))
    h = min(100 * h0, h1, T)
    k1 = f0; nacc = nrej = 0; facmax = 5.0
    dtype = complex if proj else float
    while t < T * (1 - 1e-14):
        if t + h > T:
            h = T - t
        K = np.zeros((s, len(y)), dtype)
        K[0] = k1
        for i in range(1, s):
            Yi = y + h * (A[i, :i] @ K[:i])
            K[i] = f(t + c[i] * h, Yi); nf += 1
        ynew = y + h * (b @ K); yhat = y + h * (bh @ K)
        if proj:
            ynew = ynew.real; yhat = yhat.real
        if pair.get("dop853"):
            # Hairer's DOP853 estimator: 5th- and 3rd-order differences combined to O(h^8)
            sk = sc(y, ynew)
            e5 = np.mean(((ynew - yhat) / sk) ** 2)
            e3 = np.mean((h * ((b - pair["bh3"]) @ K) / sk) ** 2)
            err = e5 / np.sqrt(e5 + 0.01 * e3) if e5 > 0 else 0.0
        else:
            err = np.sqrt(np.mean(((ynew - yhat) / sc(y, ynew)) ** 2))
        if not np.isfinite(err):
            return None
        if err <= 1.0:
            t += h; y = ynew; nacc += 1
            if fsal:
                k1 = K[-1].real
            else:
                k1 = np.real(f(t, y)) * 1.0; nf += 1
            fac = min(facmax, max(0.2, 0.9 * err ** (-1.0 / (q + 1)) if err > 0 else facmax))
            h *= fac; facmax = 5.0
        else:
            nrej += 1
            h *= max(0.2, 0.9 * err ** (-1.0 / (q + 1))); facmax = 1.0
        if nacc + nrej > max_steps or h < 1e-14 * T:
            return None
    if not fsal:
        nf -= 1          # the evaluation at the final point is not needed
    return y, nf, nacc, nrej

_P = None
_NAMES = None
def _init():
    global _P, _REF, _PROB
    sys.path.insert(0, HERE)
    import mpmath as mp
    from problems import P, references
    _PROB = P
    _P = load_pairs(_NAMES)
    _REF = {k: np.array([float(v) for v in vals]) for k, vals in references().items()}

def task(args):
    name, prob = args
    pr = _PROB[prob]
    T = float(pr["T"]); y0 = [float(v) for v in pr["y0"]]
    rows = []
    for tol in TOLS:
        out = solve(_P[name], pr["f_np"], y0, T, tol)
        if out is None:
            rows.append(dict(tol=tol, err=None)); continue
        y, nf, na, nr = out
        rows.append(dict(tol=tol, err=float(np.linalg.norm(y - _REF[prob])), nfev=nf, nacc=na, nrej=nr))
    return name, prob, rows

if __name__ == "__main__":
    sys.path.insert(0, HERE)
    from problems import ADAPTIVE
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    names = sys.argv[2].split(",") if len(sys.argv) > 2 else [n for n, _, _ in PAIRS]
    names = [n for n in names if n in load_pairs(names)]
    tasks = [(n, p) for n in names for p in ADAPTIVE]
    fn = os.path.join(DATA, "adaptive.json")
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    t0 = time.time()
    with Pool(nproc, initializer=_init) as pool:
        for name, prob, rows in pool.imap_unordered(task, tasks):
            res.setdefault(name, {})[prob] = rows
            ok = [r for r in rows if r["err"] is not None]
            print(f"[{time.time() - t0:5.0f}s] {name:11s} {prob:13s} "
                  f"tol 1e-13: err={ok[-1]['err']:.2e} nfev={ok[-1]['nfev']}" if ok else f"{name} {prob} FAILED", flush=True)
    json.dump(res, open(fn, "w"), indent=1)
