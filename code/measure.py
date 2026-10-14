"""
Measured error constants C = lim ||y_{T/h} - y(T)||_2 / h^p for every method and
every test problem, in multiple precision (45 digits), with refined coefficients
(complex tableaux: data/tableaux.json; real methods: exact, rk_tableaux.json).

For each (method, problem) the global error is computed on N = N0 * 2^k,
k = -2..6; C(h) = err/h^p is extrapolated by one Richardson step,
R_k = 2 C(h_{k+1}) - C(h_k), and the relative change of the last two R_k is the
convergence indicator.  The same runs give the empirical orders and the
fixed-step work-precision data (no round-off floor).
Writes ../data/measured.json.
"""
import json, os, sys, time
from multiprocessing import Pool
import mpmath as mp

DPS = 45
N0 = {"Dahlquist": 20, "pendulum": 20, "quadratic": 20, "exponential": 40, "nonautonomous": 40,
      "VanderPol": 40, "Rossler": 40, "Bouali": 100, "Lorenz05": 20, "Lorenz5": 200,
      "Kepler": 200, "RigidBody": 40, "Brusselator": 100}
KS = list(range(-2, 7))

# (label, tableau, level); the label of a level other than 0 ends with "/q"
METHODS = [
    # order 7
    ("CRK7-opt", "CRK7-opt", 0), ("CRK7-s", "CRK7-s", 0),
    ("CCK7(6)", "CCK7(6)", 0),
    ("Butcher7", "Butcher7", 0), ("Verner7(6)", "Verner7(6)", 0), ("Vern7", "Vern7", 0),
    ("Fehlberg7", "Fehlberg7", 0),
    # order 8
    ("CRK8-opt", "CRK8-opt", 0), ("CRK8-s", "CRK8-s", 0),
    ("CooperVerner8", "CooperVerner8", 0), ("DP8(7)", "DP8(7)", 0), ("DOP853", "DOP853", 0),
    ("Fehlberg8", "Fehlberg8", 0), ("Verner8(7)", "Verner8(7)", 0),
]
REAL = {"Butcher7", "Verner7(6)", "Vern7", "Fehlberg7", "CooperVerner8", "DP8(7)", "DOP853",
        "Fehlberg8", "Verner8(7)"}

_TAB = None
_REF = None

def _init():
    global _TAB, _REF
    mp.mp.dps = DPS
    sys.path.insert(0, os.path.dirname(__file__))
    from tableaux import load_refined
    from problems import references
    from tableaux import real_methods
    with mp.workdps(60):
        _TAB = load_refined()
        _TAB.update(real_methods())
        _REF = references()

def _run(T, L, prob, N):
    from problems import P
    pr = P[prob]
    f = pr["f_mp"]
    m, p, b = T.levels[L]
    proj = T.projected
    if proj:
        A = [[T.A[i][j] for j in range(i)] for i in range(m)]
        bb = b
        c = [mp.re(v) if mp.im(v) == 0 else v for v in T.c[:m]]
    else:
        A = [[mp.re(T.A[i][j]) for j in range(i)] for i in range(m)]
        bb = [mp.re(v) for v in b]
        c = [mp.re(v) for v in T.c[:m]]
    Tend = mp.mpf(pr["T"])
    h = Tend / N
    y = [mp.mpf(v) for v in pr["y0"]]
    d = len(y)
    t = mp.mpf(0)
    for _ in range(N):
        K = []
        for i in range(m):
            if i:
                Ai = A[i]
                yi = [y[q] + h * mp.fsum(Ai[j] * K[j][q] for j in range(i) if Ai[j] != 0) for q in range(d)]
            else:
                yi = y
            K.append(f(t + c[i] * h, yi))
        y = [y[q] + h * mp.fsum(bb[i] * K[i][q] for i in range(m) if bb[i] != 0) for q in range(d)]
        if proj:
            y = [mp.re(v) for v in y]
        t += h
        if any(not mp.isfinite(v) or abs(v) > 1e30 for v in y):
            return None
    ref = _REF[prob]
    return mp.sqrt(mp.fsum((y[q] - ref[q]) ** 2 for q in range(d)))

def task(args):
    label, key, L, prob = args
    mp.mp.dps = DPS
    T = _TAB[key]
    p = T.levels[L][1]
    Tend = mp.mpf(__import__("problems").P[prob]["T"])
    rows = []
    t0 = time.time()
    for k in KS:
        N = int(N0[prob] * 2 ** k) if k >= 0 else max(2, N0[prob] // 2 ** (-k))
        try:
            e = _run(T, L, prob, N)
        except (OverflowError, ZeroDivisionError, ValueError):
            e = None
        rows.append(dict(N=N, err=None if e is None else float(e),
                         Ch=None if e is None else float(e / (Tend / N) ** p)))
    return label, prob, rows, time.time() - t0

def summarize(rows, p):
    ok = [r for r in rows if r["err"] is not None and r["err"] > 0]
    Ch = [r["Ch"] for r in ok]
    R = [2 * Ch[k + 1] - Ch[k] for k in range(len(Ch) - 1)]
    out = dict(C=R[-1] if R else None, conv=abs(R[-1] - R[-2]) / abs(R[-1]) if len(R) > 1 and R[-1] != 0 else None,
               C_last=Ch[-1] if Ch else None)
    if len(ok) >= 2:
        import math
        out["order_last"] = math.log2(ok[-2]["err"] / ok[-1]["err"])
        out["orders"] = [math.log2(ok[k]["err"] / ok[k + 1]["err"]) for k in range(len(ok) - 1)]
    return out

def main(only=None):
    from problems import ORDER
    tasks = [(lab, key, L, prob) for (lab, key, L) in METHODS for prob in ORDER
             if only is None or lab in only]
    # longest first
    tasks.sort(key=lambda a: -N0[a[3]] * (3 if a[0] in REAL else 2))
    fn = os.path.join(os.path.dirname(__file__), "..", "data", "measured.json")
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    # cells already in measured.json are kept unless REDO is set (restart after an interruption)
    if not os.environ.get("REDO"):
        tasks = [t for t in tasks if t[3] not in res.get(t[0], {})]
    print(len(tasks), "cells", flush=True)
    t0 = time.time()
    with Pool(int(os.environ.get("NPROC", 11)), initializer=_init) as pool:
        for label, prob, rows, dt in pool.imap_unordered(task, tasks):
            res.setdefault(label, {})[prob] = dict(rows=rows)
            print(f"[{time.time() - t0:6.0f}s] {label:12s} {prob:14s} ({dt:5.0f}s) "
                  + " ".join("—" if r["Ch"] is None else f"{r['Ch']:.4e}" for r in rows[-4:]), flush=True)
            json.dump(res, open(fn, "w"), indent=1)
    json.dump(res, open(fn, "w"), indent=1)



# ----------------------------------------------------------------- extension of slow cells
def task_ext(args):
    label, key, L, prob, ks = args
    mp.mp.dps = DPS
    T = _TAB[key]
    p = T.levels[L][1]
    Tend = mp.mpf(__import__("problems").P[prob]["T"])
    rows = []
    for k in ks:
        N = int(N0[prob] * 2 ** k)
        e = _run(T, L, prob, N)
        rows.append(dict(N=N, err=None if e is None else float(e),
                         Ch=None if e is None else float(e / (Tend / N) ** p)))
    return label, prob, rows

def extend(threshold=0.01, ks=(7, 8)):
    fn = os.path.join(os.path.dirname(__file__), "..", "data", "measured.json")
    res = json.load(open(fn))
    todo = []
    for (lab, key, L) in METHODS:
        for prob, cell in res[lab].items():
            ok = [r for r in cell["rows"] if r["err"]]
            Ch = [r["Ch"] for r in ok]
            R = [2 * Ch[k + 1] - Ch[k] for k in range(len(Ch) - 1)]
            conv = abs(R[-1] - R[-2]) / abs(R[-1]) if R[-1] else 1
            have = max(r["N"] for r in cell["rows"])
            if conv > threshold and have < N0[prob] * 2 ** max(ks):
                todo.append((lab, key, L, prob, [k for k in ks if N0[prob] * 2 ** k > have]))
    print(len(todo), "cells to extend:", [(t[0], t[3]) for t in todo], flush=True)
    t0 = time.time()
    with Pool(min(int(os.environ.get("NPROC", 11)), len(todo)), initializer=_init) as pool:
        for label, prob, rows in pool.imap_unordered(task_ext, todo):
            res[label][prob]["rows"] += rows
            print(f"[{time.time() - t0:6.0f}s] {label:12s} {prob:14s} "
                  + " ".join(f"{r['Ch']:.4e}" for r in res[label][prob]["rows"][-4:]), flush=True)
            json.dump(res, open(fn, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1:2] == ["extend"]:
        extend()
    else:
        main(sys.argv[1:] or None)
