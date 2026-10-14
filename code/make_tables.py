"""
LaTeX table bodies of the order 7-8 paper generated from the data files (no number is typed by hand).
Writes ../paper/tables/*.tex and ../paper/tables/numbers.json (scalars quoted in the text).
    python make_tables.py            write the tables
    python make_tables.py splice     also replace the generated blocks in ../paper/crk78_paper.tex
"""
import glob, json, math, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
TAB = os.path.join(HERE, "..", "paper", "tables")
TEX = os.path.join(HERE, "..", "paper", "crk78_paper.tex")
os.makedirs(TAB, exist_ok=True)
NUM = {}

# evaluations per step of the propagated formula (fixed step)
EVALS = {"CRK7-opt": 7, "CRK7-s": 7, "CCK7(6)": 8,
         "Butcher7": 9, "Verner7(6)": 10, "Vern7": 9, "Fehlberg7": 11,
         "CRK8-opt": 9, "CRK8-s": 9,
         "CooperVerner8": 11, "DP8(7)": 13, "DOP853": 12, "Fehlberg8": 13, "Verner8(7)": 13}
ORDER7 = ["CRK7-opt", "CRK7-s", "CCK7(6)",
          "Butcher7", "Verner7(6)", "Vern7", "Fehlberg7"]
ORDER8 = ["CRK8-opt", "CRK8-s",
          "CooperVerner8", "DP8(7)", "DOP853", "Fehlberg8", "Verner8(7)"]
ORDER_OF = lambda m: 7 if m in ORDER7 else 8
COMPLEX = lambda m: m.startswith(("CRK", "CCK", "RSF"))
PNAME = {"Dahlquist": "Dahlquist", "pendulum": "linear pendulum", "quadratic": r"$y'=-y^{2}$",
         "exponential": r"$y'=-e^{y}$", "nonautonomous": r"$y'=4y\sin^{3}t\cos t$", "VanderPol": "Van der Pol",
         "Rossler": r"R\"ossler", "Bouali": "Bouali", "Lorenz05": r"Lorenz, $T=0.5$", "Lorenz5": r"Lorenz, $T=5$",
         "Kepler": r"Kepler, $e=0.5$", "RigidBody": "rigid body", "Brusselator": "Brusselator",
         "Arenstorf": "Arenstorf orbit"}
MNAME = {"Fehlberg7": "RKF7(8)", "Fehlberg8": "RKF8(7)"}
mname = lambda m: MNAME.get(m, m)
PROBS = ["Dahlquist", "pendulum", "quadratic", "exponential", "nonautonomous", "VanderPol", "Rossler", "Bouali",
         "Lorenz05", "Lorenz5", "Kepler", "RigidBody", "Brusselator"]
NONLIN = PROBS[2:]

def classify(rows, p):
    """Richardson-extrapolated constant C of h^p and the class: ok (1%), approx (15%), deg."""
    ok = [r for r in sorted(rows, key=lambda r: r["N"]) if r["err"] is not None and r["err"] > 0]
    Ch = [r["Ch"] for r in ok]
    R = [2 * Ch[k + 1] - Ch[k] for k in range(len(Ch) - 1)]
    order = math.log2(ok[-2]["err"] / ok[-1]["err"])
    C = R[-1]
    conv = abs(R[-1] - R[-2]) / abs(R[-1]) if R[-1] != 0 else math.inf
    if C <= 0 or order > p + 0.5 or order < p - 0.5:
        cls = "deg"
    elif conv <= 0.01:
        cls = "ok"
    elif conv <= 0.15:
        cls = "approx"
    else:
        cls = "deg"
    return dict(C=C, conv=conv, order=order, cls=cls, Nmax=ok[-1]["N"], err_last=ok[-1]["err"])

def sci(x, d=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "---"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    s = f"{m:.{d - 1}f}"
    if s.startswith("10"):
        e += 1; s = f"{m / 10:.{d - 1}f}"
    if -2 <= e <= 2:
        return f"${x:.{max(0, d - 1 - e)}f}$"
    return f"${s}{{\\cdot}}10^{{{e}}}$"

def fix3(x, approx=False):
    d = max(0, 2 - math.floor(math.log10(abs(x))))
    return "$" + (r"{\sim}" if approx else "") + f"{x:.{d}f}" + "$"

rot = lambda s: "\\rot{" + s + "}"
best = lambda s: "\\best{" + s + "}"

def mark_best(vals, cells, better=min, rel=5e-3):
    ok = [v for v in vals if v is not None and math.isfinite(v)]
    if not ok:
        return cells
    b = better(ok)
    return [best(c) if v is not None and math.isfinite(v) and abs(v - b) <= rel * abs(b) else c
            for v, c in zip(vals, cells)]

def write(name, lines):
    open(os.path.join(TAB, name + ".tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

def measured():
    meas = json.load(open(os.path.join(DATA, "measured.json")))
    S = {m: {p: classify(meas[m][p]["rows"], ORDER_OF(m)) for p in meas[m]} for m in meas if m in ORDER7 + ORDER8}
    NUM["measured"] = S
    return S

def table_measured(S):
    for name, methods in [("measured7", ORDER7), ("measured8", ORDER8)]:
        methods = [m for m in methods if m in S]
        lines = ["problem & unit & " + " & ".join(rot(mname(m)) for m in methods) + r"\\", r"\midrule"]
        for p in PROBS:
            vals, kinds = [], []
            for m in methods:
                v = S[m].get(p)
                if v is None:
                    vals.append(None); kinds.append(r"\todo{}")
                elif v["cls"] == "deg":
                    vals.append(None); kinds.append("deg.")
                else:
                    vals.append(v["C"]); kinds.append("approx" if v["cls"] == "approx" else "ok")
            k = math.floor(math.log10(max(v for v in vals if v is not None))) - 1
            cells = [kd if v is None else fix3(v / 10 ** k, kd == "approx") for v, kd in zip(vals, kinds)]
            lines.append(PNAME[p] + f" & $10^{{{k}}}$ & " + " & ".join(mark_best(vals, cells)) + r"\\")
            if p == "pendulum":
                lines.append(r"\midrule")
        write(name, lines)
    worst = {}
    for m in S:
        dev = [abs(S[m][q]["order"] - ORDER_OF(m)) for q in S[m] if S[m][q]["cls"] == "ok"]
        worst[m] = max(dev) if dev else None
    NUM["order_dev_max"] = worst
    n_ok = sum(1 for m in S for q in S[m] if S[m][q]["cls"] == "ok")
    n_01 = sum(1 for m in S for q in S[m] if S[m][q]["cls"] == "ok" and abs(S[m][q]["order"] - ORDER_OF(m)) <= 0.1)
    NUM["order_counts"] = dict(converged=n_ok, within_0p1=n_01)

KAPPA_COLS = [("CRK7-opt", "Vern7"), ("CRK7-opt", "Fehlberg7"), ("CRK7-opt", "Verner7(6)"),
              ("CCK7(6)", "Vern7"), ("CCK7(6)", "Fehlberg7"),
              ("CRK8-opt", "DOP853"), ("CRK8-opt", "DP8(7)"), ("CRK8-opt", "CooperVerner8"),
              ("CRK8-s", "DOP853"), ("CRK8-s", "DP8(7)")]

def kappa_star(S, c, r, q):
    a, b = S.get(c, {}).get(q), S.get(r, {}).get(q)
    if a is None or b is None or a["cls"] == "deg" or b["cls"] == "deg" or a["C"] <= 0 or b["C"] <= 0:
        return None
    p = ORDER_OF(c)
    return (EVALS[r] / EVALS[c]) * (b["C"] / a["C"]) ** (1 / p)

def table_kappa(S):
    kap = {}
    for c in [m for m in S if COMPLEX(m)]:
        for r in [m for m in S if not COMPLEX(m) and ORDER_OF(m) == ORDER_OF(c)]:
            for q in PROBS:
                kap[f"{c}|{r}|{q}"] = kappa_star(S, c, r, q)
    NUM["kappa_star"] = kap
    cols = [(c, r) for c, r in KAPPA_COLS if c in S and r in S]
    lines = ["problem & " + " & ".join(mname(r) for _, r in cols) + r"\\", r"\midrule"]
    for q in PROBS:
        vals = [kap.get(f"{c}|{r}|{q}") for c, r in cols]
        lines.append(PNAME[q] + " & " + " & ".join("---" if v is None else
                     (best(f"${v:.2f}$") if v >= 1.005 else f"${v:.2f}$") for v in vals) + r"\\")
        if q == "pendulum":
            lines.append(r"\midrule")
    write("kappa_star", lines)
    # header row with the complex methods (multicolumn spans)
    groups = []
    for c, r in cols:
        if groups and groups[-1][0] == c:
            groups[-1][1] += 1
        else:
            groups.append([c, 1])
    NUM["kappa_groups"] = groups

STAB_ROWS7 = [("CRK7 ($T_7$)", r"$T_7$: CRK7"), ("CCK7(6)", "CCK7(6)"), ("Butcher7", "Butcher7"),
              ("Verner7(6)", "Verner7(6)"), ("Vern7", "Vern7"), ("Fehlberg7", "RKF7(8)")]
STAB_ROWS8 = [("CRK8-opt", "CRK8-opt"), ("CRK8-s", "CRK8-s"),
              ("CooperVerner8", "CooperVerner8"), ("DP8(7)", "DP8(7)"), ("DOP853", "DOP853"),
              ("Fehlberg8", "RKF8(7)"), ("Verner8(7)", "Verner8(7)")]

def table_stability():
    """Stability and preference regions in one table (as in the order 5-6 paper): s, real interval, its
    length per evaluation, imaginary interval, area per evaluation and preferred area (tol = 1) per evaluation."""
    st = json.load(open(os.path.join(DATA, "stability.json")))
    pr = json.load(open(os.path.join(DATA, "prefregions.json")))
    lines = []
    for block in (STAB_ROWS7, STAB_ROWS8):
        rows = [(lab, st[key]) for key, lab in block if key in st]
        cols = {"len": [-v["real"] / v["evals"] for _, v in rows], "real": [-v["real"] for _, v in rows],
                "imag": [v["imag"] for _, v in rows], "a2": [v["area_left_per_eval2"] for _, v in rows],
                "p2": [pr[lab]["pref_per_eval"] for lab, _ in rows]}
        cells = {k: mark_best(cols[k], [f"${x:.3f}$" if k in ("len", "a2", "p2") else
                                        f"${x:.4f}$" for x in cols[k]], better=max) for k in cols}
        for i, (lab, v) in enumerate(rows):
            real = cells["real"][i].replace("$", "")
            realcell = (r"\best{$[-" + cols_fmt(cols["real"][i]) + r",\,0]$}") if real.startswith("\\best") else \
                       f"$[-{cols_fmt(cols['real'][i])},\\,0]$"
            lines.append(f"{lab} & {v['evals']} & {realcell} & {cells['len'][i]} & {cells['imag'][i]} & "
                         f"{cells['a2'][i]} & {cells['p2'][i]}" + r"\\")
        if block is STAB_ROWS7:
            lines.append(r"\midrule")
    write("stability", lines)

def cols_fmt(x):
    return f"{x:.4f}"

def table_prefregions():
    pr = json.load(open(os.path.join(DATA, "prefregions.json")))
    lines = []
    for block in ([r"$T_7$: CRK7", "CCK7(6)", "Butcher7", "Verner7(6)", "Vern7", "RKF7(8)"],
                  ["CRK8-opt", "CRK8-s", "CooperVerner8", "DP8(7)", "DOP853",
                   "RKF8(7)", "Verner8(7)"]):
        rows = [(k, pr[k]) for k in block if k in pr]
        cells = {}
        for key, fmt in (("stab", "{:.2f}"), ("pref", "{:.2f}"), ("stab_per_eval", "{:.3f}"), ("pref_per_eval", "{:.3f}")):
            vals = [v[key] for _, v in rows]
            cells[key] = mark_best(vals, ["$" + fmt.format(x) + "$" for x in vals], better=max)
        for i, (k, v) in enumerate(rows):
            lines.append(f"{k} & {v['evals']} & {cells['stab'][i]} & {cells['pref'][i]} & "
                         f"{cells['stab_per_eval'][i]} & {cells['pref_per_eval'][i]}" + r"\\")
        if block[0].startswith("$T_7$"):
            lines.append(r"\midrule")
    write("prefregions", lines)

ADAPT7 = ["CRK7(5)", "CCK7(6)", "Verner7(6)", "Vern7"]
ADAPT8 = ["CRK8(6)", "CRK8-s(6)", "DP8(7)", "Verner8(7)", "DOP853"]
ADAPT_VARIANTS = ["CRK7-s(5)"]
ADAPT_PROBS = ["exponential", "nonautonomous", "VanderPol", "Rossler", "Bouali", "Lorenz5", "Kepler",
               "RigidBody", "Brusselator", "Arenstorf"]

def adapt_curve(ad, m, prob):
    rows = [r for r in ad[m][prob] if r["err"] is not None and r["err"] > 0]
    return (np.array([r["nfev"] for r in rows], float), np.array([r["err"] for r in rows]),
            np.array([r["tol"] for r in rows]))

def cost_at(cost, err, eps):
    """Smallest cost reaching accuracy eps: log-log interpolation along the lower envelope."""
    o = np.argsort(cost)
    c, e = cost[o], np.minimum.accumulate(err[o])
    if eps < e[-1]:
        return np.inf
    if eps >= e[0]:
        return c[0]
    k = int(np.argmax(e <= eps))
    y0, y1 = np.log(e[k - 1]), np.log(e[k])
    if y1 == y0:
        return c[k]
    x0, x1 = np.log(c[k - 1]), np.log(c[k])
    return float(np.exp(x0 + (np.log(eps) - y0) * (x1 - x0) / (y1 - y0)))

def table_adaptive():
    fn = os.path.join(DATA, "adaptive.json")
    if not os.path.exists(fn):
        return
    ad = json.load(open(fn))
    a7 = [m for m in ADAPT7 if m in ad]; a8 = [m for m in ADAPT8 if m in ad]
    res = {}
    for eps in (1e-6, 1e-8, 1e-10):
        lines = ["problem & " + " & ".join(rot(mname(m)) for m in a7 + a8) + r"\\", r"\midrule"]
        for p in ADAPT_PROBS:
            res[(eps, p)] = {m: cost_at(*adapt_curve(ad, m, p)[:2], eps) for m in a7 + a8 + ADAPT_VARIANTS if m in ad}
            cells = []
            for group in (a7, a8):
                vals = [res[(eps, p)][m] for m in group]
                vals = [round(v) if math.isfinite(v) else v for v in vals]
                cells += mark_best(vals, ["---" if not math.isfinite(v) else f"{v:.0f}" for v in vals], rel=0)
            lines.append(PNAME[p].replace(" orbit", "") + " & " + " & ".join(cells) + r"\\")
        write(f"adaptive_{int(-math.log10(eps))}", lines)
    NUM["adaptive_cost"] = {f"{e}|{p}|{m}": v for (e, p), d in res.items() for m, v in d.items()}
    stats = {}
    for m in [m for m in a7 + a8 + ADAPT_VARIANTS if m in ad]:
        rej, slopes = [], []
        for p in ADAPT_PROBS:
            rows = [r for r in ad[m][p] if r["err"]]
            rej += [r["nrej"] / max(1, r["nacc"]) for r in rows]
            t = np.log10([r["tol"] for r in rows if r["tol"] <= 1e-5])
            e = np.log10([r["err"] for r in rows if r["tol"] <= 1e-5])
            if len(t) > 3:
                slopes.append(np.polyfit(t, e, 1)[0])
        stats[m] = dict(rej_median=float(np.median(rej)), rej_max=float(np.max(rej)),
                        slope_median=float(np.median(slopes)), slope_min=float(np.min(slopes)),
                        slope_max=float(np.max(slopes)))
    NUM["adaptive_stats"] = stats
    write("adaptive_stats", [f"{mname(m)} & ${s['slope_median']:.2f}$ & ${s['slope_min']:.2f}$--${s['slope_max']:.2f}$ & "
                             f"${100 * s['rej_median']:.0f}\\%$ & ${100 * s['rej_max']:.0f}\\%$" + r"\\"
                             for m, s in stats.items() if m in a7 + a8])

PAIRS7 = [("CRK7(5)", 7, 7), ("CRK7-s(5)", 7, 7), ("CCK7(6)", 8, 8), ("Verner7(6)", 10, 10), ("Vern7", 10, 10)]
PAIRS8 = [("CRK8(6)", 9, 9), ("CRK8-s(6)", 9, 9), ("DP8(7)", 13, 13), ("Verner8(7)", 13, 13)]
PAIR_Q = {"CCK7(6)": 6, "Verner7(6)": 6, "Vern7": 6, "DP8(7)": 7, "Verner8(7)": 7}

def table_pairs():
    emb = json.load(open(os.path.join(DATA, "embedded.json")))
    lines = []
    for block in (PAIRS7, PAIRS8):
        for name, s, ev in block:
            d = emb[name]
            q = d.get("q", PAIR_Q.get(name))
            lines.append(f"{name} & {s} & {ev} & {q} & {sci(d['Ahat'])} & ${d['B']:.2f}$ & ${d['C']:.2f}$ & "
                         f"${abs(d['E']):.2f}$ & ${d['maxbh']:.3f}$" + r"\\")
        if block is PAIRS7:
            lines.append(r"\midrule")
    write("pairs", lines)
    NUM["pairs"] = {k: v for k, v in emb.items() if isinstance(v, dict)}
    NUM["Ahat_ref"] = {k: emb[k] for k in ("Ahat_DP65", "Ahat_Vern7")}

# ---------------------------------------------------------------- CPU time (bench/bench.cpp -> data/bench.csv)
BENCH_METHODS = ["CRK7-opt", "CCK7(6)", "Butcher7", "Verner7(6)", "Vern7", "Fehlberg7",
                 "CRK8-opt", "CooperVerner8", "DP8(7)", "DOP853", "Fehlberg8", "Verner8(7)"]
# the time per step depends on the stage count and the arithmetic, not on the coefficient values
BENCH_KEY = {m: m for m in ORDER7 + ORDER8}
BENCH_KEY.update({k: "CRK7-opt" for k in ("CRK7-s", "CRK7(5)", "CRK7-s(5)")})
BENCH_KEY.update({k: "CRK8-opt" for k in ("CRK8-s", "CRK8(6)", "CRK8-s(6)")})
BENCH_PROB = {"Lorenz05": "Lorenz", "Lorenz5": "Lorenz"}
COST_PROBS = ["exponential", "nonautonomous", "VanderPol", "Rossler", "Bouali", "Lorenz5", "Kepler",
              "RigidBody", "Brusselator", "Arenstorf"]
CR_FIXED = {7: (["CRK7-opt", "CCK7(6)"], ["Butcher7", "Verner7(6)", "Vern7", "Fehlberg7"]),
            8: (["CRK8-opt", "CRK8-s"], ["CooperVerner8", "DP8(7)", "DOP853", "Fehlberg8", "Verner8(7)"])}
CR_ADAPT = {7: (["CRK7(5)", "CCK7(6)"], ["Verner7(6)", "Vern7"]),
            8: (["CRK8(6)", "CRK8-s(6)"], ["DP8(7)", "Verner8(7)", "DOP853"])}

def bench():
    """Median over the runs data/bench.csv, data/bench_run2.csv, ... (the run-to-run spread is up to 20%)."""
    files = sorted(glob.glob(os.path.join(DATA, "bench*.csv")))
    runs = []
    for fn in files:
        rows = [l.strip().split(",") for l in open(fn, encoding="ascii") if l.strip()][1:]
        runs.append({(r[0], r[1]): r[2:] for r in rows})
    if not runs:
        return {}
    med = lambda v: float(np.median(v))
    return {k: dict(stages=int(r[0]), evals=int(r[1]), ns=med([float(R[k][2]) for R in runs]),
                    fr=med([float(R[k][3]) for R in runs]), fc=med([float(R[k][4]) for R in runs]))
            for k, r in runs[0].items()}

def bench_of(B, m, q):
    return B.get((BENCH_PROB.get(q, q), BENCH_KEY.get(m, m)))

def table_bench(B):
    probs = ["exponential", "nonautonomous", "VanderPol", "Rossler", "Lorenz", "Bouali", "Kepler", "RigidBody",
             "Brusselator", "Arenstorf"]
    lines = ["problem & " + " & ".join(rot(mname(m)) for m in BENCH_METHODS)
             + r" & \rot{complex $f$ / real $f$} & $\kappa_{\mathrm{CPU}}$\\", r"\midrule"]
    kap, fratio = {}, {}
    for p in probs:
        kap[p] = (B[(p, "CRK8-opt")]["ns"] / EVALS["CRK8-opt"]) / (B[(p, "DOP853")]["ns"] / EVALS["DOP853"])
        fratio[p] = B[(p, "CRK8-opt")]["fc"] / B[(p, "CRK8-opt")]["fr"]
        name = "Lorenz" if p == "Lorenz" else PNAME[p].replace(" orbit", "")
        lines.append(name + " & " + " & ".join(f"${B[(p, m)]['ns']:.0f}$" for m in BENCH_METHODS)
                     + f" & ${fratio[p]:.2f}$ & ${kap[p]:.2f}$" + r"\\")
    write("bench", lines)
    NUM["kappa_cpu"] = kap
    NUM["f_complex_over_real"] = fratio
    NUM["stage_time_ratio"] = {f"{p}|{m}": (B[(p, m)]["ns"] / EVALS[m]) / (B[(p, "DOP853")]["ns"] / EVALS["DOP853"])
                               for p in probs for m in BENCH_METHODS}

def fixed_curve(meas, m, prob):
    rows = sorted([r for r in meas[m][prob]["rows"] if r["err"] is not None and r["err"] > 0], key=lambda r: r["N"])
    return np.array([r["N"] for r in rows], float), np.array([r["err"] for r in rows])

def table_cost_ratio(B, eps=1e-8):
    """Best complex / best real cost at eps within each order; kappa = 1 and measured CPU time."""
    ad = json.load(open(os.path.join(DATA, "adaptive.json")))
    meas = json.load(open(os.path.join(DATA, "measured.json")))
    out, lines = {}, []
    for p in COST_PROBS:
        cells = []
        for mode, groups in (("fixed", CR_FIXED), ("adaptive", CR_ADAPT)):
            for order in (7, 8):
                cplx, real = groups[order]
                for cm in ("kappa1", "cpu"):
                    vals = {}
                    for m in cplx + real:
                        if mode == "fixed":
                            if p not in meas[m]:
                                vals[m] = math.inf; continue
                            N, e = fixed_curve(meas, m, p)
                            c = EVALS[m] * N if cm == "kappa1" else N * bench_of(B, m, p)["ns"]
                        else:
                            n, e, _ = adapt_curve(ad, m, p)
                            bm = bench_of(B, m, p)
                            c = n if cm == "kappa1" else n * bm["ns"] / bm["evals"]
                        vals[m] = cost_at(c, e, eps)
                    cx, rl = min(vals[m] for m in cplx), min(vals[m] for m in real)
                    r = cx / rl if math.isfinite(cx) and math.isfinite(rl) else None
                    out[f"{mode}|{p}|{order}|{cm}"] = r
                    out[f"{mode}|{p}|{order}|{cm}|best"] = [min(cplx, key=vals.get), min(real, key=vals.get)]
                    cells.append("---" if r is None else (best(f"${r:.2f}$") if r < 0.995 else f"${r:.2f}$"))
        lines.append(PNAME[p] + " & " + " & ".join(cells) + r"\\")
    write("cost_ratio", lines)
    NUM["cost_ratio"] = out

TUNE_START = {"CRK7-opt": 7, "CRK7-s": 7, "CCK7(6)": 7, "CRK8-opt": 8, "CRK8-s": 8}

def table_tuning(S):
    """Tuning to the Rossler system (code/tune_78.py): improvement factors at N/2, N, 2N from the logs, the
    asymptotic constant before (measured.json) and after (data/tuned.json), E_{p+1} and max|coef|."""
    fn = os.path.join(DATA, "tuned.json")
    if not os.path.exists(fn):
        return
    tu = json.load(open(fn))
    TI = tableau_info()
    lines, out = [], {}
    for m in TUNE_START:
        if m not in tu:
            continue
        txt = open(os.path.join(HERE, "tune_runs", f"tune_{m}.log"), encoding="utf-8").read().splitlines()
        fac = [float(v) for v in [l for l in txt if l.startswith("improvement factors")][0].split(":")[1].split()]
        n = int(re.search(r"N_LIST \((\d+), (\d+), (\d+)\)", txt[0]).group(2))
        C0, C1 = S[m]["Rossler"]["C"], tu[m]["C"]
        lines.append(f"{mname(m)} & ${n}$ & " + " & ".join(f"${v:.2f}$" for v in fac)
                     + f" & {sci(C0)} & {sci(C1)} & ${C0 / C1:.2f}$ & {sci(TI[m]['E'])} & {sci(tu[m]['E'])} & "
                     f"${TI[m]['maxcoef']:.2f}\\to{tu[m]['maxcoef']:.2f}$" + r"\\")
        out[m] = dict(N=n, factors=fac, C0=C0, C1=C1, ratio=C0 / C1, E0=TI[m]["E"], E1=tu[m]["E"],
                      maxcoef1=tu[m]["maxcoef"], conv=tu[m]["conv"])
    write("tuning", lines)
    NUM["tuning"] = out

PAIR_OF = {"CRK7-opt": "CRK7(5)", "CCK7(6)": "CCK7(6)", "Verner7(6)": "Verner7(6)", "Vern7": "Vern7",
           "CRK8-opt": "CRK8(6)", "CRK8-s": "CRK8-s(6)", "DP8(7)": "DP8(7)", "Verner8(7)": "Verner8(7)",
           "DOP853": "DOP853"}
SUMMARY_REF = {7: "Vern7", 8: "DOP853"}

def gmean(v):
    return float(np.exp(np.mean(np.log(v)))) if len(v) else None

def tableau_info():
    """E_{p+1} from the checks; max|coef| of the complex tableaux from data/tableaux.json, i.e., after the
    normalization Im b_1 = Im b_2 = Im b_3 = 0 (which changes max|coef| but not E)."""
    info = {}
    for f in ("check_named.json", "check_added.json"):
        for k, v in json.load(open(os.path.join(DATA, f))).items():
            info[k] = dict(E=v["E"][0], maxcoef=v["maxcoef"])
    cabs = lambda z: math.hypot(float(z[0]), float(z[1]))
    for k, t in json.load(open(os.path.join(DATA, "tableaux.json"))).items():
        if k in info:
            info[k]["maxcoef"] = max([cabs(z) for row in t["A"] for z in row] + [cabs(z) for w in t["b"] for z in w])
    return info

def table_summary(S, B, eps=1e-8):
    """One row per method within each order: s, E_{p+1}, max|coef|, fixed-step cost from the measured constants
    and variable-step cost of its pair at accuracy eps, relative to Vern7 / DOP853, geometric means over the
    problems, for kappa = 1 and for the measured CPU time; wins at kappa = 1."""
    ad = json.load(open(os.path.join(DATA, "adaptive.json")))
    TI = tableau_info()
    lines, out = [], {}
    for p, methods in ((7, [m for m in ORDER7 if m in S]), (8, [m for m in ORDER8 if m in S])):
        ref = SUMMARY_REF[p]
        def wf(m, q, cpu):
            v = S[m].get(q)
            if v is None or v["cls"] == "deg" or v["C"] <= 0:
                return None
            if cpu:
                bm = bench_of(B, m, q)
                return None if bm is None else bm["ns"] * v["C"] ** (1 / p)
            return EVALS[m] * v["C"] ** (1 / p)
        def wa(m, q, cpu):
            P = PAIR_OF.get(m)
            if P is None:
                return None
            n, e, _ = adapt_curve(ad, P, q)
            if cpu:
                bm = bench_of(B, P, q)
                if bm is None:
                    return None
                n = n * bm["ns"] / bm["evals"]
            c = cost_at(n, e, eps)
            return c if math.isfinite(c) else None
        pairs = [m for m in methods if m in PAIR_OF]
        probs_a = [q for q in ADAPT_PROBS if all(wa(m, q, False) is not None for m in pairs)]
        cols = {}
        for key, fun, probs in (("fix1", lambda m, q: wf(m, q, False), NONLIN),
                                ("fixcpu", lambda m, q: wf(m, q, True), NONLIN),
                                ("ad1", lambda m, q: wa(m, q, False), probs_a),
                                ("adcpu", lambda m, q: wa(m, q, True), probs_a)):
            rel, wins = {}, {}
            for m in methods:
                r = [fun(m, q) / fun(ref, q) for q in probs if fun(m, q) is not None and fun(ref, q) is not None]
                rel[m] = gmean(r) if r else None
            for q in probs:
                vals = {m: fun(m, q) for m in methods if fun(m, q) is not None}
                if vals:
                    w = min(vals, key=vals.get)
                    wins[w] = wins.get(w, 0) + 1
            cols[key] = (rel, wins, probs)
        out[p] = {k: dict(rel=v[0], wins=v[1], problems=v[2]) for k, v in cols.items()}
        cell = lambda v: "---" if v is None else f"${v:.2f}$"
        rel_cells = {k: mark_best([cols[k][0][m] for m in methods], [cell(cols[k][0][m]) for m in methods])
                     for k in cols}
        Ecells = mark_best([TI[m]["E"] for m in methods], [sci(TI[m]["E"], 3) for m in methods])
        Mcells = mark_best([TI[m]["maxcoef"] for m in methods], [f"${TI[m]['maxcoef']:.2f}$" for m in methods])
        win_cells = {k: [f"{cols[k][1].get(m, 0)}" if (k == "fix1" or m in PAIR_OF) else "---" for m in methods]
                     for k in ("fix1", "ad1")}
        for i, m in enumerate(methods):
            lines.append(f"{mname(m)} & {EVALS[m]} & {Ecells[i]} & {Mcells[i]} & {rel_cells['fix1'][i]} & "
                         f"{win_cells['fix1'][i]} & {rel_cells['fixcpu'][i]} & "
                         f"{rel_cells['ad1'][i]} & {win_cells['ad1'][i]} & {rel_cells['adcpu'][i]}" + r"\\")
        if p == 7:
            lines.append(r"\midrule")
    write("summary", lines)
    NUM["summary"] = out

def splice():
    tex = open(TEX, encoding="utf-8").read()
    n = 0
    for f in sorted(os.listdir(TAB)):
        if not f.endswith(".tex"):
            continue
        body = open(os.path.join(TAB, f), encoding="utf-8").read()
        pat = re.compile(r"(%% BEGIN generated: tables/" + re.escape(f) + r"\n)(.*?)(%% END generated: tables/"
                         + re.escape(f) + ")", re.S)
        tex, k = pat.subn(lambda m: m.group(1) + body + m.group(3), tex)
        n += k
    open(TEX, "w", encoding="utf-8").write(tex)
    print("spliced", n, "blocks")

if __name__ == "__main__":
    S = measured()
    table_measured(S); table_kappa(S); table_stability(); table_prefregions(); table_adaptive(); table_pairs()
    B = bench()
    if B:
        table_bench(B); table_cost_ratio(B); table_summary(S, B)
    table_tuning(S)
    json.dump(NUM, open(os.path.join(TAB, "numbers.json"), "w"), indent=1, default=str)
    if sys.argv[1:2] == ["splice"]:
        splice()
