"""
All figures of the order 7-8 paper (PDF), in the style of the order 5-6 paper.
Usage:  python figures.py [name ...]      (no argument: all figures)
"""
import json, os, sys
from math import factorial
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, FormatStrFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIG = os.path.join(HERE, "..", "paper", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.5, "lines.linewidth": 1.2,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42})

# fixed colour / marker / line style for every method
ST = {
    # order 7, complex
    "CRK7-opt": ("#1f5fbf", "o", "-"), "CRK7s7b": ("#5b8fd9", "o", "--"), "CRK7-s": ("#5b8fd9", "o", "--"), "CRK7-B0": ("#8fb3e8", "o", ":"),
    "CRK7-Bbal": ("#2bb3c9", "d", "--"), "CRK7-Blo": ("#9aa7b8", "o", "-."),
    "CCK7(6)": ("#d1495b", "s", "-"), "CCK7-casc": ("#e07b14", "D", "-"),
    # order 8, complex
    "CRK8-opt": ("#1a8a4a", "^", "-"), "CRK8-b": ("#2e6b3f", "v", "-."), "CRK8-s": ("#55b57d", "v", "--"), "CRK8s9": ("#55b57d", "^", "--"), "RSF-CRK8-9": ("#a3b81a", "d", "--"),
    "RSF-CRK8-10": ("#8fd1a8", "^", ":"), "CCK8(7)": ("#b5285b", "s", "-"),
    # order 7, real
    "Butcher7": ("#8e5bb5", "v", ":"), "Verner7(6)": ("#6a2c91", "P", "-."), "Vern7": ("#222222", "x", "--"),
    "Fehlberg7": ("#777777", "+", "--"),
    # order 8, real
    "CooperVerner8": ("#b59a2b", "h", ":"), "DP8(7)": ("#8c5a2b", "*", "--"), "DOP853": ("#000000", "x", "-."),
    "Fehlberg8": ("#777777", "+", "--"), "Verner8(7)": ("#6a2c91", "P", "-."),
}
LBL = {"CCK7-casc": "CCK7-cascade", "Fehlberg7": "RKF7(8)", "Fehlberg8": "RKF8(7)"}
lbl = lambda k: LBL.get(k, k)

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    plt.close(fig)
    print("wrote", name)

def Tm(m):
    return [1.0 / factorial(k) for k in range(m + 1)]

# ---------------------------------------------------------------- stability regions
def fig_stability():
    st = json.load(open(os.path.join(DATA, "stability.json")))
    X, Y = np.meshgrid(np.linspace(-6.8, 1.6, 1000), np.linspace(-6.2, 6.2, 1200)); Z = X + 1j * Y
    P = lambda co: np.abs(np.polynomial.Polynomial(co)(Z))
    co = lambda k: st[k]["coeffs"]
    has = lambda k: k in st
    # complex projected methods with s = p: P = T_p; with s > p the free tail of P is shown
    o7 = [(r"$T_7$: CRK7", Tm(7), "CRK7-opt")]
    o7 += [(lbl(k), co(k), k) for k in ["CCK7(6)", "Butcher7", "Verner7(6)", "Vern7", "Fehlberg7"] if has(k)]
    o8 = [(r"$T_8$", Tm(8), None)]
    o8 += [(lbl(k), co(k), k) for k in ["CRK8-opt", "CRK8-s", "CooperVerner8", "DOP853",
                                         "DP8(7)", "Verner8(7)", "Fehlberg8"] if has(k)]
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 3.5), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
    for a, title, curves, fill in [(ax[0], "(a) order 7", o7, "#e6edf7"), (ax[1], "(b) order 8", o8, "#e8f3ea")]:
        a.contourf(X, Y, P(curves[0][1]), levels=[0, 1], colors=[fill])
        for lab, c, key in curves:
            col, _, ls = ST[key] if key else ("#404040", None, "-")
            a.contour(X, Y, P(c), [1], colors=[col], linestyles=[ls], linewidths=1.1)
            a.plot([], [], color=col, ls=ls, label=lab)
        a.set_aspect("equal"); a.set_title(title); a.set_xlabel(r"Re $z$")
        a.axhline(0, color="#999999", lw=0.4); a.axvline(0, color="#999999", lw=0.4)
        a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.2, handlelength=2.0,
                 columnspacing=0.8)
        a.set_xlim(-6.8, 1.6); a.set_ylim(-6.2, 6.2)
    ax[0].set_ylabel(r"Im $z$")
    x = np.linspace(-7, 0.5, 2000)
    for lab, c, key in [(r"$T_7$", Tm(7), "CRK7-opt"), (r"$T_8$", Tm(8), None)] + \
                       [(lbl(k), co(k), k) for k in ["CCK7(6)", "Vern7", "CRK8-opt", "CRK8-s", "DOP853"] if has(k)]:
        col, _, ls = ST[key] if key else ("#404040", None, "-")
        ax[2].plot(x, np.abs(np.polynomial.Polynomial(c)(x)), color=col, ls=ls, label=lab)
    ax[2].axhline(1, color="k", lw=0.6)
    ax[2].set_ylim(0, 1.6); ax[2].set_xlim(-7, 0.3)
    ax[2].set_xlabel(r"$z\in\mathbb{R}$"); ax[2].set_ylabel(r"$|P(z)|$"); ax[2].set_title("(c) real axis")
    ax[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=6.2, ncol=2, handlelength=2.0,
                 columnspacing=0.8)
    fig.tight_layout(w_pad=0.4)
    save(fig, "fig_stability")

# ---------------------------------------------------------------- E_min(B) fronts
def tableau_EM():
    """E_{p+1} from the checks and max|coef| (complex tableaux: after the inert normalization of tableaux.json)."""
    info = {}
    for f in ("check_named.json", "check_added.json"):
        for k, v in json.load(open(os.path.join(DATA, f))).items():
            info[k] = (v["E"][0], v["maxcoef"])
    cabs = lambda z: float(np.hypot(float(z[0]), float(z[1])))
    for k, t in json.load(open(os.path.join(DATA, "tableaux.json"))).items():
        if k in info:
            info[k] = (info[k][0], max([cabs(z) for row in t["A"] for z in row] + [cabs(z) for w in t["b"] for z in w]))
    return info

def front_E(fam, files=(), bmax=1000.0):
    """E_min(B) from data/front_E*.json: for every bound B of the chains, the smallest E over all points of all
    chains of the family and all B = 5 minima of the files with normalized max|coef| <= B (front_E.py --norm
    adds maxcoef_n)."""
    d = [c for fn in ("front_E.json", "front_E_CCK7.json") if os.path.exists(os.path.join(DATA, fn))
         for c in json.load(open(os.path.join(DATA, fn)))]
    pts = [p for ch in d if ch["family"] == fam for p in ch["points"] if p["E"] is not None]
    grid = sorted({p["B"] for p in pts if p["B"] <= bmax})
    pts += [dict(E=r["E"], maxcoef=r["maxcoef"], maxcoef_n=r.get("maxcoef_n", r["maxcoef"]))
            for fn in files for r in json.load(open(os.path.join(DATA, fn)))["results"]
            if r.get("E") and r.get("resid", 0) < 1e-10]
    B, E = [], []
    for b in grid:
        ok = [p["E"] for p in pts if p.get("maxcoef_n", p["maxcoef"]) <= b * 1.001]
        if ok:
            B.append(b); E.append(min(ok))
    return np.array(B), np.array(E)

def fig_pareto():
    TI = tableau_EM()
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 3.5))
    fkw = dict(color="k", lw=1.1, drawstyle="steps-post", label=r"$E_{\min}(B)$")
    panels = [(ax[0], "CRK7", ["Emin_77_5.json", "Emin_77_5_polish.json"],
               ["CRK7-opt", "CRK7-s", "Butcher7", "Verner7(6)", "Vern7", "Fehlberg7"],
               "(a) CRK7, $E_8$"),
              (ax[1], "CRK8", ["Emin_98_5.json", "Emin_98_5_polish.json"],
               ["CRK8-opt", "CRK8-s", "CooperVerner8", "DP8(7)", "DOP853", "Fehlberg8",
                "Verner8(7)"], "(b) CRK8, $E_9$"),
              (ax[2], "CCK7", ["Emin_87_76_5.json", "Emin_87_76_5_front.json", "Emin_87_76_5_front2.json",
                               "Emin_87_76_5_pol_r1.json", "Emin_87_76_5_final.json"],
               ["CCK7(6)", "Verner7(6)", "Vern7"], "(c) CCK7(6), $E_8$")]
    for a, fam, files, marks, title in panels:
        E, M = [], []
        for fn in files:
            for r in json.load(open(os.path.join(DATA, fn)))["results"]:
                if r.get("E") and r.get("resid", 0) < 1e-10:
                    E.append(r["E"]); M.append(r.get("maxcoef_n", r["maxcoef"]))
        a.scatter(M, E, s=4, color="#bbbbbb", lw=0, label="local minima, $B=5$")
        a.plot(*front_E(fam, files), **fkw)
        for k in marks:
            if k in TI:
                a.plot(TI[k][1], TI[k][0], ST[k][1], color=ST[k][0], ms=5, mec="k", mew=0.4, label=lbl(k))
        a.set_title(title)
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel(r"$\max|$coefficient$|$")
        a.set_xlim(0.3, 300)
        a.xaxis.set_minor_formatter(NullFormatter())
        a.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.5, 1, 2, 5, 10, 20, 50, 100, 200]))
        a.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.0, handletextpad=0.3,
                 columnspacing=0.8, handlelength=1.6)
    ax[0].set_ylabel(r"$E_{p+1}=\|e\|_2$")
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig_pareto")

ALL = {"stability": fig_stability, "pareto": fig_pareto}

if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    for n in names:
        ALL[n]()
