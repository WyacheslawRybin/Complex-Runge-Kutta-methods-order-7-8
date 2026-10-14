"""
Cost-based figures of the order 7-8 paper: fixed-step and variable-step
work-precision diagrams with the cost relative to a reference method, and the
tolerance proportionality.
Usage:  python figures_cost.py [name ...]      (no argument: all figures)
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from figures import ST, lbl, save, DATA

PTITLE = {"exponential": r"$y'=-e^{y}$", "nonautonomous": r"$y'=4y\sin^3t\cos t$", "VanderPol": "Van der Pol",
          "Rossler": "Rössler", "Bouali": "Bouali", "Lorenz05": r"Lorenz, $T=0.5$", "Lorenz5": r"Lorenz, $T=5$",
          "Kepler": r"Kepler, $e=0.5$", "RigidBody": "rigid body", "Brusselator": "Brusselator",
          "quadratic": r"$y'=-y^2$", "Arenstorf": "Arenstorf orbit", "Dahlquist": r"Dahlquist $y'=-y$",
          "pendulum": "linear pendulum"}
EVALS = {"CRK7-opt": 7, "CRK7-s": 7, "CCK7(6)": 8,
         "Butcher7": 9, "Verner7(6)": 10, "Vern7": 9, "Fehlberg7": 11,
         "CRK8-opt": 9, "CRK8-s": 9,
         "CooperVerner8": 11, "DP8(7)": 13, "DOP853": 12, "Fehlberg8": 13, "Verner8(7)": 13}
# a pair is drawn in the style of the tableau that carries it
STYLE_OF = {"CRK7(5)": "CRK7-opt", "CRK7-s(5)": "CRK7-s", "CRK8(6)": "CRK8-opt", "CRK8-s(6)": "CRK8-s"}
sty = lambda m: ST[STYLE_OF.get(m, m)]

FIG7 = ["CRK7-opt", "CRK7-s", "CCK7(6)", "Verner7(6)", "Vern7", "Fehlberg7"]      # Butcher7: tables only
FIG8 = ["CRK8-opt", "CRK8-s", "CooperVerner8", "DP8(7)", "DOP853", "Verner8(7)"]   # RKF8(7): tables only
ADAPT7 = ["CRK7(5)", "CCK7(6)", "Verner7(6)", "Vern7"]
ADAPT8 = ["CRK8(6)", "CRK8-s(6)", "DP8(7)", "Verner8(7)", "DOP853"]
REF = {7: "Vern7", 8: "DOP853"}

def fixed_curve(meas, m, prob):
    rows = sorted([r for r in meas[m][prob]["rows"] if r["err"] is not None and r["err"] > 0], key=lambda r: r["N"])
    return np.array([r["N"] for r in rows], float), np.array([r["err"] for r in rows])

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

def ratio_curve(c, e, cr, er, lo=1e-30, hi=1e-2, n=200):
    """cost(eps) / cost_ref(eps) along the lower envelopes of two work-precision curves."""
    env = lambda cc, ee: np.minimum.accumulate(ee[np.argsort(cc)])
    a, b = env(c, e), env(cr, er)
    emax, emin = min(a[0], b[0], hi), max(a[-1], b[-1], lo)
    if emax <= emin:
        return np.array([]), np.array([])
    eps = np.geomspace(emax, emin, n)
    return eps, np.array([cost_at(c, e, x) / cost_at(cr, er, x) for x in eps])

def _wp_block(sf, curves, methods, ref, probs, title, xlab="RHS evaluations", ylim=None, ncol=7, lo=1e-30):
    """One order: work-precision row on top, cost relative to the reference method below."""
    axs = sf.subplots(2, len(probs), gridspec_kw=dict(height_ratios=[1.45, 1.0]))
    for j, p in enumerate(probs):
        a, b = axs[0, j], axs[1, j]
        cr, er = curves(ref, p)
        for m in methods:
            c, e = curves(m, p)
            col, mk, ls = sty(m)
            a.loglog(c, e, ls=ls, marker=mk, ms=2.4, color=col, mfc="none", label=lbl(m), lw=1.0)
            if m != ref:
                x, r = ratio_curve(c, e, cr, er, lo=lo)
                b.semilogx(x, r, ls=ls, color=col, lw=1.2)
        b.axhline(1, color=sty(ref)[0], ls=sty(ref)[2], lw=1.0)
        b.invert_xaxis()
        a.set_title(f"{title}: {PTITLE[p]}", fontsize=7.5)
        a.xaxis.set_minor_formatter(NullFormatter())
        if ylim:
            a.set_ylim(*ylim)
        a.set_xlabel(xlab, fontsize=7, labelpad=1)
        b.set_xlabel(r"error at $T$", fontsize=7, labelpad=1)
    axs[0, 0].set_ylabel(r"error at $T$")
    axs[1, 0].set_ylabel(f"cost / cost of {lbl(ref)}", fontsize=7)
    h, l = axs[0, 0].get_legend_handles_labels()
    sf.legend(h, l, loc="outside lower center", ncol=ncol, fontsize=6.6, columnspacing=1.0, handlelength=2.4)
    return axs

def fig_wp_fixed():
    meas = json.load(open(os.path.join(DATA, "measured.json")))
    probs = ["VanderPol", "Bouali", "Lorenz5", "Kepler"]
    def curves(m, p):
        N, e = fixed_curve(meas, m, p)
        ok = e < 1
        return EVALS[m] * N[ok], e[ok]
    fig = plt.figure(figsize=(7.4, 8.4), layout="constrained")
    sfs = fig.subfigures(2, 1, hspace=0.04)
    _wp_block(sfs[0], curves, FIG7, REF[7], probs, "order 7", ylim=(1e-32, 1), ncol=6)
    _wp_block(sfs[1], curves, FIG8, REF[8], probs, "order 8", ylim=(1e-32, 1), ncol=7)
    save(fig, "fig_wp_fixed")

def fig_adaptive():
    ad = json.load(open(os.path.join(DATA, "adaptive.json")))
    probs = ["Arenstorf", "Kepler", "Lorenz5", "Brusselator"]
    curves = lambda m, p: adapt_curve(ad, m, p)[:2]
    fig = plt.figure(figsize=(7.4, 8.4), layout="constrained")
    sfs = fig.subfigures(2, 1, hspace=0.04)
    xl = "RHS evaluations incl. rejections"
    _wp_block(sfs[0], curves, ADAPT7, REF[7], probs, "order-7 pairs", xl, ncol=4, lo=1e-14)
    _wp_block(sfs[1], curves, ADAPT8, REF[8], probs, "order-8 pairs", xl, ncol=5, lo=1e-14)
    save(fig, "fig_wp_adaptive")
    # tolerance proportionality, one row per order
    fig = plt.figure(figsize=(7.2, 4.9), layout="constrained")
    sfs = fig.subfigures(2, 1, hspace=0.04)
    for sf, methods, title in [(sfs[0], ADAPT7, "order-7 pairs"), (sfs[1], ADAPT8, "order-8 pairs")]:
        axs = sf.subplots(1, 3)
        for a, p in zip(axs, ["Arenstorf", "Kepler", "VanderPol"]):
            a.loglog([1e-13, 1e-3], [1e-13, 1e-3], color="#999999", lw=0.6, label="error = tol")
            for m in methods:
                _, e, tol = adapt_curve(ad, m, p)
                col, mk, ls = sty(m)
                a.loglog(tol, e, ls=ls, marker=mk, ms=2.2, color=col, label=lbl(m), mfc="none", lw=1.0)
            a.set_title(f"{title}: {PTITLE[p]}", fontsize=7.5); a.set_xlabel("tolerance", fontsize=7, labelpad=1)
        axs[0].set_ylabel(r"error at $T$")
        h, l = axs[0].get_legend_handles_labels()
        sf.legend(h[1:] + h[:1], l[1:] + l[:1], loc="outside lower center", ncol=7, fontsize=6.6,
                  columnspacing=1.0, handlelength=2.4)
    save(fig, "fig_tolerance")

ALL = {"wp_fixed": fig_wp_fixed, "adaptive": fig_adaptive}

if __name__ == "__main__":
    for n in (sys.argv[1:] or list(ALL)):
        ALL[n]()
