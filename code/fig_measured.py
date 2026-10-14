"""Figure fig_measured: measured error constants of all fixed-step methods relative to CRK7-opt (order seven)
and CRK8-opt (order eight) on the eleven nonlinear problems.  It replaces Tables measured7/measured8 in the
main text (they are in the appendix).  The constants and their classification are those of the tables
(make_tables.measured / classify); degenerate constants are omitted, approximate ones (15%) are plotted
with open markers.
Usage:  python fig_measured.py      -> ../paper/figures/fig_measured.pdf/.png
"""
import numpy as np

import figures as F
import make_tables as MT

plt = F.plt
PROBS = MT.NONLIN
SHORT = {"quadratic": r"$y'=-y^2$", "exponential": r"$y'=-e^{y}$", "nonautonomous": "non-aut.",
         "VanderPol": "Van der Pol", "Rossler": "Rössler", "Bouali": "Bouali", "Lorenz05": "Lorenz $T{=}0.5$",
         "Lorenz5": "Lorenz $T{=}5$", "Kepler": "Kepler", "RigidBody": "rigid body", "Brusselator": "Brusselator"}


def constants(S, m):
    return {p: (v["C"], v["cls"]) for p, v in S[m].items() if p in PROBS and v["cls"] != "deg"}


def panel(ax, S, methods, ref, title):
    cref = constants(S, ref)
    n = len(methods)
    for k, m in enumerate(methods):
        c = constants(S, m)
        col, mk, _ = F.ST[m]
        xs, ys, approx = [], [], []
        for i, p in enumerate(PROBS):
            if p in c and p in cref:
                xs.append(i + (k - (n - 1) / 2) * 0.07)
                ys.append(c[p][0] / cref[p][0])
                approx.append(c[p][1] == "approx" or cref[p][1] == "approx")
        xs, ys, approx = np.array(xs), np.array(ys), np.array(approx, bool)
        real = not MT.COMPLEX(m)
        if mk in ("x", "+"):
            ax.scatter(xs, ys, s=22, marker=mk, c=col, linewidths=1.0, label=F.lbl(m), zorder=3)
        else:
            face = "none" if real else col
            ax.scatter(xs[~approx], ys[~approx], s=16, marker=mk, facecolors=face, edgecolors=col, linewidths=0.8,
                       label=F.lbl(m), zorder=3)
            if approx.any():
                ax.scatter(xs[approx], ys[approx], s=16, marker=mk, facecolors="none", edgecolors=col,
                           linewidths=0.5, linestyle=":", zorder=3)
    ax.axhline(1.0, color="#999999", lw=0.6, zorder=1)
    ax.set_yscale("log")
    ax.set_xticks(range(len(PROBS)))
    ax.set_xticklabels([SHORT[p] for p in PROBS], rotation=40, ha="right")
    ax.set_xlim(-0.6, len(PROBS) - 0.4)
    ax.grid(axis="x", visible=False)
    ax.set_title(title)
    ax.set_ylabel(r"$C/C_{\mathrm{ref}}$")
    ax.legend(ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5), handletextpad=0.2, borderaxespad=0.3)


def main():
    S = MT.measured()
    fig, ax = plt.subplots(2, 1, figsize=(7.0, 5.2))
    panel(ax[0], S, MT.ORDER7, "CRK7-opt", "(a) order seven: constant relative to CRK7-opt")
    panel(ax[1], S, MT.ORDER8, "CRK8-opt", "(b) order eight: constant relative to CRK8-opt")
    fig.tight_layout()
    F.save(fig, "fig_measured")


if __name__ == "__main__":
    main()
