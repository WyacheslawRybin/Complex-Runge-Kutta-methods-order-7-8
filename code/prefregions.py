"""
Stability and preference regions in the sense of Fedoseev, Karimov, Legat and
Butusov (Mathematics 10:4327, 2022), as implemented in U-CUDA:
a point z = h*lambda of the complex plane is PREFERRED when the step is stable,
|P(z)| <= 1, and its relative one-step error is below a tolerance,
    |e^z - P(z)| / |e^z| <= tol,     tol = 1 as in the original paper.
For a projected method on the real test system with eigenvalues sigma +- i*omega
the step matrix is P(hA) with the real polynomial P of stability.py; A is normal,
so the spectral-norm error of U-CUDA reduces to the scalar formula above.
Methods of order seven and of order eight are compared separately.
Writes ../paper/figures/fig_prefregions.pdf and ../data/prefregions.json.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from figures import ST, lbl, save, DATA, Tm

TOL = 1.0
STAB_FILL, PREF_FILL, PREF_EDGE = "#d9d9d9", "#6cc76c", "#1e7b1e"
XLIM, YLIM = (-7.0, 2.0), (-6.5, 6.5)

def grid(xlim=XLIM, ylim=YLIM, n=(1100, 1500)):
    x = np.linspace(*xlim, n[0]); y = np.linspace(*ylim, n[1])
    X, Y = np.meshgrid(x, y)
    return X, Y, X + 1j * Y, (x[1] - x[0]) * (y[1] - y[0])

def masks(co, Z):
    P = np.polynomial.Polynomial(co)(Z)
    rho = np.abs(P)
    err = np.abs(np.exp(Z) - P) / np.abs(np.exp(Z))
    return rho, err

def areas(co, Z, dA):
    rho, err = masks(co, Z)
    st = rho <= 1
    return {"stab": float(st.sum() * dA), "pref": float((st & (err <= TOL)).sum() * dA)}

# (label, polynomial or None = from stability.json, evaluations per step, style key, stability.json key)
ORDER7 = [("$T_7$: CRK7", Tm(7), 7, "CRK7-opt", None), ("CCK7(6)", None, 8, "CCK7(6)", "CCK7(6)"),
          ("Butcher7", None, 9, "Butcher7", "Butcher7"), ("Verner7(6)", None, 10, "Verner7(6)", "Verner7(6)"),
          ("Vern7", None, 9, "Vern7", "Vern7"), ("RKF7(8)", None, 11, "Fehlberg7", "Fehlberg7")]
ORDER8 = [("CRK8-opt", None, 9, "CRK8-opt", "CRK8-opt"), ("CRK8-s", None, 9, "CRK8-s", "CRK8-s"),
          ("CooperVerner8", None, 11, "CooperVerner8", "CooperVerner8"), ("DOP853", None, 12, "DOP853", "DOP853"),
          ("DP8(7)", None, 13, "DP8(7)", "DP8(7)"), ("Verner8(7)", None, 13, "Verner8(7)", "Verner8(7)"),
          ("RKF8(7)", None, 13, "Fehlberg8", "Fehlberg8")]
# panels drawn in full (the rest only in the per-evaluation comparison)
SHOW7 = ["$T_7$: CRK7", "CCK7(6)", "Vern7"]
SHOW8 = ["CRK8-opt", "CRK8-s", "DOP853"]

def resolve(lst, st):
    out = []
    for lab, co, ev, key, skey in lst:
        if co is None:
            if skey not in st:
                print(f"  {lab}: missing in stability.json")
                continue
            co = st[skey]["coeffs"]
        out.append((lab, co, ev, key))
    return out

def main():
    st = json.load(open(os.path.join(DATA, "stability.json")))
    o7, o8 = resolve(ORDER7, st), resolve(ORDER8, st)
    X, Y, Z, dA = grid()
    res = {}
    for lab, co, ev, key in o7 + o8:
        a = areas(co, Z, dA)
        a.update(evals=ev, stab_per_eval=a["stab"] / ev ** 2, pref_per_eval=a["pref"] / ev ** 2)
        res[lab] = a
        print(f"{lab:26s} stab {a['stab']:6.2f}  pref {a['pref']:6.2f}   per eval: "
              f"{a['stab_per_eval']:.4f} {a['pref_per_eval']:.4f}")
    json.dump(res, open(os.path.join(DATA, "prefregions.json"), "w"), indent=1)

    fig, axs = plt.subplots(2, 4, figsize=(7.4, 4.55))
    def panel(a, lab, co):
        rho, err = masks(co, Z)
        stab = rho <= 1
        a.contourf(X, Y, stab.astype(float), levels=[0.5, 1.5], colors=[STAB_FILL])
        a.contourf(X, Y, (stab & (err <= TOL)).astype(float), levels=[0.5, 1.5], colors=[PREF_FILL])
        a.contour(X, Y, rho, [1], colors=["k"], linewidths=0.6)
        a.contour(X, Y, np.where(stab, err, 10.0), [TOL], colors=[PREF_EDGE], linewidths=0.8)
        a.set_title(lab, fontsize=8)
        a.set_aspect("equal"); a.grid(False)
        a.set_xlim(*XLIM); a.set_ylim(*YLIM)
        a.axhline(0, color="#999999", lw=0.3); a.axvline(0, color="#999999", lw=0.3)
    for r, (lst, show) in enumerate([(o7, SHOW7), (o8, SHOW8)]):
        shown = [t for t in lst if t[0] in show][:3]
        for c, (lab, co, ev, key) in enumerate(shown):
            panel(axs[r, c], f"({'abcdefgh'[4 * r + c]}) {lab}", co)
        for c in range(len(shown), 3):
            axs[r, c].axis("off")
        # per-evaluation comparison: boundaries of the preferred regions in z/s for every method of this order
        a = axs[r, 3]
        for lab, co, ev, key in lst:
            rho, err = masks(co, Z * ev)
            col, _, ls = ST[key]
            a.contour(X, Y, np.where(rho <= 1, err, 10.0), [TOL], colors=[col], linestyles=[ls], linewidths=1.2)
        a.set_xlim(-0.45, 0.08); a.set_ylim(-0.55, 0.55); a.set_aspect("equal")
        a.axhline(0, color="#999999", lw=0.3); a.axvline(0, color="#999999", lw=0.3)
        a.set_title(f"({'abcdefgh'[4 * r + 3]}) per evaluation", fontsize=8)
        handles = [Line2D([], [], color=ST[key][0], ls=ST[key][2], lw=1.2, label=lab.split(":")[0])
                   for lab, co, ev, key in lst]
        a.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=5.8, handlelength=2.0,
                 borderaxespad=0.3, labelspacing=0.25)
        a.set_xlabel(r"Re $z/s$")
        a.set_ylabel(r"Im $z/s$", labelpad=1)
    for a in axs[:, 0]:
        a.set_ylabel(r"Im $z$")
    for a in axs[1, :3]:
        a.set_xlabel(r"Re $z$")
    leg = [Patch(facecolor=STAB_FILL, edgecolor="k", lw=0.6, label=r"stable, $|P(z)|\leq1$"),
           Patch(facecolor=PREF_FILL, edgecolor=PREF_EDGE, lw=0.8, label=r"preferred, tol $=1$")]
    fig.legend(handles=leg, loc="lower center", ncol=2, fontsize=7)
    fig.tight_layout(rect=(0, 0.045, 1, 1), w_pad=0.3, h_pad=0.6)
    save(fig, "fig_prefregions")

if __name__ == "__main__":
    main()
