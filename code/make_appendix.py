"""Appendix with all coefficients (refined, 30 significant digits) -> ../paper/tables/appendix_tableaux.tex"""
import json, os, sys
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tableaux import load_refined

DATA = os.path.join(HERE, "..", "data")
TAB = os.path.join(HERE, "..", "paper", "tables")
D = 30

def fmt(x):
    """30 significant digits, mantissa and power of ten for |x| < 1e-4."""
    if x != 0 and abs(x) < mp.mpf(10) ** -4:
        e = int(mp.floor(mp.log10(abs(x))))
        return mp.nstr(x / mp.mpf(10) ** e, D, strip_zeros=False) + rf"\cdot10^{{{e}}}"
    return mp.nstr(x, D, min_fixed=-6, max_fixed=6, strip_zeros=False)

def num(z, real_only=False):
    re_v, im_v = mp.re(z), mp.im(z)
    r = ("+" if re_v >= 0 else "-") + fmt(abs(re_v))
    if real_only or abs(im_v) < mp.mpf(10) ** -40:
        return r
    return r + (" + " if im_v >= 0 else " - ") + fmt(abs(im_v)) + r"\,\mathrm{i}"

def tex(x, d=2):
    x = mp.mpf(x)
    if x == 0:
        return "0"
    e = int(mp.floor(mp.log10(abs(x))))
    return rf"{mp.nstr(x / mp.mpf(10) ** e, d, strip_zeros=False)}\cdot10^{{{e}}}"

def zero(z):
    return abs(z) < mp.mpf(10) ** -40

def block(T, title, wlabels=None, extra_b=None):
    L = [rf"\subsubsection*{{{title}}}", note(T), r"{\small\allowdisplaybreaks", r"\begin{align*}"]
    s = T.s
    for i in range(s):
        for j in range(i):
            if not zero(T.A[i][j]):
                L.append(rf"a_{{{i + 1},{j + 1}}} &= {num(T.A[i][j])}\\")
    L[-1] += r"[1mm]"
    for lev, (m, p, b) in enumerate(T.levels):
        sup = "" if wlabels is None else f"^{{({wlabels[lev]})}}"
        for i, v in enumerate(b):
            if not zero(v):
                L.append(rf"b{sup}_{{{i + 1}}} &= {num(v)}\\")
        L[-1] += r"[1mm]"
    if extra_b:
        lab, bh = extra_b
        for i, v in enumerate(bh):
            if not zero(v):
                L.append(rf"{lab}_{{{i + 1}}} &= {num(v)}\\")
        L[-1] += r"[1mm]"
    for i in range(1, s):
        L.append(rf"c_{{{i + 1}}} &= {num(mp.fsum(T.A[i][j] for j in range(i)))}\\")
    L[-1] = L[-1].rstrip("\\")
    L += [r"\end{align*}", "}", ""]
    return "\n".join(L)

def note(T):
    Es = "$, $".join(rf"E_{{{p + 1}}}={tex(T.E(L), 6)}" for L, (m, p, _) in enumerate(T.levels))
    r = max(abs(v) for lev in T.residuals() for v in lev)
    return rf"${Es}$, $\max|\text{{coef}}|={mp.nstr(T.maxcoef(), 6, strip_zeros=False)}$, residual of~\eqref{{eq:relaxed}} ${tex(r)}$."

def emb(E, name):
    return [mp.mpc(mp.mpf(a), mp.mpf(b)) for a, b in E[name]["bh"]]

def l_block():
    d = json.load(open(os.path.join(DATA, "confluent_L.json"), encoding="utf-8"))
    c = {k: mp.mpc(mp.mpf(a), mp.mpf(b)) for k, (a, b) in d["coef"].items()}
    L = [r"\subsubsection*{The confluent limit scheme $L$ of~\eqref{eq:Lscheme}}",
         rf"Order eight, $70$ real parameters, Jacobian rank ${d['rank']}$ of ${d['n']}$ essential unknowns, "
         rf"$\|e^{{(9)}}\|_2={tex(d['E9'], 3)}$, $\max|\text{{coef}}|={d['maxcoef']:.6f}$, residual of the $200$ "
         rf"projected conditions ${tex(d['resid'])}$. The values of the note of Section~\ref{{sec:confluent}}, "
         r"refined in $90$-digit arithmetic; $\Imag\beta_1=\Imag\beta_G=0$ (inert).",
         r"{\small\allowdisplaybreaks", r"\begin{align*}"]
    for i in range(3, 9):
        L.append(rf"\gamma_{{{i}}} &= {num(c[f'γ_{i}'])}\\")
        L.append(rf"\alpha_{{{i}}} &= {num(c[f'α_{i}'])}\\")
        for j in range(3, i):
            L.append(rf"a_{{{i},{j}}} &= {num(c[f'a_{i}{j}'])}\\")
    L[-1] += r"[1mm]"
    L.append(rf"\beta_1 &= {num(c['β_1'])}\\")
    L.append(rf"\beta_G &= {num(c['β_G'])}\\")
    for i in range(3, 9):
        L.append(rf"b_{{{i}}} &= {num(c[f'b_{i}'])}\\")
    L[-1] = L[-1].rstrip("\\")
    L += [r"\end{align*}", "}", ""]
    return "\n".join(L)

def main():
    mp.mp.dps = 50
    T = load_refined()
    E = json.load(open(os.path.join(DATA, "embedded.json")))
    out = [r"\subsection*{Order seven}", ""]
    out.append(block(T["CRK7-opt"], r"CRK7-opt --- minimum of $E_8$ under $\max|\text{coef}|\le5$",
                     extra_b=(r"\hat b", emb(E, "CRK7(5)"))))
    out.append(r"The weights $\hat b$ give the embedded fifth-order formula of the pair CRK7(5) (Section~\ref{sec:embedded}).")
    out.append(block(T["CRK7-s"], r"CRK7-s --- interior local minimum of $E_8$ with small coefficients",
                     extra_b=(r"\hat b", emb(E, "CRK7-s(5)"))))
    out.append(r"The weights $\hat b$ give the embedded fifth-order formula of the pair CRK7-s(5).")
    out.append(block(T["CCK7(6)"], r"CCK7(6) --- complex Cash--Karp pair $7(6)$; $b^{(7)}$ on all eight stages, $b^{(6)}$ on the first seven",
                     wlabels=["7", "6"]))
    out += [r"\subsection*{Order eight}", ""]
    out.append(block(T["CRK8-opt"], r"CRK8-opt --- minimum of $E_9$ (interior)",
                     extra_b=(r"\hat b", emb(E, "CRK8(6)"))))
    out.append(r"The weights $\hat b$ give the embedded sixth-order formula of the pair CRK8(6) (Section~\ref{sec:embedded}).")
    out.append(block(T["CRK8-s"], r"CRK8-s --- minimum of $E_9$ under $\max|\text{coef}|\le1$",
                     extra_b=(r"\hat b", emb(E, "CRK8-s(6)"))))
    out.append(r"The weights $\hat b$ give the embedded sixth-order formula of the pair CRK8-s(6).")
    out.append(l_block())
    fn = os.path.join(TAB, "appendix_tableaux.tex")
    open(fn, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("wrote", fn, sum(len(x) for x in out), "chars")

if __name__ == "__main__":
    main()
