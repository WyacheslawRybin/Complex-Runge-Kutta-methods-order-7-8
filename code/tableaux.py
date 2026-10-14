"""
All tableaux of the order 7-8 paper, read from their primary sources so that
nothing is retyped by hand.

Complex (real projection), provisional names until the paper fixes them:
  CRK7-B0, CRK7-Bbal, CRK7-Blo   appendix of the draft "CRK7s7 (1).pdf"
                                 (data/crk7s7_draft_tableaux.txt, 30 digits)
  CRK7s7b                        best77_mp.txt of the search archive (40 digits)
  CRK8s9                         crk8s9_mp.txt of the search archive (55 digits)
  pool77[k], k = 0..31           h77.bin   (double, 32 roots of 7 stages / order 7)
  pool98[k], k = 0..12           h98pool.bin (double, 13 roots of 9 stages / order 8)

Real (exact, rk_tableaux.json; expressions with sqrt(21) are evaluated in mp):
  Butcher7 (9 stages), Verner7(6) (1978, 10), Vern7 (Verner 2010, 10),
  Fehlberg7 (RKF7(8) seventh-order formula), CooperVerner8 (11), DP8(7)13M,
  DOP853, Fehlberg8, Verner8(7).
"""
import json, os, re
import numpy as np
import mpmath as mp
import sympy
from crk import Tableau

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, ".."))   # the repository root
ARCH = os.path.join(PAPER, "data", "search_archive")
DRAFT = os.path.join(PAPER, "data", "crk7s7_draft_tableaux.txt")
RK_JSON = os.path.join(PAPER, "data", "rk_tableaux.json")

# ----------------------------------------------------------------- helpers
def _from_flat(x, s, name, p, conv=mp.mpf):
    """Archive layout: [Re a (s(s-1)/2, row-wise) | Re b (s) | Im a | Im b]."""
    na, half = s * (s - 1) // 2, s * (s + 1) // 2
    assert len(x) == 2 * half, (len(x), s)
    A = [[mp.mpc(0)] * s for _ in range(s)]
    k = 0
    for i in range(1, s):
        for j in range(i):
            A[i][j] = mp.mpc(conv(x[k]), conv(x[half + k]))
            k += 1
    b = [mp.mpc(conv(x[na + i]), conv(x[half + na + i])) for i in range(s)]
    return Tableau(name, A, [(s, p, b)], True)

def _num(tok):
    return tok.replace("−", "-").replace(" ", "")

_CPLX = re.compile(r"^([+-]?[\d.]+(?:e[+-]?\d+)?)(?:([+-][\d.]+(?:e[+-]?\d+)?)i)?$")

def _parse_cplx(txt):
    m = _CPLX.match(_num(txt))
    assert m, txt
    return mp.mpc(mp.mpf(m.group(1)), mp.mpf(m.group(2) or "0"))

# ----------------------------------------------------------------- complex
def draft():
    blocks, cur = {}, None
    for ln in open(DRAFT, encoding="utf-8"):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if ln in ("B0", "Bbal", "Blo"):
            cur = blocks.setdefault(ln, {"a": {}, "b": {}})
            continue
        key, val = [u.strip() for u in ln.split("=", 1)]
        if key[0] == "a":
            cur["a"][(int(key[1]), int(key[2]))] = _parse_cplx(val)
        elif key[0] == "b":
            cur["b"][int(key[1])] = _parse_cplx(val)
    out = {}
    for tag, d in blocks.items():
        assert len(d["a"]) == 21 and len(d["b"]) == 7, tag
        A = [[d["a"].get((i + 1, j + 1), mp.mpc(0)) for j in range(7)] for i in range(7)]
        b = [d["b"][i + 1] for i in range(7)]
        nm = "CRK7-" + tag
        out[nm] = Tableau(nm, A, [(7, 7, b)], True)
    return out

def _mp_txt(fn, s, p, name):
    vals = [v.strip() for v in open(os.path.join(ARCH, fn)) if v.strip()]
    return _from_flat(vals, s, name, p)

def archive():
    return {"CRK7s7b": _mp_txt("best77_mp.txt", 7, 7, "CRK7s7b"),
            "CRK8s9": _mp_txt("crk8s9_mp.txt", 9, 8, "CRK8s9")}

def pool(kind):
    fn, s, p = {"77": ("h77.bin", 7, 7), "98": ("h98pool.bin", 9, 8)}[kind]
    X = np.fromfile(os.path.join(ARCH, fn), dtype=np.float64).reshape(-1, s * (s + 1))
    return [_from_flat([float(v) for v in x], s, f"pool{kind}[{k}]", p) for k, x in enumerate(X)]

def _casc_flat(x, S, MH, p, q, name):
    """cascsearch.c layout: [Re a (row-wise) | Re b (S) | Re bh (MH) | Im a | Im b | Im bh]."""
    na = S * (S - 1) // 2
    H = len(x) // 2
    assert H == na + S + MH, (len(x), S, MH)
    z = [mp.mpc(mp.mpf(x[k]), mp.mpf(x[H + k])) for k in range(H)]
    A = [[mp.mpc(0)] * S for _ in range(S)]
    k = 0
    for i in range(1, S):
        for j in range(i):
            A[i][j] = z[k]
            k += 1
    levels = [(S, p, z[na:na + S])]
    if MH:
        levels.append((MH, q, z[na + S:]))
    return Tableau(name, A, levels, True)

def clean_inert(T, tol=1e-12):
    """Stages whose elementary weights are real (to tol): snap Im a_jk of their rows to 0
    and set Im b_j = 0 on every level (inert directions, Prop. 2); returns (T, real stages)."""
    ph = T.phi(max(p for _, p, _ in T.levels))
    real = [j for j in range(T.s) if max(abs(mp.im(v[j])) for v in ph.values()) < tol]
    A = [row[:] for row in T.A]
    for j in real:
        A[j] = [mp.mpc(mp.re(v)) for v in A[j]]
    levels = [(m, p, [mp.mpc(mp.re(v)) if i in real else v for i, v in enumerate(b)])
              for m, p, b in T.levels]
    return Tableau(T.name, A, levels, T.projected, T.cost), real

def casc_hits(fn, S, MH, p, q, prefix, clean=True):
    """All hits of a cascsearch.c output file as Tableaux with levels [(S,p,b), (MH,q,bh)]."""
    X = np.fromfile(fn, dtype=np.float64).reshape(-1, 2 * (S * (S - 1) // 2 + S + MH))
    out = []
    for k, x in enumerate(X):
        T = _casc_flat([float(v) for v in x], S, MH, p, q, f"{prefix}[{k}]")
        out.append(clean_inert(T)[0] if clean else T)
    return out

# ----------------------------------------------------------------- real
def _exact(v):
    """'3*sqrt(21)/56 + 3/8' -> mpf at the working precision."""
    e = sympy.sympify(str(v))
    return mp.mpf(sympy.N(e, mp.mp.dps + 10).__str__())

def _rk_json(key, name, embedded=False, cost=None):
    d = json.load(open(RK_JSON, encoding="utf-8"))[key]
    s = d["stages"]
    A = [[mp.mpc(_exact(d["A_exact"][i][j])) if j < len(d["A_exact"][i]) else mp.mpc(0)
          for j in range(s)] for i in range(s)]
    b = [mp.mpc(_exact(v)) for v in d["b_exact"]]
    m = max(i + 1 for i, v in enumerate(b) if v != 0)       # stages actually used
    levels = [(s if embedded else m, d["order"], b if embedded else b[:m])]
    if embedded and d.get("bhat_exact"):
        levels.append((s, d["embedded_order"], [mp.mpc(_exact(v)) for v in d["bhat_exact"]]))
    if embedded:
        T = Tableau(name, A, levels, projected=False, cost=cost or s)
    else:
        T = Tableau(name, [row[:m] for row in A[:m]], levels, projected=False, cost=cost or m)
    return T

def real_methods():
    return {
        "Butcher7": _rk_json("Butcher7_sqrt21", "Butcher7"),
        "Verner7(6)": _rk_json("Verner7_1978", "Verner7(6)", embedded=True),
        "Vern7": _rk_json("VernerEfficient7", "Vern7", embedded=True),
        "Fehlberg7": _rk_json("Fehlberg7", "Fehlberg7"),
        "CooperVerner8": _rk_json("CooperVerner8", "CooperVerner8"),
        "DP8(7)": _rk_json("DormandPrince8", "DP8(7)", embedded=True),
        "DOP853": _rk_json("DOP853", "DOP853"),
        "Fehlberg8": _rk_json("Fehlberg8", "Fehlberg8"),
        "Verner8(7)": _rk_json("Verner8_1978", "Verner8(7)", embedded=True),
    }

def complex_named():
    T = {}
    T.update(draft()); T.update(archive())
    return T

def load_refined(fn=None):
    """Tableaux after Newton refinement (written by check_tableaux.py)."""
    fn = fn or os.path.join(PAPER, "data", "tableaux.json")
    d = json.load(open(fn))
    out = {}
    for name, info in d.items():
        s = info["s"]
        A = [[mp.mpc(0)] * s for _ in range(s)]
        for i in range(s):
            for j in range(i):
                A[i][j] = mp.mpc(mp.mpf(info["A"][i][j][0]), mp.mpf(info["A"][i][j][1]))
        levels = [(m, p, [mp.mpc(mp.mpf(v[0]), mp.mpf(v[1])) for v in b])
                  for (m, p), b in zip(info["levels"], info["b"])]
        out[name] = Tableau(name, A, levels, info["projected"], cost=info["cost"])
    return out

if __name__ == "__main__":
    mp.mp.dps = 60
    allT = {}
    allT.update(complex_named()); allT.update(real_methods())
    for k, T in allT.items():
        r = max(abs(v) for lev in T.residuals() for v in lev)
        print(f"{k:14s} s={T.s:2d} cost={T.cost:2d} levels={[(m, p) for m, p, _ in T.levels]} "
              f"resid={mp.nstr(r, 3):>9s} E={mp.nstr(T.E(), 6):>12s} maxcoef={mp.nstr(T.maxcoef(), 6)}",
              flush=True)
