"""
Behaviour of the error-constant descent without an effective bound (B = 1e3), continued from the point of a
front chain with the smallest E: E, max|coef| (normalized), the smallest distance of two abscissae, the
largest |a_ij| and |b_j| and the indices of the largest coefficient, every block of iterations.
Usage:  python drift_large_B.py FAMILY BLOCKS [ITERS_PER_BLOCK]
Writes ../data/drift_<FAMILY>.json.
"""
import json, os, sys, time
import numpy as np
import front_E as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

def start(fam):
    pts = []
    for fn in ("front_E.json", "front_E_CCK7.json"):
        if os.path.exists(os.path.join(DATA, fn)):
            for ch in json.load(open(os.path.join(DATA, fn))):
                if ch["family"] == fam:
                    pts += [(p["E"], ch["levels"], p["x"], ch["chain"], p["B"]) for p in ch["points"]
                            if p["E"] is not None and p["B"] >= 5]
    return min(pts, key=lambda t: t[0])

def where(Fm, x):
    A, W = Fm.P.unpack(np.array(x)[None])
    A = A[0]
    i, j = np.unravel_index(np.argmax(np.abs(A)), A.shape)
    L, k = max(((L, k) for L in range(len(W)) for k in range(W[L].shape[1])), key=lambda t: abs(W[t[0]][0][t[1]]))
    c = A.sum(1)
    S = len(c)
    g = min(((abs(c[a] - c[b]), a, b) for a in range(S) for b in range(a)))
    return dict(argA=[int(i) + 1, int(j) + 1], argb=[L, k + 1], gap_pair=[g[2] + 1, g[1] + 1],
                c=[[float(v.real), float(v.imag)] for v in c])

if __name__ == "__main__":
    fam, blocks = sys.argv[1], int(sys.argv[2])
    its = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    E0, levels, x, chain, B = start(fam)
    Fm = F.family(levels)
    x = np.array(x)
    print(f"{fam}: start {chain} @B{B:g}, E {E0:.6e}", flush=True)
    out = []
    t0 = time.time()
    for k in range(blocks):
        x, E, it = Fm.minimize(x, 1e3, rho=10.0, iters=its)
        rec = dict(block=k, iters=it, E=E, Mn=F.norm_maxcoef(Fm, x), resid=float(np.abs(Fm.residual(x)).max()))
        rec.update({q: v for q, v in F.info(Fm, x).items() if q != "c"})
        rec.update(where(Fm, x))
        out.append(rec)
        print(f"[{time.time() - t0:6.0f}s] {k:3d} E {E:.7e} Mn {rec['Mn']:8.3f} gap {rec['gap']:.4f} {rec['gap_pair']} "
              f"maxA {rec['maxA']:.3f} {rec['argA']} maxb {rec['maxb']:.3f} {rec['argb']} it {it}", flush=True)
        json.dump(dict(start=chain, B=B, E0=E0, blocks=out, x=x.tolist()),
                  open(os.path.join(DATA, f"drift_{fam}.json"), "w"), indent=1)
        if it < its - 1:
            break
