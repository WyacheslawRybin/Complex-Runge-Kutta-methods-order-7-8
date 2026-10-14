"""
Verify the hits of cascsearch.c (complex Cash-Karp tableaux) in multiple precision.

For every hit: inert directions removed (Im b_j on real stages), minimum-norm Newton
refinement of all levels jointly at 60 digits, then
  - residuals of both levels, true orders (E of the next order is nonzero),
  - local dimension of the solution set (Jacobian rank at two difference steps),
    with the number of inert directions it contains,
  - nondegeneracy: |b_S| and max|Re(bh - b[:MH])| (the estimate must not vanish),
  - real stages, nodes, max|coef|, E_{p+1} of the main and E_{q+1} of the embedded level.
Usage:  python check_casc.py HITFILE S MH p q PREFIX [NPROC]
Writes ../data/casc_<PREFIX>.json
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
import mpmath as mp
import crk
import tableaux as TB
import check_tableaux as CT

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

def analyse(T):
    mp.mp.dps = CT.DPS
    t0 = time.time()
    T, real = TB.clean_inert(T)
    r0 = max(abs(v) for lev in T.residuals() for v in lev)
    R, shift, r1 = T.refine(iters=14)
    R, real = TB.clean_inert(R, tol=mp.mpf(10) ** -40)
    (S, p, b), (MH, q, bh) = R.levels[0], R.levels[1]
    rk, n, gap, stable, sv = CT.jac_rank(R)
    inert = sum(1 for j in real if j > 0 and j < S) + sum(1 for j in real if j > 0 and j < MH)
    c = R.c
    d = dict(name=R.name, S=S, MH=MH, p=p, q=q, resid_start=float(r0), resid=float(r1), shift=float(shift),
             E_main=float(R.E(0)), E_emb=float(R.E(1)), maxcoef=float(R.maxcoef()),
             b_last=float(abs(b[S - 1])), dRe=float(max(abs(mp.re(bh[i] - b[i])) for i in range(MH))),
             real_stages=real, c=[[float(mp.re(v)), float(mp.im(v))] for v in c],
             jac_rank=rk, n_free=n, dim=n - rk, inert=inert, gap=float(gap), stable=stable,
             structure=CT.structure(R),
             A=[[CT.cstr(R.A[i][j]) for j in range(i)] for i in range(R.s)],
             b=[[CT.cstr(v) for v in bb] for _, _, bb in R.levels], time=time.time() - t0)
    return d

def line(d):
    cs = " ".join(f"{re:.3f}{im:+.3f}i" if abs(im) > 1e-30 else f"{re:.3f}" for re, im in d["c"])
    return (f"{d['name']:14s} resid {d['resid_start']:.1e}->{d['resid']:.1e}  dim {d['dim']:2d} (J {d['jac_rank']}/{d['n_free']}, "
            f"inert {d['inert']}, gap {d['gap']:.0e}{'' if d['stable'] else ' UNSTABLE'})  E {d['E_main']:.4e} / {d['E_emb']:.4e}  "
            f"max|c| {d['maxcoef']:.3f}  |b_S| {d['b_last']:.3f}  dRe {d['dRe']:.3f}  real {d['real_stages']}  c: {cs}  [{d['time']:.0f}s]")

def main(fn, S, MH, p, q, prefix, nproc=4):
    mp.mp.dps = CT.DPS
    Ts = TB.casc_hits(fn, S, MH, p, q, prefix)
    print(f"{fn}: {len(Ts)} hits", flush=True)
    out = []
    with Pool(nproc) as pool:
        for d in pool.imap(analyse, Ts):
            print(line(d), flush=True)
            out.append(d)
    json.dump(out, open(os.path.join(DATA, f"casc_{prefix}.json"), "w"), indent=1)

if __name__ == "__main__":
    a = sys.argv
    main(a[1], int(a[2]), int(a[3]), int(a[4]), int(a[5]), a[6], int(a[7]) if len(a) > 7 else 4)
