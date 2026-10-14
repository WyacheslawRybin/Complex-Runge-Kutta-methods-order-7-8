"""
CRK8-9 and CRK8-10 of the RSF report (Раздел_4_продолжение_CRK7_CRK8.docx, tables 4.7 and 4.8,
20 digits; text dump in ../data/rsf_section4.txt): parsed, refined at 60 digits and stored in
tableaux.json under the keys RSF-CRK8-9 and RSF-CRK8-10 (provisional names).
"""
import os, re, sys
import mpmath as mp
from crk import Tableau
import add_tableau as AT
import check_tableaux as CT

HERE = os.path.dirname(os.path.abspath(__file__))
mp.mp.dps = CT.DPS
lines = open(os.path.join(HERE, "..", "data", "rsf_section4.txt"), encoding="utf-8").read().split("\n")
tabs, cur = [], None
for ln in lines:
    m = re.match(r"(a|b)_(\d+) \| (\S+) \| (\S+)$", ln.strip())
    if m is None:
        if cur:
            tabs.append(cur); cur = None
        continue
    cur = cur or {"a": {}, "b": {}}
    cur[m.group(1)][m.group(2)] = mp.mpc(mp.mpf(m.group(3)), mp.mpf(m.group(4)))
for name, d in zip(("RSF-CRK8-9", "RSF-CRK8-10"), tabs[:2]):
    s = len(d["b"])
    A = [[mp.mpc(0)] * s for _ in range(s)]
    for k, v in d["a"].items():
        i, j = (int(k[:-1]), int(k[-1])) if len(k) == 2 or s < 10 else ((10, int(k[2:])) if k.startswith("10") else (int(k[0]), int(k[1:])))
        A[i - 1][j - 1] = v
    b = [d["b"][str(i)] for i in range(1, s + 1)]
    T = Tableau(name, A, [(s, 8, b)], True)
    print(name, "s =", s, "printed residual", mp.nstr(max(abs(v) for lev in T.residuals() for v in lev), 3))
    AT.add(name, T)
