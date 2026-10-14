"""
Measured constants of the front candidates (data/measured_front.json) against the tableaux of the paper
(data/measured.json): C per nonlinear problem and the geometric mean of the ratios over the problems on
which both constants converged (class ok or approx of make_tables.classify).
Usage:  python compare_front.py
"""
import json, math, os
import make_tables as MT

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
PAIRS = [("CRK7-L", "CRK7-opt"), ("CRK7-L", "CRK7s7b"), ("CRK7-L", "Vern7"),
         ("CCK7-B2", "CCK7(6)"), ("CCK7-B5", "CCK7(6)"), ("CCK7-L", "CCK7(6)"),
         ("CCK7-B5", "CRK7-opt"), ("CCK7-B5", "Vern7"), ("CCK7(6)", "Vern7")]

def consts():
    M = json.load(open(os.path.join(DATA, "measured.json")))
    F = json.load(open(os.path.join(DATA, "measured_front.json")))
    out = {}
    for src in (M, F):
        for m, cells in src.items():
            for q, cell in cells.items():
                c = MT.classify(cell["rows"], 7 if m.startswith(("CRK7", "CCK7", "Butcher7", "Verner7", "Vern7", "Fehlberg7")) else 8)
                out.setdefault(m, {})[q] = c
    return out

if __name__ == "__main__":
    C = consts()
    names = ["CRK7-opt", "CRK7s7b", "CRK7-L", "CCK7(6)", "CCK7-B2", "CCK7-B5", "CCK7-L", "Vern7"]
    print(f"{'problem':14s}" + "".join(f"{n:>11s}" for n in names))
    for q in MT.NONLIN:
        print(f"{q:14s}" + "".join(f"{C[n][q]['C']:11.3e}" + ("" if C[n][q]["cls"] == "ok" else "*") if q in C.get(n, {}) else f"{'-':>11s}"
                                   for n in names))
    print()
    for a, b in PAIRS:
        r = [C[a][q]["C"] / C[b][q]["C"] for q in MT.NONLIN
             if q in C.get(a, {}) and q in C.get(b, {}) and C[a][q]["cls"] != "deg" and C[b][q]["cls"] != "deg"]
        if r:
            g = math.exp(sum(math.log(v) for v in r) / len(r))
            print(f"{a:8s} / {b:9s}: geometric mean {g:.3f} over {len(r)} problems, smaller on {sum(v < 1 for v in r)}, "
                  f"range {min(r):.3f}-{max(r):.3f}")
