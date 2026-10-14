"""
Report on the fronts E_min(B) of data/front_E*.json: for every bound B the smallest E over all points of all
chains and over the B = 5 multistart minima (data/Emin_*.json) whose normalized max|coef| (inert Im b_1..3 = 0)
does not exceed B, the chain that attains it, and the large-B points of the chains (gap, max|A|, max|b|).
Usage:  python front_report.py [FAMILY ...]
"""
import json, os, sys
import numpy as np
import front_E as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FAM = {"CRK7": ("7,7", ["Emin_77_5.json", "Emin_77_5_polish.json"]),
       "CRK8": ("9,8", ["Emin_98_5.json", "Emin_98_5_polish.json"]),
       "CCK7": ("8,7/7,6", ["Emin_87_76_5.json", "Emin_87_76_5_front.json", "Emin_87_76_5_front2.json",
                            "Emin_87_76_5_pol_r1.json", "Emin_87_76_5_final.json"])}

def points(fam):
    levels, files = FAM[fam]
    Fm = F.family(levels)
    pts = []
    for fn in ("front_E.json", "front_E_CCK7.json"):
        if not os.path.exists(os.path.join(DATA, fn)):
            continue
        for ch in json.load(open(os.path.join(DATA, fn))):
            if ch["family"] != fam:
                continue
            for p in ch["points"]:
                if p["E"] is not None:
                    mn = p.get("maxcoef_n") or F.norm_maxcoef(Fm, p["x"])
                    pts.append(dict(src=ch["chain"], B=p["B"], E=p["E"], M=p["maxcoef"], Mn=mn, gap=p["gap"],
                                    maxA=p["maxA"], maxb=p["maxb"], it=p["iters"]))
    for fn in files:
        for r in json.load(open(os.path.join(DATA, fn)))["results"]:
            if r.get("E") and r["resid"] < 1e-10:
                x = np.array(r["x"])
                inf = F.info(Fm, x)
                pts.append(dict(src=f"{fn}:{r['start']}", B=5.0, E=r["E"], M=r["maxcoef"], Mn=F.norm_maxcoef(Fm, x),
                                gap=inf["gap"], maxA=inf["maxA"], maxb=inf["maxb"], it=r["iters"]))
    return pts

def report(fam):
    pts = points(fam)
    print(f"== {fam}: {len(pts)} points")
    grid = sorted(set(F.BOUNDS))
    for b in grid:
        ok = [p for p in pts if p["Mn"] <= b * 1.001]
        if not ok:
            print(f"B {b:7g}  none (smallest normalized max|coef| {min(p['Mn'] for p in pts):.4f})")
            continue
        p = min(ok, key=lambda q: q["E"])
        print(f"B {b:7g}  E_min {p['E']:.5e}  Mn {p['Mn']:.4f}  gap {p['gap']:.3g}  maxA {p['maxA']:.3g}  "
              f"maxb {p['maxb']:.3g}  it {p['it']}  [{p['src']} @B{p['B']:g}]")
    print("-- chain points at B >= 100:")
    for p in pts:
        if p["B"] >= 100 and p["src"].startswith(("down", "up")):
            print(f"   {p['src']:52s} B {p['B']:6g} E {p['E']:.6e} Mn {p['Mn']:8.3f} gap {p['gap']:.3g} "
                  f"maxA {p['maxA']:.3g} maxb {p['maxb']:.3g} it {p['it']}")

if __name__ == "__main__":
    for fam in sys.argv[1:] or list(FAM):
        report(fam)
