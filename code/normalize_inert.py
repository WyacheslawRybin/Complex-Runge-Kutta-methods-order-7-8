"""
One-off: set Im b_j = 0 on the leading real stages of every level in ../data/tableaux.json and
Im bh_j = 0 on the same stages in ../data/embedded.json (Proposition 2: these parts are inert,
so neither the projected order, the error constants nor the projected solutions change).
check_tableaux.py applies the same normalization when it rewrites tableaux.json.
    python normalize_inert.py
"""
import json, os, shutil
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

def real_stages(t):
    m = 1
    while m < t["s"] and all(abs(mp.mpf(t["A"][m][j][1])) < mp.mpf(10) ** -40 for j in range(m)):
        m += 1
    return m

def main():
    ft, fe = os.path.join(DATA, "tableaux.json"), os.path.join(DATA, "embedded.json")
    T, E = json.load(open(ft)), json.load(open(fe))
    for fn in (ft, fe):
        shutil.copy(fn, fn + ".bak_inert")
    mreal = {}
    for k, t in T.items():
        m = mreal[k] = real_stages(t)
        for b in t["b"]:
            for i in range(min(m, len(b))):
                if mp.mpf(b[i][1]) != 0:
                    print(f"{k}: Im b_{i + 1} = {mp.nstr(mp.mpf(b[i][1]), 6)} -> 0")
                    b[i][1] = "0.0"
    for k, d in E.items():
        if not isinstance(d, dict) or d.get("base") not in T:
            continue
        m = mreal[d["base"]]
        for i in range(m):
            if mp.mpf(d["bh"][i][1]) != 0:
                print(f"{k}: Im bh_{i + 1} = {mp.nstr(mp.mpf(d['bh'][i][1]), 6)} -> 0")
                d["bh"][i][1] = "0.0"
        d["maxbh"] = float(max(abs(mp.mpc(mp.mpf(a), mp.mpf(b))) for a, b in d["bh"]))
        print(f"{k}: max|bh| = {d['maxbh']:.4f}")
    json.dump(T, open(ft, "w"), indent=1)
    json.dump(E, open(fe, "w"), indent=1)

if __name__ == "__main__":
    main()
