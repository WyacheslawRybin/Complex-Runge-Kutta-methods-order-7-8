"""
Repeated polishing of the minima of optimize_E.py under a bound: every round starts from the K best results
of the previous round and random tangent displacements of them, until the best E improves by less than TOL.
Usage:  python polish_rounds.py LEVELS BOUND FIRST_JSON TAG [ROUNDS] [K] [NRANDOM] [NPROC]
Writes ../data/Emin_<TAG>_r<k>.json for every round.
"""
import json, os, sys
import optimize_E as O

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

if __name__ == "__main__":
    levels, bound, first, tag = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
    rounds = int(sys.argv[5]) if len(sys.argv) > 5 else 6
    K = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    nrand = int(sys.argv[7]) if len(sys.argv) > 7 else 8
    nproc = int(sys.argv[8]) if len(sys.argv) > 8 else 6
    TOL = 0.005
    src = os.path.join(DATA, first)
    best = min(r["E"] for r in json.load(open(src))["results"])
    for k in range(1, rounds + 1):
        res = json.load(open(src))["results"]
        top = sorted(range(len(res)), key=lambda i: res[i]["E"])[:K]
        spec = "json:" + src + "#" + "+".join(str(i) for i in top)
        O.main(levels, bound, spec, nrandom=nrand, nproc=nproc, iters=1500, out_tag=f"{tag}_r{k}")
        src = os.path.join(DATA, f"Emin_{tag}_r{k}.json")
        new = min(r["E"] for r in json.load(open(src))["results"])
        print(f"round {k}: best E {new:.6e} (previous {best:.6e})", flush=True)
        if new > best * (1 - TOL):
            break
        best = new
