"""
Seeds for cascsearch.c: the leading block of a cascade is taken from known roots of the
smaller problem, the embedded weights bh are the roots' own weights (so the embedded
level holds exactly at the seed), and the new last row a_S. and the new weights b are
random (b = bh padded with a random b_S, plus noise).

    python make_seeds.py SRC S_src NREP S_tgt MH_tgt OUT [sigma_row sigma_b seed]

SRC is a raw double file in the archive layout [Re a | Re b | Im a | Im b] of S_src stages
(seeds87.bin, h77.bin, hits of cascsearch.c with MH = 0); NREP random completions per root.
"""
import sys
import numpy as np

def main(src, s0, nrep, S, MH, out, sig_row=0.5, sig_b=0.2, seed=0):
    rng = np.random.default_rng(seed)
    na0 = s0 * (s0 - 1) // 2
    X = np.fromfile(src).reshape(-1, 2 * (na0 + s0))
    na = S * (S - 1) // 2
    H = na + S + MH
    assert MH == s0, "the embedded level is the source root itself"
    rows = []
    for x in X:
        H0 = na0 + s0
        a0 = x[:na0] + 1j * x[H0:H0 + na0]
        b0 = x[na0:H0] + 1j * x[H0 + na0:]
        for _ in range(nrep):
            a = np.zeros(na, complex)
            a[:na0] = a0                                   # rows 2..s0 (row-wise layout is a prefix)
            k = na0
            for i in range(s0, S):                         # new rows
                ci = rng.uniform(0.2, 1.0) + 0.3j * rng.standard_normal()
                r = sig_row * (rng.standard_normal(i) + 1j * rng.standard_normal(i))
                r += (ci - r.sum()) / i                    # row sum = node c_i
                a[k:k + i] = r
                k += i
            b = np.zeros(S, complex)
            b[:s0] = b0 + sig_b * (rng.standard_normal(s0) + 1j * rng.standard_normal(s0))
            b[s0:] = sig_b * (rng.standard_normal(S - s0) + 1j * rng.standard_normal(S - s0))
            z = np.concatenate([a, b, b0])
            rows.append(np.concatenate([z.real, z.imag]))
    Y = np.array(rows)
    assert Y.shape[1] == 2 * H
    Y.tofile(out)
    print(f"{out}: {len(Y)} seeds of length {Y.shape[1]} from {len(X)} roots")

if __name__ == "__main__":
    a = sys.argv
    main(a[1], int(a[2]), int(a[3]), int(a[4]), int(a[5]), a[6],
         *(float(v) for v in a[7:9]), *(int(v) for v in a[9:10]))
