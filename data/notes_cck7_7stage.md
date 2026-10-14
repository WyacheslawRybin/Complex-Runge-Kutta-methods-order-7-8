# Seven-stage complex Cash–Karp pairs of order 7 (2026-10-12)

Levels: main (7,7); embedded on the leading m stages with order q.

| embedded level | free − conditions on the (7,7) family (dim 7) | result |
|---|---|---|
| (6,6) | 7 + 11 − 37 = −19 | no finite solution found; the near-misses degenerate (see below) |
| (6,5) | 7 + 11 − 17 = +1 | no finite solution found (600 random starts with casc77_6, LM from the 167 archive roots, LM from unfolded limit schemes); the near-misses degenerate the same way |
| (5,5) | 7 + 9 − 17 = −1 | LM from the 167 archive roots: best 4.1e-7 |
| (6,4), (5,4) | +10, +8 | exist on the whole (7,7) family (weights by least squares, residual 1e-16) |

## Confluent limit of the pair 7(6)

LM on (7,7)+(6,6) from the 167 archive (7,7) roots (bh by least squares, scratch/s77_lm.py) stalls at
max|r| = 6.7e-7 … 2e-6. All 20 best points show the mechanism of the CRK8-8 note
(D:\HDD\Науко\2026\Complex Methods\CRK8-8_конфлюэнтный_предел_доказательство.docx): |a21| = |c2 − c1|
= 1e-3 … 1e-2, the large coefficients are in column 2 (a42, a52), M·|a21| ≈ 0.05.

code/confluent.py implements the limit scheme L (stage 2 → pseudo-stage G = h f'(y)f(y),
Phi_G(t) = [t = [tau]]). It reproduces the note: L(8,8) of its Table 2 has residual 1.1e-16,
Jacobian rank 62; CRK(eps) residuals 8.697e-5 (eps = 1e-2) and 2.526e-6 (eps = 1e-2 i).

L for the pair (7,7)+(6,6): the 20 near-misses mapped by (8) of the note have L-residuals
7.7e-6 … 8e-3; LM in L converges for 6 of them to max|r| = 8e-17 … 2.2e-16 (scratch/L77_66.npy),
Jacobian rank 54 of 62 (dimension 8), max|coef| 0.77 … 6.3, alpha_i ~ 1e-2, beta_G ~ 1e-3.

Conclusion so far: a 7-stage 7(6) Cash–Karp pair exists in the confluent limit (6 evaluations of f
+ one f'f product); finite 7-stage tableaux only approach it with residual O(eps) and
coefficients O(1/eps). The smallest finite CCK7 pair found is the 8-stage 7(6) (h87_76_s11.bin,
22 hits in 6000 starts), which also carries the full cascade 7(6)(5)(4)(3)(2)(1).
casc77_6, 40000 random starts for (7,7)+(6,6) (h77_66_s201.bin, 6951 s): 0 hits, best max|r| 1.05e-5;
final max|r| by decade 1e-5:1889 1e-4:13072 1e-3:8177 1e-2:2218 1e-1:98 1e+0:36. No finite 7-stage 7(6) pair.
