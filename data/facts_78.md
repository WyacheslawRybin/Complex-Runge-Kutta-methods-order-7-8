# Verified facts for the order 7-8 paper (2026-10-12)

All tableaux refined at 60 digits (add_tableau.py / check_tableaux.py); numbers from
data/check_named.log, data/check_added.json, data/add_CRK8-opt.log, data/stability.json.

## Condition counts
N(5)=17, N(6)=37, N(7)=85, N(8)=200 (trees of order <= p).
Free real unknowns (Im b_1 dropped): s=7: 55, s=8: 71, s=9: 89, s=10: 109.

## CRK7 (s = p = 7), single level (7,7)
| tableau | E_8 | max|coef| | J rank / free | dim |
|---|---|---|---|---|
| CRK7-opt | 6.112116e-5 | 5.00000 | 48/55 | 7 (mp rank, rank_mp.log; double-precision SVD gave 47/8) |
| CRK7s7b | 9.774699e-5 | 0.85512 (after Im b_1..3 = 0; 0.89344 before) | 48/55 | 7 |
| CRK7-Bbal | 1.560126e-4 | 1.18464 | 48/55 | 7 |
| CRK7-Blo | 1.687857e-4 | 0.96499 | 48/55 | 7 |
| CRK7-B0 | 1.844817e-4 | 1.43258 | 48/55 | 7 |
Structure of every CRK7 tableau: stages 1-3 real, C(2) for i >= 3, Re D(1) (and full D(1)
after the normalization Im b_1..b_3 = 0, RSF identity (4.6)-(4.7)), b_2 = 0, c_3 = 2/7, c_7 = 1.
CRK7-opt refinement needed pinv rcond 1e-14 and 40 iterations (rank loss at b_2 = 0).
E-min front: CRK7-opt at B = 5: 6.1121e-5, at B = 4: 6.1287e-5 (+0.3%).

## CRK8 (s = 9, p = 8), single level (9,8)
| tableau | E_9 | max|coef| | J rank / free | dim | notes |
|---|---|---|---|---|---|
| CRK8-opt | 7.000004e-6 | 5.00000 (|A| <= 1.571) | 79/89 | 10 | bound active (b_8) |
| CRK8-b | 8.132565e-6 | 1.25177 | 79/89 | 10 | interior local minimum |
| CRK8s9 | 1.844812e-5 | 3.91010 | 79/89 | 10 | (earlier draft) |
| RSF-CRK8-9 | 1.067059e-4 | 0.92384 | 79/89 | 10 | RSF report Table 4.7 |
| RSF-CRK8-10 | 2.513846e-3 | 3.38826 | 85/109 | 24 | s = 10 |
Structure of CRK8-opt, CRK8-b, CRK8s9: stages 1-3 real, C(2) for i >= 3, Re D(1), b_2 = 0, c_9 = 1.
E-opt (optimize_E 9,8 bound 5, 23 starts pool98+CRK8s9+RSF-CRK8-9, then polish 3000 it):
CRK8s9 and its neighbours only crawl to 9.08e-6 (flat valley); RSF-CRK8-9 start -> 7.69e-5 (other component).

## s = p = 8: only the confluent limit
User's note CRK8-8_конфлюэнтный_предел (Theorems A, B): infimum of the residual is 0, attained
by CRK(eps) with rho <= C|eps|, M ~ K/|eps|; limit scheme L (7 f-evals + one G = h f'f) has an
exact solution (residual 6.1e-53). Existence of a finite 8-stage tableau is OPEN (not disproved);
RSF section 4.13: 621 + 81 direct runs, none converged, best 1.4e-7.
code/confluent.py reproduces the note: L(8,8) residual 1.1e-16, Jacobian rank 62/68.

## CCK7
7-stage 7(6) pair (7,7)+(6,6): no finite solution. casc77_6 40000 random starts: 0 hits, best 1.05e-5.
LM from the 167 (7,7) roots stalls at 6.7e-7 ... 2e-6 with |a21| ~ 1e-3..1e-2 (stage merging).
Confluent limit L for (7,7)+(6,6): exact solutions (max|r| ~ 1e-16), rank 54/62, dim 8, max|coef| 0.77..6.3.
7(5) on 7 stages: (7,7)+(6,5) best 1.7e-7, (7,7)+(5,5) best 4.1e-7 (same merging); 7(4) exists everywhere.
8-stage 7(6) pair (8,7)+(7,6): 22 hits in 6000 starts (h87_76_s11.bin), refined to ~1e-60
(cck87[18]: 3.4e-16 only), dims 5-15; best raw E 3.64e-5 (cck87[14], max|c| 1.39).
E-opt of the pair (bound 5) running: Emin_87_76_5.json.

## Stability (stability.json, prefregions.json, tol = 1)
P = T_7 for every CRK7 (s = p). CRK8 (s = 9): P = T_8 + c z^9, 9! c = 0.9411 (CRK8-opt),
0.64465 (CRK8-b), 0.49307 (CRK8s9), -5.59589 (RSF-CRK8-9).
Real interval / imag / area-left; per eval:
T_7 (CRK7): [-3.954,0] / 1.764 / 23.22; 0.565
CRK8-opt: [-4.796,0] / 0 / 35.17; 0.533
CRK8-b: [-6.084,0] / 3.999 / 41.35; 0.676
DOP853: [-6.394,0] / 5.960 / 65.69; 0.533 (12 evals)
Preferred area per eval^2: T_7 0.2970, Vern7 0.2662; CRK8-b 0.2871, CRK8-opt 0.2787, DOP853 0.2783.
Bug fixed: imag_interval root finding failed for Verner8(7) (zero top coeff) and DOP853 (4.72 -> 5.96);
the 5-6 paper values were re-checked by direct evaluation and are correct.

## kappa = 1 efficiency s * E^(1/p) (propagated stages)
CRK7-opt 1.75, Vern7 (9 used) 1.87, RKF7 2.15; CCK7(6) (8 stages, E_8 1.95648e-5) 1.70;
CRK8-opt 2.04, DOP853 2.68, DP8(7) 2.79.

## Update 2026-10-13
- Inert normalization Im b_1 = Im b_2 = Im b_3 = 0 applied to ALL complex tableaux (tableaux.json) and
  Im bh_j = 0 on the real stages of the embedded formulas (embedded.json); code/normalize_inert.py,
  check_tableaux.inert_normalized (threshold 1e-40). Changes neither order nor E; changes max|coef|
  of CRK7s7b (0.893 -> 0.855) and max|bh| of CRK7s7b(5) (-> 0.710).
- CCK7(6) final: E_8 = 1.95648e-5, E_7 (level 6) = 1.84836e-4, max|coef| = 5.000 (bound active),
  3 real stages, full cascade 7(6)(5)(4)(3)(2)(1) on the leading stages (cascade_levels.json),
  C = 1.83, E = 0.11 (DP measures), 8!c8 = 1.200.
- Confluent scheme L refined in 90 digits (confluent_mp.py -> confluent_L.json): residual 6.1e-92,
  Jacobian rank 62 of 68 essential unknowns (gap 5.8e-7 -> 4.8e-51), E_9 (sigma-weighted) = 4.185e-5,
  max|coef| = 1.2080 (Im beta_1 = Im beta_G = 0).
- Appendix coefficients: make_appendix.py -> tables/appendix_tableaux.tex (30 digits, |x| < 1e-4 as m*10^e).
- CCK8 searches: (9,8)+(8,7) 8662 starts, best 8.0e-6; (10,8)+(9,7) 480 (t31, best 2.937e-6) + 4828 (r32, interrupted at 80.5%, best 3.10e-6) = 5308 starts.
- Cross-order (variable step, best pair of each order, adaptive_cost): 8 vs 7 cheaper on 6/10 at 1e-6
  (by 1-22%), 8/10 at 1e-8 (7-23%), 9/10 at 1e-10 (2-41%, Brusselator +3%). 7 vs 6 (5-6 numbers.json):
  9/10 at 1e-6 (6-37%, Kepler +15%), 10/10 at 1e-10 (19-55%); 8 vs 6 at 1e-10: 30-65%.
- Fixed-step cross-order at 1e-8 (s C^(1/p)): 8 cheaper on 8/11 nonlinear (2-41%), dearer on exponential,
  VdP, rigid body (3-9%); at 1e-10 cheaper on 10/11 (1-46%), VdP equal.

## Update 2026-10-13 (2): CPU, cost ratios, summary
- bench: data/bench.csv, bench_run2.csv, bench_run3.csv (each best of 5 x 4e5 steps); make_tables.bench() takes
  the median (run-to-run spread median 4.4%, max 23%).
- kappa_CPU (CRK8-opt stage / DOP853 stage): polynomial 1.50-1.69; exp 1.87; nonautonomous 3.25; Kepler 3.16;
  Arenstorf 4.03. complex f / real f: polynomial 0.92-1.54, exp 2.13, trig 8.81, Kepler 6.22, Arenstorf 10.96.
  Step time CRK8-opt/DOP853: polynomial 1.13-1.27, others 1.40-3.02; CRK7-opt/Vern7: 1.08-1.22, 1.34-2.98.
  CRK7s7b, CRK8-b take the time of CRK7-opt, CRK8-opt within 2.6%.
- Stage-time ratios exceed kappa* vs Vern7, DOP853, DP8(7) on all problems (fixed step, CPU).
- Cost ratio at 1e-8 (best complex / best real): fixed kappa=1 order 7 cheaper on 6/9 (8-42%), order 8 on 8/9 (9-42%);
  variable kappa=1 order 7 on 8/10 (19-43%), order 8 on 10/10 (9-32%); CPU: complex wins only VdP fixed 7 (0.91),
  Brusselator fixed 7 (0.96), Brusselator variable 7 (0.93); Bouali fixed 7 = 1.00.
- Summary (geom. means, rel. Vern7 / DOP853): variable kappa=1 CRK7(5) 0.78, CCK7(6) 0.78, CRK8(6) 0.77, CRK8-b(6) 0.77;
  fixed kappa=1 CRK8-opt 0.80, CRK7 family 1.06-1.13, CCK7(6) 1.08; CPU fixed 1.50-2.25, variable 1.50-1.64.

## Update 2026-10-13 (3): CCK7(6) replaced by the front minimum (supersedes the CCK7 values above)
- Old pair (E_8 = 1.95648e-5) data backed up in data/bak_cck7_old/ and tables_bak_cck7_old/; tex backups
  crk78_paper.tex.bak_before_cck7new, .bak_prose_cck7new.
- New CCK7(6) (Emin_87_76_5_final.json): E_8 = 8.699e-6 (smallest of all order-7 methods; 7.0x below CRK7-opt,
  1.9x below Vern7, 21% below RKF7(8)), E_7 = 2.047e-4, max|coef| = 5.000 at a_{8,4}, weights <= 4.60;
  s E^(1/p) = 1.51. c3 = 0.2354, c8 = 0.9813 (real to 1e-59), D(1) fails (max|Re g1| = 0.011), span dim 12.
  Rank 70, smallest kept sv 1.9e-6, 23 decades gap. Kernel on all stages: q=6 1 (FSAL 1), q=5 3 (FSAL 4);
  full cascade, levels m=7, m=6 unique up to inert; min-norm cascade weights resid < 5e-46, max|b| <= 17.7.
- DP measures: C = 2.04, E = 0.04, Ahat = 2.05e-4. 8!c8 = 1.098 (linear error 0.098), real interval
  [-4.1840, 0] (0.523/eval), area/eval 0.428, preferred area/eval 0.272.
- Cascade empirical orders (7,6)..(1,1): 6.03 5.09 4.09 3.00 2.00 2.00 1.00 (y'=-y^2);
  6.29 5.00 4.00 2.82 2.55 2.07 1.00 (non-autonomous; (4,3),(3,2) converge from below/above, N = 1280-20480).
- Tuning (Rossler): constant 2.11e-3 -> 1.13e-3 (1.87x), E_8 x1.31, resid 1.7e-19, conv 0.4%.
  From the old first minimum: 35x, to 2.30e-4.
- Summary: variable kappa=1 CRK7(5) 0.78, CCK7(6) 0.76 (best 4 + 5 = 9/10); fixed kappa=1 CCK7(6) 0.99,
  complex best on 9/11 (Vern7 2); CPU variable 1.50-1.61.
- Cost ratio at 1e-8: fixed order 7 cheaper on 7/9 (4-42%; exp +8%, Rossler +10%); variable order 7 on 9/10
  (2-43%; Lorenz +1%); CPU: complex wins only VdP and Brusselator, fixed, order 7 (0.91, 0.96).
- Cross-order: 8 vs 7 variable 1e-6 cheaper 5/10 (1-19%), dearer 4 (4-31%), Arenstorf equal; 1e-8 9/10 (6-23%,
  non-autonomous +7%); 1e-10 10/10 (1-41%). Fixed 1e-8: 9/11 (1-38%; exp +7%, VdP +9%); 1e-10 10/11 (1-43%).
  7 vs 6: 1e-6 10/10 (6-30%), 1e-10 9-51%; 8 vs 6 at 1e-10: 30-64%.
