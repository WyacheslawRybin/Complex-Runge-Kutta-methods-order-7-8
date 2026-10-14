"""
Core algebra for explicit Runge-Kutta tableaux with complex coefficients and
real projection, with an optional cascade of embedded levels (Cash-Karp type).

A tableau is a strictly lower triangular complex matrix A (s x s) and a list of
levels (m, p, b): level uses the first m stages, weights b in C^m, and has order p
in the projected sense  Re(b^T Phi_t[:m]) = 1/gamma(t)  for rho(t) <= p.
The first level is the propagated solution.

Everything here works in mpmath; numpy is used only for linear algebra in the
Newton refinement (the Jacobian is formed in mp and converted once per step).
"""
import mpmath as mp
import numpy as np
from fractions import Fraction
from math import factorial

mp.mp.dps = 50

# --------------------------------------------------------------------- trees
def _order(t):
    return 1 + sum(_order(u) for u in t)

def _trees_by_order(pmax):
    """Rooted trees as sorted nested tuples; () is the single node."""
    by = {1: [()]}
    for n in range(2, pmax + 1):
        pool = [t for k in range(1, n) for t in by[k]]
        out, seen = [], set()
        def ms(total, start):
            if total == 0:
                yield []
                return
            for i in range(start, len(pool)):
                w = _order(pool[i])
                if w <= total:
                    for rest in ms(total - w, i):
                        yield [pool[i]] + rest
        for m in ms(n - 1, 0):
            key = tuple(sorted(m, key=repr))
            if key not in seen:
                seen.add(key)
                out.append(key)
        by[n] = out
    return by

TREES = _trees_by_order(10)
assert [len(TREES[k]) for k in range(1, 10)] == [1, 1, 2, 4, 9, 20, 48, 115, 286]

def order(t):
    return _order(t)

def gamma(t):
    g = _order(t)
    for u in t:
        g *= gamma(u)
    return g

def sigma(t):
    out = 1
    for u in set(t):
        k = t.count(u)
        out *= sigma(u) ** k * factorial(k)
    return out

def trees_upto(p):
    return [t for k in range(1, p + 1) for t in TREES[k]]

# --------------------------------------------------------------------- numbers
def to_mpc(v):
    """Accepts mpc, (re, im) of strings/numbers, Fraction, str, number."""
    if isinstance(v, mp.mpc):
        return v
    if isinstance(v, tuple):
        return mp.mpc(to_mpf(v[0]), to_mpf(v[1]))
    return mp.mpc(to_mpf(v))

def to_mpf(v):
    if isinstance(v, Fraction):
        return mp.mpf(v.numerator) / v.denominator
    if isinstance(v, str) and "/" in v:
        a, b = v.split("/")
        return mp.mpf(int(a)) / int(b)
    return mp.mpf(v)

# --------------------------------------------------------------------- tableau
class Tableau:
    """name, A (list of lists of mpc), levels [(m, p, [mpc]*m)], cost = evaluations per step."""

    def __init__(self, name, A, levels, projected=True, cost=None, written=None):
        self.name = name
        self.s = len(A)
        self.A = [[to_mpc(A[i][j]) if j < i else mp.mpc(0) for j in range(self.s)] for i in range(self.s)]
        self.levels = [(m, p, [to_mpc(v) for v in b]) for (m, p, b) in levels]
        self.projected = projected
        self.cost = cost if cost is not None else self.s
        self.c = [mp.fsum(self.A[i][:i]) if i else mp.mpc(0) for i in range(self.s)]

    @property
    def p(self):
        return self.levels[0][1]

    def level(self, L=0):
        m, p, b = self.levels[L]
        sub = Tableau(f"{self.name}[{m},{p}]", [row[:m] for row in self.A[:m]], [(m, p, b)],
                      self.projected, cost=m)
        return sub

    # elementary weights for all trees up to order pmax
    def phi(self, pmax):
        s, A = self.s, self.A
        cache, acache = {}, {}
        def APhi(u):
            if u not in acache:
                pu = Phi(u)
                acache[u] = [mp.fsum(A[i][j] * pu[j] for j in range(i)) for i in range(s)]
            return acache[u]
        def Phi(t):
            if t in cache:
                return cache[t]
            v = [mp.mpc(1)] * s
            for u in t:
                w = APhi(u)
                v = [v[i] * w[i] for i in range(s)]
            cache[t] = v
            return v
        for t in trees_upto(pmax):
            Phi(t)
        return cache

    def residuals(self):
        """List over levels of the residuals Re(b^T Phi_t) - 1/gamma(t), rho(t) <= p."""
        pmax = max(p for _, p, _ in self.levels)
        ph = self.phi(pmax)
        out = []
        for m, p, b in self.levels:
            out.append([mp.re(mp.fsum(b[i] * ph[t][i] for i in range(m))) - mp.mpf(1) / gamma(t)
                        for t in trees_upto(p)])
        return out

    def full_complex_residual(self, L=0):
        """max |b^T Phi_t - 1/gamma(t)| without projection (for the 'barrier over C' column)."""
        m, p, b = self.levels[L]
        ph = self.phi(p)
        return max(abs(mp.fsum(b[i] * ph[t][i] for i in range(m)) - mp.mpf(1) / gamma(t)) for t in trees_upto(p))

    def error_coefficients(self, L=0, extra=1):
        """e(t) = (Re(b^T Phi_t) - 1/gamma(t))/sigma(t) over trees of order p+extra."""
        m, p, b = self.levels[L]
        ph = self.phi(p + extra)
        return [(mp.re(mp.fsum(b[i] * ph[t][i] for i in range(m))) - mp.mpf(1) / gamma(t)) / sigma(t)
                for t in TREES[p + extra]]

    def E(self, L=0):
        """Leading error constant E_{p+1} = ||e||_2 over the trees of order p+1."""
        return mp.sqrt(mp.fsum(v ** 2 for v in self.error_coefficients(L)))

    def maxcoef(self):
        """max modulus over all a_ij and all weights of all levels."""
        vals = [abs(self.A[i][j]) for i in range(self.s) for j in range(i)]
        for _, _, b in self.levels:
            vals += [abs(v) for v in b]
        return max(vals)

    def stab_coeffs(self, L=0):
        """Coefficients of P(z) = 1 + sum_k Re(b^T A^{k-1} 1) z^k (real linear problems)."""
        m, p, b = self.levels[L]
        v = [mp.mpc(1)] * m
        co = [mp.mpf(1)]
        for k in range(1, m + 1):
            co.append(mp.re(mp.fsum(b[i] * v[i] for i in range(m))))
            v = [mp.fsum(self.A[i][j] * v[j] for j in range(i)) for i in range(m)]
        return co

    # ----------------------------------------------------------- real unknowns
    def _slots(self):
        """Complex coefficient slots: ('a', i, j) and ('b', L, i)."""
        sl = [("a", i, j) for i in range(self.s) for j in range(i)]
        for L, (m, _, _) in enumerate(self.levels):
            sl += [("b", L, i) for i in range(m)]
        return sl

    def _real_vars(self):
        """(slot, part) pairs that are free: Re and Im of everything, except
        Im b_1 of every level (inert) and all Im parts for a real method."""
        out = []
        for sl in self._slots():
            out.append((sl, 0))
            if self.projected and not (sl[0] == "b" and sl[2] == 0):
                out.append((sl, 1))
        return out

    def get_x(self):
        x = []
        for sl, part in self._real_vars():
            z = self.A[sl[1]][sl[2]] if sl[0] == "a" else self.levels[sl[1]][2][sl[2]]
            x.append(mp.re(z) if part == 0 else mp.im(z))
        return x

    def set_x(self, x):
        A = [row[:] for row in self.A]
        levels = [(m, p, b[:]) for m, p, b in self.levels]
        # zero the inert / imaginary parts that are not free
        for i in range(self.s):
            for j in range(i):
                if not self.projected:
                    A[i][j] = mp.mpc(mp.re(A[i][j]))
        for L, (m, p, b) in enumerate(levels):
            b[0] = mp.mpc(mp.re(b[0]))
            if not self.projected:
                for i in range(m):
                    b[i] = mp.mpc(mp.re(b[i]))
        for (sl, part), v in zip(self._real_vars(), x):
            if sl[0] == "a":
                z = A[sl[1]][sl[2]]
                A[sl[1]][sl[2]] = mp.mpc(v, mp.im(z)) if part == 0 else mp.mpc(mp.re(z), v)
            else:
                z = levels[sl[1]][2][sl[2]]
                levels[sl[1]][2][sl[2]] = mp.mpc(v, mp.im(z)) if part == 0 else mp.mpc(mp.re(z), v)
        return Tableau(self.name, A, levels, self.projected, self.cost)

    def normalized(self):
        """Same tableau with Im b_1 = 0 on every level (changes nothing, Prop. 2)."""
        return self.set_x(self.get_x())

    def refine(self, tol_digits=None, iters=12, verbose=False, fixed=(), rcond=1e-11):
        """Minimum-norm Newton on the projected order conditions, in mp.
        fixed: indices of get_x() held at their values (exact zero weights).
        Returns the refined tableau and the size of the correction."""
        tol = mp.mpf(10) ** (-(tol_digits or mp.mp.dps - 6))
        T = self.normalized()
        x0 = T.get_x()
        x = x0[:]
        def F(xx):
            return [v for lev in T.set_x(xx).residuals() for v in lev]
        r = F(x)
        for it in range(iters):
            nr = max(abs(v) for v in r)
            if verbose:
                print(f"    refine {self.name}: it {it} residual {mp.nstr(nr, 3)}")
            if nr < tol:
                break
            hstep = mp.mpf(10) ** (-(mp.mp.dps // 2))
            J = np.zeros((len(r), len(x)))
            for k in range(len(x)):
                xp = x[:]
                xp[k] += hstep
                rp = F(xp)
                J[:, k] = [float((a - b) / hstep) for a, b in zip(rp, r)]
            J[:, list(fixed)] = 0.0
            Jp = np.linalg.pinv(J, rcond=rcond)
            dx = [mp.fsum(mp.mpf(float(Jp[q, k])) * r[k] for k in range(len(r))) for q in range(len(x))]
            x = [a - d for a, d in zip(x, dx)]
            r = F(x)
        shift = max(abs(a - b) for a, b in zip(x, x0))
        return T.set_x(x), shift, max(abs(v) for v in r)

# --------------------------------------------------------------------- integration (mp)
def step_mp(T, f, t, y, h, levels=(0,)):
    """One projected step; returns the list of new states for the requested levels."""
    s, A, c = T.s, T.A, T.c
    K = []
    d = len(y)
    mmax = max(T.levels[L][0] for L in levels)
    for i in range(mmax):
        if i:
            yi = [y[q] + h * mp.fsum(A[i][j] * K[j][q] for j in range(i)) for q in range(d)]
        else:
            yi = [mp.mpc(v) for v in y]
        K.append(f(t + c[i] * h, yi))
    out = []
    for L in levels:
        m, _, b = T.levels[L]
        ynew = [y[q] + h * mp.fsum(b[i] * K[i][q] for i in range(m)) for q in range(d)]
        out.append([mp.re(v) for v in ynew] if T.projected else [mp.re(v) for v in ynew])
    return out

def integrate_mp(T, f, y0, t0, tend, N, L=0):
    h = (mp.mpf(tend) - t0) / N
    y = [mp.mpf(v) for v in y0]
    t = mp.mpf(t0)
    for _ in range(N):
        y = step_mp(T, f, t, y, h, (L,))[0]
        t += h
    return y

# --------------------------------------------------------------------- numpy view
def as_numpy(T, L=0):
    """(A, b, c) as complex128 arrays for level L (A restricted to its stages)."""
    m, _, b = T.levels[L]
    A = np.array([[complex(T.A[i][j]) for j in range(m)] for i in range(m)])
    bb = np.array([complex(v) for v in b])
    return A, bb, A.sum(1)
