"""
Confluent limit of a projected tableau (Kastlunger-Wanner multiple nodes; the CRK8-8 note):
when a_21 = eps -> 0 with a_i2 = alpha_i/eps, a_i1 = gamma_i - alpha_i/eps, b_2 = beta_G/eps,
b_1 = beta_1 - beta_G/eps, stage 2 turns into the pseudo-stage G = h f'(y_n) f(y_n), whose
elementary weight is the indicator of the tree [tau].  ConfluentProblem is cascade_search.Problem
with slot 2 replaced by G: Phi_G(t) = [t = [tau]], a_21 unused, Im of the weights of slot 2
inert.  Coefficients of slot 2 in rows i >= 3 are the alpha_i, of slot 1 the gamma_i.
"""
import numpy as np
from cascade_search import Problem, TREES, gamma

BUSH1 = ((),)                       # the tree [tau]

class ConfluentProblem(Problem):
    def __init__(self, levels, S=None, extra=()):
        super().__init__(levels, S, extra)
        drop = {0, self.nz + 0} | {self.nz + o + 1 for o in self.woff}   # a_21 (both parts), Im w_2
        self.keep = np.array([q for q in self.keep if q not in drop])
        self.n = len(self.keep)

    def phi(self, A, pmax=None, deriv=True):
        pmax = pmax or self.pmax
        B, S = A.shape[0], self.S
        P, dP, AP, dAP = {}, {}, {}, {}
        def ap(u):
            if u not in AP:
                pu = get(u)
                AP[u] = np.einsum('bij,bj->bi', A, pu)
                if deriv:
                    d = np.einsum('bij,bjk->bik', A, dP[u])
                    d[:, self.I, self.K] += pu[:, self.Jc]
                    dAP[u] = d
            return AP[u]
        def get(t):
            if t in P:
                return P[t]
            v = np.ones((B, S), complex)
            dv = np.zeros((B, S, self.NA), complex) if deriv else None
            for u in t:
                w = ap(u)
                if deriv:
                    dv = dv * w[:, :, None] + v[:, :, None] * dAP[u]
                v = v * w
            v[:, 1] = 1.0 if t == BUSH1 else 0.0          # pseudo-stage G
            if deriv:
                dv[:, 1, :] = 0.0
            P[t] = v
            if deriv:
                dP[t] = dv
            return v
        for k in range(1, pmax + 1):
            for t in TREES[k]:
                get(t)
        return P, dP

def to_confluent(A, W):
    """(8) of the note: ordinary tableau with small a_21 -> parameters of the limit scheme."""
    A = A.copy(); W = [w.copy() for w in W]
    eps = A[:, 1, 0].copy()
    A[:, 2:, 0] = A[:, 2:, 0] + A[:, 2:, 1]
    A[:, 2:, 1] = A[:, 2:, 1] * eps[:, None]
    A[:, 1, 0] = 0
    for w in W:
        w[:, 0] = w[:, 0] + w[:, 1]
        w[:, 1] = w[:, 1] * eps
    return A, W

def from_confluent(A, W, eps):
    """(9) of the note: limit scheme -> eight-stage-type tableau CRK(eps)."""
    A = A.copy(); W = [w.copy() for w in W]
    al = A[:, 2:, 1].copy()
    A[:, 1, 0] = eps
    A[:, 2:, 1] = al / eps
    A[:, 2:, 0] = A[:, 2:, 0] - al / eps
    for w in W:
        bg = w[:, 1].copy()
        w[:, 1] = bg / eps
        w[:, 0] = w[:, 0] - bg / eps
    return A, W
