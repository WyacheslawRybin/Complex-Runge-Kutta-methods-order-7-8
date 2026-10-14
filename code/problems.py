"""
Test problems of the paper: right-hand sides in mpmath (lists) and numpy
(complex arrays), initial data, horizons, exact solutions where available, and
high-precision reference solutions by a Taylor-series integrator for the
polynomial systems.
"""
import json, os
import numpy as np
import mpmath as mp

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

_KC = {}
def K(s):
    """Problem constant at the CURRENT working precision (cached per precision).
    Module-level mpf constants would be frozen at the precision of the import."""
    key = (s, mp.mp.prec)
    v = _KC.get(key)
    if v is None:
        if "/" in s:
            a, b = s.split("/")
            v = mp.mpf(int(a)) / int(b)
        else:
            v = mp.mpf(s)
        _KC[key] = v
    return v

def _mp(v):
    return mp.mpf(v) if not isinstance(v, (list, tuple)) else [mp.mpf(x) for x in v]

# ------------------------------------------------------------------ definitions
# each problem: f_mp(t, y) -> list, f_np(t, y) -> ndarray, y0 (strings), T, exact(T) or None
P = {}

P["Dahlquist"] = dict(
    f_mp=lambda t, y: [-y[0]], f_np=lambda t, y: -y, y0=["1"], T="5",
    exact=lambda T: [mp.exp(-T)], label="Dahlquist $y'=-y$", short="Dahlquist")
P["pendulum"] = dict(
    f_mp=lambda t, y: [y[1], -y[0]], f_np=lambda t, y: np.array([y[1], -y[0]]), y0=["1", "0"], T="5",
    exact=lambda T: [mp.cos(T), -mp.sin(T)], label="linear pendulum", short="pendulum")
P["quadratic"] = dict(
    f_mp=lambda t, y: [-y[0] ** 2], f_np=lambda t, y: -y ** 2, y0=["1"], T="5",
    exact=lambda T: [1 / (1 + T)], label="$y'=-y^{2}$", short="$y'=-y^2$")
P["exponential"] = dict(
    f_mp=lambda t, y: [-mp.exp(y[0])], f_np=lambda t, y: -np.exp(y), y0=["1"], T="5",
    exact=lambda T: [-mp.log(T + mp.exp(-1))], label="$y'=-e^{y}$", short="$y'=-e^y$")
P["nonautonomous"] = dict(
    f_mp=lambda t, y: [4 * y[0] * mp.sin(t) ** 3 * mp.cos(t)],
    f_np=lambda t, y: 4 * y * np.sin(t) ** 3 * np.cos(t), y0=["1"], T="5",
    exact=lambda T: [mp.exp(mp.sin(T) ** 4)], label="$y'=4y\\sin^{3}t\\cos t$", short="non-autonomous")

def _vdp_mp(t, y):
    return [y[1], (1 - y[0] ** 2) * y[1] - y[0]]
def _vdp_np(t, y):
    return np.array([y[1], (1 - y[0] ** 2) * y[1] - y[0]])
P["VanderPol"] = dict(f_mp=_vdp_mp, f_np=_vdp_np, y0=["2", "0"], T="5", exact=None,
                      label="Van der Pol", short="Van der Pol")

def _ros_mp(t, y):
    return [-y[1] - y[2], y[0] + K("0.2") * y[1], K("0.2") + y[2] * (y[0] - K("5.7"))]
def _ros_np(t, y):
    return np.array([-y[1] - y[2], y[0] + 0.2 * y[1], 0.2 + y[2] * (y[0] - 5.7)])
P["Rossler"] = dict(f_mp=_ros_mp, f_np=_ros_np, y0=["1", "1", "1"], T="5", exact=None,
                    label="R\\\"ossler", short="Rössler")

def _lor_mp(t, y):
    return [10 * (y[1] - y[0]), y[0] * (28 - y[2]) - y[1], y[0] * y[1] - K("8/3") * y[2]]
def _lor_np(t, y):
    return np.array([10 * (y[1] - y[0]), y[0] * (28 - y[2]) - y[1], y[0] * y[1] - 8 / 3 * y[2]])
P["Lorenz05"] = dict(f_mp=_lor_mp, f_np=_lor_np, y0=["1", "1", "20"], T="0.5", exact=None,
                     label="Lorenz, $T=0.5$", short="Lorenz, T=0.5")
P["Lorenz5"] = dict(f_mp=_lor_mp, f_np=_lor_np, y0=["1", "1", "20"], T="5", exact=None,
                    label="Lorenz, $T=5$", short="Lorenz, T=5")

def _bou_mp(t, y):
    return [K("3") * y[0] * (1 - y[1]) - K("2.2") * y[2], -K("1") * y[1] * (1 - y[0] ** 2), K("0.001") * y[0]]
def _bou_np(t, y):
    return np.array([3 * y[0] * (1 - y[1]) - 2.2 * y[2], -y[1] * (1 - y[0] ** 2), 0.001 * y[0]])
P["Bouali"] = dict(f_mp=_bou_mp, f_np=_bou_np, y0=["1", "0.1", "0"], T="5", exact=None,
                   label="Bouali", short="Bouali")

# --- standard non-stiff benchmarks -------------------------------------------
# Kepler two-body problem, eccentricity 0.5 (DETEST D3), T = 20; exact via Kepler's equation
def _kep_mp(t, y):
    r3 = (y[0] ** 2 + y[1] ** 2) ** mp.mpf(1.5)
    return [y[2], y[3], -y[0] / r3, -y[1] / r3]
def _kep_np(t, y):
    r3 = (y[0] ** 2 + y[1] ** 2) ** 1.5
    return np.array([y[2], y[3], -y[0] / r3, -y[1] / r3])
def _kep_exact(T):
    E = mp.findroot(lambda E: E - K("0.5") * mp.sin(E) - T, T)
    w = 1 - K("0.5") * mp.cos(E)
    s = mp.sqrt(1 - K("0.5") ** 2)
    return [mp.cos(E) - K("0.5"), s * mp.sin(E), -mp.sin(E) / w, s * mp.cos(E) / w]
P["Kepler"] = dict(f_mp=_kep_mp, f_np=_kep_np,
                   y0=["0.5", "0", "0", "1.732050807568877293527446341505872366942805253810380628055806979451933"], T="20", exact=_kep_exact,
                   label="Kepler, $e=0.5$", short="Kepler")

# Euler equations of a rigid body (DETEST B5), T = 12; exact: sn, cn, dn with m = 0.51
P["RigidBody"] = dict(
    f_mp=lambda t, y: [y[1] * y[2], -y[0] * y[2], -K("0.51") * y[0] * y[1]],
    f_np=lambda t, y: np.array([y[1] * y[2], -y[0] * y[2], -0.51 * y[0] * y[1]]),
    y0=["0", "1", "1"], T="12",
    exact=lambda T: [mp.ellipfun("sn", T, m=K("0.51")), mp.ellipfun("cn", T, m=K("0.51")), mp.ellipfun("dn", T, m=K("0.51"))],
    label="rigid body", short="rigid body")

# Brusselator (A = 1, B = 3), T = 20
P["Brusselator"] = dict(
    f_mp=lambda t, y: [1 + y[0] ** 2 * y[1] - 4 * y[0], 3 * y[0] - y[0] ** 2 * y[1]],
    f_np=lambda t, y: np.array([1 + y[0] ** 2 * y[1] - 4 * y[0], 3 * y[0] - y[0] ** 2 * y[1]]),
    y0=["1.5", "3"], T="20", exact=None, label="Brusselator", short="Brusselator")

# Arenstorf orbit (restricted three-body problem), one period; exact: y(T) = y(0)
def _are_mp(t, y):
    D1 = ((y[0] + K("0.012277471")) ** 2 + y[1] ** 2) ** mp.mpf(1.5)
    D2 = ((y[0] - (1 - K("0.012277471"))) ** 2 + y[1] ** 2) ** mp.mpf(1.5)
    return [y[2], y[3],
            y[0] + 2 * y[3] - (1 - K("0.012277471")) * (y[0] + K("0.012277471")) / D1 - K("0.012277471") * (y[0] - (1 - K("0.012277471"))) / D2,
            y[1] - 2 * y[2] - (1 - K("0.012277471")) * y[1] / D1 - K("0.012277471") * y[1] / D2]
def _are_np(t, y):
    mu, mup = 0.012277471, 1 - 0.012277471
    D1 = ((y[0] + mu) ** 2 + y[1] ** 2) ** 1.5
    D2 = ((y[0] - mup) ** 2 + y[1] ** 2) ** 1.5
    return np.array([y[2], y[3],
                     y[0] + 2 * y[3] - mup * (y[0] + mu) / D1 - mu * (y[0] - mup) / D2,
                     y[1] - 2 * y[2] - mup * y[1] / D1 - mu * y[1] / D2])
_are_y0 = ["0.994", "0", "0", "-2.00158510637908252240537862224"]
P["Arenstorf"] = dict(f_mp=_are_mp, f_np=_are_np, y0=_are_y0, T="17.0652165601579625588917206249",
                      exact=lambda T: [mp.mpf(v) for v in _are_y0], label="Arenstorf orbit", short="Arenstorf")

ORDER = ["Dahlquist", "pendulum", "quadratic", "exponential", "nonautonomous",
         "VanderPol", "Rossler", "Bouali", "Lorenz05", "Lorenz5", "Kepler", "RigidBody", "Brusselator"]
ADAPTIVE = ["exponential", "nonautonomous", "VanderPol", "Rossler", "Bouali", "Lorenz5",
            "Kepler", "RigidBody", "Brusselator", "Arenstorf"]
LINEAR = ["Dahlquist", "pendulum"]

# ------------------------------------------------------------------ Taylor reference
def _conv(a, b, k):
    return mp.fsum(a[i] * b[k - i] for i in range(k + 1))

def _taylor_coeffs(name, y, nterms):
    """Taylor coefficients y_k (k = 0..K) at the current point for the polynomial systems."""
    d = len(y)
    Y = [[y[q]] for q in range(d)]
    for k in range(nterms):
        if name == "VanderPol":
            x, v = Y
            x2 = _conv(x, x, k) if False else None
            # (1 - x^2) v - x : need coefficient k of x^2 v
            xx = [_conv(x, x, i) for i in range(k + 1)]
            xxv = mp.fsum(xx[i] * v[k - i] for i in range(k + 1))
            F = [v[k], v[k] - xxv - x[k]]
        elif name == "Rossler":
            x, yv, z = Y
            F = [-yv[k] - z[k], x[k] + K("0.2") * yv[k], (K("0.2") if k == 0 else 0) + _conv(z, x, k) - K("5.7") * z[k]]
        elif name in ("Lorenz05", "Lorenz5"):
            x, yv, z = Y
            F = [10 * (yv[k] - x[k]), 28 * x[k] - _conv(x, z, k) - yv[k], _conv(x, yv, k) - K("8/3") * z[k]]
        elif name == "Brusselator":
            x, yv = Y
            xx = [_conv(x, x, i) for i in range(k + 1)]
            xxy = mp.fsum(xx[i] * yv[k - i] for i in range(k + 1))
            F = [(1 if k == 0 else 0) + xxy - 4 * x[k], 3 * x[k] - xxy]
        elif name == "Bouali":
            x, yv, z = Y
            xx = [_conv(x, x, i) for i in range(k + 1)]
            F = [K("3") * (x[k] - _conv(x, yv, k)) - K("2.2") * z[k],
                 -K("1") * (yv[k] - mp.fsum(yv[i] * xx[k - i] for i in range(k + 1))),
                 K("0.001") * x[k]]
        else:
            raise ValueError(name)
        for q in range(d):
            Y[q].append(F[q] / (k + 1))
    return Y

def taylor_reference(name, nterms=40, h="0.01", dps=80):
    with mp.workdps(dps):
        pr = P[name]
        T = mp.mpf(pr["T"]); h = mp.mpf(h)
        n = int(mp.nint(T / h))
        h = T / n
        y = [mp.mpf(v) for v in pr["y0"]]
        for _ in range(n):
            Y = _taylor_coeffs(name, y, nterms)
            y = [mp.polyval(list(reversed(Yq)), h) for Yq in Y]
        return y

def references(recompute=False):
    fn = os.path.join(DATA, "references.json")
    if os.path.exists(fn) and not recompute:
        d = json.load(open(fn))
        return {k: [mp.mpf(v) for v in vals] for k, vals in d["ref"].items()}
    out, check = {}, {}
    for name in ORDER + ["Arenstorf"]:
        pr = P[name]
        if pr["exact"] is not None:
            with mp.workdps(80):
                out[name] = pr["exact"](mp.mpf(pr["T"]))
            continue
        y1 = taylor_reference(name, nterms=40, h="0.01")
        y2 = taylor_reference(name, nterms=48, h="0.005")
        diff = max(abs(a - b) for a, b in zip(y1, y2))
        out[name] = y2
        check[name] = mp.nstr(diff, 3)
        print(f"reference {name}: self-consistency {mp.nstr(diff, 3)}", flush=True)
    json.dump(dict(ref={k: [mp.nstr(v, 60) for v in vals] for k, vals in out.items()}, check=check),
              open(fn, "w"), indent=1)
    return out

if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    R = references(recompute=True)
    old = json.load(open(os.path.join(DATA, "..", "..", "report_figs", "refs.json"), encoding="utf-8"))
    names = {"Ван дер Поль": "VanderPol", "Рёсслер": "Rossler", "Лоренц": "Lorenz05", "Буали": "Bouali"}
    for k, v in old.items():
        nm = names[k]
        print(f"vs old 30-digit refs.json {nm}: {mp.nstr(max(abs(mp.mpf(a) - b) for a, b in zip(v, R[nm])), 3)}")
