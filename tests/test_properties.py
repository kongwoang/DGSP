"""Property tests of the guarantee (main Sec. 3, supp. S1) on random small scenes, fixed seeds, CPU only.

1. Max-composable planners (Corollary 1) and bottleneck selectors (Corollary 2): the closed-form field equals the
   definition (Eq. 2), insertion stability, and certification of the support (Eq. 3) for every subset D of the
   objects outside the support, with no insertion or with one inserted object outside the field.
2. Theorem 1: the loop (Algorithm 1) on synthetic scenes that satisfy locality and coverage returns pi(F).
3. The braking controller, TTC warning and curved-path braking of dgsp/planner.py (margin 0): field cells equal the
   probe definition, the support is certifying.
Run: pytest -q tests/test_properties.py (counts are printed with -s)."""
from __future__ import annotations

import itertools
import os

import numpy as np

from dgsp import planner as P

SCALE = float(os.environ.get("PROP_SCALE", "1"))
N_SCENES = {k: max(1, int(v * SCALE)) for k, v in {"maxcomp": 32000, "bottleneck": 32000, "thm_maxcomp": 20000, "thm_bottleneck": 15000, "dgsp_planners": 1200}.items()}
COUNTS = {}


def _masks(m):
    """All 2^m subsets of m items as a bool matrix (2^m, m)."""
    return np.array(list(itertools.product((False, True), repeat=m)), bool).reshape(2 ** m, m)


# ---------------------------------------------------------------- max-composable (Definition 1)
def mc_act(d, tau):
    """Action of each row of demands d (..., n) with thresholds tau (tau_1 < ... < tau_K): q(max demand)."""
    m = d.max(-1) if d.shape[-1] else np.zeros(d.shape[:-1])
    return (m[..., None] >= tau).sum(-1)


def test_max_composable():
    rng, viol, cases = np.random.default_rng(0), {"field": 0, "insertion": 0, "certification": 0}, 0
    for _ in range(N_SCENES["maxcomp"]):
        K = int(rng.integers(1, 4))
        tau = np.sort(rng.choice(np.arange(1, 10), K, replace=False)).astype(float)   # integer grid -> ties with demands
        n = int(rng.integers(0, 7))
        d = rng.integers(0, 11, n).astype(float)
        a = int(mc_act(d, tau))
        U = np.arange(0, 11, dtype=float)                                     # every probe state (demand value)
        field_def = mc_act(np.c_[np.tile(d, (len(U), 1)), U], tau) != a       # Eq. 2
        field_cf = U >= (tau[a] if a < K else np.inf)                         # Corollary 1
        viol["field"] += int((field_def != field_cf).any())
        out = U[~field_def]
        X = rng.choice(out, (8, int(rng.integers(1, 5)))) if len(out) else np.zeros((0, 1))   # finite sets outside the field
        viol["insertion"] += int((mc_act(np.c_[np.tile(d, (len(X), 1)), X], tau) != a).any())
        sup = d >= tau[a - 1] if a >= 1 else np.zeros(n, bool)                # Corollary 1 support
        rest = np.flatnonzero(~sup)
        M = _masks(len(rest))                                                 # True = removed (D)
        keep = np.ones((len(M), n), bool)
        keep[:, rest] = ~M
        dk = np.where(keep, d, -np.inf)
        O = np.r_[-np.inf, out]                                               # no insertion, or one object outside the field
        acts = mc_act(np.concatenate([np.repeat(dk[:, None], len(O), 1), np.broadcast_to(O[None, :, None], (len(M), len(O), 1))], -1), tau)
        viol["certification"] += int((acts != a).any())
        cases += 1
    COUNTS["maxcomp"] = (cases, viol)
    assert not any(viol.values()), viol


# ---------------------------------------------------------------- bottleneck selector (Corollary 2)
def bn_pi(beta, Hm):
    """pi for each scene: Hm (..., n, C) per-object scores; first maximizer of min(beta, min_o h)."""
    s = np.minimum(beta, Hm.min(-2)) if Hm.shape[-2] else np.broadcast_to(beta, Hm.shape[:-2] + beta.shape)
    return np.argmax(s, -1), s


def test_bottleneck():
    rng, viol, cases = np.random.default_rng(1), {"field": 0, "insertion": 0, "certification": 0}, 0
    for _ in range(N_SCENES["bottleneck"]):
        C, n = int(rng.integers(2, 6)), int(rng.integers(0, 6))
        beta = rng.integers(2, 7, C).astype(float)
        univ = rng.integers(0, 8, (24, C)).astype(float)                      # probe states = score vectors (ties likely)
        univ[rng.random(univ.shape) < 0.3] = np.inf                           # no overlap with some candidates
        Hs = univ[rng.integers(0, len(univ), n)]
        c0, s0 = bn_pi(beta, Hs)
        c0, th = int(c0), s0[c0]
        ins = bn_pi(beta, np.concatenate([np.broadcast_to(Hs, (len(univ), n, C)), univ[:, None]], 1))[0]
        field_def = ins != c0                                                 # Eq. 2
        # closed form (supp. Eq. S-bnfield): some c != c* overtakes, or ties and precedes c*
        sc = np.minimum(s0[None], univ)
        hc = univ[:, c0:c0 + 1]
        other = np.arange(C) != c0
        field_cf = (((sc > hc) | ((sc == hc) & (np.arange(C) < c0))) & other).any(1)
        viol["field"] += int((field_def != field_cf).any())
        out = univ[~field_def]
        if len(out):                                                          # 24 sets of 1-4 objects (+inf rows = no object)
            X = out[rng.integers(0, len(out), (24, 4))]
            X[np.arange(4)[None, :] >= rng.integers(1, 5, 24)[:, None]] = np.inf
            got = bn_pi(beta, np.concatenate([np.broadcast_to(Hs, (24, n, C)), X], 1))[0]
            viol["insertion"] += int((got != c0).any())
        sup = (Hs[:, other] <= th).any(1) if n else np.zeros(0, bool)         # Corollary 2 support
        rest = np.flatnonzero(~sup)
        M = _masks(len(rest))
        keep = np.ones((len(M), n), bool)
        keep[:, rest] = ~M
        Hk = np.where(keep[:, :, None], Hs[None], np.inf)                     # removed objects -> +inf scores
        O = np.r_[np.full((1, C), np.inf), out]
        Hall = np.concatenate([np.repeat(Hk[:, None], len(O), 1), np.broadcast_to(O[None, :, None], (len(M), len(O), 1, C))], 2)
        viol["certification"] += int((bn_pi(beta, Hall)[0] != c0).any())
        cases += 1
    COUNTS["bottleneck"] = (cases, viol)
    assert not any(viol.values()), viol


# ---------------------------------------------------------------- Theorem 1 on synthetic scenes
# States are cells (i, j) of a ground grid; an object's image box is a function of its state (the lift inverts the box),
# so a crop that covers the image position of a state covers the box centre of every object with that state.
GI, GJ = 8, 7


def cell_pos(i, j):
    """Image position (u, v) of a cell: farther rows higher, lateral offset scaled by range."""
    r = 6.0 + 7.0 * i
    return 500.0 - 900.0 * (j - 3) / r, 200.0 + 1500.0 / r


POS = np.array([[cell_pos(i, j) for j in range(GJ)] for i in range(GI)])


def inside(c, crops):
    return any(c[0] >= w[0] and c[1] >= w[1] and c[2] <= w[2] and c[3] <= w[3] for w in crops)


def centre_in(p, crops):
    return any(w[0] <= p[0] <= w[2] and w[1] <= p[1] <= w[3] for w in crops)


def bbox(pts, pad=4.0):
    pts = np.asarray(pts, float).reshape(-1, 2)
    return (pts[:, 0].min() - pad, pts[:, 1].min() - pad, pts[:, 0].max() + pad, pts[:, 1].max() + pad)


def run_loop(draft, F, field_cells, support_idx, decide):
    """Algorithm 1 with an idealized verification: ver(R) = F|R (locality). Objects are cells (i, j). The footprint is the
    bounding crop of the image positions of all field cells (coverage), support crops are small boxes at the objects."""
    R, M = [], list(draft)
    for _ in range(200):
        req = []
        cells = field_cells(M)
        if len(cells):
            req.append(bbox(POS[cells[:, 0], cells[:, 1]]))
        req += [bbox(POS[M[k]], 2.0) for k in support_idx(M)]
        new = [c for c in req if not inside(c, R)]
        if not new:
            return decide(M), True
        R += new
        M = [o for o in draft if not centre_in(POS[o], R)] + [o for o in F if centre_in(POS[o], R)]
    return decide(M), False


def random_scene(rng):
    nF = int(rng.integers(0, 6))
    F = [tuple(x) for x in np.c_[rng.integers(0, GI, nF), rng.integers(0, GJ, nF)]]
    draft = [o for o in F if rng.random() < 0.6]                                         # misses
    draft = [(int(np.clip(o[0] + rng.integers(-1, 2), 0, GI - 1)), int(np.clip(o[1] + rng.integers(-1, 2), 0, GJ - 1)))
             if rng.random() < 0.3 else o for o in draft]                                # mislocalized
    draft += [tuple(x) for x in np.c_[rng.integers(0, GI, 3), rng.integers(0, GJ, 3)][: int(rng.integers(0, 3))]]  # spurious
    return F, draft


def test_theorem_max_composable():
    rng, viol, cases, nc = np.random.default_rng(2), 0, 0, 0
    for _ in range(N_SCENES["thm_maxcomp"]):
        K = int(rng.integers(1, 4))
        tau = np.sort(rng.choice(np.arange(1, 10), K, replace=False)).astype(float)
        dem = rng.integers(0, 11, (GI, GJ)).astype(float) * (rng.random((GI, GJ)) < 0.6)
        F, draft = random_scene(rng)
        dd = lambda S: np.array([dem[o] for o in S])                                     # noqa: E731
        decide = lambda S: int(mc_act(dd(S), tau))                                       # noqa: E731

        def field_cells(S):
            a = decide(S)
            return np.argwhere(dem >= tau[a]) if a < K else np.zeros((0, 2), int)

        def support_idx(S):
            a = decide(S)
            return [k for k, o in enumerate(S) if a >= 1 and dem[o] >= tau[a - 1]]

        act, conv = run_loop(draft, F, field_cells, support_idx, decide)
        nc += int(not conv)
        viol += int(conv and act != decide(F))
        cases += 1
    COUNTS["thm_maxcomp"] = (cases, {"theorem": viol, "not_converged": nc})
    assert viol == 0 and nc == 0


def test_theorem_bottleneck():
    rng, viol, cases, nc = np.random.default_rng(3), 0, 0, 0
    for _ in range(N_SCENES["thm_bottleneck"]):
        C = int(rng.integers(2, 6))
        beta = rng.integers(2, 7, C).astype(float)
        h = rng.integers(0, 8, (GI, GJ, C)).astype(float)
        h[rng.random(h.shape) < 0.5] = np.inf
        F, draft = random_scene(rng)
        Hof = lambda S: np.array([h[o] for o in S]).reshape(-1, C)                      # noqa: E731
        decide = lambda S: int(bn_pi(beta, Hof(S))[0])                                  # noqa: E731

        def field_cells(S):
            c0 = decide(S)
            ins = bn_pi(beta, np.concatenate([np.broadcast_to(Hof(S), (GI * GJ, len(S), C)), h.reshape(-1, 1, C)], 1))[0]
            return np.argwhere((ins != c0).reshape(GI, GJ))

        def support_idx(S):
            c0, s = bn_pi(beta, Hof(S))
            th, other = s[int(c0)], np.arange(C) != int(c0)
            return [k for k, o in enumerate(S) if (h[o][other] <= th).any()]

        act, conv = run_loop(draft, F, field_cells, support_idx, decide)
        nc += int(not conv)
        viol += int(conv and act != decide(F))
        cases += 1
    COUNTS["thm_bottleneck"] = (cases, {"theorem": viol, "not_converged": nc})
    assert viol == 0 and nc == 0


# ---------------------------------------------------------------- the planners of dgsp/planner.py (margin 0)
R_G, Y_G, WPROBE = np.arange(4.0, 81.0, 4.0), np.arange(-4.0, 4.01, 0.5), 1.8


def test_dgsp_planners():
    rng, viol, cases = np.random.default_rng(4), {"field": 0, "certification": 0, "insertion": 0}, 0
    for t in range(N_SCENES["dgsp_planners"]):
        kind = ("braking", "ttc", "curved")[t % 3]
        v = float(rng.uniform(2.0, 25.0))
        vc = float(rng.choice([0.0, 5.0]))
        if kind == "braking":
            pl = P.Braking(P.P0, 0.0)
        elif kind == "ttc":
            pl, vc = P.TTCWarning(P.P0, 0.0), v                                       # the TTC probe closes at ego speed
        else:
            pl = P.CurvedBraking(float(rng.uniform(-0.05, 0.05)), P.P0)
        n = int(rng.integers(0, 6))
        z = rng.uniform(3.0, 90.0, n)
        yc, wd = rng.uniform(-4.0, 4.0, n), rng.uniform(0.5, 2.5, n)
        ttc = np.where(rng.random(n) < 0.5, 1e3, z / rng.uniform(0.5, 15.0, n))
        S0 = S = (z, yc - wd / 2, yc + wd / 2, ttc)
        if kind == "curved":                                   # curved braking = max-composable in path coordinates (s, d)
            S, decide, field = pl.along(S), (lambda s_, v_: P.Braking.decide(pl, s_, v_)), pl.field
        else:
            decide, field = pl.decide, pl.field
        a = decide(S, v)
        sub = lambda keep, extra=(): tuple(np.r_[x[keep], [e[i] for e in extra]] for i, x in enumerate(S))  # noqa: E731

        def probe(r, y):
            if kind == "curved":                               # the field cell's probe in path coordinates
                s_, d_ = P.path_sd(r, y, pl.k)
                s_ = float(s_) if float(s_) > 0 else 1e6
                return (s_, float(d_) - WPROBE / 2, float(d_) + WPROBE / 2, (s_ / vc) if vc > 0 else 1e3)
            return (r, y - WPROBE / 2, y + WPROBE / 2, (r / vc) if vc > 0 else 1e3)
        # field cells vs the definition
        fc = field(S0, v, R_G, Y_G, WPROBE, vc)
        cells = [(i, j) for i in range(len(R_G)) for j in range(len(Y_G))]
        for i, j in cells[t % 14::14]:                                               # a fourteenth of the cells per scene
            viol["field"] += int(bool(fc[i, j]) != (decide(sub(np.ones(n, bool), [probe(R_G[i], Y_G[j])]), v) != a))
        outs = [probe(R_G[i], Y_G[j]) for i, j in cells if not fc[i, j]]
        # certification: every subset of the non-support objects, with no insertion or one probe outside the field
        sup = set(pl.support(S0, v).tolist())
        rest = [k for k in range(n) if k not in sup]
        pick = [outs[k] for k in rng.integers(0, len(outs), min(4, len(outs)))] if outs else []
        for m in _masks(len(rest)):
            keep = np.ones(n, bool)
            keep[np.array(rest, int)[m]] = False
            for o in [None] + pick:
                viol["certification"] += int(decide(sub(keep, [] if o is None else [o]), v) != a)
        if len(pick) >= 2:                                                            # insertion stability: all picks at once
            viol["insertion"] += int(decide(sub(np.ones(n, bool), pick), v) != a)
        cases += 1
    COUNTS["dgsp_planners"] = (cases, viol)
    assert not any(viol.values()), viol


def teardown_module(module):
    tot = sum(c for c, _ in COUNTS.values())
    for k, (c, v) in COUNTS.items():
        print(f"\n{k}: {c} scenes, violations {v}", end="")
    print(f"\ntotal scenes {tot}")
