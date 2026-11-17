"""Counterfactual importance (own rule): context-2 crops around the draft objects whose removal, range x (1 -/+ 0.5) or
TTC -/+ 2 s changes the braking action (removal-critical first, then range, then TTC; by demand within a class); one
pass, same windows, guard, merge and lift as DGSP."""
from __future__ import annotations

import numpy as np

from baselines.ana import RegionRule
from dgsp import planner as P


def _act(m, v, p):
    return P.action(P.required_decel(*m, v, p)[0], p)


class Counterfactual(RegionRule):
    iters, rel, shift = 1, 0.5, 2.0                # range x (1 -/+ rel), TTC -/+ shift [s]

    def targets(self, scene, draft, r, v):
        m = tuple(np.asarray(a, float) for a in scene["mono"])
        n = len(m[0])
        if not n:
            return scene["xyxy"], np.zeros(0, int)
        a0, dem = _act(m, v, self.p), np.asarray(P.required_decel(*m, v, self.p)[1], float)
        cls = np.full(n, 9)
        for i in range(n):
            keep = np.arange(n) != i
            def with_(j, val):
                x = [a.copy() for a in m]; x[j][i] = val
                return tuple(x)
            if _act(tuple(a[keep] for a in m), v, self.p) != a0:
                cls[i] = 0
            elif any(_act(with_(0, m[0][i] * f), v, self.p) != a0 for f in (1 - self.rel, 1 + self.rel)):
                cls[i] = 1
            elif np.isfinite(m[3][i]) and any(_act(with_(3, max(m[3][i] + d, 0.1)), v, self.p) != a0 for d in (-self.shift, self.shift)):
                cls[i] = 2
        idx = np.flatnonzero(cls < 9)
        return scene["xyxy"], idx[np.lexsort((-dem[idx], cls[idx]))]
