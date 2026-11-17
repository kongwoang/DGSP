"""The planners: braking controller (main), TTC warning and curved-path braking (other planners). Each exposes decide(), the
field on the ground grid (closed form: where one probe object changes the action) and the support (the objects the
action rests on). A scene is the tuple (z, lat_min, lat_max, ttc) of per-object arrays in the ego frame."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KEEP, DECELERATE, HARD_BRAKE = 0, 1, 2


@dataclass(frozen=True)
class Params:
    corridor_half_w: float = 1.2     # ego half-width plus a lateral margin [m]
    standoff: float = 2.0            # distance kept when stopped [m]
    reaction_t: float = 0.4          # reaction time folded into the stopping distance [s]
    thr_decel: float = 1.0           # tau_1: required deceleration for DECELERATE [m/s^2]
    thr_hard: float = 3.5            # tau_2: required deceleration for HARD_BRAKE [m/s^2]
    max_range: float = 80.0
    a_cap: float = 9.0
    ttc_warn: float = 4.0            # TTC planner: warn at ttc <= 4 s, brake at ttc <= 2 s
    ttc_brake: float = 2.0


P0 = Params()


def in_corridor(lo, hi, hw):
    return (np.asarray(lo) < hw) & (np.asarray(hi) > -hw)


def required_decel(z, lo, hi, ttc, v, p=P0):
    """Deceleration needed to stop behind the most constraining in-corridor object; also per object."""
    z = np.asarray(z, float)
    a_obj = np.zeros(z.shape)
    if z.size == 0:
        return 0.0, a_obj
    sel = in_corridor(lo, hi, p.corridor_half_w) & (z <= p.max_range)
    if not sel.any() or v <= 0.1:
        return 0.0, a_obj
    a = v ** 2 / (2.0 * np.maximum(z[sel] - p.standoff - v * p.reaction_t, 0.05))
    v_rel = np.clip(z[sel] / np.clip(np.asarray(ttc, float)[sel], 0.1, 1e3), 0.0, 60.0)
    a = np.maximum(a, v_rel ** 2 / (2.0 * np.maximum(z[sel] - p.standoff, 0.05)))
    a_obj[sel] = np.clip(a, 0.0, p.a_cap)
    return float(a_obj.max()), a_obj


def action(a_req, p=P0):
    return HARD_BRAKE if a_req >= p.thr_hard else DECELERATE if a_req >= p.thr_decel else KEEP


def probe_demand(r, v, p=P0, v_close=0.0):
    """Required deceleration of a probe object alone at range r, closing at v_close (0: static)."""
    r = np.asarray(r, float)
    if v <= 0.1:
        return np.zeros_like(r)
    a = v ** 2 / (2.0 * np.maximum(r - p.standoff - v * p.reaction_t, 0.05))
    ttc = np.clip(r / v_close, 0.1, 1e3) if v_close > 0 else np.full_like(r, 1e3)
    a = np.maximum(a, np.clip(r / ttc, 0.0, 60.0) ** 2 / (2.0 * np.maximum(r - p.standoff, 0.05)))
    return np.where(r <= p.max_range, np.clip(a, 0.0, p.a_cap), 0.0)


def _support_of(a_obj, act, p):
    if act == KEEP or not len(a_obj):
        return np.zeros(0, int)
    return np.flatnonzero(np.asarray(a_obj) >= (p.thr_decel if act == DECELERATE else p.thr_hard))


class Braking:
    """Longitudinal braking controller (max over objects of the required deceleration)."""

    def __init__(self, p=P0, margin=0.0):
        self.p, self.margin = p, margin

    def decide(self, s, v):
        return action(required_decel(*s, v, self.p)[0], self.p)

    def field(self, s, v, r_grid, y_grid, w, v_close=0.0):
        """(R, Y) bool: a probe of width w at (r, y) reaches (1 - margin) x the next action's threshold."""
        p, act = self.p, self.decide(s, v)
        if act == HARD_BRAKE:
            return np.zeros((len(r_grid), len(y_grid)), bool)
        thr = (p.thr_decel if act == KEEP else p.thr_hard) * (1.0 - self.margin)
        y = np.asarray(y_grid, float)
        return (probe_demand(r_grid, v, p, v_close)[:, None] >= thr) & in_corridor(y - w / 2, y + w / 2, p.corridor_half_w)[None, :]

    def support(self, s, v):
        a_max, a_obj = required_decel(*s, v, self.p)
        return _support_of(a_obj, action(a_max, self.p), self.p)

    def pivots(self, s, v):
        """Objects whose removal alone changes the action (ablation 'pivots')."""
        a_max, a_obj = required_decel(*s, v, self.p)
        if not len(a_obj):
            return np.zeros(0, int)
        o = np.argsort(-a_obj, kind="stable")
        second = a_obj[o[1]] if len(o) > 1 else 0.0
        return np.array([o[0]]) if action(second, self.p) != action(a_max, self.p) and a_obj[o[0]] > second else np.zeros(0, int)


class TTCWarning(Braking):
    """Warn / brake when the smallest in-corridor TTC falls below 4 / 2 s. The probe is static (closing at ego speed)."""

    def demand(self, s):
        z, lo, hi, ttc = (np.asarray(a, float) for a in s)
        d = np.zeros(len(z))
        sel = in_corridor(lo, hi, self.p.corridor_half_w) & (z <= self.p.max_range) & (z > 0)
        d[sel] = 1.0 / np.clip(ttc[sel], 0.05, np.inf)
        return d

    def decide(self, s, v=None):
        m = self.demand(s).max() if len(s[0]) else 0.0
        return 2 if m >= 1 / self.p.ttc_brake else 1 if m >= 1 / self.p.ttc_warn else 0

    def field(self, s, v, r_grid, y_grid, w, v_close=0.0):
        a = self.decide(s)
        if a == 2:
            return np.zeros((len(r_grid), len(y_grid)), bool)
        thr = (1 / self.p.ttc_warn, 1 / self.p.ttc_brake)[a] * (1.0 - self.margin)
        rr, yy = np.meshgrid(np.asarray(r_grid, float), np.asarray(y_grid, float), indexing="ij")
        return in_corridor(yy - w / 2, yy + w / 2, self.p.corridor_half_w) & (v / rr >= thr) & (rr <= self.p.max_range)

    def support(self, s, v=None):
        a = self.decide(s)
        return np.zeros(0, int) if a == 0 else np.flatnonzero(self.demand(s) >= (0, 1 / self.p.ttc_warn, 1 / self.p.ttc_brake)[a])


KMAX = 0.1


def curvature(yaw_rate, v):
    return float(np.clip(yaw_rate / max(v, 1.0), -KMAX, KMAX)) if np.isfinite(yaw_rate) else 0.0


def path_sd(x, y, k):
    """ego (x forward, y left) -> (arc length s along the constant-curvature path, lateral offset d)"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if abs(k) < 1e-6:
        return x, y
    R = 1.0 / k
    rho = np.hypot(x, y - R)
    d = np.sign(R) * (abs(R) - rho)
    s = abs(R) * np.arctan2(x, np.sign(R) * (R - y))
    return np.where(x > 0, s, -1.0), d


class CurvedBraking(Braking):
    """The braking controller along the ego's constant-curvature path (curvature from the yaw rate). Its field uses the
    true thresholds (no margin)."""

    def __init__(self, k, p=P0):
        super().__init__(p, 0.0)
        self.k = k

    def along(self, s):
        z, lo, hi, ttc = (np.asarray(a, float) for a in s)
        if abs(self.k) < 1e-6 or not len(z):
            return z, lo, hi, ttc
        a, _ = path_sd(z, (lo + hi) / 2, self.k)
        _, d1 = path_sd(z, lo, self.k)
        _, d2 = path_sd(z, hi, self.k)
        return np.where(a < 0, 1e6, a), np.minimum(d1, d2), np.maximum(d1, d2), ttc

    def decide(self, s, v):
        return super().decide(self.along(s), v)

    def support(self, s, v):
        return super().support(self.along(s), v)

    def field(self, s, v, r_grid, y_grid, w, v_close=0.0):
        p, act = self.p, self.decide(s, v)
        if act == HARD_BRAKE:
            return np.zeros((len(r_grid), len(y_grid)), bool)
        rr, yy = np.meshgrid(np.asarray(r_grid, float), np.asarray(y_grid, float), indexing="ij")
        a, d = path_sd(rr, yy, self.k)
        dem = probe_demand(np.where(a > 0, a, 1e6), v, p, v_close)
        return (dem >= (p.thr_decel if act == KEEP else p.thr_hard)) & in_corridor(d - w / 2, d + w / 2, p.corridor_half_w) & (a > 0)
