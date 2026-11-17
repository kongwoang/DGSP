"""Annotated ground-truth geometry: one row per annotated road user per frame, in the ego frame."""
from __future__ import annotations

import numpy as np

GEOM_FIELDS = [
    ("seq", "U16"), ("frame", "i4"), ("track_id", "i4"), ("type", "U16"), ("coarse", "U8"),
    ("x1", "f4"), ("y1", "f4"), ("x2", "f4"), ("y2", "f4"),
    ("truncated", "f4"), ("occluded", "i4"),
    ("long_near", "f4"),     # nearest longitudinal distance of the footprint [m]
    ("long_cen", "f4"),      # footprint centre longitudinal distance [m]
    ("lat_cen", "f4"),       # footprint centre lateral offset, + left [m]
    ("lat_min", "f4"),       # footprint lateral extent, low edge
    ("lat_max", "f4"),       # footprint lateral extent, high edge
    ("range_rate", "f4"),    # d(long_near)/dt [m/s], negative = closing
    ("ttc", "f4"),           # long_near / closing rate [s], inf if opening
    ("has_rate", "?"),
]
MIN_BOX_H = 10.0


def fill_range_rate(geom: np.ndarray, halfwidth: int, dt: float) -> None:
    """Range rate per track by a local linear fit of long_near against time (temporally contiguous neighbours)."""
    geom["ttc"] = np.inf
    for tid in np.unique(geom["track_id"]):
        if tid < 0:
            continue
        idx = np.flatnonzero(geom["track_id"] == tid)
        order = idx[np.argsort(geom["frame"][idx])]
        f = geom["frame"][order].astype(float) * dt
        d = geom["long_near"][order].astype(float)
        n = len(order)
        for j in range(n):
            lo, hi = max(0, j - halfwidth), min(n, j + halfwidth + 1)
            while lo < j and f[j] - f[lo] > (halfwidth + 0.5) * dt:
                lo += 1
            while hi - 1 > j and f[hi - 1] - f[j] > (halfwidth + 0.5) * dt:
                hi -= 1
            if hi - lo < 2:
                continue
            slope = np.polyfit(f[lo:hi], d[lo:hi], 1)[0]
            k = order[j]
            geom["range_rate"][k] = slope
            geom["has_rate"][k] = True
            closing = -slope
            geom["ttc"][k] = d[j] / closing if closing > 1e-3 else np.inf


def to_array(rows, seq_len: int = 16) -> np.ndarray:
    """rows: (seq, frame, track, type, coarse, x1, y1, x2, y2, long_near, long_cen, lat_cen, lat_min, lat_max)."""
    if not rows:
        return np.zeros(0, dtype=GEOM_FIELDS)
    out = np.zeros(len(rows), dtype=GEOM_FIELDS)
    for i, r in enumerate(rows):
        out[i]["seq"], out[i]["frame"], out[i]["track_id"] = r[0][:seq_len], r[1], r[2]
        out[i]["type"], out[i]["coarse"] = r[3], r[4]
        for j, c in enumerate(("x1", "y1", "x2", "y2")):
            out[i][c] = r[5 + j]
        out[i]["truncated"], out[i]["occluded"] = 0.0, 0
        for j, c in enumerate(("long_near", "long_cen", "lat_cen", "lat_min", "lat_max")):
            out[i][c] = r[9 + j]
        out[i]["range_rate"], out[i]["ttc"] = 0.0, np.inf
    return out


def frame_geometry(geom: np.ndarray, fr: int) -> np.ndarray:
    g = geom[geom["frame"] == fr] if len(geom) else geom
    return g[(g["y2"] - g["y1"]) >= MIN_BOX_H] if len(g) else g


def yaw_rates(Rs, ts):
    """Ego yaw rate per frame [rad/s]: central difference of the ego heading (ego -> world rotations Rs, times ts [s])."""
    h = np.unwrap(np.array([np.arctan2(R[1, 0], R[0, 0]) for R in Rs], float))
    ts, out = np.asarray(ts, float), np.zeros(len(h))
    for i in range(len(h)):
        a, b = max(0, i - 1), min(len(h) - 1, i + 1)
        out[i] = (h[b] - h[a]) / max(ts[b] - ts[a], 1e-3) if b > a else 0.0
    return out
