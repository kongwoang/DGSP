"""Field footprint: the image region covering the probe objects at every sensitive cell of the planner's field."""
from __future__ import annotations

import numpy as np

from .config import CAR, R_GRID, Y_GRID


def footprint(planner, scene, v, cam, W, H, closing_speed=0.0, mode="exact", r_grid=R_GRID, y_grid=Y_GRID, probe=CAR):
    """Bounding box (x1, y1, x2, y2) of the probes at the sensitive cells, or None if the field is empty.
    exact: the probe's bottom / top corners projected by the camera's ego -> camera transform;
    pinhole: bottom row cy + fy h_cam / r, top fy h / r higher, centre cx - fx y / r."""
    w, h = probe
    sens = planner.field(scene, v, r_grid, y_grid, w, closing_speed)
    if not sens.any():
        return None
    rr, yy = np.meshgrid(np.asarray(r_grid, float), np.asarray(y_grid, float), indexing="ij")
    rr, yy = rr[sens], yy[sens]
    c = cam.calib
    if mode == "pinhole":
        vb, uc, hw = c.cy + c.fy * cam.cam_h / rr, c.cx - c.fx * yy / rr, c.fx * w / (2 * rr)
        return np.array([max((uc - hw).min(), 0), max((vb - c.fy * h / rr).min(), 0), min((uc + hw).max(), W), min(vb.max(), H)])
    R, t = np.asarray(cam.R, float), np.asarray(cam.t, float)
    zg = float(t[2]) - float(cam.cam_h)                      # ground height in the ego frame
    us, vs = [], []
    for r_, y_ in zip(rr, yy):
        for dy in (-w / 2, w / 2):
            for dz in (0.0, h):
                pc = (np.array([r_, y_ + dy, zg + dz]) - t) @ R
                if pc[2] <= 0.1:
                    continue
                us.append(c.cx + c.fx * pc[0] / pc[2]); vs.append(c.cy + c.fy * pc[1] / pc[2])
    if not us:
        return None
    return np.array([max(min(us), 0), max(min(vs), 0), min(max(us), W), min(max(vs), H)])


def box_crop(b, W, H, ctx=2.0):
    """A crop centred on a box, ctx times its size, clipped to the image."""
    x1, y1, x2, y2 = b
    cw, ch = (x2 - x1) * ctx, (y2 - y1) * ctx
    return np.array([max((x1 + x2) / 2 - cw / 2, 0), max((y1 + y2) / 2 - ch / 2, 0), min((x1 + x2) / 2 + cw / 2, W),
                     min((y1 + y2) / 2 + ch / 2, H)])


def centres_in(xyxy, crops):
    xyxy = np.asarray(xyxy, float).reshape(-1, 4)
    m = np.zeros(len(xyxy), bool)
    if not len(xyxy) or not len(crops):
        return m
    u, w = (xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2
    for c in crops:
        m |= (u >= c[0]) & (u <= c[2]) & (w >= c[1]) & (w <= c[3])
    return m


def covered(crop, crops, tol=0.95):
    """Is `crop` (almost entirely) inside one of the earlier crops?"""
    if not len(crops):
        return False
    x1, y1, x2, y2 = crop
    area = max((x2 - x1) * (y2 - y1), 1e-6)
    return max(max(0.0, min(x2, c[2]) - max(x1, c[0])) * max(0.0, min(y2, c[3]) - max(y1, c[1])) for c in crops) / area >= tol
