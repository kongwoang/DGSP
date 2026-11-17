"""Metrics on per-frame records (scored frames only): braking episodes (precision / recall / F1, missed, false runs),
agreement with pi(F) (Agr. on all frames, Rep. on frames where the draft disagrees with F, Trans. on frames where F's
action changes), px, recovery of relevant objects the draft missed; paired unit bootstrap (2,000 draws, seed 0)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from dgsp.lift import box_iou

COMPAT = {"vehicle": {"vehicle"}, "person": {"person", "cyclist"}, "cyclist": {"cyclist", "person"}}


def matched(gt_boxes, gt_cls, det_xyxy, det_cls, thr=0.3):
    """Per annotated box: index of the matched detection (class-compatible, IoU >= thr, Hungarian) or -1."""
    res = -np.ones(len(gt_boxes), int)
    if not len(gt_boxes) or not len(det_xyxy):
        return res
    M = box_iou(np.asarray(gt_boxes, float), np.asarray(det_xyxy, float))
    for i, c in enumerate(gt_cls):
        for j, dc in enumerate(det_cls):
            if str(dc) not in COMPAT.get(str(c), {str(c)}):
                M[i, j] = 0
    for i, j in zip(*linear_sum_assignment(-M)):
        if M[i, j] >= thr:
            res[i] = j
    return res


def bootstrap(units):
    """Unit resampling counts: (sorted units, (2000, n_units) multiplicities), seed 0."""
    u = np.unique(units)
    draws = np.random.default_rng(0).integers(0, len(u), (2000, len(u)))
    return u, np.stack([np.bincount(x, minlength=len(u)) for x in draws]).astype(float)


def runs(mask):
    """Maximal runs of True after merging gaps of at most one frame: (start, length)."""
    idx = np.flatnonzero(np.asarray(mask, bool))
    if not len(idx):
        return []
    out, s, e = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - e <= 2:
            e = i
        else:
            out.append((s, e - s + 1)); s = e = i
    out.append((s, e - s + 1))
    return out


def episodes(d, col, gt_col="act_gt", level=1):
    """Per unit: required episodes (runs of required action >= level, one-frame gaps merged), missed ones (no frame of
    the run with the method's action >= level) and false runs (runs of action >= level with no required frame)."""
    out = []
    for u, g in d.groupby("unit", sort=True):
        gt, a = g[gt_col].to_numpy(int), g[col].to_numpy(int)
        ep = runs(gt >= level)
        out.append({"unit": u, "frames": len(g), "episodes": len(ep), "missed": sum(not (a[s:s + n] >= level).any() for s, n in ep),
                    "false": sum((gt[s:s + n] < level).all() for s, n in runs(a >= level))})
    return pd.DataFrame(out).set_index("unit")


def statistics(d, units, C, a, px, F, gt_col="act_gt", gf=None):
    """Point value and bootstrap replicates of every statistic for one row (action column a, px column px); d holds the
    scored frames with F's action in column F and the flag F_changes."""
    g = d.unit.to_numpy()
    U = lambda v: pd.Series(np.asarray(v, float)).groupby(g).sum().reindex(units, fill_value=0).to_numpy()  # noqa: E731
    q = lambda x, y: (x.sum() / max(y.sum(), 1e-12), (C @ x) / np.maximum(C @ y, 1e-12))                  # noqa: E731
    av, Fv = d[a].to_numpy(int), d[F].to_numpy(int)
    eq, rep = (av == Fv).astype(float), (d["a_draft"].to_numpy(int) != Fv).astype(float)
    tr = d["F_changes"].to_numpy(float)          # F's action differs from its previous frame (set before frame selection)
    e = episodes(d, a, gt_col).reindex(units)
    fn, ep, fp = (e[c].fillna(0).to_numpy(float) for c in ("missed", "episodes", "false"))
    tp, n = ep - fn, U(np.ones(len(d)))
    out = {"px": q(U(d[px]), n)}
    if gf is not None and gf in d:
        out["GF"] = q(U(d[gf]), n)
    return {**out, "Agr.": q(U(eq), n), "Rep.": q(U(eq * rep), U(rep)), "Trans.": q(U(eq * tr), U(tr)),
            "Precision": q(tp, tp + fp), "Recall": q(tp, tp + fn), "F1": q(2 * tp, 2 * tp + fp + fn),
            "FN": (fn.sum(), C @ fn), "FP": (fp.sum(), C @ fp), "False": q(1e3 * fp, n)}


def recovery(d, units, C, found_col, classes=None):
    """Share of relevant annotated objects missed by the draft that the row's final scene contains."""
    num, den = [], []
    for f, fd, cl in zip(d[found_col], d["found_draft"], d["rel_class"]):
        m = ~np.asarray(fd, bool)
        if classes is not None and len(m):
            m &= np.array([str(c) in classes for c in cl], bool)
        num.append(float(np.asarray(f, bool)[m].sum()) if len(m) else 0.0); den.append(float(m.sum()))
    g = pd.DataFrame({"u": d.unit.to_numpy(), "n": num, "d": den}).groupby("u").sum().reindex(units, fill_value=0)
    n_, d_ = g.n.to_numpy(float), g.d.to_numpy(float)
    return n_.sum() / max(d_.sum(), 1e-12), (C @ n_) / np.maximum(C @ d_, 1e-12)


def compare(stats, ref):
    """Rows of (row, statistic, value, ref - row, 95 % interval, mark): dagger = ref significantly higher, * = lower."""
    out = []
    for row, s in stats.items():
        for k, (pt, bs) in s.items():
            r = {"row": row, "statistic": k, "value": float(pt)}
            if row != ref and ref in stats:
                diff = stats[ref][k][1] - bs
                lo, hi = np.percentile(diff[np.isfinite(diff)], [2.5, 97.5])
                r.update({"ref_minus_row": float(stats[ref][k][0] - pt), "ci_lo": lo, "ci_hi": hi,
                          "mark": "†" if lo > 0 else ("*" if hi < 0 else "")})
            out.append(r)
    return out
