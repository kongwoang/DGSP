"""DGSP per frame: draft -> loop (field + support crops, guard, merge; at most `rounds` rounds) -> lifted final scene
(eval/run.py applies the refiner and the planners). Every pipeline keeps its own history (its verified boxes at t-1 / t-2)
for the looming estimate."""
from __future__ import annotations

import numpy as np

from . import planner as P
from .config import DRAFT_CANVAS, Config
from .crops import Detector, Mode, Windows, guarded, keep_confident
from .field import box_crop, centres_in, covered, footprint
from .lift import box_iou, lift as geometry, mono

_EMPTY = {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}


def _nms(full):
    """Overlapping crop detections: keep the more confident of any pair with IoU >= 0.5."""
    if len(full["conf"]) <= 1:
        return full
    iou, keep = box_iou(full["xyxy"], full["xyxy"]), []
    for i in np.argsort(-full["conf"], kind="stable"):
        if not keep or iou[i, keep].max() < 0.5:
            keep.append(i)
    keep = np.sort(np.array(keep))
    return {k: full[k][keep] for k in full}


def merge(draft, full, crops, grown, lift):
    """Every draft object whose centre lies in a crop (or that overlaps a guard re-detection, IoU >= 0.5) is replaced by
    the crop detections; the crop detections are lifted."""
    out = centres_in(draft["xyxy"], crops)
    if len(grown) and len(full["conf"]) and len(draft["xyxy"]):
        out |= box_iou(np.asarray(draft["xyxy"], float), full["xyxy"]).max(1) >= 0.5
    fz = lift(full) if len(full["conf"]) else tuple(np.zeros(0) for _ in range(4))
    return {"xyxy": np.r_[draft["xyxy"][~out], full["xyxy"]], "conf": np.r_[draft["conf"][~out], full["conf"]],
            "coarse": np.r_[draft["coarse"][~out], full["coarse"]],
            "mono": tuple(np.r_[a[~out], b] for a, b in zip(draft["mono"], fz)),
            "from_full": np.r_[np.zeros(int((~out).sum()), bool), np.ones(len(full["conf"]), bool)]}


def run(draft, v, cam, W, H, detect, lift, planner, cfg: Config, seeds=(), support_fn=None):
    """The loop (Algorithm 1). detect(crop) -> (detections, grown crops). Returns scene, crops, grown, rounds, converged
    (False = stopped at the round cap while a further round would still add a crop)."""
    support_fn = support_fn or planner.support
    crops, grown, full = [], [], dict(_EMPTY)
    scene, rounds = dict(draft), 0

    def requests(scene, first):
        new = []
        if cfg.field:
            fc = footprint(planner, scene["mono"], v, cam, W, H, cfg.closing_speed, cfg.footprint, cfg.r_grid)
            if fc is not None and not covered(fc, crops):
                new.append(fc)
        if first:
            new += [c for c in seeds if not covered(c, crops + grown + new)]
        for k in support_fn(scene["mono"], v):
            bc = box_crop(scene["xyxy"][k], W, H, cfg.context)
            if not covered(bc, crops + grown + new):
                new.append(bc)
        return new

    converged = False
    for _ in range(cfg.rounds):
        new = requests(scene, rounds == 0)
        if not new:
            converged = True
            break
        rounds += 1
        for c in new:
            d, ex = detect(c)
            grown.extend(ex)
            full = {k: np.concatenate([full[k], np.asarray(d[k])]) for k in full}
        crops.extend(new)
        full = _nms(full)
        scene = merge(draft, full, crops, grown, lift)
    if not converged:
        converged = not requests(scene, False)
    scene.setdefault("from_full", np.zeros(len(scene["conf"]), bool))
    return scene, crops, grown, rounds, converged


class Pipeline:
    """One pipeline over the frames of a unit. planner: "braking" | "ttc" | "curved" (curvature passed to step)."""

    def __init__(self, cfg: Config, dataset: str, detector: Detector, looming, planner: str = "braking", params: P.Params = P.P0):
        self.cfg, self.dataset, self.det, self.looming = cfg, dataset, detector, looming
        self.kind, self.p = planner, params
        self.braking = P.Braking(params, cfg.margin)
        h, w = DRAFT_CANVAS[dataset]
        self.draft_mode = Mode(f"u320_{h}x{w}", h, w)
        self.win = Windows(detector, dataset, cfg.window_long_side)
        self.new_unit(None)

    def new_unit(self, cam):
        self.cam, self.prev_draft, self.hist, self.imgs, self.track = cam, None, {"s1": None, "s2": None}, [], []
        self.win.lb = None

    def planner_for(self, k=0.0):
        if self.kind == "ttc":
            return P.TTCWarning(self.p, self.cfg.margin)
        if self.kind == "curved":
            return P.CurvedBraking(k, self.p)
        return self.braking

    def draft(self, img):
        d = keep_confident(self.det.detect(img, self.draft_mode))
        d["mono"] = mono(d, self.prev_draft, self.cam)
        self.prev_draft = {"xyxy": d["xyxy"], "coarse": d["coarse"]}
        return {k: d[k] for k in ("xyxy", "conf", "coarse", "mono")}

    def retro(self, base_lift, W, H):
        """Before the looming estimate: a crop detection in the ego corridor within retro_range with no same-scale match
        (IoU >= 0.3) in the history s1 is re-detected on the same window of the previous frame's letterbox; the best
        match (IoU >= 0.3) extends s1, so the object gets a 3-observation chain one frame earlier."""
        cam, p, cfg = self.cam, self.p, self.cfg

        def f(full):
            X = np.asarray(full["xyxy"], float).reshape(-1, 4)
            if len(X) and self.win.prev_lb is not None:
                g = geometry(X, np.asarray(full["coarse"]), None, cam)
                sel = P.in_corridor(g["lat_min"], g["lat_max"], p.corridor_half_w) & (np.asarray(g["z"], float) <= cfg.retro_range)
                s1 = np.zeros((0, 4)) if self.hist["s1"] is None else np.asarray(self.hist["s1"], float).reshape(-1, 4)
                add = []
                for b, s in zip(X, sel):
                    if not s or (len(s1) and box_iou(b[None], s1).max() >= 0.3):
                        continue
                    d = self.win.previous(box_crop(b, W, H, 2.0))
                    if len(d["conf"]):
                        io = box_iou(b[None], d["xyxy"])[0]
                        if io.max() >= 0.3:
                            add.append(d["xyxy"][int(np.argmax(io))])
                if add:
                    self.hist["s1"] = np.vstack([s1, np.asarray(add, float)])
            return base_lift(full)
        return f

    def cannot_matter(self, b, rect, W, H):
        """Filtered guard: a box truncated only at a side / top edge that lies beyond the range limit or wholly outside
        the corridor on the side away from that edge cannot change the action."""
        tol, c, cam, p = 2.0, self.cam.calib, self.cam, P.P0         # nominal planner limits in every setting
        e_l = b[0] - rect[0] < tol and rect[0] > tol
        e_r = rect[2] - b[2] < tol + 1 and rect[2] < W - tol - 1
        if rect[3] - b[3] < tol + 1 and rect[3] < H - tol - 1:
            return False
        R = np.asarray(cam.R, float)
        D = -R[2, 0] * ((c.fy / c.fx) * (0.5 * (b[0] + b[2]) - c.cx)) - R[2, 1] * (b[3] - c.cy) - R[2, 2] * c.fy
        if D < 30:
            return False
        g = geometry(np.asarray(b, float)[None], np.array(["vehicle"]), None, cam)
        z, lo, hi, hw = float(g["z"][0]), float(g["lat_min"][0]), float(g["lat_max"][0]), p.corridor_half_w
        return bool(z > p.max_range or (lo > hw and not e_r) or (hi < -hw and not e_l))

    def step(self, img, v, k=0.0, draft=None):
        """One frame. Returns the final scene, crops, guard re-detections, rounds, px."""
        cam, cfg, (H, W) = self.cam, self.cfg, img.shape[:2]
        self.win.new_frame(img)
        draft = draft if draft is not None else self.draft(img)
        cons = self.planner_for(k)
        lift = self.looming.lifter(self.hist, img, self.imgs, cam)
        if cfg.retro:
            lift = self.retro(lift, W, H)
        if cfg.guard is None:
            detect = lambda c: (self.win(c), [])  # noqa: E731
            skipped = set()
        else:
            detect = guarded(self.win, self.win.native, W, H, cfg.max_grow,
                             (lambda b, r: self.cannot_matter(b, r, W, H)) if cfg.guard == "filtered" else None)
            skipped = detect.skipped
        support_fn = {"support": cons.support, "pivots": self.braking.pivots, "none": lambda s, v_: np.zeros(0, int)}[cfg.support]
        seeds = [box_crop(b, W, H, cfg.context) for b in self.track] if cfg.track else []
        scene, crops, grown, rounds, conv = run(draft, v, cam, W, H, detect, lift, cons, cfg, seeds, support_fn)
        out = {"scene": scene, "crops": crops, "grown": grown, "rounds": rounds, "converged": conv,
               "px": (self.draft_mode.net_h * self.draft_mode.net_w + self.win.px) / (W * H)}
        ff = np.asarray(scene["from_full"], bool)
        self.hist["s2"], self.hist["s1"] = self.hist["s1"], np.asarray(scene["xyxy"], float).reshape(-1, 4)[ff]
        if skipped and len(self.hist["s1"]) and self.kind != "ttc":   # boxes kept by the filtered guard seed no looming chain
            self.hist["s1"] = self.hist["s1"][np.array([tuple(b.tolist()) not in skipped for b in self.hist["s1"]], bool)]
        self.track = [np.asarray(scene["xyxy"][j], float) for j in self.braking.support(scene["mono"], v) if ff[j]]
        self.imgs = (self.imgs + [img])[-2:]
        return out
