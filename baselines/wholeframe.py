"""Whole-frame baselines: one detector pass on the frame letterboxed to a long side (u320 ... native), the same lift
(looming estimate with the row's own history) and refiner. u960 is F, the reference whose action defines pi(F).
'u320+' = the draft's detections lifted like a verified pass (no new pixels)."""
from __future__ import annotations

import numpy as np

from dgsp.config import DRAFT_CANVAS, NATIVE_LONG
from dgsp.crops import Mode, keep_confident, wholeframe_mode


class WholeFrame:
    def __init__(self, dataset, detector, looming, long_side: int | str):
        self.dataset, self.det, self.looming, self.long_side = dataset, detector, looming, long_side
        self.new_unit(None)

    def new_unit(self, cam):
        self.cam, self.hist, self.imgs = cam, {"s1": None, "s2": None}, []

    def mode(self, W, H):
        if self.long_side in ("u320+", 320):
            h, w = DRAFT_CANVAS[self.dataset]
            return Mode(f"u320_{h}x{w}", h, w)
        L = NATIVE_LONG[self.dataset] if self.long_side == "native" else int(self.long_side)
        return wholeframe_mode(self.dataset, L, W, H)

    def step(self, img, v, draft=None):
        """Returns the final scene (all objects verified at this canvas) and px. 'u320+' reuses the draft detections."""
        H, W = img.shape[:2]
        md = self.mode(W, H)
        d = draft if (self.long_side == "u320+" and draft is not None) else keep_confident(self.det.detect(img, md))
        d = {k: d[k] for k in ("xyxy", "conf", "coarse")}
        scene = dict(d, mono=self.looming.lifter(self.hist, img, self.imgs, self.cam)(d), from_full=np.ones(len(d["conf"]), bool))
        self.hist["s2"], self.hist["s1"] = self.hist["s1"], np.asarray(d["xyxy"], float).reshape(-1, 4)
        self.imgs = (self.imgs + [img])[-2:]
        return {"scene": scene, "px": md.net_h * md.net_w / (W * H)}
